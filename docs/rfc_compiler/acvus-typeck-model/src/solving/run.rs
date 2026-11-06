//! The loop: judge, record what was decided, fall back at the fixpoint.

use std::collections::BTreeMap;

use crate::coerce::coerce;
use crate::decl::{Decls, Node};
use crate::read::ReadValue;
use crate::ty::HasTerm;
use crate::join::le;
use crate::lit::Family;
use crate::program::Projection;
use crate::ty::{Constructor, FloatTy, IntTy, Member, Mutability, Name, PartKey, Scalar, Term, Ty};

use super::judge::{Head, Judged, judge, literal_admits};
use super::read::read_value;
use super::record::{ClassId, ClassName, HintDecision, Record, Refusal, Relation, Relations, Requirement};
use super::{choice, hint};

fn skeleton(record: &mut Record, decls: &Decls, root: ClassId, head: &Head) -> Term<ClassId> {
    let mut part = |key: PartKey| record.classes.part(root, key);
    let named = |members: &std::collections::BTreeSet<Member>| -> Vec<Name> {
        members
            .iter()
            .map(|member| match member {
                Member::Field(name) | Member::Variant(name) => name.clone(),
            })
            .collect()
    };
    match &head.constructor {
        Constructor::Ref(mutability) => Term::Ref(*mutability, part(PartKey::Target)),
        Constructor::Tuple(n) => Term::Tuple((0..*n).map(|i| part(PartKey::Component(i))).collect()),
        Constructor::Array(n) => Term::Array(*n, part(PartKey::Element)),
        Constructor::Slice => Term::Slice(part(PartKey::Element)),
        Constructor::Option => Term::Option(part(PartKey::Payload(Name::new(crate::ty::SOME)))),
        Constructor::Result => Term::Result {
            ok: part(PartKey::Payload(Name::new(crate::ty::OK))),
            err: part(PartKey::Payload(Name::new(crate::ty::ERR))),
        },
        Constructor::Function(arity) => Term::FunctionValue { arity: *arity },
        Constructor::Object => Term::Object(
            named(&head.members)
                .into_iter()
                .map(|field| (field.clone(), part(PartKey::Field(field))))
                .collect(),
        ),
        Constructor::Enum(name) => Term::Enum(
            name.clone(),
            named(&head.members)
                .into_iter()
                .map(|variant| (variant.clone(), part(PartKey::Payload(variant))))
                .collect(),
        ),
        Constructor::ErrorOf => Term::ErrorOf(part(PartKey::Wrapped)),
        Constructor::Struct(module) => {
            let fields = decls
                .types()
                .struct_decl(module)
                .map_or(0, |decl| decl.fields.len());
            Term::Struct(module.clone(), (0..fields).map(|i| part(PartKey::Arg(i))).collect())
        }
        Constructor::Extern(name) => {
            let decl = decls.types().extern_decl(name);
            Term::Extern {
                name: name.clone(),
                args: (0..decl.map_or(0, |d| d.arity))
                    .map(|i| part(PartKey::Arg(i)))
                    .collect(),
                identity: decl
                    .is_some_and(|d| d.identity)
                    .then(|| part(PartKey::Identity)),
            }
        }
        Constructor::Scalar(scalar) => Term::Scalar(*scalar),
        Constructor::String => Term::String,
        Constructor::Str => Term::Str,
        Constructor::Unit => Term::Unit,
        Constructor::Never => Term::Never,
        Constructor::Identity(name) => Term::Identity(name.clone()),
    }
}

fn satisfies(
    decls: &Decls,
    requirement: &Requirement,
    structure: &Term<ClassId>,
) -> Option<Option<ClassId>> {
    let constructor = structure.constructor();
    if constructor == Constructor::Never {
        return Some(None);
    }
    let part = match (&requirement.projection, &constructor) {
        (Projection::Deref, Constructor::Ref(_)) => structure.part(&PartKey::Target),
        (Projection::Field(field), Constructor::Object) => {
            structure.part(&PartKey::Field(field.clone()))
        }
        (Projection::Field(field), Constructor::Struct(module)) => {
            let decl = decls.types().struct_decl(module)?;
            let i = decl.fields.iter().position(|f| f == field)?;
            structure.part(&PartKey::Arg(i))
        }
        (Projection::Component(i), Constructor::Tuple(n)) if i < n => {
            structure.part(&PartKey::Component(*i))
        }
        (Projection::Elem, Constructor::Array(_) | Constructor::Slice) => {
            structure.part(&PartKey::Element)
        }
        (Projection::Payload { variant, .. }, Constructor::Option | Constructor::Result) => {
            let key = constructor.payload_part(variant)?;
            structure.part(&key)
        }
        (Projection::Payload { variant, .. }, Constructor::Enum(_)) => {
            structure.part(&PartKey::Payload(variant.clone()))
        }
        _ => return None,
    };
    part.copied().map(Some)
}

fn take(record: &mut Record, decls: &Decls, root: ClassId, head: &Head) {
    let structure = skeleton(record, decls, root, head);
    let requirements: Vec<Requirement> = record.requirements(root).into_iter().cloned().collect();
    for requirement in requirements {
        match satisfies(decls, &requirement, &structure) {
            Some(Some(part)) => {
                record.classes.equate(requirement.part, part);
            }
            Some(None) => {}
            None => {
                record.refused.insert(root, Refusal::Conflict);
                return;
            }
        }
    }
    record.solved.insert(root, structure);
}

fn check_requirements(record: &mut Record, decls: &Decls) -> bool {
    let mut pairs: Vec<(ClassId, ClassId)> = Vec::new();
    let mut conflicts: Vec<ClassId> = Vec::new();
    for relation in &record.relations {
        let Relation::Requires { class, requirement } = relation else {
            continue;
        };
        let root = record.root(*class);
        let Some(structure) = record.structure(root) else {
            continue;
        };
        match satisfies(decls, requirement, structure) {
            Some(Some(part)) => pairs.push((requirement.part, part)),
            Some(None) => {}
            None => conflicts.push(root),
        }
    }
    let mut changed = false;
    for root in conflicts {
        changed |= record.refused.insert(root, Refusal::Conflict).is_none();
    }
    for (a, b) in pairs {
        changed |= record.classes.equate(a, b);
    }
    changed
}

fn written_of_ty(record: &mut Record, maker: &Name, ty: &Ty) -> ClassId {
    let structure = ty
        .term()
        .clone()
        .map(|_, part| written_of_ty(record, maker, &part));
    let class = record.classes.fresh(maker);
    record.relations.push(Relation::Written { class, structure });
    class
}

// A value passing into a written type reaches it by a coercion, and the
// coercion gives the written type's variables their values. Where the value
// has a constructor, it is raised along the casts to the written type's
// node and matched part by part; a variable meets the part it stands for.
fn bind_coercion(
    record: &Record,
    decls: &Decls,
    value: &ReadValue,
    written: ClassId,
    bindings: &mut Vec<(ClassId, Ty)>,
) -> Result<(), ()> {
    let written = record.root(written);
    let Some(structure) = record.structure(written).cloned() else {
        if let Some(ty) = value.to_ty() {
            bindings.push((written, ty));
        }
        return Ok(());
    };
    let Some(term) = value.decided() else {
        return Ok(());
    };
    let raised: ReadValue = if term.constructor() == structure.constructor() {
        value.clone()
    } else {
        let node = match &structure {
            Term::Ref(m, target) => {
                let Some(target) = record.structure(*target) else {
                    return Ok(());
                };
                Node::Ref(*m, target.constructor())
            }
            other => Node::Owned(other.constructor()),
        };
        match decls.raise(value, &node) {
            Some((_, raised)) => raised,
            None => return Err(()),
        }
    };
    let Some(raised_term) = raised.decided() else {
        return Ok(());
    };
    for (key, part) in raised_term.parts() {
        let Some(other) = structure.part(&key) else {
            continue;
        };
        bind_coercion(record, decls, part, *other, bindings)?;
    }
    Ok(())
}

fn bind_coercions(record: &mut Record, decls: &Decls) -> bool {
    let mut bindings: Vec<(ClassId, Ty)> = Vec::new();
    let mut failed: Vec<ClassId> = Vec::new();
    let relations = Relations::of(record);
    for relation in &record.relations {
        let Relation::Into { value, written } = relation else {
            continue;
        };
        if record.is_refused(*value) || record.is_refused(*written) {
            continue;
        }
        let read = read_value(record, &relations, *value);
        if bind_coercion(record, decls, &read, *written, &mut bindings).is_err() {
            failed.push(record.root(*value));
        }
    }
    let mut changed = false;
    for root in failed {
        changed |= record.refused.insert(root, Refusal::Conflict).is_none();
    }
    for (class, ty) in bindings {
        if record.is_decided(class) {
            continue;
        }
        let maker = Name::new("coercion");
        let holder = written_of_ty(record, &maker, &ty);
        changed |= record.classes.equate(class, holder);
    }
    changed
}

fn unify_written(record: &mut Record) -> bool {
    let mut pairs: Vec<(ClassId, ClassId)> = Vec::new();
    let mut conflicts: Vec<ClassId> = Vec::new();
    for root in record.classes.roots() {
        if record.is_refused(root) {
            continue;
        }
        let written = record.written(root);
        let Some(first) = written.first() else {
            continue;
        };
        for other in &written[1..] {
            if Head::of(first) != Head::of(other) {
                conflicts.push(root);
                break;
            }
            for (key, part) in first.parts() {
                if let Some(same) = other.part(&key) {
                    pairs.push((*part, *same));
                }
            }
        }
    }
    let mut changed = false;
    for root in conflicts {
        changed |= record.refused.insert(root, Refusal::Conflict).is_none();
    }
    for (a, b) in pairs {
        changed |= record.classes.equate(a, b);
    }
    changed
}

fn decompose(record: &mut Record) -> bool {
    let mut pairs: Vec<(ClassId, ClassId)> = Vec::new();
    for relation in Relations::of(record).all.iter().cloned() {
        let Relation::Enters { lower, upper, .. } = relation else {
            continue;
        };
        let (Some(below), Some(above)) = (record.structure(lower), record.structure(upper)) else {
            continue;
        };
        let constructor = below.constructor();
        if constructor != above.constructor() || constructor.is_structural() {
            continue;
        }
        let above: BTreeMap<PartKey, ClassId> =
            above.parts().into_iter().map(|(k, p)| (k, *p)).collect();
        for (key, part) in below.parts() {
            if let Some(other) = above.get(&key) {
                pairs.push((*part, *other));
            }
        }
    }
    let mut changed = false;
    for (a, b) in pairs {
        changed |= record.classes.equate(a, b);
    }
    changed
}

fn judge_all(record: &mut Record, decls: &Decls) -> bool {
    let roots: Vec<ClassId> = record
        .classes
        .roots()
        .filter(|&root| !record.is_decided(root) && !record.is_refused(root))
        .collect();
    let relations = Relations::of(record);
    let judged: Vec<(ClassId, Judged)> = roots
        .into_iter()
        .map(|root| (root, judge(record, &relations, root)))
        .collect();
    // A decision by occurrences is recorded only in a round without a decision
    // by lower bounds: an occurrence read as undecided inside the component may
    // be decided by its own lower bounds in this very round.
    let by_lower = judged.iter().any(|(_, judged)| {
        matches!(
            judged,
            Judged::Head { by_lower: true, .. } | Judged::Refuse(_)
        )
    });
    let mut changed = false;
    for (root, judged) in judged {
        match judged {
            Judged::Head { head, by_lower: true } => {
                take(record, decls, root, &head);
                changed = true;
            }
            Judged::Head {
                head,
                by_lower: false,
            } if !by_lower => {
                take(record, decls, root, &head);
                changed = true;
            }
            Judged::Refuse(refusal) => {
                record.refused.insert(root, refusal);
                changed = true;
            }
            Judged::Head { .. } | Judged::Wait { .. } | Judged::Nothing => {}
        }
    }
    changed
}

fn family_default(family: Family) -> Constructor {
    Constructor::Scalar(match family {
        Family::Int => Scalar::Int(IntTy::I32),
        Family::Float => Scalar::Float(FloatTy::F64),
    })
}

fn fall_back(record: &mut Record, decls: &Decls) -> bool {
    let relations = Relations::of(record);
    let undecided: Vec<ClassId> = record
        .classes
        .roots()
        .filter(|&root| !record.is_decided(root) && !record.is_refused(root))
        .collect();
    let with_literals: Vec<ClassId> = undecided
        .iter()
        .copied()
        .filter(|&root| record.lower_bounds(&relations, root).is_empty() && !record.literals(root).is_empty())
        .filter(|&root| {
            matches!(
                judge(record, &relations, root),
                Judged::Nothing | Judged::Wait { occurrence_only: true }
            )
        })
        .collect();
    if !with_literals.is_empty() {
        for root in with_literals {
            let literals = record.literals(root);
            let family = literals[0].family();
            let default = family_default(family);
            let fits = literals.iter().all(|lit| literal_admits(lit, &default));
            if fits {
                take(record, decls, root, &Head::bare(default));
            } else if literals.iter().all(|lit| lit.family() == family) {
                record.refused.insert(root, Refusal::Conflict);
            } else {
                record.refused.insert(root, Refusal::NoJoin);
            }
        }
        return true;
    }
    let mut changed = false;
    for root in undecided {
        if record.may_require(root) {
            continue;
        }
        if !record.requirements(root).is_empty() && record.is_empty(&relations, root) {
            record.refused.insert(root, Refusal::NoValue);
            changed = true;
        } else if record.is_empty(&relations, root) {
            take(record, decls, root, &Head::bare(Constructor::Never));
            changed = true;
        }
    }
    changed
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub values: BTreeMap<Name, Option<Ty>>,
    pub refusals: BTreeMap<Name, Refusal>,
    pub decisions: BTreeMap<Name, Option<String>>,
    pub failing: Vec<String>,
    pub accepted: bool,
}

fn decision_text(decision: &HintDecision) -> String {
    let reference = |m: Mutability| match m {
        Mutability::Shared => "&",
        Mutability::Exclusive => "&mut",
    };
    match decision {
        HintDecision::Place => "place".to_string(),
        HintDecision::Through { mutability } => format!("through {}", reference(*mutability)),
        HintDecision::Applied => "applied".to_string(),
        HintDecision::AppliedToTarget => "applied through".to_string(),
        HintDecision::Read => "read".to_string(),
    }
}

pub fn value(record: &Record, id: ClassId) -> Option<Ty> {
    fn value_in(record: &Record, id: ClassId, visiting: &mut Vec<ClassId>) -> Option<Ty> {
        let root = record.root(id);
        if visiting.contains(&root) {
            return None;
        }
        visiting.push(root);
        let structure = record.structure(root)?.clone();
        let term = structure
            .try_map(|_, part| value_in(record, part, visiting).ok_or(()))
            .ok()?;
        visiting.pop();
        Ty::new(term).ok()
    }
    value_in(record, id, &mut Vec::new())
}

pub fn solve(mut record: Record, decls: &Decls) -> (Record, Outcome) {
    loop {
        loop {
            let unified = unify_written(&mut record)
                | check_requirements(&mut record, decls)
                | bind_coercions(&mut record, decls);
            let decomposed = decompose(&mut record);
            let hinted = hint::decide(&mut record, decls);
            let narrowed = choice::narrow(&mut record, decls) | choice::never(&mut record);
            let judged = judge_all(&mut record, decls);
            if !unified && !decomposed && !hinted && !narrowed && !judged {
                break;
            }
        }
        if !fall_back(&mut record, decls) {
            break;
        }
    }
    let outcome = outcome(&record, decls);
    (record, outcome)
}

fn outcome(record: &Record, decls: &Decls) -> Outcome {
    let mut values = BTreeMap::new();
    let mut refusals = BTreeMap::new();
    for id in (0..record.classes.len()).map(ClassId) {
        let ClassName::Text(name) = record.classes.name(id) else {
            continue;
        };
        values.insert(name.clone(), value(record, id));
        if let Some(refusal) = record.refused.get(&record.root(id)) {
            refusals.insert(name.clone(), refusal.clone());
        }
    }
    let mut failing = Vec::new();
    for relation in Relations::of(record).all.iter().cloned() {
        let (lower, upper, into_written) = match &relation {
            Relation::Enters { lower, upper, .. } => (*lower, *upper, false),
            Relation::Into { value, written } => (*value, *written, true),
            _ => continue,
        };
        let (Some(s), Some(t)) = (value(record, lower), value(record, upper)) else {
            failing.push(format!("{relation:?}"));
            continue;
        };
        let holds = if into_written {
            coerce(decls, &s, &t).is_ok()
        } else {
            le(decls, &s, &t)
        };
        if !holds {
            failing.push(format!("{relation:?}"));
        }
    }
    let mut decisions: BTreeMap<Name, Option<String>> = BTreeMap::new();
    let relations = Relations::of(record);
    for choice in &record.choices {
        let chosen = choice::chosen(record, &relations, choice).map(|chosen| match chosen {
            choice::Chosen::Candidate(id) => format!("{}#{}", id.name.as_str(), id.overload),
            choice::Chosen::Never => "never".to_string(),
        });
        decisions.insert(choice.name.clone(), chosen);
    }
    for hint in &record.hints {
        decisions.insert(hint.name.clone(), hint.decision.as_ref().map(decision_text));
    }
    let every_class_decided = record
        .classes
        .roots()
        .all(|root| record.is_decided(root));
    let every_decision_made = decisions.values().all(Option::is_some);
    let accepted = record.refused.is_empty()
        && every_class_decided
        && every_decision_made
        && failing.is_empty();
    Outcome {
        values,
        refusals,
        decisions,
        failing,
        accepted,
    }
}
