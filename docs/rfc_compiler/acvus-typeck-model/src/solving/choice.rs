//! A use of an overloaded name: what its candidates share is recorded at
//! once, and the candidates are read off the classes at its positions.

use std::collections::{BTreeMap, BTreeSet};

use crate::coerce::{coerce_read, le_read};
use crate::decl::Decls;
use crate::disjunction::generalization::Generalization;
use crate::read::ReadValue;
use crate::sig::{SigTy, SigVar, Signature};
use crate::ty::Name;

use super::read::{read_value, read_value_for};
use super::record::{Choice, ClassId, Record, Refusal, Relation, Relations};

fn instantiate(
    record: &mut Record,
    maker: &Name,
    ty: &SigTy,
    vars: &mut BTreeMap<SigVar, ClassId>,
) -> ClassId {
    match ty {
        SigTy::Var(var) => *vars
            .entry(*var)
            .or_insert_with(|| record.classes.fresh(maker)),
        SigTy::Term(term) => {
            let structure = (**term)
                .clone()
                .map(|_, part| instantiate(record, maker, &part, vars));
            let class = record.classes.fresh(maker);
            record.relations.push(Relation::Written { class, structure });
            class
        }
    }
}

fn passes(record: &mut Record, value: ClassId, into: ClassId, into_written: bool) {
    record.relations.push(if into_written {
        Relation::Into {
            value,
            written: into,
        }
    } else {
        Relation::Enters {
            lower: value,
            upper: into,
            part: false,
        }
    });
}

fn record_signature(
    record: &mut Record,
    maker: &Name,
    params: &[SigTy],
    result: &SigTy,
    args: &[ClassId],
    into: ClassId,
    vars: &mut BTreeMap<SigVar, ClassId>,
) {
    for (arg, param) in args.iter().zip(params) {
        let class = instantiate(record, maker, param, vars);
        passes(record, *arg, class, matches!(param, SigTy::Term(_)));
    }
    let class = instantiate(record, maker, result, vars);
    passes(record, class, into, false);
}

pub fn record_use(
    record: &mut Record,
    decls: &Decls,
    name: &Name,
    callee: &Name,
    args: Vec<ClassId>,
    result: ClassId,
) {
    let candidates: Vec<&Signature> = decls
        .candidates(callee)
        .iter()
        .filter(|decl| decl.sig.params().len() == args.len())
        .map(|decl| &decl.sig)
        .collect();
    let ids = decls
        .candidates(callee)
        .iter()
        .filter(|decl| decl.sig.params().len() == args.len())
        .map(|decl| decl.id.clone())
        .collect();
    let mut vars: BTreeMap<SigVar, ClassId> = BTreeMap::new();
    let (generalized_vars, result_var_indices) = match candidates.as_slice() {
        [] => (Vec::new(), Vec::new()),
        [one] => {
            record_signature(record, name, one.params(), one.result(), &args, result, &mut vars);
            (Vec::new(), Vec::new())
        }
        several => {
            let generalization = Generalization::of(several);
            record_signature(
                record,
                name,
                &generalization.params,
                &generalization.result,
                &args,
                result,
                &mut vars,
            );
            let generalized: Vec<ClassId> = (0..generalization.vars.len())
                .map(|k| {
                    *vars
                        .entry(SigVar::Type(k))
                        .or_insert_with(|| record.classes.fresh(name))
                })
                .collect();
            let in_result = (0..generalization.vars.len())
                .filter(|&k| generalization.writes_in_result(k))
                .collect();
            (generalized, in_result)
        }
    };
    let remaining = (0..candidates.len()).collect();
    record.choices.push(Choice {
        name: name.clone(),
        candidates: ids,
        remaining,
        args,
        result,
        vars: generalized_vars,
        result_var_indices,
    });
}

fn signatures<'a>(decls: &'a Decls, choice: &Choice) -> Vec<&'a Signature> {
    choice
        .candidates
        .iter()
        .map(|id| &decls.decl(id).sig)
        .collect()
}

pub enum Chosen<'a> {
    Candidate(&'a crate::decl::DeclId),
    Never,
}

fn never_argument(record: &Record, relations: &Relations, choice: &Choice) -> bool {
    choice.args.iter().any(|&arg| {
        matches!(read_value(record, relations, arg), ReadValue::Term(term) if matches!(*term, crate::ty::Term::Never))
    })
}

pub fn chosen<'a>(record: &Record, relations: &Relations, choice: &'a Choice) -> Option<Chosen<'a>> {
    match choice.remaining.iter().collect::<Vec<_>>().as_slice() {
        [one] => Some(Chosen::Candidate(&choice.candidates[**one])),
        [] => None,
        _ if never_argument(record, relations, choice) => Some(Chosen::Never),
        _ => None,
    }
}

pub fn never(record: &mut Record) -> bool {
    let mut changed = false;
    let relations = Relations::of(record);
    for at in 0..record.choices.len() {
        let choice = record.choices[at].clone();
        if choice.remaining.len() < 2 || !never_argument(record, &relations, &choice) {
            continue;
        }
        if record.is_decided(choice.result) {
            continue;
        }
        let never = record.classes.fresh(&choice.name);
        record.relations.push(Relation::Written {
            class: never,
            structure: crate::ty::Term::Never,
        });
        passes(record, never, choice.result, false);
        changed = true;
    }
    changed
}

pub fn narrow(record: &mut Record, decls: &Decls) -> bool {
    let mut changed = false;
    let relations = Relations::of(record);
    for at in 0..record.choices.len() {
        if record.choices[at].remaining.len() < 2 {
            continue;
        }
        let choice = record.choices[at].clone();
        if never_argument(record, &relations, &choice) {
            continue;
        }
        let sigs = signatures(decls, &choice);
        let generalization = Generalization::of(&sigs);
        let mut remaining = choice.remaining.clone();
        for (k, &var) in choice.vars.iter().enumerate() {
            let value = read_value_for(record, &relations, var, Some(&choice.name));
            if value == ReadValue::Undecided {
                continue;
            }
            let at_param = generalization.writes_in_params(k);
            let at_result = generalization.writes_in_result(k);
            remaining.retain(|&i| {
                let written = ReadValue::of_sig(&generalization.vars[k][i]);
                (!at_param || coerce_read(decls, &value, &written).is_ok())
                    && (!at_result || le_read(decls, &written, &value).is_ok())
            });
        }
        if remaining == choice.remaining {
            continue;
        }
        changed = true;
        record.choices[at].remaining = remaining.clone();
        let left: Vec<usize> = remaining.iter().copied().collect();
        match left.as_slice() {
            [] => {
                for &var in &choice.vars {
                    record.refused.insert(record.root(var), Refusal::NoJoin);
                }
            }
            &[one] => {
                let mut vars: BTreeMap<SigVar, ClassId> = BTreeMap::new();
                for (k, &var) in choice.vars.iter().enumerate() {
                    let candidate = instantiate(record, &choice.name, &generalization.vars[k][one], &mut vars);
                    let written = matches!(generalization.vars[k][one], SigTy::Term(_));
                    if generalization.writes_in_params(k) {
                        passes(record, var, candidate, written);
                    }
                    if generalization.writes_in_result(k) {
                        passes(record, candidate, var, false);
                    }
                }
            }
            _ => {}
        }
    }
    changed
}

pub fn undecided(record: &Record) -> BTreeSet<Name> {
    record
        .choices
        .iter()
        .filter(|choice| choice.remaining.len() != 1)
        .map(|choice| choice.name.clone())
        .collect()
}
