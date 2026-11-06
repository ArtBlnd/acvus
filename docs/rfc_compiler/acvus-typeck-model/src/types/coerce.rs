//! Coercion, the steps of the join's `≤` each taken by a row of the Coercion
//! table, and rank.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::decl::{DeclId, Decls};
use crate::read::ReadValue;
use crate::ty::{HasTerm, Member, Mutability, Name, PartKey, Term, Ty};

/// A stage of a composition, ordered `reborrow ≺ cast ≺ structural`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Stage {
    Reborrow,
    Cast,
    Structural,
}

/// The stages a coercion takes, the structural stage counted down into the
/// parts' coercions, which take no other, and its number of casts;
/// `empty` where it takes none.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Rank {
    stages: BTreeSet<Stage>,
    casts: usize,
}

impl Rank {
    pub fn empty() -> Rank {
        Rank::default()
    }

    pub fn highest(&self) -> Option<Stage> {
        self.stages.last().copied()
    }

    pub fn casts(&self) -> usize {
        self.casts
    }

    fn take(&mut self, stage: Stage) {
        self.stages.insert(stage);
    }

    fn add(&mut self, other: Rank) {
        self.stages.extend(other.stages);
        self.casts += other.casts;
    }

    /// The highest stage, then the number of casts, then the lower stages from
    /// the highest down, a stage taken above one not taken.
    fn key(&self) -> (Option<Stage>, usize, Vec<bool>) {
        let highest = self.highest();
        let lower = [Stage::Structural, Stage::Cast, Stage::Reborrow]
            .into_iter()
            .filter(|stage| Some(*stage) < highest)
            .map(|stage| self.stages.contains(&stage))
            .collect();
        (highest, self.casts, lower)
    }
}

impl Ord for Rank {
    fn cmp(&self, other: &Rank) -> Ordering {
        self.key().cmp(&other.key())
    }
}

impl PartialOrd for Rank {
    fn partial_cmp(&self, other: &Rank) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// How a value is changed into a value of the type it passes into, with the
/// functions called.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Coercion {
    /// None is needed.
    Identity,
    /// A position that decides nothing, which reaches every type.
    Undecided,
    /// `reborrow? ; cast* ; structural?`, taking at least one.
    Composition {
        reborrow: bool,
        casts: Vec<DeclId>,
        structural: Option<Structural>,
    },
    /// `! ⇝ U`.
    Never,
    /// A declared conversion.
    Conversion(DeclId),
}

/// The structural stage.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Structural {
    /// At one constructor: the object fields the type passed into does not
    /// name are dropped, the members it names besides are added, and each part
    /// both have is changed by its coercion. Along `≤`, the fields an object
    /// gains are added and not changed.
    Shape {
        dropped: BTreeSet<Member>,
        added: BTreeSet<Member>,
        parts: Vec<(PartKey, Coercion)>,
    },
}

impl Coercion {
    /// The rank, where the coercion is a composition. A position that decides
    /// nothing, `!` included, adds nothing; a declared conversion has none.
    pub fn rank(&self) -> Option<Rank> {
        match self {
            Coercion::Identity | Coercion::Undecided | Coercion::Never => Some(Rank::empty()),
            Coercion::Conversion(_) => None,
            Coercion::Composition {
                reborrow,
                casts,
                structural,
            } => {
                let mut rank = Rank::empty();
                if *reborrow {
                    rank.take(Stage::Reborrow);
                }
                if !casts.is_empty() {
                    rank.take(Stage::Cast);
                    rank.casts += casts.len();
                }
                if let Some(structural) = structural {
                    rank.add(structural.rank()?);
                }
                Some(rank)
            }
        }
    }
}

impl Structural {
    /// A shape changes nothing where no member changes and no part's
    /// coercion takes a stage, so it then takes no structural stage either.
    fn rank(&self) -> Option<Rank> {
        match self {
            Structural::Shape {
                dropped,
                added,
                parts,
            } => {
                let mut rank = Rank::empty();
                for (_, part) in parts {
                    rank.add(part.rank()?);
                }
                if !dropped.is_empty() || !added.is_empty() || rank.highest().is_some() {
                    rank.take(Stage::Structural);
                }
                Some(rank)
            }
        }
    }
}

/// Where a coercion gives no result.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CoerceError {
    /// No row, and no composition of them, reaches the type, and no declared
    /// conversion goes there.
    None,
    /// More than one composition reaches the type.
    Compositions(Vec<Coercion>),
    /// No composition reaches the type, and more than one declared conversion
    /// goes there.
    Conversions(Vec<DeclId>),
}

/// The order a value is changed along.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Way {
    /// `⇝`, by the Coercion table, conversions included where `conversions`.
    Coercion { conversions: bool },
    /// The join's `≤`, each step taken by a row of the Coercion table.
    Le,
}

/// `from ⇝ to`, a declared conversion taken where no composition reaches.
pub fn coerce(decls: &Decls, from: &Ty, to: &Ty) -> Result<Coercion, CoerceError> {
    walk(
        decls,
        &ReadValue::of_ty(from),
        &ReadValue::of_ty(to),
        Way::Coercion { conversions: true },
    )
}

/// `from ⇝ to` between values a use reads, taking no conversion, as narrowing
/// takes none.
pub fn coerce_read(
    decls: &Decls,
    from: &ReadValue,
    to: &ReadValue,
) -> Result<Coercion, CoerceError> {
    walk(decls, from, to, Way::Coercion { conversions: false })
}

/// `from ≤ to` between values read, a position either leaves undecided
/// deciding nothing, as `along_le` takes it between closed types.
pub fn le_read(decls: &Decls, from: &ReadValue, to: &ReadValue) -> Result<Coercion, CoerceError> {
    walk(decls, from, to, Way::Le)
}

/// How `from` is changed at each step of `from ≤ to`.
pub fn along_le(decls: &Decls, from: &Ty, to: &Ty) -> Result<Coercion, CoerceError> {
    walk(
        decls,
        &ReadValue::of_ty(from),
        &ReadValue::of_ty(to),
        Way::Le,
    )
}

fn walk(
    decls: &Decls,
    from: &ReadValue,
    to: &ReadValue,
    way: Way,
) -> Result<Coercion, CoerceError> {
    if let Some(reached) = deciding_nothing(from, to) {
        return reached;
    }
    let mut starts = vec![(false, from.clone())];
    if let (
        Some(Term::Ref(Mutability::Exclusive, target)),
        Some(Term::Ref(Mutability::Shared, _)),
    ) = (from.decided(), to.decided())
    {
        let shared = ReadValue::from_term(Term::Ref(Mutability::Shared, target.clone()))
            .expect("the target of a reference is never a reference");
        starts.push((true, shared));
    }
    let mut found = Vec::new();
    for (reborrow, start) in starts {
        let mut at = start;
        let mut casts = Vec::new();
        loop {
            if (reborrow || !casts.is_empty()) && agrees(&at, to) {
                found.push(Coercion::Composition {
                    reborrow,
                    casts: casts.clone(),
                    structural: None,
                });
            } else if let Some(structural) = structural(&at, to, way) {
                found.push(Coercion::Composition {
                    reborrow,
                    casts: casts.clone(),
                    structural: Some(structural),
                });
            }
            let Some((cast, up)) = decls.step_up(&at) else {
                break;
            };
            casts.push(cast);
            at = up;
        }
    }
    match found.len() {
        1 => Ok(found.pop().expect("one composition was found")),
        0 => conversion(decls, from, to, way),
        _ => Err(CoerceError::Compositions(found)),
    }
}

/// The result where no stage is taken: the two are equal, `from` is `!`, or
/// a position decides nothing, being undecided on either side, or a domain,
/// which reaches a domain it shares a constructor with and a type of a
/// constructor it admits.
fn deciding_nothing(from: &ReadValue, to: &ReadValue) -> Option<Result<Coercion, CoerceError>> {
    if from == to {
        return Some(Ok(Coercion::Identity));
    }
    match (from, to) {
        (ReadValue::Undecided, _) | (_, ReadValue::Undecided) => Some(Ok(Coercion::Undecided)),
        (ReadValue::Term(term), _) if matches!(**term, Term::Never) => Some(Ok(Coercion::Never)),
        (ReadValue::Domain(_), _) | (_, ReadValue::Domain(_)) if agrees(from, to) => {
            Some(Ok(Coercion::Undecided))
        }
        (ReadValue::Domain(_), _) | (_, ReadValue::Domain(_)) => Some(Err(CoerceError::None)),
        _ if agrees(from, to) => Some(Ok(Coercion::Undecided)),
        _ => None,
    }
}

/// Equal at every position both decide.
fn agrees(a: &ReadValue, b: &ReadValue) -> bool {
    match (a, b) {
        (ReadValue::Undecided, _) | (_, ReadValue::Undecided) => true,
        (ReadValue::Domain(a), ReadValue::Domain(b)) => a.meet(b).is_some(),
        (ReadValue::Domain(domain), ReadValue::Term(term))
        | (ReadValue::Term(term), ReadValue::Domain(domain)) => domain.admits(&term.constructor()),
        (ReadValue::Term(s), ReadValue::Term(t)) => {
            s.constructor() == t.constructor()
                && s.members() == t.members()
                && s.parts()
                    .into_iter()
                    .zip(t.parts())
                    .all(|((_, s), (_, t))| agrees(s, t))
        }
    }
}

fn conversion(
    decls: &Decls,
    from: &ReadValue,
    to: &ReadValue,
    way: Way,
) -> Result<Coercion, CoerceError> {
    if way != (Way::Coercion { conversions: true }) {
        return Err(CoerceError::None);
    }
    let (Some(from), Some(to)) = (from.to_ty(), to.to_ty()) else {
        return Err(CoerceError::None);
    };
    let mut conversions = decls.conversions_between(&from, &to);
    match conversions.len() {
        0 => Err(CoerceError::None),
        1 => Ok(Coercion::Conversion(
            conversions.pop().expect("one conversion was found"),
        )),
        _ => Err(CoerceError::Conversions(conversions)),
    }
}

/// The structural stage from `at` to `to`, where one reaches: the rows
/// `shape` takes.
fn structural(at: &ReadValue, to: &ReadValue, way: Way) -> Option<Structural> {
    let (Some(a), Some(b)) = (at.decided(), to.decided()) else {
        return None;
    };
    shape(a, b, way)
}

/// A part's coercion `⇝ₚ`, or a part's steps along `≤`: the
/// rows `shape` takes and `! ⇝ U` alone, so no reborrow, cast or
/// conversion. None where no row reaches, the only way `deciding_nothing`
/// gives no result.
fn part(x: &ReadValue, y: &ReadValue, way: Way) -> Option<Coercion> {
    if let Some(reached) = deciding_nothing(x, y) {
        return reached.ok();
    }
    let (Some(a), Some(b)) = (x.decided(), y.decided()) else {
        return None;
    };
    shape(a, b, way).map(|structural| Coercion::Composition {
        reborrow: false,
        casts: Vec::new(),
        structural: Some(structural),
    })
}

/// The object, enum and part rows at one structural constructor, each part
/// changed by its `⇝ₚ`, where they reach.
fn shape(a: &Term<ReadValue>, b: &Term<ReadValue>, way: Way) -> Option<Structural> {
    let constructor = a.constructor();
    if constructor != b.constructor() || !constructor.is_structural() {
        return None;
    }
    let (members, dropped, added) = match (a, b, way) {
        (Term::Object(f), Term::Object(g), Way::Coercion { .. }) => {
            let dropped = narrowed(f, g)?;
            (
                shared_parts(f, g, PartKey::Field),
                field_members(dropped),
                BTreeSet::new(),
            )
        }
        (Term::Object(f), Term::Object(g), Way::Le) => {
            let gained = narrowed(g, f)?;
            (
                shared_parts(f, g, PartKey::Field),
                BTreeSet::new(),
                field_members(gained),
            )
        }
        (Term::Enum(_, v), Term::Enum(_, w), _) => {
            let widened = narrowed(w, v)?;
            let added = widened.into_iter().map(Member::Variant).collect();
            (shared_parts(v, w, PartKey::Payload), BTreeSet::new(), added)
        }
        _ => {
            let pairs = a
                .parts()
                .into_iter()
                .zip(b.parts())
                .map(|((key, x), (_, y))| (key, x.clone(), y.clone()))
                .collect();
            (pairs, BTreeSet::new(), BTreeSet::new())
        }
    };
    let parts = members
        .into_iter()
        .map(|(key, x, y)| Some((key, part(&x, &y, way)?)))
        .collect::<Option<Vec<_>>>()?;
    Some(Structural::Shape {
        dropped,
        added,
        parts,
    })
}

/// The members `wide` has that `narrow` has not, where `wide` has every
/// member `narrow` has.
fn narrowed(
    wide: &BTreeMap<Name, ReadValue>,
    narrow: &BTreeMap<Name, ReadValue>,
) -> Option<BTreeSet<Name>> {
    narrow
        .keys()
        .all(|member| wide.contains_key(member))
        .then(|| {
            wide.keys()
                .filter(|member| !narrow.contains_key(*member))
                .cloned()
                .collect()
        })
}

/// The parts of the members both have, each from `from`'s to `to`'s, in the
/// order of the members' names.
fn shared_parts<K>(
    from: &BTreeMap<Name, ReadValue>,
    to: &BTreeMap<Name, ReadValue>,
    key: K,
) -> Vec<(PartKey, ReadValue, ReadValue)>
where
    K: Fn(Name) -> PartKey,
{
    from.iter()
        .filter_map(|(member, x)| {
            let y = to.get(member)?;
            Some((key(member.clone()), x.clone(), y.clone()))
        })
        .collect()
}

fn field_members(fields: BTreeSet<Name>) -> BTreeSet<Member> {
    fields.into_iter().map(Member::Field).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decl::{DeclsBuilder, Role};
    use crate::domain::Domain;
    use crate::fixture::*;
    use crate::join::le;
    use crate::lit::{LitVar, Literal};
    use crate::sig::{DeclaredTypes, ExternDecl, SigTy, SigVar, Signature};
    use crate::ty::{IntTy, Scalar};

    fn read(ty: &Ty) -> ReadValue {
        ReadValue::of_ty(ty)
    }

    fn cast_id(i: usize) -> DeclId {
        DeclId {
            name: name(&format!("cast{i}")),
            overload: 0,
        }
    }

    fn casts(ids: &[usize]) -> Vec<DeclId> {
        ids.iter().copied().map(cast_id).collect()
    }

    fn rank(from: &Ty, to: &Ty) -> Rank {
        coerce(&decls(), from, to).unwrap().rank().unwrap()
    }

    fn plain_sig(text: &str) -> SigTy {
        SigTy::term(Term::Extern {
            name: name(text),
            args: vec![],
            identity: None,
        })
    }

    /// `A`, `B` and `D` with the casts `A → B` and `B → D`, and the conversions
    /// from `u16` to `u8` named by `conversions`.
    fn chain(conversions: &[&str]) -> Decls {
        let mut types = DeclaredTypes::default();
        for text in ["A", "B", "D"] {
            types
                .declare_extern(
                    name(text),
                    ExternDecl {
                        arity: 0,
                        identity: false,
                    },
                )
                .unwrap();
        }
        let mut builder = DeclsBuilder::new(types);
        for (i, (from, to)) in [("A", "B"), ("B", "D")].into_iter().enumerate() {
            let sig = Signature::new(
                vec![],
                false,
                vec![plain_sig(from)],
                plain_sig(to),
                builder.types(),
            )
            .unwrap();
            builder.declare(name(&format!("cast{i}")), sig, Role::Cast);
        }
        for text in conversions {
            let sig = Signature::new(
                vec![],
                false,
                vec![SigTy::term(Term::Scalar(Scalar::Int(IntTy::U16)))],
                SigTy::term(Term::Scalar(Scalar::Int(IntTy::U8))),
                builder.types(),
            )
            .unwrap();
            builder.declare(name(text), sig, Role::Conversion);
        }
        builder.finish().unwrap()
    }

    #[test]
    fn a_type_passing_into_itself_needs_no_coercion() {
        assert_eq!(
            coerce(&decls(), &named("A"), &named("A")),
            Ok(Coercion::Identity)
        );
        assert_eq!(rank(&named("A"), &named("A")), Rank::empty());
    }

    #[test]
    fn a_cast_steps_up_the_tree_and_a_reborrow_comes_before_it() {
        let decls = decls();
        assert_eq!(
            coerce(&decls, &named("A"), &named("B")),
            Ok(Coercion::Composition {
                reborrow: false,
                casts: casts(&[0]),
                structural: None,
            })
        );
        let mut_a = reference(Mutability::Exclusive, named("A"));
        assert_eq!(
            coerce(&decls, &mut_a, &reference(Mutability::Shared, named("B"))),
            Ok(Coercion::Composition {
                reborrow: true,
                casts: casts(&[2]),
                structural: None,
            })
        );
        assert_eq!(
            coerce(
                &decls,
                &mut_a,
                &reference(Mutability::Exclusive, named("B"))
            ),
            Ok(Coercion::Composition {
                reborrow: false,
                casts: casts(&[3]),
                structural: None,
            })
        );
        assert_eq!(
            coerce(&decls, &mut_a, &reference(Mutability::Shared, named("A"))),
            Ok(Coercion::Composition {
                reborrow: true,
                casts: Vec::new(),
                structural: None,
            })
        );
        assert_eq!(
            coerce(&decls, &named("B"), &named("A")),
            Err(CoerceError::None)
        );
    }

    #[test]
    fn an_object_drops_fields_and_changes_those_it_keeps_by_their_part_rows() {
        let decls = decls();
        let v = enumeration("E", &[("V", never())]);
        let vw = enumeration("E", &[("V", never()), ("W", never())]);
        let from = object(&[("x", v), ("y", int(IntTy::U8))]);
        let to = object(&[("x", vw)]);
        let coercion = coerce(&decls, &from, &to).unwrap();
        assert_eq!(
            coercion,
            Coercion::Composition {
                reborrow: false,
                casts: Vec::new(),
                structural: Some(Structural::Shape {
                    dropped: [Member::Field(name("y"))].into(),
                    added: BTreeSet::new(),
                    parts: vec![(
                        PartKey::Field(name("x")),
                        Coercion::Composition {
                            reborrow: false,
                            casts: Vec::new(),
                            structural: Some(Structural::Shape {
                                dropped: BTreeSet::new(),
                                added: [Member::Variant(name("W"))].into(),
                                parts: vec![(PartKey::Payload(name("V")), Coercion::Identity)],
                            }),
                        }
                    )],
                }),
            }
        );
        let rank = coercion.rank().unwrap();
        assert_eq!(rank.highest(), Some(Stage::Structural));
        assert_eq!(rank.casts(), 0);
        assert_eq!(
            coerce(&decls, &to, &from),
            Err(CoerceError::None),
            "no coercion gains a field"
        );
        assert_eq!(
            coerce(
                &decls,
                &object(&[("x", named("A"))]),
                &object(&[("x", named("B"))])
            ),
            Err(CoerceError::None),
            "no cast at a part"
        );
    }

    #[test]
    fn an_enum_widens_and_no_value_is_wrapped_in_an_error() {
        let decls = decls();
        let v = enumeration("E", &[("V", named("A"))]);
        let vw = enumeration("E", &[("V", named("A")), ("W", never())]);
        assert!(matches!(
            coerce(&decls, &v, &vw),
            Ok(Coercion::Composition {
                structural: Some(Structural::Shape { .. }),
                ..
            })
        ));
        let b_c = error(&[named("B"), named("C")]);
        assert_eq!(
            coerce(&decls, &named("A"), &b_c),
            Err(CoerceError::None),
            "no row wraps a value in an `Error`, which `?` alone makes"
        );
        let b_d = error(&[named("B"), named("D")]);
        assert_eq!(
            coerce(&chain(&[]), &named("A"), &b_d),
            Err(CoerceError::None),
            "no composition reaches an `Error`"
        );
    }

    #[test]
    fn never_coerces_to_every_type_and_at_parts() {
        let decls = decls();
        assert_eq!(coerce(&decls, &never(), &named("A")), Ok(Coercion::Never));
        let none = ty(Term::Option(never()));
        let some_u8 = ty(Term::Option(int(IntTy::U8)));
        let coercion = coerce(&decls, &none, &some_u8).unwrap();
        assert_eq!(
            coercion,
            Coercion::Composition {
                reborrow: false,
                casts: Vec::new(),
                structural: Some(Structural::Shape {
                    dropped: BTreeSet::new(),
                    added: BTreeSet::new(),
                    parts: vec![(PartKey::Payload(name("Some")), Coercion::Never)],
                }),
            }
        );
        assert_eq!(coercion.rank(), Some(Rank::empty()));
    }

    #[test]
    fn a_conversion_is_taken_only_where_no_composition_reaches_and_exactly_one_applies() {
        let u16_ = int(IntTy::U16);
        let u8_ = int(IntTy::U8);
        assert_eq!(coerce(&chain(&[]), &u16_, &u8_), Err(CoerceError::None));
        let one = chain(&["narrow"]);
        let narrow = DeclId {
            name: name("narrow"),
            overload: 0,
        };
        assert_eq!(
            coerce(&one, &u16_, &u8_),
            Ok(Coercion::Conversion(narrow.clone()))
        );
        assert_eq!(coerce(&one, &u16_, &u8_).unwrap().rank(), None);
        let two = chain(&["narrow", "truncate"]);
        assert!(matches!(
            coerce(&two, &u16_, &u8_),
            Err(CoerceError::Conversions(found)) if found.len() == 2
        ));
        assert_eq!(
            coerce_read(&one, &read(&u16_), &read(&u8_)),
            Err(CoerceError::None),
            "narrowing takes no conversion"
        );
        let some = |t: &Ty| ty(Term::Option(t.clone()));
        assert_eq!(
            coerce(&one, &some(&u16_), &some(&u8_)),
            Err(CoerceError::None),
            "no conversion at a part"
        );
    }

    #[test]
    fn ranks_compare_by_highest_stage_then_casts_then_lower_stages() {
        let rank = |decls: &Decls, from: Ty, to: Ty| coerce(decls, &from, &to).unwrap().rank().unwrap();
        let chain = chain(&[]);
        let one_cast = rank(&chain, named("A"), named("B"));
        let two_casts = rank(&chain, named("A"), named("D"));
        let decls = decls();
        let mut_a = reference(Mutability::Exclusive, named("A"));
        let shared_a = reference(Mutability::Shared, named("A"));
        let shared_b = reference(Mutability::Shared, named("B"));
        let v = enumeration("E", &[("V", never())]);
        let vw = enumeration("E", &[("V", never()), ("W", never())]);
        let reborrow = rank(&decls, mut_a.clone(), shared_a);
        let reborrow_cast = rank(&decls, mut_a, shared_b);
        let structural = rank(&decls, v, vw.clone());
        let mut types = DeclaredTypes::default();
        types
            .declare_extern(
                name("A"),
                ExternDecl {
                    arity: 0,
                    identity: false,
                },
            )
            .unwrap();
        let mut builder = DeclsBuilder::new(types);
        let up = Signature::new(
            vec![],
            false,
            vec![plain_sig("A")],
            SigTy::term(Term::Enum(
                name("E"),
                [(name("V"), SigTy::term(Term::Never))].into(),
            )),
            builder.types(),
        )
        .unwrap();
        builder.declare(name("up"), up, Role::Cast);
        let enum_decls = builder.finish().unwrap();
        let cast_structural = rank(&enum_decls, named("A"), vw.clone());
        assert!(Rank::empty() < reborrow);
        assert!(reborrow < one_cast);
        assert!(one_cast < two_casts);
        assert!(reborrow < reborrow_cast);
        assert!(one_cast < reborrow_cast);
        assert!(reborrow_cast < structural);
        assert!(two_casts < structural);
        assert!(structural < cast_structural);
    }

    #[test]
    fn a_position_that_decides_nothing_reaches_every_type() {
        let decls = decls();
        let u8_ = read(&int(IntTy::U8));
        assert_eq!(
            coerce_read(&decls, &ReadValue::Undecided, &u8_),
            Ok(Coercion::Undecided)
        );
        let candidate = ReadValue::of_sig(&SigTy::term(Term::Option(SigTy::Var(SigVar::Type(0)))));
        assert_eq!(
            coerce_read(&decls, &read(&ty(Term::Option(named("A")))), &candidate),
            Ok(Coercion::Undecided)
        );
        let big = ReadValue::Domain(Domain::of_literal(&LitVar::of(Literal::Int(300))));
        assert_eq!(coerce_read(&decls, &big, &u8_), Err(CoerceError::None));
        assert_eq!(
            coerce_read(&decls, &big, &read(&int(IntTy::U16))),
            Ok(Coercion::Undecided)
        );
        assert_eq!(
            coerce_read(&decls, &big, &read(&named("A"))),
            Err(CoerceError::None)
        );
        let float = ReadValue::Domain(Domain::of_literal(&LitVar::of(Literal::float(1.5))));
        assert_eq!(coerce_read(&decls, &float, &big), Err(CoerceError::None));
    }

    #[test]
    fn along_le_gains_fields_unchanged_and_takes_no_cast_or_wrap_at_a_part() {
        let decls = decls();
        let from = object(&[("x", named("A"))]);
        let to = object(&[("x", named("A")), ("y", int(IntTy::U8))]);
        assert_eq!(
            along_le(&decls, &from, &to),
            Ok(Coercion::Composition {
                reborrow: false,
                casts: Vec::new(),
                structural: Some(Structural::Shape {
                    dropped: BTreeSet::new(),
                    added: [Member::Field(name("y"))].into(),
                    parts: vec![(PartKey::Field(name("x")), Coercion::Identity)],
                }),
            })
        );
        assert_eq!(
            along_le(&decls, &from, &object(&[("x", named("B"))])),
            Err(CoerceError::None),
            "no cast at a part"
        );
        let u8_ = int(IntTy::U8);
        let a_c = error(&[named("A"), named("C")]);
        let ok_a = result(u8_.clone(), named("A"));
        assert_eq!(
            along_le(&decls, &ok_a, &result(u8_.clone(), a_c.clone())),
            Err(CoerceError::None),
            "no wrap at a part"
        );
        let Ok(Coercion::Composition {
            structural: Some(Structural::Shape { parts, .. }),
            ..
        }) = along_le(
            &decls,
            &result(u8_.clone(), error(&[named("A")])),
            &result(u8_, a_c.clone()),
        )
        else {
            panic!("`Result⟨u8, Error{{A(A)}}⟩ ≤ Result⟨u8, Error{{A(A), C(C)}}⟩` by the enum row");
        };
        assert!(matches!(
            &parts[1],
            (_, Coercion::Composition { structural: Some(Structural::Shape { added, .. }), .. })
                if *added == [Member::Variant(name("C"))].into()
        ));
        assert_eq!(along_le(&decls, &named("A"), &a_c), Err(CoerceError::None));
    }

    #[test]
    fn along_le_reaches_exactly_where_the_join_orders_over_the_universe() {
        let decls = decls();
        let types = universe();
        for s in &types {
            for t in &types {
                assert_eq!(
                    along_le(&decls, s, t).is_ok(),
                    le(&decls, s, t),
                    "{s} ≤ {t}"
                );
            }
        }
    }
}
