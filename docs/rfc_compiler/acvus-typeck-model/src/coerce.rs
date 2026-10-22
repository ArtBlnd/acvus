//! Coercion, the steps of the join's `≤` each taken by a row of the Coercion
//! table, and rank (l.93–111, l.282).

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::decl::{DeclId, Decls};
use crate::join::{ERROR, is_error};
use crate::read::ReadValue;
use crate::ty::{HasTerm, Member, Mutability, Name, PartKey, Term, Ty};

#[cfg(test)]
mod tests;

/// A stage of a composition, ordered `reborrow ≺ cast ≺ structural` (l.111).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Stage {
    Reborrow,
    Cast,
    Structural,
}

/// The stages a coercion takes, the structural stage counted down into the
/// parts' coercions, which take no other (l.111), and its number of casts;
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
    /// A position that decides nothing, which reaches every type (l.335).
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
    /// gains are added and not changed (l.282).
    Shape {
        dropped: BTreeSet<Member>,
        added: BTreeSet<Member>,
        parts: Vec<(PartKey, Coercion)>,
    },
    /// Wrapped in the `Error` at this part: the variant named after the
    /// value's type, or `Error⟨E⟩`'s one part where the type has no name
    /// (l.104, l.105), after the value is changed into that part's type.
    Wrap(PartKey, Box<Coercion>),
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
            Structural::Wrap(_, inner) => {
                let mut rank = inner.rank()?;
                rank.take(Stage::Structural);
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
    /// More than one composition reaches the type (`holes.md`, H2).
    Compositions(Vec<Coercion>),
    /// No composition reaches the type, and more than one declared conversion
    /// goes there (types §7).
    Conversions(Vec<DeclId>),
}

/// The order a value is changed along.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Way {
    /// `⇝`, by the Coercion table, conversions included where `conversions`.
    Coercion { conversions: bool },
    /// The join's `≤`, each step taken by a row of the Coercion table (l.282).
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
/// takes none (l.335).
pub fn coerce_read(
    decls: &Decls,
    from: &ReadValue,
    to: &ReadValue,
) -> Result<Coercion, CoerceError> {
    walk(decls, from, to, Way::Coercion { conversions: false })
}

/// How `from` is changed at each step of `from ≤ to` (l.282).
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
/// a position decides nothing, being undecided on either side, or a LitVar,
/// which reaches a LitVar of its family and a scalar admitting its values
/// (l.164, l.335).
fn deciding_nothing(from: &ReadValue, to: &ReadValue) -> Option<Result<Coercion, CoerceError>> {
    if from == to {
        return Some(Ok(Coercion::Identity));
    }
    match (from, to) {
        (ReadValue::Undecided, _) | (_, ReadValue::Undecided) => Some(Ok(Coercion::Undecided)),
        (ReadValue::Term(term), _) if matches!(**term, Term::Never) => Some(Ok(Coercion::Never)),
        (ReadValue::Lit(_), _) | (_, ReadValue::Lit(_)) if agrees(from, to) => {
            Some(Ok(Coercion::Undecided))
        }
        (ReadValue::Lit(_), _) | (_, ReadValue::Lit(_)) => Some(Err(CoerceError::None)),
        _ if agrees(from, to) => Some(Ok(Coercion::Undecided)),
        _ => None,
    }
}

/// Equal at every position both decide.
fn agrees(a: &ReadValue, b: &ReadValue) -> bool {
    match (a, b) {
        (ReadValue::Undecided, _) | (_, ReadValue::Undecided) => true,
        (ReadValue::Lit(l), ReadValue::Lit(m)) => l.family() == m.family(),
        (ReadValue::Lit(lit), ReadValue::Term(term))
        | (ReadValue::Term(term), ReadValue::Lit(lit)) => {
            matches!(**term, Term::Scalar(scalar) if lit.admits(scalar))
        }
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

/// The structural stage from `at` to `to`, where one reaches: an `Error` row
/// of the whole value, or the rows `shape` takes.
fn structural(at: &ReadValue, to: &ReadValue, way: Way) -> Option<Structural> {
    let (Some(a), Some(b)) = (at.decided(), to.decided()) else {
        return None;
    };
    if let (Way::Coercion { .. }, Term::Enum(name, variants)) = (way, b)
        && name.as_str() == ERROR
        && !is_error(a)
    {
        return variants
            .iter()
            .find(|(_, payload)| agrees(at, payload))
            .map(|(variant, _)| {
                Structural::Wrap(
                    PartKey::Payload(variant.clone()),
                    Box::new(Coercion::Identity),
                )
            });
    }
    if let (Way::Coercion { .. }, Term::ErrorOf(wrapped)) = (way, b)
        && !is_error(a)
        && !matches!(a, Term::Enum(..) | Term::Struct(..) | Term::Extern { .. })
    {
        return agrees(at, wrapped)
            .then(|| Structural::Wrap(PartKey::Wrapped, Box::new(Coercion::Identity)));
    }
    shape(a, b, way)
}

/// A part's coercion `⇝ₚ`, or a part's steps along `≤` (l.89, l.111): the
/// rows `shape` takes and `! ⇝ U` alone, so no reborrow, cast, `Error` wrap or
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
