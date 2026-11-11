//! The contracts between the three operations and the rules they ask.
//!
//! A law reads the store and answers. It merges nothing, decides nothing and
//! finishes nothing: the operations do those, from what it answers. So a rule
//! cannot leave a constraint half done, and every rule is one answer in one place.
//!
//! | Who asks | What it asks | The answer | What is done with it |
//! |---|---|---|---|
//! | looking | what a constraint comes to ([`Law::look`]) | [`Looked`] | the constraint is finished or left waiting, and its pairs are connected |
//! | connecting | how a class is compared with one a declaration writes ([`Law::same`]) | a constraint | it is recorded in place of a union |
//! | connecting | where a part lies that a head has no place for ([`Law::place`]) | [`Place`] | the part is named there, or refused |
//! | connecting | whether a literal's class may take a head ([`Law::admits`]) | yes or no | it is refused where not |
//! | settling | which classes are one type by the constraints as a whole ([`Law::one`]) | pairs | they are connected |
//! | deciding | which classes take which heads now ([`Law::heads`]) | [`Heads`] | each head is given, and what that leaves is connected |
//! | settling, at its end | what is refused of a class ([`Law::end`]) | refusals | they are gathered |
//!
//! Every question but the first is answered from the classes as they stand. The
//! first may need something made before it can be answered, a signature at a
//! boundary or the copy of a template, and is given a [`Scribe`] to make it
//! with.

use std::collections::{BTreeMap, BTreeSet};

use crate::primitive::ItemId;
use crate::primitive::adjustment::Adjustment;
use crate::primitive::class::ClassKind;
use crate::primitive::constraint::ConstraintInto;
use crate::primitive::domain::Domain;
use crate::primitive::refusal::Refused;
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{PartKey, Shape};
use crate::primitive::{ClosureId, LambdaId, ScopeId};
use crate::store::Store;
use crate::store::classes::Classes;
use crate::store::closures::Closures;
use crate::store::template::Fate;

/// Two classes that are one type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Pair<K>
where
    K: ClassKind,
{
    Classes(K::Id, K::Id),
    /// `part` is one type with the part `key` of `of`. Where `of` is undecided it
    /// is given the name, and where it is decided with no place for it there is
    /// nothing to connect.
    Part { part: K::Id, of: K::Id, key: PartKey },
}

/// What a constraint comes to, as its two ends stand.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Looked<K>
where
    K: ClassKind,
{
    /// It holds, and is finished. These pairs are one type by it, these
    /// adjustments are what it takes, and these constraints are recorded in its place,
    /// by the item that recorded it.
    Holds {
        pairs: Vec<Pair<K>>,
        adjustments: Vec<Adjustment>,
        constraints: Vec<K::Constraint>,
    },
    /// It does not hold, and is finished.
    Refused(Refused<K>),
    /// It cannot be told yet. It is looked at again when a class of `on` is
    /// merged or decided. Until it is finished it may still merge what lies below
    /// each class of `apart`, though not those classes themselves, so none of
    /// what lies there is decided. `pairs` are one type whatever it comes to.
    Waits {
        on: BTreeSet<K::Id>,
        apart: BTreeSet<K::Id>,
        pairs: Vec<Pair<K>>,
    },
}

/// The heads classes take now, and what was refused in finding them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Heads<K>
where
    K: ClassKind,
{
    pub heads: Vec<Head<K>>,
    pub refused: Vec<Refused<K>>,
}

/// A head a class takes. Only its constructor and members are taken: the parts
/// of the head decided are the class's own.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Head<K>
where
    K: ClassKind,
{
    pub class: K::Id,
    pub shape: Shape<K::Id>,
    /// The items that give it, for a refusal to name.
    pub by: BTreeSet<ItemId>,
}

impl<K> Heads<K>
where
    K: ClassKind,
{
    /// The heads of a ladder: those of its first rung that offers a head to any
    /// class of `asked` that is not held, with what that rung and those before
    /// it refused, of a class held or not. A rung is a rule that offers one class a head; a later rung is
    /// one that holds only where no earlier one gives anything, so it is asked
    /// only then.
    pub fn of_ladder<'a, R>(ladder: R, asked: &[K::Id], held: &dyn Fn(K::Id) -> bool) -> Self
    where
        R: IntoIterator<Item = &'a dyn Fn(K::Id) -> Offer<K>>,
        K: 'a,
    {
        let mut refused = Vec::new();
        for rung in ladder {
            let mut heads = Vec::new();
            for class in asked.iter().copied() {
                match rung(class) {
                    Offer::Head { shape, by } if !held(class) => heads.push(Head { class, shape, by }),
                    Offer::Head { .. } | Offer::Nothing => {}
                    Offer::Refused(refusal) => refused.push(refusal),
                }
            }
            if !heads.is_empty() {
                return Self { heads, refused };
            }
        }
        Self { heads: Vec::new(), refused }
    }
}

/// What a rung offers an undecided class.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Offer<K>
where
    K: ClassKind,
{
    Nothing,
    Head { shape: Shape<K::Id>, by: BTreeSet<ItemId> },
    /// The class cannot take a head by this rung, and the program is refused.
    Refused(Refused<K>),
}

/// Where a part named on a class lies, once the class is decided as a head that
/// has no place of that name.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Place {
    /// It is the part of that name of the head's part `through`, and each
    /// projection that named it takes `adjustment`.
    Within { through: PartKey, adjustment: Adjustment },
    Nowhere,
}

/// The store as a law may use it while it looks at a constraint: all of it to
/// read, and to add to. It merges nothing, decides no class that was there, and
/// changes no scope and no closure that was there.
///
/// | What is added | For what |
/// |---|---|
/// | a class with a head ([`Scribe::written`]) | a head a rule writes |
/// | the classes of a template ([`Scribe::anew`]) | a signature at a boundary, the copy of a lambda |
/// | a scope for a copy ([`Scribe::copy_of`]) | the copy of a lambda lies in a scope of its own |
/// | a closure ([`Scribe::name`]) | a copy names a closure of each lambda its body defines |
#[derive(Debug)]
pub struct Scribe<'a, K>
where
    K: ClassKind,
{
    store: &'a mut Store<K>,
}

impl<'a, K> Scribe<'a, K>
where
    K: ClassKind,
{
    pub(super) fn new(store: &'a mut Store<K>) -> Self {
        Self { store }
    }

    pub fn classes(&self) -> &Classes<K> {
        &self.store.classes
    }

    pub fn scopes(&self) -> &Scopes {
        &self.store.scopes
    }

    pub fn closures(&self) -> &Closures<K> {
        &self.store.closures
    }

    /// Makes a scope within `scope` for a copy of the template of `closure`.
    pub fn copy_of(&mut self, closure: ClosureId, scope: ScopeId) -> ScopeId {
        let copy = self.store.scopes.child(scope);
        self.store.closures.copied(copy, closure);
        copy
    }

    /// Names a closure anew, `lambda` over `captures`.
    pub fn name(&mut self, lambda: LambdaId, captures: Vec<K::Id>) -> ClosureId {
        self.store.closures.name(lambda, captures)
    }

    /// Makes a class with the head `shape`, as the text writes one, in `scope`, by
    /// `by`.
    pub fn written(&mut self, scope: ScopeId, shape: Shape<K::Id>, by: BTreeSet<ItemId>) -> K::Id {
        self.store.classes.decided(scope, shape, by)
    }

    /// Makes every class of `template` anew in `scope`, by `at`, and gives each
    /// class of the template with the class it became.
    pub fn anew<S>(&mut self, template: &Classes<S>, scope: ScopeId, at: ItemId) -> BTreeMap<S::Id, K::Id>
    where
        S: ClassKind,
        S::Constraint: ConstraintInto<S, K>,
    {
        self.store
            .classes
            .copy_from(template, &self.store.scopes, scope, at, |_| Fate::Anew)
            .classes
    }
}

/// The rules solving asks.
pub trait Law<K>
where
    K: ClassKind,
{
    /// What `constraint`, recorded by `at`, comes to as its ends stand.
    fn look(&self, scribe: &mut Scribe<'_, K>, constraint: &K::Constraint, at: ItemId) -> Looked<K>;

    /// The bound that says `part` has the head of `declared`, a class a
    /// declaration writes. Such a class never changes, so nothing is merged with
    /// it, and what is one type with it is compared with it by this bound.
    fn same(&self, part: K::Id, declared: K::Id) -> K::Constraint;

    /// The classes that are one type by how the constraints stand as a whole, and by
    /// no one constraint, each pair with the item it goes by.
    fn one(&self, classes: &Classes<K>) -> Vec<(K::Id, K::Id, ItemId)>;

    /// Where the part `key` lies on a class decided as `head`, which has no place
    /// of that name.
    fn place(&self, head: &Shape<K::Id>, key: &PartKey) -> Place;

    /// Whether a class a literal registered `domain` in may be decided as `head`.
    fn admits(&self, head: &Shape<K::Id>, domain: &Domain) -> bool;

    /// The heads the classes of `asked`, each undecided, take now. A class that
    /// `held` says is held may still be merged with another, and takes none.
    fn heads(&self, classes: &Classes<K>, asked: &[K::Id], held: &dyn Fn(K::Id) -> bool) -> Heads<K>;

    /// What is refused of `class` when nothing more can be done.
    fn end(&self, classes: &Classes<K>, class: K::Id) -> Vec<Refused<K>>;
}
