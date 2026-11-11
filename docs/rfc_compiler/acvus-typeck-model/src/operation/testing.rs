//! A world to record a program in and settle it, for the tests of the law.

use std::collections::BTreeSet;

use crate::primitive::constraint::{Bound, Constraint, Suspended};
use crate::primitive::class::{Live, Side, State};
use crate::primitive::domain::Domain;
use crate::primitive::id::Arena;
use crate::primitive::scope::Scopes;
use crate::primitive::ty::{Mutability, PartKey, Scalar, Shape, VariantOf};
use crate::primitive::Name;
use crate::primitive::{ClassId, ItemId, LambdaId};
use crate::store::classes::Classes;
use crate::store::closures::Closures;

use super::Outcome;
use super::law::Law;
use super::settle::settle;

pub(super) const SHARED: Mutability = Mutability::Shared;
pub(super) const EXCLUSIVE: Mutability = Mutability::Exclusive;
pub(super) const U8: Shape<ClassId> = Shape::Scalar(Scalar::U8);

pub(super) struct World {
    pub(super) classes: Classes<Live>,
    pub(super) scopes: Scopes,
    pub(super) closures: Closures<Live>,
    pub(super) items: Arena<ItemId, &'static str>,
    pub(super) outcome: Outcome<Live>,
    /// The classes made here, in the order settling asks them in, ahead of any
    /// other class.
    pub(super) open: Vec<ClassId>,
}

impl World {
    pub(super) fn new() -> Self {
        Self {
            classes: Classes::default(),
            scopes: Scopes::new(),
            closures: Closures::default(),
            items: Arena::default(),
            outcome: Outcome::default(),
            open: Vec::new(),
        }
    }

    /// A variable, or a hole of a signature.
    pub(super) fn undecided(&mut self) -> ClassId {
        let class = self.classes.undecided(self.scopes.root());
        self.open.push(class);
        class
    }

    /// The class of an expression that writes the head `shape`: the head the text
    /// writes enters it.
    pub(super) fn text(&mut self, shape: Shape<ClassId>) -> ClassId {
        let by = self.items.push("written");
        let written = self.classes.decided(self.scopes.root(), shape, BTreeSet::from([by]));
        let class = self.undecided();
        self.enter(written, class);
        class
    }

    /// A class a declaration writes.
    pub(super) fn external(&mut self, shape: Shape<ClassId>) -> ClassId {
        let by = self.items.push("declared");
        self.classes.external(self.scopes.root(), shape, BTreeSet::from([by]))
    }

    pub(super) fn record(&mut self, constraint: Constraint<Live>) -> ItemId {
        let at = self.items.push("bound");
        self.classes.constrain(constraint, at);
        at
    }

    pub(super) fn enter(&mut self, lower: ClassId, upper: ClassId) -> ItemId {
        self.record(Bound::Enter { lower, upper }.into())
    }

    /// `let name = value`, or a branch of an `if`: a place `value` enters.
    pub(super) fn meets(&mut self, value: ClassId) -> ClassId {
        let place = self.undecided();
        self.enter(value, place);
        place
    }

    /// `value as _`: the class of the expression.
    pub(super) fn alone(&mut self, value: ClassId) -> ClassId {
        let place = self.undecided();
        self.record(
            Suspended::Alone {
                lower: value,
                upper: place,
            }
            .into(),
        );
        place
    }

    /// `operand?` in a body whose result is `result`: the class of the expression.
    ///
    /// The payload of `Ok` is the value, and that of `Err` is what it exits with.
    /// A `Result` whose error is a class of its own enters the result, and the
    /// exit says what enters that class.
    pub(super) fn question(&mut self, operand: ClassId, result: ClassId) -> ClassId {
        let payload = |variant: &str| PartKey::Payload(VariantOf::Result, Name::new(variant));
        let (value, _) = self.project(operand, payload("Ok"), Side::Upper);
        let (exits_with, _) = self.project(operand, payload("Err"), Side::Upper);
        let (ok, error) = (self.undecided(), self.undecided());
        let exit = self.text(Shape::Result(ok, error));
        self.enter(exit, result);
        self.record(
            Suspended::Exit {
                lower: exits_with,
                upper: error,
            }
            .into(),
        );
        value
    }

    /// `argument` passed to a parameter of the type `given`: the item of the
    /// boundary. The argument enters the boundary's class, which is passed.
    pub(super) fn pass(&mut self, argument: ClassId, given: ClassId) -> ItemId {
        let boundary = self.meets(argument);
        self.record(
            Bound::Change {
                lower: boundary,
                upper: given,
            }
            .into(),
        )
    }

    /// A lambda over `captures`, where the text defines it: the class of the
    /// expression, which the function value of a closure named anew enters. A
    /// definition is a use that passes nothing, so it is called with arguments
    /// nothing enters.
    pub(super) fn define(&mut self, lambda: LambdaId, captures: Vec<ClassId>, parameters: usize) -> ClassId {
        let closure = self.closures.name(lambda, captures);
        let value = self.text(Shape::FnValue(BTreeSet::from([closure])));
        let nothing: Vec<ClassId> = (0..parameters).map(|_| self.undecided()).collect();
        self.call(value, &nothing);
        value
    }

    /// `callee(arguments)`: the class of the expression. The callee is passed to
    /// the function type the call writes.
    pub(super) fn call(&mut self, callee: ClassId, arguments: &[ClassId]) -> ClassId {
        let result = self.undecided();
        let by = self.items.push("call");
        let written = self
            .classes
            .decided(self.scopes.root(), Shape::Fn(arguments.to_vec(), result), BTreeSet::from([by]));
        self.classes.constrain(
            Bound::Change {
                lower: callee,
                upper: written,
            }
            .into(),
            by,
        );
        result
    }

    /// `&place` or `&mut place`: the class of the expression, and the item by
    /// which the reference the text writes enters it.
    pub(super) fn lend(&mut self, place: ClassId, mutability: Mutability) -> (ClassId, ItemId) {
        let by = self.items.push("written");
        let written = self
            .classes
            .decided(self.scopes.root(), Shape::Ref(mutability, place), BTreeSet::from([by]));
        let class = self.undecided();
        let at = self.enter(written, class);
        (class, at)
    }

    /// `operand.key`, read or written: the class of the part, and the item of the
    /// projection.
    pub(super) fn project(&mut self, operand: ClassId, key: PartKey, side: Side) -> (ClassId, ItemId) {
        let at = self.items.push("projection");
        let part = self.classes.part(operand, key, side, at).expect("the operand is not decided when it is projected");
        self.open.push(part);
        (part, at)
    }

    /// A literal with no suffix: a class with a domain, and nothing entering it.
    pub(super) fn literal(&mut self, domain: &[Scalar]) -> ClassId {
        let class = self.undecided();
        let at = self.items.push("literal");
        assert_eq!(self.classes.domain(class, Domain(domain.iter().copied().collect()), at), None);
        class
    }

    /// Settles by `law`, asking classes in the order `open` holds them.
    pub(super) fn settle<L>(&mut self, law: &L)
    where
        L: Law<Live>,
    {
        let at = self.items.push("settling");
        let open = self.open.clone();
        let visit = |classes: &Classes<Live>| -> Vec<ClassId> {
            open.iter().copied().chain(classes.roots().map(|root| root.id())).collect()
        };
        settle(&mut self.classes, &mut self.scopes, &mut self.closures, law, &mut self.outcome, at, &visit);
    }

    pub(super) fn head(&self, class: ClassId) -> Option<Shape<ClassId>> {
        match &self.classes.get(self.classes.find(class)).state {
            State::Decided(decided) => Some(decided.shape.clone()),
            State::Undecided(_) => None,
        }
    }

    pub(super) fn same(&self, a: ClassId, b: ClassId) -> bool {
        self.classes.find(a) == self.classes.find(b)
    }

    /// The type of `class` written out, `?` standing for a class not decided.
    pub(super) fn ty(&self, class: ClassId) -> String {
        self.ty_to(class, 6)
    }

    fn ty_to(&self, class: ClassId, depth: usize) -> String {
        if depth == 0 {
            return "…".to_owned();
        }
        let Some(shape) = self.head(class) else {
            return "?".to_owned();
        };
        let at = |part: &ClassId| self.ty_to(*part, depth - 1);
        match &shape {
            Shape::Option(payload) => format!("Option<{}>", at(payload)),
            Shape::Ref(SHARED, target) => format!("&{}", at(target)),
            Shape::Ref(EXCLUSIVE, target) => format!("&mut {}", at(target)),
            Shape::Struct(name, arguments) if arguments.is_empty() => name.0.clone(),
            Shape::Struct(name, arguments) => {
                let arguments: Vec<String> = arguments.iter().map(at).collect();
                format!("{}<{}>", name.0, arguments.join(", "))
            }
            Shape::Result(ok, error) => format!("Result<{}, {}>", at(ok), at(error)),
            Shape::Error(variants) => {
                let variants: Vec<String> = variants.iter().map(|(variant, payload)| format!("{}({})", variant.0, at(payload))).collect();
                format!("Error{{{}}}", variants.join(", "))
            }
            Shape::ErrorOf(error) => format!("Error<{}>", at(error)),
            Shape::Slice(element) => format!("[{}]", at(element)),
            Shape::Array(length, element) => format!("[{}; {length}]", at(element)),
            Shape::Object(fields) => {
                let fields: Vec<String> = fields.iter().map(|(name, part)| format!("{}: {}", name.0, at(part))).collect();
                format!("{{{}}}", fields.join(", "))
            }
            Shape::Enum(name, variants) => {
                let variants: Vec<String> = variants.iter().map(|(variant, payload)| format!("{}({})", variant.0, at(payload))).collect();
                format!("{}{{{}}}", name.0, variants.join(", "))
            }
            Shape::Scalar(scalar) => format!("{scalar:?}").to_lowercase(),
            Shape::Never => "!".to_owned(),
            other => format!("{:?}", other.layer()),
        }
    }

    /// The type of `class`, or the causes the program is refused by.
    pub(super) fn outcome_of(&self, class: ClassId) -> String {
        let causes: BTreeSet<String> = self.outcome.refusals.iter().map(|refusal| format!("{:?}", refusal.cause)).collect();
        if causes.is_empty() {
            self.ty(class)
        } else {
            format!("refused: {}", causes.into_iter().collect::<Vec<_>>().join(", "))
        }
    }
}

impl Drop for World {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert_eq!(self.classes.verify(), Ok(()));
        }
    }
}

/// The order `order` names for the classes of a world: as made, reversed, and
/// shuffled.
pub(super) fn reorder(open: &mut [ClassId], order: u64) {
    match order {
        0 => {}
        1 => open.reverse(),
        seed => {
            let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            for i in (1..open.len()).rev() {
                state = state.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                let j = usize::try_from(state >> 33).expect("a shifted word fits") % (i + 1);
                open.swap(i, j);
            }
        }
    }
}
