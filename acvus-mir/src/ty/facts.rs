//! Facts of a concrete type alone: what copies (ownership §1), whether some
//! value has it, and where a value holds loans (ownership §5).

use acvus_utils::QualifiedRef;
use rustc_hash::FxHashMap;

use super::arg::LenTerm;
use super::kind::{Holds, Mutability, TyKind};
use super::structural::structure;
use super::{Concrete, Phase, Ty, TyId, Types};

/// The lifetimes each extern-defined type declares (extern-fn §8). The
/// declarations answer; a type does not carry the count.
pub trait LifetimesOf {
    fn lifetimes_of(&self, id: QualifiedRef) -> usize;
}

/// A reader of one phase's types that follows a variable to what it is
/// bound to: a concrete table has none to follow, and solving follows its
/// own.
pub trait Resolve<V: Phase> {
    fn types(&self) -> &Types<V>;
    /// `ty` with every bound variable at its root followed.
    fn resolve(&self, ty: TyId<V>) -> TyId<V>;
}

impl Resolve<Concrete> for Types<Concrete> {
    fn types(&self) -> &Types<Concrete> {
        self
    }

    fn resolve(&self, ty: Ty) -> Ty {
        ty
    }
}

/// Whether a value of `ty` copies: `!`, a primitive, a shared reference,
/// or a transparent structured type whose every part copies (ownership §1).
/// `None` where an unbound variable stands where the answer is read. The
/// one copy rule, for every phase.
pub fn copies_in<V, R>(reader: &R, ty: TyId<V>) -> Option<bool>
where
    V: Phase,
    R: Resolve<V>,
{
    acvus_utils::grow(|| copies_level(reader, ty))
}

fn copies_level<V, R>(reader: &R, ty: TyId<V>) -> Option<bool>
where
    V: Phase,
    R: Resolve<V>,
{
    let ty = reader.resolve(ty);
    let types = reader.types();
    let Some(structure) = structure(types, ty) else {
        return match types.kind(ty) {
            TyKind::Var(_) => None,
            kind => Some(copies_alone(kind)),
        };
    };
    let mut open = false;
    for part in structure.parts() {
        match copies_in(reader, part.ty) {
            Some(false) => return Some(false),
            Some(true) => {}
            None => open = true,
        }
    }
    (!open).then_some(true)
}

/// Whether a value of the concrete `ty` copies (ownership §1).
pub fn copies(types: &Types<Concrete>, ty: Ty) -> bool {
    copies_in(types, ty).expect("a concrete type holds no variable")
}

/// Whether a value of a type that is not structural copies: `!`, a
/// primitive or a shared reference (ownership §1).
pub fn copies_alone<V>(kind: &TyKind<V>) -> bool
where
    V: Phase,
{
    matches!(
        kind,
        TyKind::Never
            | TyKind::Int(_)
            | TyKind::Float
            | TyKind::Bool
            | TyKind::Char
            | TyKind::Unit
            | TyKind::Ref(Mutability::Shared, _)
    )
}

/// Whether some value has the concrete `ty`. `!` has none (types §4). A
/// value holds every part of a tuple, an object and an array, so each has
/// one where every part does, and an array of no element always; it holds
/// one variant of an enum, an `Option` and a `Result` (types §1), so each
/// has one where some variant does, a variant of no payload always. A
/// reference lends storage holding a value of its target (ownership §5), so
/// it has one where its target does; a slice may be empty, so it always
/// does. Every other type has one: a primitive, `String`, `str`, a function
/// value, an order token, a started call, and an extern-defined type, whose
/// values the language knows only through its declarations.
///
/// Cost: O(the types `ty` reaches), each judged once.
pub fn inhabited(types: &Types<Concrete>, ty: Ty) -> bool {
    inhabited_in(types, ty, &mut FxHashMap::default())
}

/// `inhabited`, each type already judged answered from `judged`.
fn inhabited_in(types: &Types<Concrete>, ty: Ty, judged: &mut FxHashMap<Ty, bool>) -> bool {
    if let Some(known) = judged.get(&ty) {
        return *known;
    }
    let answer = acvus_utils::grow(|| inhabited_level(types, ty, judged));
    judged.insert(ty, answer);
    answer
}

fn inhabited_level(types: &Types<Concrete>, ty: Ty, judged: &mut FxHashMap<Ty, bool>) -> bool {
    match types.kind(ty) {
        TyKind::Never => false,
        TyKind::Tuple(items, _) => items.iter().all(|item| inhabited_in(types, *item, judged)),
        TyKind::Object { fields, .. } => fields.iter().all(|(_, field)| inhabited_in(types, *field, judged)),
        TyKind::Array(element, len, _) => match len {
            LenTerm::Known(0) => true,
            LenTerm::Known(_) => inhabited_in(types, *element, judged),
            LenTerm::Var(var) => match *var {},
        },
        TyKind::Enum { .. } | TyKind::Option(..) | TyKind::Result(..) => structure(types, ty)
            .and_then(|structure| structure.variants())
            .expect("an enum, an `Option` and a `Result` are variant types")
            .iter()
            .any(|variant| variant.payload.is_none_or(|payload| inhabited_in(types, payload, judged))),
        TyKind::Ref(_, target) => inhabited_in(types, *target, judged),
        TyKind::Slice(_)
        | TyKind::Int(_)
        | TyKind::Float
        | TyKind::Char
        | TyKind::String
        | TyKind::Bool
        | TyKind::Unit
        | TyKind::Str
        | TyKind::Order
        | TyKind::Handle(_)
        | TyKind::Fn { .. }
        | TyKind::Closures(_)
        | TyKind::UserDefined { .. } => true,
        TyKind::Var(var) => match *var {},
    }
}

/// What one position of a type is (ownership §5, effects §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionKind {
    /// A reference's own position, and how many positions its target
    /// has; the target's follow it.
    Ref(Mutability, usize),
    /// A function value's one position: what its captures hold.
    Captures(Holds),
    /// One lifetime an extern-defined type declares.
    Lifetime,
    /// A started call's arguments, held until it is taken.
    InFlight,
}

/// Every position a value of `ty` holds loans at, in order (ownership §5):
/// a reference's own before its target's; a function value's one; a
/// started call's one before its result's; an extern-defined type's
/// lifetimes before its arguments'; a structural type's parts in order,
/// `Result`'s `Ok` before its `Err`; one element's for a sequence. Each
/// position is given with the type it is the position of. This is the only
/// walk of positions: `positions`, `position_kinds` and `lent_types` read
/// it.
///
/// The callback is a trait object: the walk recurses, and a generic one would
/// be instantiated anew at every depth.
pub fn for_each_position(
    types: &Types<Concrete>,
    lifetimes: &dyn LifetimesOf,
    ty: Ty,
    on_position: &mut dyn FnMut(PositionKind, Ty),
) {
    acvus_utils::grow(|| for_each_position_level(types, lifetimes, ty, on_position))
}

fn for_each_position_level(
    types: &Types<Concrete>,
    lifetimes: &dyn LifetimesOf,
    ty: Ty,
    on_position: &mut dyn FnMut(PositionKind, Ty),
) {
    match types.kind(ty) {
        TyKind::Int(_)
        | TyKind::Float
        | TyKind::Char
        | TyKind::String
        | TyKind::Bool
        | TyKind::Unit
        | TyKind::Never
        | TyKind::Order
        | TyKind::Str => {}
        TyKind::Ref(mutability, target) => {
            let mut width = 0;
            for_each_position(types, lifetimes, *target, &mut |_, _| width += 1);
            on_position(PositionKind::Ref(*mutability, width), ty);
            for_each_position(types, lifetimes, *target, on_position);
        }
        TyKind::Fn { holds, .. } => on_position(PositionKind::Captures(*holds), ty),
        // One environment, whichever signature it is called at: what its
        // captures hold, as a function value's (functions §2).
        TyKind::Closures(signatures) => {
            let holds = signatures
                .iter()
                .filter_map(|signature| match types.kind(*signature) {
                    TyKind::Fn { holds, .. } => Some(*holds),
                    _ => None,
                })
                .fold(Holds::Nothing, Holds::join);
            on_position(PositionKind::Captures(holds), ty);
        }
        TyKind::Handle(result) => {
            on_position(PositionKind::InFlight, ty);
            for_each_position(types, lifetimes, *result, on_position);
        }
        TyKind::Array(element, _, _) | TyKind::Slice(element) | TyKind::Option(element, _) => {
            for_each_position(types, lifetimes, *element, on_position);
        }
        TyKind::Result(ok, err, _) => {
            for_each_position(types, lifetimes, *ok, on_position);
            for_each_position(types, lifetimes, *err, on_position);
        }
        TyKind::Tuple(items, _) => {
            for item in items.iter() {
                for_each_position(types, lifetimes, *item, on_position);
            }
        }
        TyKind::Object { fields, .. } => {
            for (_, field) in fields {
                for_each_position(types, lifetimes, *field, on_position);
            }
        }
        TyKind::Enum { variants, .. } => {
            for payload in variants.iter().filter_map(|(_, payload)| *payload) {
                for_each_position(types, lifetimes, payload, on_position);
            }
        }
        TyKind::UserDefined { id, type_args, .. } => {
            for _ in 0..lifetimes.lifetimes_of(*id) {
                on_position(PositionKind::Lifetime, ty);
            }
            for arg in type_args.iter() {
                for_each_position(types, lifetimes, *arg, on_position);
            }
        }
        TyKind::Var(var) => match *var {},
    }
}

/// The kind of every position of `ty`, in order.
pub fn position_kinds(types: &Types<Concrete>, lifetimes: &impl LifetimesOf, ty: Ty) -> Vec<PositionKind> {
    let mut kinds = Vec::new();
    for_each_position(types, lifetimes, ty, &mut |kind, _| kinds.push(kind));
    kinds
}

/// How many positions a value of `ty` holds loans at.
pub fn positions(types: &Types<Concrete>, lifetimes: &impl LifetimesOf, ty: Ty) -> usize {
    let mut count = 0;
    for_each_position(types, lifetimes, ty, &mut |_, _| count += 1);
    count
}

/// What each position of a value of `ty` lends, in order: a reference's
/// target at its own position, and no typed target at any other.
pub fn lent_types(types: &Types<Concrete>, lifetimes: &impl LifetimesOf, ty: Ty) -> Vec<Option<Ty>> {
    let mut lent = Vec::new();
    for_each_position(types, lifetimes, ty, &mut |_, of| {
        lent.push(match types.kind(of) {
            TyKind::Ref(_, target) => Some(*target),
            _ => None,
        })
    });
    lent
}

#[cfg(test)]
mod tests {
    use acvus_utils::{Interner, QualifiedRef};

    use super::inhabited;
    use crate::ty::{Concrete, IntTy, LenTerm, Mark, MarkTerm, Mutability, Ty, TyKind, Types};

    struct Fixture {
        names: Interner,
        types: Types<Concrete>,
    }

    impl Fixture {
        fn new() -> Self {
            Fixture { names: Interner::new(), types: Types::new() }
        }

        fn ty(&mut self, kind: TyKind<Concrete>) -> Ty {
            self.types.intern(kind)
        }

        fn never(&mut self) -> Ty {
            self.ty(TyKind::Never)
        }

        fn int(&mut self) -> Ty {
            self.ty(TyKind::Int(IntTy::I64))
        }

        /// The enum `E` of `variants`, each a tag and its payload.
        fn enum_of(&mut self, variants: &[(&str, Option<Ty>)]) -> Ty {
            let variants = variants.iter().map(|(tag, payload)| (self.names.intern(tag), *payload)).collect();
            let name = QualifiedRef::root(self.names.intern("E"));
            self.ty(TyKind::Enum { name, variants, label: () })
        }

        fn holds_a_value(&self, ty: Ty) -> bool {
            inhabited(&self.types, ty)
        }
    }

    #[test]
    fn never_has_no_value() {
        let mut fx = Fixture::new();
        let never = fx.never();
        assert!(!fx.holds_a_value(never));
    }

    /// A tuple or an object has a value where every part does.
    #[test]
    fn a_tuple_or_an_object_has_none_where_a_part_has_none() {
        let mut fx = Fixture::new();
        let (int, never) = (fx.int(), fx.never());
        let (a, b) = (fx.names.intern("a"), fx.names.intern("b"));
        let held = fx.ty(TyKind::Tuple(Box::new([int, never]), ()));
        let unit = fx.ty(TyKind::Tuple(Box::new([]), ()));
        let ints = fx.ty(TyKind::Tuple(Box::new([int, int]), ()));
        let object = fx.ty(TyKind::Object { fields: Box::new([(a, int), (b, never)]), label: () });
        assert!(!fx.holds_a_value(held));
        assert!(!fx.holds_a_value(object));
        assert!(fx.holds_a_value(unit));
        assert!(fx.holds_a_value(ints));
    }

    /// An array of no element has its one value whatever its element type;
    /// a longer one holds an element.
    #[test]
    fn an_array_has_none_only_where_it_holds_an_element_that_has_none() {
        let mut fx = Fixture::new();
        let (int, never) = (fx.int(), fx.never());
        let empty = fx.ty(TyKind::Array(never, LenTerm::Known(0), ()));
        let two = fx.ty(TyKind::Array(never, LenTerm::Known(2), ()));
        let ints = fx.ty(TyKind::Array(int, LenTerm::Known(2), ()));
        assert!(fx.holds_a_value(empty));
        assert!(!fx.holds_a_value(two));
        assert!(fx.holds_a_value(ints));
    }

    /// A value holds one variant: a sum has a value where some variant
    /// does, and a variant of no payload always does.
    #[test]
    fn a_sum_has_a_value_where_some_variant_does() {
        let mut fx = Fixture::new();
        let (int, never) = (fx.int(), fx.never());
        let none_only = fx.ty(TyKind::Option(never, ()));
        let ok_only = fx.ty(TyKind::Result(int, never, ()));
        let neither = fx.ty(TyKind::Result(never, never, ()));
        assert!(fx.holds_a_value(none_only));
        assert!(fx.holds_a_value(ok_only));
        assert!(!fx.holds_a_value(neither));
        let no_variant = fx.enum_of(&[]);
        let bare = fx.enum_of(&[("A", None)]);
        let held = fx.enum_of(&[("A", Some(never))]);
        let one_held = fx.enum_of(&[("A", Some(never)), ("B", None)]);
        assert!(!fx.holds_a_value(no_variant));
        assert!(fx.holds_a_value(bare));
        assert!(!fx.holds_a_value(held));
        assert!(fx.holds_a_value(one_held));
    }

    /// A reference lends storage holding a value of its target; a slice
    /// may be empty.
    #[test]
    fn a_reference_has_a_value_where_its_target_does() {
        let mut fx = Fixture::new();
        let (int, never) = (fx.int(), fx.never());
        let shared = fx.ty(TyKind::Ref(Mutability::Shared, never));
        let exclusive = fx.ty(TyKind::Ref(Mutability::Mut, never));
        let lent_int = fx.ty(TyKind::Ref(Mutability::Shared, int));
        let slice = fx.ty(TyKind::Slice(never));
        let lent_slice = fx.ty(TyKind::Ref(Mutability::Shared, slice));
        assert!(!fx.holds_a_value(shared));
        assert!(!fx.holds_a_value(exclusive));
        assert!(fx.holds_a_value(lent_int));
        assert!(fx.holds_a_value(lent_slice));
    }

    /// An extern-defined type has a value whatever its arguments: the
    /// language knows its values only through its declarations.
    #[test]
    fn an_extern_defined_type_has_a_value() {
        let mut fx = Fixture::new();
        let never = fx.never();
        let id = QualifiedRef::root(fx.names.intern("Vec"));
        let vec = fx.ty(TyKind::UserDefined {
            id,
            type_args: Box::new([never]),
            effect: None,
            identity: None,
            marked: MarkTerm::Known(Mark::Unmarked),
        });
        assert!(fx.holds_a_value(vec));
    }

    /// compilation §2: a type written in k steps is judged in O(k), so a
    /// tuple of two copies of the one before it, 64 deep, is judged at
    /// once.
    #[test]
    fn a_type_repeating_its_parts_is_judged_once_per_type() {
        let mut fx = Fixture::new();
        let mut ty = fx.never();
        for _ in 0..64 {
            ty = fx.ty(TyKind::Tuple(Box::new([ty, ty]), ()));
        }
        assert!(!fx.holds_a_value(ty));
    }
}
