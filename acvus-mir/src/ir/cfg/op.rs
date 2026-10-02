//! The instructions of the CFG form. Control flow is only in terminators
//! (`body.rs`); nothing here names a block. `Nop` and `Poison` are gone: a
//! removal removes, and a refused program is never lowered (refusals).

use acvus_ast::Literal;
use acvus_utils::{Astr, QualifiedRef};

use super::BodyId;
use crate::ir::{
    BinOp, Callee, IndexBound, IndexMode, PathSeg, RefTarget, UnaryOp, ValueId,
};
use crate::ty::{CastTy, Concrete, Mutability, Ty, Types, copies};

#[derive(Debug, Clone)]
pub enum Op {
    // Constants
    Const {
        value: Literal,
    },
    /// Obligation across artifacts: the bytes the result names are not in this
    /// instruction. `acvus-interpreter`'s preparation copies each distinct
    /// text of a module once and the module's prepared code owns that copy,
    /// so a `&str` constant is the pointer and length of a run whose owner
    /// is the code holding this operation (RFC-0062 rule 2).
    ConstStr {
        text: String,
    },
    /// A template's output: the parts joined into one `String`. A part is a
    /// `String`, moved in, or a `&String`, read through.
    StringConcat {
        parts: Vec<ValueId>,
    },
    /// A template's append: the bytes of `part` are written onto the end of
    /// the `String` that `target`, a `&mut String`, names. `part` is a
    /// `String`, moved in, or a `&str`/`&String`, read through, as a
    /// `StringConcat` part is. The accumulator is not copied, which is what
    /// keeps a loop body's appends linear in the text they write
    /// (RFC-0071).
    StringAppend {
        target: ValueId,
        part: ValueId,
    },
    /// Whether the two strings `a` and `b` name hold the same bytes.
    StringEq {
        a: ValueId,
        b: ValueId,
    },
    /// A new `String` with the bytes of the one `src` names.
    StringClone {
        src: ValueId,
    },

    // -- Storage (RFC-0018) -----------------------------------------
    /// A reference to a storage: the result is a `&T` or `&mut T` naming the
    /// value at `path` under `target`. `path: vec![]` is the storage itself,
    /// `vec![a, b]` the field `a.b` of it.
    Ref {
        target: RefTarget,
        path: Vec<PathSeg>,
        mutability: Mutability,
    },
    /// Move the value out of a storage into the result; a primitive is copied
    /// and the storage keeps it. A read of a variable.
    /// A `Take` of a `&mut` is a reborrow, not a move (ownership §5): a `&mut`
    /// does not copy, and each use reborrows it, so the storage keeps it.
    Take {
        target: RefTarget,
        path: Vec<PathSeg>,
        /// The take-out of a place lent to a call through a conversion
        /// (RFC-0041): the place is taken out, a word's as any other's,
        /// until the `Assign` that restores it after the call.
        taken_out: bool,
    },
    /// Move `value` into a storage; it drops nothing. The old content, where
    /// the program holds it, is dropped by an explicit `Drop` before this
    /// store (ownership §6). An assignment to a variable.
    Assign {
        target: RefTarget,
        path: Vec<PathSeg>,
        value: ValueId,
        /// The store that puts back a place a `Take` took out for a call.
        restores: bool,
    },
    // -- Slices (RFC-0047) ------------------------------------------
    /// Take the whole run of `container`'s elements: the result is a
    /// `Ref(mutability, Slice(T))` holding the loan `container` holds, and
    /// `impl` is the impl of `core::as_slice` / `core::as_slice_mut` the
    /// checker chose for the container.
    ///
    /// It is not a `FunctionCall` because the kind itself states that it
    /// is a pure, infallible borrow projection.
    AsSlice {
        container: ValueId,
        mutability: Mutability,
        r#impl: QualifiedRef,
    },
    /// The whole run of the elements of the array `src` names: a
    /// `&[T; N]` becomes a `&[T]`, and a `&mut [T; N]` a `&mut [T]`. It is
    /// the language's own view, since no declaration carries an array
    /// (extern-fn §1), so no impl runs; it neither traps nor has an effect.
    ArrayView {
        src: ValueId,
        mutability: Mutability,
    },
    /// The number of elements of the slice `src`, a `&[T]` or `&mut [T]`,
    /// references: the language's own, as a `u64`. It neither traps nor has
    /// an effect.
    SliceLen {
        src: ValueId,
    },
    /// Element `index` of `slice`. The index is `u64`; the one check is
    /// `index < len`, made unless `bound` is `Proven`.
    Index {
        slice: ValueId,
        index: ValueId,
        mode: IndexMode,
        bound: IndexBound,
    },
    /// Write `value` into element `index` of a `&mut [T]`, dropping the
    /// element that was there.
    IndexSet {
        slice: ValueId,
        index: ValueId,
        value: ValueId,
        bound: IndexBound,
    },

    // -- Scalar field access --------------------------------------
    /// Extract a field from a scalar value. 1+ depth via `field` + `rest`.
    FieldGet {
        object: ValueId,
        field: Astr,
        rest: Vec<Astr>,
    },
    /// Replace a field in a scalar value, producing a new value. 1+ depth.
    FieldSet {
        object: ValueId,
        field: Astr,
        rest: Vec<Astr>,
        value: ValueId,
    },

    // Arithmetic / logic
    BinOp {
        op: BinOp,
        left: ValueId,
        right: ValueId,
    },
    UnaryOp {
        op: UnaryOp,
        operand: ValueId,
    },
    /// `src as to` (RFC-0049), total, with Rust's `as` values.
    ///
    /// It carries no source type, and that is a decision: `src`'s own type
    /// is the one every reader already holds, and a second copy of it here
    /// could disagree with it.
    ///
    /// A declared coercion between user-defined types (RFC-0023 rule 8)
    /// is not this: the lowering issues it as a call of the pure ExternFn
    /// performing it, which is a call, not a width.
    Cast {
        src: ValueId,
        to: CastTy,
    },
    /// A value a call gave at its declared result type, taken at the type
    /// its uses settled, which differs from it only in marks raised to
    /// unmarked (types §8): the one place its representation changes. A
    /// backend treating `#F<A>` as `F<A>` makes it a copy.
    Unmark {
        src: ValueId,
    },

    // Functions
    /// Unified function call. Callee can be a direct graph function or an indirect value.
    /// Semantically equivalent to Spawn + TakeEffect (synchronous call = spawn then immediately eval).
    /// A call whose effect is not Pure has an order edge on its instruction
    /// record; a Pure call has none.
    FunctionCall {
        callee: Callee,
        callee_ty: Ty,
        args: Vec<ValueId>,
    },
    /// Issue a call and receive a Handle<T> for its result. The work starts
    /// here; its record consumes the `Order` it waits for when the call is
    /// effectful. The result is a Handle whose type carries the callee's
    /// return type.
    Spawn {
        callee: Callee,
        callee_ty: Ty,
        args: Vec<ValueId>,
    },
    /// Evaluate (force) a Handle, consuming it.
    /// `src` must be a Handle<T>. The result is the T; its record yields the
    /// `Order` that follows the call when the call is effectful.
    TakeEffect {
        src: ValueId,
    },
    /// Join orders: the result follows every order in `orders`. Associative and
    /// commutative; a value instruction, not control flow.
    Merge {
        orders: Vec<ValueId>,
    },

    // Composite constructors
    /// An array of no elements, `Array<T, 0>`, its storage allocated for
    /// `capacity` elements. An array is built from it by `ArrayPush`, one
    /// element at a time as each is computed, so an element's value is
    /// live only from its definition to its push.
    ArrayBegin {
        capacity: usize,
    },
    /// `array`, an `Array<T, k>`, moved in with `value` moved onto its end:
    /// the result is the `Array<T, k + 1>`. Every value either writes is a whole
    /// array of the length its type states.
    ArrayPush {
        array: ValueId,
        value: ValueId,
    },
    MakeObject {
        fields: Vec<(Astr, ValueId)>,
    },
    /// An intrinsic of the language over `args`, as `Intrinsic` states.
    Intrinsic {
        intrinsic: crate::ir::Intrinsic,
        args: Vec<ValueId>,
    },
    MakeTuple {
        elements: Vec<ValueId>,
    },
    TupleIndex {
        tuple: ValueId,
        index: usize,
    },

    // Pattern matching (decision tree)
    /// Whether `src` is `value`. `src` is the value itself, or a reference
    /// to it, read through as `TestVariant` reads a tag: a test on a place
    /// takes nothing out of it (D89).
    TestLiteral {
        src: ValueId,
        value: Literal,
    },
    TestObjectKey {
        src: ValueId,
        key: Astr,
    },
    ArrayIndex {
        array: ValueId,
        index: usize,
    },
    ObjectGet {
        object: ValueId,
        key: Astr,
    },

    // Closures
    /// A closure over the environment `captures`, taken once where the
    /// lambda is written (functions §2), with code at each signature its
    /// result's type states: one body for a `Fn` type, and for a `Closures`
    /// type one per signature, each paired with its signature. The pairing
    /// is written in the set's canonical order (`TyKind::Closures`); that
    /// is its only representation, and no pass reads a position as
    /// meaning anything else.
    MakeClosure {
        bodies: Vec<BodyId>,
        captures: Vec<ValueId>,
    },
    /// `closure`, of a `Closures` type, kept at the set of signatures
    /// `signatures`, some of those it has, selected by signature, over the
    /// same environment: a use of a generalized binding at its instance,
    /// one signature, or where the value enters another generalized
    /// binding, that binding's signatures (functions §2). `signatures` is
    /// written in the set's canonical order. The result is a `Fn` type for
    /// one signature, and a `Closures` type of that set otherwise.
    Specialize {
        closure: ValueId,
        signatures: Vec<Ty>,
    },

    // Variant (tagged union)
    MakeVariant {
        tag: Astr,
        payload: Option<ValueId>,
    },
    /// Whether the variant `src` holds is `tag`'s. `src` is the variant
    /// itself, or a reference to it, read through as a `Switch` reads its
    /// tag through one: a pattern past a reference tests the variant the
    /// reference names without taking it out.
    TestVariant {
        src: ValueId,
        tag: Astr,
    },
    UnwrapVariant {
        src: ValueId,
    },
    /// Undefined value - valid to move/copy, UB to read as a concrete value.
    /// Used as initial value for SSA variables that are defined inside loops
    /// (iteration bindings, write-only contexts).
    Undef,

    /// Drop a value, releasing its resources.
    ///
    /// Inserted by the compiler at the end of a value's live range (last use
    /// or scope exit). Drop only consumes, never produces.
    ///
    /// At runtime, the interpreter calls the Owned vtable's drop function.
    /// Copy types (SBO) have a no-op drop. Boxed types invoke their destructor.
    Drop {
        src: ValueId,
    },
}

/// Walks every operand of an `Op`, by shared or by mutable reference: one
/// list, so the two walks cannot disagree on what an operand is.
macro_rules! walk_operands {
    ($op:expr, $on:ident, $($r:tt)*) => {{
        let on_target = |target: $($r)* RefTarget, $on: &mut dyn FnMut($($r)* ValueId)| match target {
            RefTarget::Var(v) | RefTarget::Param(v) | RefTarget::Through(v) => $on(v),
            RefTarget::Context(_) => {}
        };
        let on_callee = |callee: $($r)* Callee, $on: &mut dyn FnMut($($r)* ValueId)| match callee {
            Callee::Indirect(v) => $on(v),
            Callee::Direct(_) | Callee::Extern { .. } => {}
        };
        match $op {
            Op::Const { .. }
            | Op::ConstStr { .. }
            | Op::ArrayBegin { .. }
            | Op::Undef => {}
            Op::StringConcat { parts } => parts.into_iter().for_each(|v| $on(v)),
            Op::StringAppend { target, part } => {
                $on(target);
                $on(part);
            }
            Op::StringEq { a, b } => {
                $on(a);
                $on(b);
            }
            Op::StringClone { src }
            | Op::Cast { src, .. }
            | Op::Unmark { src }
            | Op::TakeEffect { src }
            | Op::TestLiteral { src, .. }
            | Op::TestObjectKey { src, .. }
            | Op::TestVariant { src, .. }
            | Op::SliceLen { src }
            | Op::UnwrapVariant { src }
            | Op::Drop { src } => $on(src),
            Op::Ref { target, .. } | Op::Take { target, .. } => on_target(target, $on),
            Op::Assign { target, value, .. } => {
                on_target(target, $on);
                $on(value);
            }
            Op::AsSlice { container, .. } => $on(container),
            Op::ArrayView { src, .. } => $on(src),
            Op::Index { slice, index, .. } => {
                $on(slice);
                $on(index);
            }
            Op::IndexSet { slice, index, value, .. } => {
                $on(slice);
                $on(index);
                $on(value);
            }
            Op::FieldGet { object, .. } | Op::ObjectGet { object, .. } => $on(object),
            Op::FieldSet { object, value, .. } => {
                $on(object);
                $on(value);
            }
            Op::BinOp { left, right, .. } => {
                $on(left);
                $on(right);
            }
            Op::UnaryOp { operand, .. } => $on(operand),
            Op::FunctionCall { callee, args, .. } | Op::Spawn { callee, args, .. } => {
                on_callee(callee, $on);
                args.into_iter().for_each(|v| $on(v));
            }
            Op::Merge { orders } => orders.into_iter().for_each(|v| $on(v)),
            Op::ArrayPush { array, value } => {
                $on(array);
                $on(value);
            }
            Op::MakeObject { fields } => fields.into_iter().for_each(|(_, v)| $on(v)),
            Op::MakeTuple { elements } => elements.into_iter().for_each(|v| $on(v)),
            Op::Intrinsic { args, .. } => args.into_iter().for_each(|v| $on(v)),
            Op::TupleIndex { tuple, .. } => $on(tuple),
            Op::ArrayIndex { array, .. } => $on(array),
            Op::MakeClosure { captures, .. } => captures.into_iter().for_each(|v| $on(v)),
            Op::Specialize { closure, .. } => $on(closure),
            Op::MakeVariant { payload, .. } => {
                if let Some(payload) = payload {
                    $on(payload);
                }
            }
        }
    }};
}

impl Op {
    /// Every value the operation reads, once per operand. The order token
    /// an effectful instruction consumes is on its record, not here.
    pub(super) fn for_each_operand(&self, mut on_use: impl FnMut(ValueId)) {
        let on_use: &mut dyn FnMut(&ValueId) = &mut |v| on_use(*v);
        walk_operands!(self, on_use, &)
    }

    pub(super) fn for_each_operand_mut(&mut self, mut on_use: impl FnMut(&mut ValueId)) {
        let on_use: &mut dyn FnMut(&mut ValueId) = &mut on_use;
        walk_operands!(self, on_use, &mut)
    }

    /// The place a `Take`, an `Assign` or a `Ref` names: its target and its
    /// path. Cost: O(1).
    pub fn place(&self) -> Option<(&RefTarget, &[PathSeg])> {
        match self {
            Op::Take { target, path, .. } | Op::Assign { target, path, .. } | Op::Ref { target, path, .. } => {
                Some((target, path))
            }
            _ => None,
        }
    }

    /// As `place`, to rewrite it. Cost: O(1).
    pub(super) fn place_mut(&mut self) -> Option<(&mut RefTarget, &mut Vec<PathSeg>)> {
        match self {
            Op::Take { target, path, .. } | Op::Assign { target, path, .. } | Op::Ref { target, path, .. } => {
                Some((target, path))
            }
            _ => None,
        }
    }

    /// Whether the operation yields a value; the ones that do not only
    /// write.
    pub fn has_result(&self) -> bool {
        match self {
            Op::StringAppend { .. }
            | Op::Assign { .. }
            | Op::IndexSet { .. }
            | Op::Drop { .. } => false,
            Op::Const { .. }
            | Op::ConstStr { .. }
            | Op::StringConcat { .. }
            | Op::StringEq { .. }
            | Op::StringClone { .. }
            | Op::Ref { .. }
            | Op::Take { .. }
            | Op::AsSlice { .. }
            | Op::ArrayView { .. }
            | Op::Index { .. }
            | Op::FieldGet { .. }
            | Op::FieldSet { .. }
            | Op::BinOp { .. }
            | Op::UnaryOp { .. }
            | Op::Cast { .. }
            | Op::Unmark { .. }
            | Op::FunctionCall { .. }
            | Op::Spawn { .. }
            | Op::TakeEffect { .. }
            | Op::Merge { .. }
            | Op::ArrayBegin { .. }
            | Op::ArrayPush { .. }
            | Op::MakeObject { .. }
            | Op::MakeTuple { .. }
            | Op::Intrinsic { .. }
            | Op::TupleIndex { .. }
            | Op::TestLiteral { .. }
            | Op::TestObjectKey { .. }
            | Op::ArrayIndex { .. }
            | Op::ObjectGet { .. }
            | Op::MakeClosure { .. }
            | Op::Specialize { .. }
            | Op::MakeVariant { .. }
            | Op::TestVariant { .. }
            | Op::SliceLen { .. }
            | Op::UnwrapVariant { .. }
            | Op::Undef => true,
        }
    }

    /// Whether running the operation on more runs or fewer than its place
    /// runs on is observable, apart from an effect its callee's type
    /// declares: it may trap (compilation §1), it writes storage, or it is
    /// a `Take` of a type that does not copy, which moves the value out and
    /// so empties its place (ownership §1-2). `result` is the type of what
    /// it yields. Such an operation, like an effectful one, moves only
    /// between places that run on exactly the same runs. Cost: O(the
    /// result type's size) for a `Take`, O(1) otherwise.
    ///
    /// # Panics
    /// The operation is a `Take` and `result` is none: a `Take` yields the
    /// value it reads.
    pub fn bound_to_its_paths(&self, result: Option<Ty>, types: &Types<Concrete>) -> bool {
        let moves_out = match self {
            Op::Take { .. } => {
                let Some(result) = result else { panic!("a Take yields the value it reads") };
                !copies(types, result)
            }
            _ => false,
        };
        self.may_trap() || self.writes_storage() || moves_out
    }

    /// Whether the operation may end the run with a trap (compilation §1:
    /// whether a run traps is its meaning), decided by its kind and its
    /// checked flags alone. A `Cast` is Rust's total `as` (RFC-0049) and
    /// never traps.
    pub fn may_trap(&self) -> bool {
        match self {
            Op::UnwrapVariant { .. } => true,
            Op::BinOp { op, .. } => op.can_trap_on_integers(),
            Op::Index { bound, .. } | Op::IndexSet { bound, .. } => *bound == IndexBound::Checked,
            Op::Const { .. }
            | Op::ConstStr { .. }
            | Op::StringConcat { .. }
            | Op::StringAppend { .. }
            | Op::StringEq { .. }
            | Op::StringClone { .. }
            | Op::Ref { .. }
            | Op::Take { .. }
            | Op::Assign { .. }
            | Op::AsSlice { .. }
            | Op::ArrayView { .. }
            | Op::FieldGet { .. }
            | Op::FieldSet { .. }
            | Op::UnaryOp { .. }
            | Op::Cast { .. }
            | Op::Unmark { .. }
            | Op::FunctionCall { .. }
            | Op::Spawn { .. }
            | Op::TakeEffect { .. }
            | Op::Merge { .. }
            | Op::ArrayBegin { .. }
            | Op::ArrayPush { .. }
            | Op::MakeObject { .. }
            | Op::MakeTuple { .. }
            | Op::TupleIndex { .. }
            | Op::TestLiteral { .. }
            | Op::TestObjectKey { .. }
            | Op::ArrayIndex { .. }
            | Op::ObjectGet { .. }
            | Op::MakeClosure { .. }
            | Op::Specialize { .. }
            | Op::MakeVariant { .. }
            | Op::TestVariant { .. }
            | Op::SliceLen { .. }
            | Op::Undef
            | Op::Drop { .. } => false,
            // `Gather` only assembles references; `ArrayMap` traps where
            // its function does, which that function's type states, as a
            // `FunctionCall`'s does.
            Op::Intrinsic { .. } => false,
        }
    }

    /// Whether the operation writes storage, or ends a value it holds:
    /// removing one still changes what the storage holds. A `FieldSet`
    /// yields a new value and writes nothing.
    pub fn writes_storage(&self) -> bool {
        matches!(self, Op::Assign { .. } | Op::IndexSet { .. } | Op::StringAppend { .. } | Op::Drop { .. })
    }

    /// Whether the instruction reads or writes a storage.
    pub(super) fn touches_storage(&self) -> bool {
        matches!(self, Op::Ref { .. } | Op::Take { .. } | Op::Assign { .. } | Op::IndexSet { .. } | Op::StringAppend { .. })
    }
}
