//! The instructions of the CFG form. Control flow is only in terminators
//! (`body.rs`); nothing here names a block. `Nop` and `Poison` are gone: a
//! removal removes, and a refused program is never lowered (refusals).

use acvus_ast::Literal;
use acvus_utils::{Astr, QualifiedRef};

use super::BodyId;
use crate::ir::{
    BinOp, Callee, Checked, Chosen, ExternImpl, IndexBound, IndexMode, OrderEdge, PathSeg, RefTarget,
    UnaryOp, ValueId,
};
use crate::ty::{CastTy, Mutability, Ty};

#[derive(Debug, Clone)]
pub enum Op {
    // Constants
    Const {
        dst: ValueId,
        value: Literal,
    },
    /// Obligation across artifacts: the bytes `dst` names are not in this
    /// instruction. `acvus-interpreter`'s preparation copies each distinct
    /// text of a module once and the module's prepared code owns that copy,
    /// so a `&str` constant is the pointer and length of a run whose owner
    /// is the code holding this operation (RFC-0062 rule 2).
    ConstStr {
        dst: ValueId,
        text: String,
    },
    /// A template's output: the parts joined into one `String`. A part is a
    /// `String`, moved in, or a `&String`, read through.
    StringConcat {
        dst: ValueId,
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
        dst: ValueId,
        a: ValueId,
        b: ValueId,
    },
    /// A new `String` with the bytes of the one `src` names.
    StringClone {
        dst: ValueId,
        src: ValueId,
    },
    /// RFC-0020. `leaves` is in `structural::structural_leaves` order over
    /// the type `a` lends, the order `acvus_interpreter::prepare` reads it in.
    StructuralEq {
        dst: ValueId,
        a: ValueId,
        b: ValueId,
        leaves: Vec<Chosen>,
    },
    /// RFC-0020. `leaves` is in `structural::structural_leaves` order over
    /// the type `src` lends, the order `acvus_interpreter::prepare` reads it
    /// in.
    StructuralClone {
        dst: ValueId,
        src: ValueId,
        leaves: Vec<Chosen>,
    },

    // -- Storage (RFC-0018) -----------------------------------------
    /// A reference to a storage: `dst` is a `&T` or `&mut T` naming the
    /// value at `path` under `target`. `path: vec![]` is the storage itself,
    /// `vec![a, b]` the field `a.b` of it.
    Ref {
        dst: ValueId,
        target: RefTarget,
        path: Vec<PathSeg>,
        mutability: Mutability,
    },
    /// Move the value out of a storage into `dst`; a primitive is copied
    /// and the storage keeps it. A read of a variable.
    Take {
        dst: ValueId,
        target: RefTarget,
        path: Vec<PathSeg>,
        /// The take-out of a place lent to a call through a conversion
        /// (RFC-0041): the place is taken out, a word's as any other's,
        /// until the `Assign` that restores it after the call.
        taken_out: bool,
    },
    /// Move `value` into a storage; the storage's old value is dropped. An
    /// assignment to a variable.
    Assign {
        target: RefTarget,
        path: Vec<PathSeg>,
        value: ValueId,
        /// The store that puts back a place a `Take` took out for a call.
        restores: bool,
    },
    // -- Slices (RFC-0047) ------------------------------------------
    /// Take the whole run of `container`'s elements: `dst` is a
    /// `Ref(mutability, Slice(T))` holding the loan `container` holds, and
    /// `impl` is the `core::as_slice` / `core::as_slice_mut` the
    /// container's own evidence settled on.
    ///
    /// It is not a `FunctionCall` because the kind itself states what the
    /// hoist must know: a pure, infallible borrow projection, so
    /// `code_motion` lifts a shared one out of every loop that does not
    /// write the container, which it may never do for a call.
    AsSlice {
        dst: ValueId,
        container: ValueId,
        mutability: Mutability,
        r#impl: ExternImpl,
    },
    /// Element `index` of `slice`. The index is `u64`; the one check is
    /// `index < len`, made unless `bound` is `Proven`.
    Index {
        dst: ValueId,
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

    /// Move a context's whole value out of the page into `dst` (RFC-0025).
    Fetch {
        dst: ValueId,
        context: QualifiedRef,
    },
    /// Move `value` into the page as the context's whole value (RFC-0025).
    /// `wrote` holds where the body may have written the variable since its
    /// fetch (RFC-0025 rules 2, 4); a commit that did not write hands the
    /// value back and stores nothing.
    Commit {
        context: QualifiedRef,
        value: ValueId,
        wrote: bool,
    },
    // -- Scalar field access --------------------------------------
    /// Extract a field from a scalar value. 1+ depth via `field` + `rest`.
    FieldGet {
        dst: ValueId,
        object: ValueId,
        field: Astr,
        rest: Vec<Astr>,
    },
    /// Replace a field in a scalar value, producing a new value. 1+ depth.
    FieldSet {
        dst: ValueId,
        object: ValueId,
        field: Astr,
        rest: Vec<Astr>,
        value: ValueId,
    },

    // Arithmetic / logic
    BinOp {
        dst: ValueId,
        op: BinOp,
        left: ValueId,
        right: ValueId,
    },
    UnaryOp {
        dst: ValueId,
        op: UnaryOp,
        operand: ValueId,
    },
    /// Ends the run with `op`'s trap where `left op right` does not fit the
    /// operands' width. A pass that rewrites a trapping operation's value
    /// into a computation that wraps writes one at the operation's place, so
    /// the operation's trap stays where it was (RFC-0037 rule 3).
    Check {
        op: Checked,
        left: ValueId,
        right: ValueId,
    },
    /// Ends the run with `+`'s trap where `from + count·step`, over the
    /// integers, does not fit the width `from` and `step` share; `count` is
    /// a `u64`. It is, once, the `Check { op: Add }` of a value that starts
    /// at `from` and advances by `step` on each of `count` iterations: that
    /// value is monotone, so every step fits exactly where the last one
    /// does. The removal of a loop whose only work was such checks writes
    /// it at the loop's place, so their trap stays on exactly the runs that
    /// had it (RFC-0088 rule 8).
    CheckSteps {
        from: ValueId,
        step: ValueId,
        count: ValueId,
    },
    /// `src as to` (RFC-0049), total, with Rust's `as` values.
    ///
    /// It carries no source type, and that is a decision: `src`'s own type
    /// is the one every reader already holds, and a second copy of it here
    /// could disagree with it.
    ///
    /// `CastKind` above is not reused because it is not this: it names a
    /// pure ExternFn performing a declared coercion between user-defined
    /// types (RFC-0023 rule 8), which is a call, not a width.
    Cast {
        dst: ValueId,
        src: ValueId,
        to: CastTy,
    },

    // Functions
    /// Load a graph-level function into a value (for passing as argument, storing, etc.)
    LoadFunction {
        dst: ValueId,
        id: QualifiedRef,
    },
    /// Unified function call. Callee can be a direct graph function or an indirect value.
    /// Semantically equivalent to Spawn + TakeEffect (synchronous call = spawn then immediately eval).
    /// A call whose effect is not Pure carries an `OrderEdge`; a Pure call carries none.
    FunctionCall {
        dst: ValueId,
        callee: Callee,
        callee_ty: Ty,
        args: Vec<ValueId>,
        order: Option<OrderEdge>,
    },
    /// Issue a call and receive a Handle<T> for its result. The work starts
    /// here; `order` is the `Order` it waits for when the call is effectful.
    /// `dst` receives a Handle whose type carries the callee's return type.
    Spawn {
        dst: ValueId,
        callee: Callee,
        callee_ty: Ty,
        args: Vec<ValueId>,
        order: Option<ValueId>,
    },
    /// Evaluate (force) a Handle, consuming it.
    /// `src` must be a Handle<T>. `dst` receives T; `order` receives the
    /// `Order` that follows the call when the call is effectful.
    TakeEffect {
        dst: ValueId,
        src: ValueId,
        order: Option<ValueId>,
    },
    /// Join orders: `dst` follows every order in `orders`. Associative and
    /// commutative; a value instruction, not control flow.
    Merge {
        dst: ValueId,
        orders: Vec<ValueId>,
    },

    // Composite constructors
    /// An array of no elements, `Array<T, 0>`, its storage allocated for
    /// `capacity` elements. An array is built from it by `ArrayPush`, one
    /// element at a time as each is computed, so an element's value is
    /// live only from its definition to its push.
    ArrayBegin {
        dst: ValueId,
        capacity: usize,
    },
    /// `array`, an `Array<T, k>`, moved in with `value` moved onto its end:
    /// `dst` is the `Array<T, k + 1>`. Every value either writes is a whole
    /// array of the length its type states.
    ArrayPush {
        dst: ValueId,
        array: ValueId,
        value: ValueId,
    },
    MakeObject {
        dst: ValueId,
        fields: Vec<(Astr, ValueId)>,
    },
    MakeTuple {
        dst: ValueId,
        elements: Vec<ValueId>,
    },
    TupleIndex {
        dst: ValueId,
        tuple: ValueId,
        index: usize,
    },

    // Pattern matching (decision tree)
    TestLiteral {
        dst: ValueId,
        src: ValueId,
        value: Literal,
    },
    TestObjectKey {
        dst: ValueId,
        src: ValueId,
        key: Astr,
    },
    ArrayIndex {
        dst: ValueId,
        array: ValueId,
        index: usize,
    },
    ObjectGet {
        dst: ValueId,
        object: ValueId,
        key: Astr,
    },

    // Closures
    MakeClosure {
        dst: ValueId,
        body: BodyId,
        captures: Vec<ValueId>,
    },

    // Variant (tagged union)
    MakeVariant {
        dst: ValueId,
        tag: Astr,
        payload: Option<ValueId>,
    },
    TestVariant {
        dst: ValueId,
        src: ValueId,
        tag: Astr,
    },
    UnwrapVariant {
        dst: ValueId,
        src: ValueId,
    },
    /// Undefined value - valid to move/copy, UB to read as a concrete value.
    /// Used as initial value for SSA variables that are defined inside loops
    /// (iteration bindings, write-only contexts).
    Undef {
        dst: ValueId,
    },

    /// Drop a value, releasing its resources.
    ///
    /// Inserted by the compiler at the end of a value's live range (last use
    /// or scope exit). No `dst` - Drop only consumes, never produces.
    ///
    /// At runtime, the interpreter calls the Owned vtable's drop function.
    /// Copy types (SBO) have a no-op drop. Boxed types invoke their destructor.
    Drop {
        src: ValueId,
    },
}
