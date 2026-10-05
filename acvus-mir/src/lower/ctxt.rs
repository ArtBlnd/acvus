//! What lowering one body holds: the Builder, what typeck decided, the home
//! of each binding, the targets of the loops it is inside, and the home of
//! each context it names. A lambda's body and a map's function are lowered
//! with their own `Lcx` over a nested Builder.
//!
//! A form, and a value built from a form's by a coercion or by a structure
//! tree, yields `Lowered`: its value where a run gets past it, the block it
//! ends in open, and `Unreached` where none does, the block it ends in
//! terminated and no block open (types §4, control-flow).
//!
//! The Builder is this module's alone. Every value lowering makes comes from
//! one of three points here: an instruction's result (`emit`, `made`), a
//! block's parameters (`enter_block`), a body's parameters and captures
//! (`make_body`). No value has a type such as `!`
//! (`ty::inhabited`), so no run gets past a point that gives one: each of
//! these ends the block there, and every value lowering goes on with has a
//! type some value has. An effect is appended as one by what its op is
//! (`Builder::add`), not by the caller's choice.

use rustc_hash::FxHashMap;

use acvus_ast::{AstId, Literal, Span};
use acvus_utils::{Astr, QualifiedRef};

use crate::decl::{Declarations, PairMember};
use crate::ir::cfg::{BlockId, BodyId, Builder, Op, Terminator};
use crate::ir::{Callee, Chosen as IrChosen, Intrinsic, RefTarget, ValueId};
use crate::ty::{Concrete, Effect, IntTy, Mutability, Reissue, Ty, TyKind, Types, inhabited};
use crate::typeck::{Chosen, Impl, Typeck};

/// No run gets past the form that yields this: the block lowering appended
/// to is terminated, and no block is open.
#[derive(Debug, Clone, Copy)]
pub(super) struct Unreached;

/// A form's value where a run gets past it, the block it ends in open;
/// `Unreached` where none does.
pub(super) type Lowered<T> = Result<T, Unreached>;

/// Where `break` and `continue` go: the innermost loop's exit and header
/// (control-flow).
#[derive(Debug, Clone, Copy)]
pub(super) struct LoopTargets {
    pub exit: BlockId,
    pub header: BlockId,
}

/// Where a binding lives: a slot of this body, or, for a name a lambda
/// captured by reference, the storage its captured reference names
/// (functions §2).
#[derive(Debug, Clone, Copy)]
pub(super) enum Home {
    Slot(ValueId),
    Through(ValueId),
}

pub(super) struct Lcx<'a, 'm> {
    pub decls: &'a Declarations,
    /// The run's typeck, for what is decided once where a lambda is
    /// written (its captures) and for the run as a whole.
    pub typeck: &'a Typeck,
    /// What this body reads at each of its sites: the run's, or the
    /// instance's it is lowered at (functions §2).
    pub decided: super::decided::Decided<'a>,
    /// Each generalized lambda's body at each signature, made once for the
    /// module (functions §2: a signature determines its code).
    pub bodies: std::rc::Rc<std::cell::RefCell<FxHashMap<(AstId, Ty), crate::ir::cfg::BodyId>>>,
    builder: Builder<'m>,
    pub bindings: FxHashMap<AstId, Home>,
    /// The span of each binder, where its binding was made.
    pub binder_spans: FxHashMap<AstId, Span>,
    pub loops: Vec<LoopTargets>,
    /// Where each input the body names lives (programs §4): a slot of the
    /// entry body, or a lambda's capture of it.
    pub inputs: FxHashMap<Astr, Home>,
    /// How many `anyorder` scopes are open around the current position; a
    /// lambda written inside one belongs to it (effects §3).
    pub anyorder: usize,
    /// A template's output, the `String` slot text is appended to.
    pub out: Option<ValueId>,
}

impl<'a, 'm> Lcx<'a, 'm> {
    pub fn new(decls: &'a Declarations, decided: super::decided::Decided<'a>, builder: Builder<'m>, anyorder: usize) -> Self {
        Lcx {
            decls,
            typeck: decided.typeck,
            decided,
            bodies: std::rc::Rc::default(),
            builder,
            bindings: FxHashMap::default(),
            binder_spans: FxHashMap::default(),
            loops: Vec::new(),
            inputs: FxHashMap::default(),
            anyorder,
            out: None,
        }
    }

    /// A body nested in this one's, lowered over `decided`, ordered where
    /// `ordered`, returning a `ret`: a lambda's at one signature, or a map's
    /// function. It belongs to every `anyorder` scope this one is in
    /// (effects §3), and makes each generalized lambda's body at each
    /// signature once with this one (functions §2).
    pub fn nested<'d>(&mut self, decided: super::decided::Decided<'d>, ordered: bool, ret: Ty) -> Lcx<'d, '_>
    where
        'a: 'd,
    {
        let (decls, bodies, anyorder) = (self.decls, self.bodies.clone(), self.anyorder);
        let builder = self.builder.build_nested(ordered, ret);
        let mut nested = Lcx::new(decls, decided, builder, anyorder);
        nested.bodies = bodies;
        nested
    }

    /// The body this `Lcx` builds, one skeleton for every body: an entry, a
    /// lambda at a signature, a map's function. Its `anyorder` scope opens
    /// where it belongs to one (effects §3); its captures, its parameters
    /// and the contexts it names are declared (context §3). No run enters a
    /// body given a value of a type no value has, so where a capture or a
    /// parameter has one, its entry ends there, before any instruction.
    /// Otherwise `content`, given the captures and the parameters, lowers
    /// it, and its value is returned where a run gets past it.
    pub fn make_body(
        mut self,
        entry: Entry<'_>,
        content: impl FnOnce(&mut Self, &[ValueId], &[ValueId]) -> Lowered<ValueId>,
        span: Span,
    ) -> BodyId {
        if self.anyorder > 0 {
            self.builder.open_anyorder();
        }
        let captures: Vec<ValueId> =
            entry.captures.iter().map(|declared| self.builder.capture(declared.name, declared.ty, declared.span)).collect();
        let params: Vec<ValueId> =
            entry.params.iter().map(|declared| self.builder.param(declared.name, declared.ty, declared.span)).collect();
        for (context, ty) in entry.contexts {
            let volatile = self
                .decls
                .context_is_volatile(*context)
                .unwrap_or_else(|| panic!("the context {context:?} an admitted program names is declared"));
            self.builder.context(*context, *ty, volatile);
        }
        let entered = captures
            .iter()
            .zip(&entry.captures)
            .chain(params.iter().zip(&entry.params))
            .try_for_each(|(value, declared)| self.judged(*value, declared.span).map(|_| ()));
        if let Ok(value) = entered.and_then(|()| content(&mut self, &captures, &params)) {
            self.return_value(value, span);
        }
        if self.anyorder > 0 {
            self.builder.close_anyorder();
        }
        self.builder.seal().unwrap_or_else(|open| panic!("a lowered body is sealed whole: {open:?}"))
    }

    /// The module's one type table, for reading.
    pub fn types(&self) -> &Types<Concrete> {
        self.builder.types()
    }

    /// The id of `kind` in the module's one type table (`Builder::intern`).
    pub fn intern(&mut self, kind: TyKind<Concrete>) -> Ty {
        self.builder.intern(kind)
    }

    /// An instruction whose result is a part or a target of an operand, or
    /// a callee's result (`Yields::Taken`), appended as what it is, an
    /// effect or not (`Builder::add`), its result judged: where no value
    /// has its type, no run gets past it, the block ends there, and it is
    /// `Unreached`. A `Ref` passes here whatever it lends: whether it is a
    /// reborrow, whose result is made, is its place's to decide.
    ///
    /// # Panics
    /// `op` is neither of those: one made from its operands is `made`'s.
    pub fn emit(&mut self, op: Op, ty: Ty, span: Span) -> Lowered<ValueId> {
        assert!(
            matches!(yields(&op), Yields::Taken) || matches!(op, Op::Ref { .. }),
            "{op:?} yields no part, target or callee's result"
        );
        let value = self.builder.add(op, ty, span);
        self.judged(value, span)
    }

    /// An instruction whose result is made from its operands
    /// (`Yields::Made`), appended as what it is. Its operands are values
    /// that exist, so some value has its type.
    ///
    /// # Panics
    /// `op` does not make its result from its operands, or no value has
    /// `ty`.
    pub fn made(&mut self, op: Op, ty: Ty, span: Span) -> ValueId {
        assert!(matches!(yields(&op), Yields::Made), "{op:?} does not make its result from its operands");
        assert!(inhabited(self.builder.types(), ty), "{op:?} is made of values, so some value has its type, and not {ty:?}");
        self.builder.add(op, ty, span)
    }

    /// An instruction that yields nothing, appended as what it is, an
    /// effect or not (`Builder::add_unit`).
    ///
    /// # Panics
    /// `op` yields a value.
    pub fn emit_unit(&mut self, op: Op, span: Span) {
        assert!(matches!(yields(&op), Yields::Nothing), "{op:?} yields a value");
        self.builder.add_unit(op, span);
    }

    /// A block of this body, entered later, taking parameters of `params`.
    pub fn new_block(&mut self, params: &[Ty]) -> BlockId {
        self.builder.new_block(params)
    }

    /// Goes on in `block`, its parameters what each edge into it gives. No
    /// run enters a block given a value of a type no value has: where a
    /// parameter has one, the block ends there, before any instruction, and
    /// it is `Unreached`.
    pub fn enter_block(&mut self, block: BlockId, span: Span) -> Lowered<Vec<ValueId>> {
        self.builder.switch_to(block);
        let params = self.builder.block_params(block).to_vec();
        params.iter().try_for_each(|param| self.judged(*param, span).map(|_| ()))?;
        Ok(params)
    }

    /// Ends the current block with a terminator into a region's blocks: a
    /// `Diamond`, a `Switch`, a `While`, a `For` or a `JumpIf`.
    ///
    /// # Panics
    /// `terminator` leads into no region: a run past it is ended by
    /// `end_block`, and an edge to one block made by `jump`.
    pub fn terminate(&mut self, terminator: Terminator, span: Span) {
        assert!(
            matches!(
                terminator,
                Terminator::Diamond { .. } | Terminator::Switch { .. } | Terminator::While { .. } | Terminator::For { .. } | Terminator::JumpIf { .. }
            ),
            "{terminator:?} leads into no region"
        );
        self.builder.terminate(terminator, span);
    }

    /// A storage slot of the body holding a `ty` (`Builder::slot`).
    pub fn slot(&mut self, ty: Ty, span: Span) -> ValueId {
        self.builder.slot(ty, span)
    }

    /// `value` where some value has its type; where none does, no run gets
    /// past the point that gives it, and the block ends there, in
    /// `Diverge`.
    fn judged(&mut self, value: ValueId, span: Span) -> Lowered<ValueId> {
        match inhabited(self.builder.types(), self.value_ty(value)) {
            true => Ok(value),
            false => Err(self.end_block(Terminator::Diverge, span)),
        }
    }

    /// The type typeck recorded at `site`.
    ///
    /// # Panics
    /// typeck recorded none: every site of an admitted program is typed.
    pub fn ty(&self, site: AstId) -> Ty {
        self.decided.ty(site)
    }

    pub fn kind(&self, ty: Ty) -> &TyKind<crate::ty::Concrete> {
        self.builder.types().kind(ty)
    }

    pub fn value_ty(&self, value: ValueId) -> Ty {
        self.builder.ty_of(value)
    }

    pub fn bool_ty(&mut self) -> Ty {
        self.intern(TyKind::Bool)
    }

    pub fn unit_ty(&mut self) -> Ty {
        self.intern(TyKind::Unit)
    }

    pub fn u64_ty(&mut self) -> Ty {
        self.intern(TyKind::Int(IntTy::U64))
    }

    pub fn ref_ty(&mut self, mutability: Mutability, target: Ty) -> Ty {
        self.intern(TyKind::Ref(mutability, target))
    }

    pub fn constant(&mut self, value: Literal, ty: Ty, span: Span) -> ValueId {
        self.made(Op::Const { value }, ty, span)
    }

    pub fn bool_const(&mut self, value: bool, span: Span) -> ValueId {
        let ty = self.bool_ty();
        self.constant(Literal::Bool(value), ty, span)
    }

    pub fn unit(&mut self, span: Span) -> ValueId {
        let ty = self.unit_ty();
        self.constant(Literal::Unit, ty, span)
    }

    /// Ends the current block with a terminator no run passes on from, such
    /// as a `Return` or an edge out of a loop, and opens none.
    pub fn end_block(&mut self, terminator: Terminator, span: Span) -> Unreached {
        self.builder.terminate(terminator, span);
        Unreached
    }

    /// Ends the current block with an edge to `to`.
    pub fn jump(&mut self, to: BlockId, args: Vec<ValueId>, span: Span) {
        self.builder.terminate(Terminator::Jump { to, args }, span);
    }

    /// The body's return of `value`.
    pub fn return_value(&mut self, value: ValueId, span: Span) -> Unreached {
        self.end_block(Terminator::Return { value, order: None, span }, span)
    }

    /// `body` lowered inside the loop `targets` names, which its `break`
    /// and `continue` reach; the loop is left whatever `body` yields.
    pub fn in_loop<T>(&mut self, targets: LoopTargets, body: impl FnOnce(&mut Self) -> T) -> T {
        self.loops.push(targets);
        let lowered = body(self);
        self.loops.pop();
        lowered
    }

    /// `body` lowered inside an `anyorder` scope (effects §3), which is
    /// closed whatever `body` yields.
    pub fn in_anyorder<T>(&mut self, body: impl FnOnce(&mut Self) -> T) -> T {
        self.builder.open_anyorder();
        self.anyorder += 1;
        let lowered = body(self);
        self.anyorder -= 1;
        self.builder.close_anyorder();
        lowered
    }

    /// The binding `binder` is, in this body.
    ///
    /// # Panics
    /// The binder was not lowered in this body: a name reaches only a
    /// binding in scope, and a lambda's captures are its own bindings.
    pub fn home(&self, binder: AstId) -> Home {
        *self
            .bindings
            .get(&binder)
            .unwrap_or_else(|| panic!("the binder {binder:?} a name reaches is lowered before the name"))
    }

    /// Makes `binder`'s binding live at `home`.
    pub fn bind(&mut self, binder: AstId, home: Home, span: Span) {
        self.bindings.insert(binder, home);
        self.binder_spans.insert(binder, span);
    }

    /// The declaration `chosen` reached: the function itself, or the impl
    /// of a generic function typeck chose (extern-fn §6).
    ///
    /// # Panics
    /// `chosen` is a lifted impl: a call answered structurally is lowered by
    /// its tree (`part_tree::lifted`), and a requirement so answered is
    /// refused by typeck (`Unimplemented::LiftedImplAsRequirement`), so no
    /// declaration reached here is one.
    pub fn reached_function(&self, chosen: &Chosen) -> QualifiedRef {
        let r#impl = match chosen.r#impl {
            None => return chosen.declaration,
            Some(r#impl) => r#impl,
        };
        let (_, impls) = self
            .decls
            .generic_function(chosen.declaration)
            .unwrap_or_else(|| panic!("a declaration {:?} reached at an impl is a generic function", chosen.declaration));
        match r#impl {
            Impl::Concrete(at) => impls.concrete[at],
            Impl::Generic => impls.generic.expect("a declaration reached at its generic impl has one"),
            Impl::Structural => unreachable!(
                "a lifted impl {:?} is no callee: typeck refuses one as a requirement, and lower calls its parts",
                chosen.declaration
            ),
        }
    }

    /// # Panics
    /// A required impl is at no impl: a requirement names a generic
    /// function, which is reached only at one of its impls.
    pub fn ir_chosen(&self, chosen: &Chosen) -> IrChosen {
        assert!(
            chosen.r#impl.is_some(),
            "a requirement names a generic function reached at one of its impls, not {:?}",
            chosen.declaration
        );
        IrChosen {
            function: self.reached_function(chosen),
            required: chosen.required.iter().map(|required| self.ir_chosen(required)).collect(),
        }
    }

    /// The call of `chosen` as `member` of its `#` pair (types §8).
    pub fn callee(&self, chosen: &Chosen, member: PairMember) -> Callee {
        Callee::Extern {
            id: self.reached_function(chosen),
            required: chosen.required.iter().map(|required| self.ir_chosen(required)).collect(),
            member,
        }
    }

    /// The impl typeck reached at `site`.
    ///
    /// # Panics
    /// None was reached there: a form that calls is admitted only once its
    /// call reaches.
    pub fn reached(&self, site: AstId) -> &'a Chosen {
        self.decided
            .reached(site)
            .unwrap_or_else(|| panic!("typeck reaches a declaration at every call site of an admitted program, and not at {site:?}"))
    }
}

/// One capture or parameter a body declares: its name, its type, and the
/// span of its binder.
#[derive(Debug, Clone, Copy)]
pub(super) struct Declared {
    pub name: Astr,
    pub ty: Ty,
    pub span: Span,
}

/// What a body declares at its entry (`Lcx::make_body`): its captures, its
/// parameters, and the contexts it names (context §3, `Typeck::contexts`),
/// each at its declared type.
pub(super) struct Entry<'n> {
    pub captures: Vec<Declared>,
    pub params: Vec<Declared>,
    pub contexts: &'n [(QualifiedRef, Ty)],
}

/// How an instruction's result comes to be, which decides what lowering
/// knows of its type before judging it.
enum Yields {
    /// A part or a target of an operand, or a callee's result: its type may
    /// be one no value has (`Lcx::emit`).
    Taken,
    /// Made from its operands, values that exist: some value has its type
    /// (`Lcx::made`).
    Made,
    /// No result (`Lcx::emit_unit`).
    Nothing,
}

/// The way `op`'s result comes to be. A `Ref` through a value by an empty
/// path is a reborrow of an existing reference, whose target it names, so
/// its result is made; any other `Ref` lends a place, its target.
///
/// # Panics
/// `op` is one lowering never writes: the core writes the order chain's
/// (effects §3, §6), and a pass writes `Undef`.
fn yields(op: &Op) -> Yields {
    match op {
        Op::Ref { target: RefTarget::Through(_), path, .. } if path.is_empty() => Yields::Made,
        Op::Ref { .. }
        | Op::Take { .. }
        | Op::Index { .. }
        | Op::FieldGet { .. }
        | Op::ObjectGet { .. }
        | Op::TupleIndex { .. }
        | Op::ArrayIndex { .. }
        | Op::UnwrapVariant { .. }
        | Op::FunctionCall { .. }
        | Op::Unmark { .. }
        | Op::Intrinsic { intrinsic: Intrinsic::ArrayMap, .. } => Yields::Taken,
        Op::Const { .. }
        | Op::ConstStr { .. }
        | Op::StringConcat { .. }
        | Op::StringEq { .. }
        | Op::StringClone { .. }
        | Op::AsSlice { .. }
        | Op::ArrayView { .. }
        | Op::SliceLen { .. }
        | Op::BinOp { .. }
        | Op::UnaryOp { .. }
        | Op::Cast { .. }
        | Op::ArrayBegin { .. }
        | Op::ArrayPush { .. }
        | Op::MakeObject { .. }
        | Op::MakeTuple { .. }
        | Op::MakeVariant { .. }
        | Op::MakeClosure { .. }
        | Op::Specialize { .. }
        | Op::TestLiteral { .. }
        | Op::TestObjectKey { .. }
        | Op::TestVariant { .. }
        | Op::Intrinsic { intrinsic: Intrinsic::Gather | Intrinsic::StringClone, .. } => Yields::Made,
        Op::StringAppend { .. } | Op::Assign { .. } | Op::IndexSet { .. } | Op::FieldSet { .. } | Op::Drop { .. } => Yields::Nothing,
        Op::Spawn { .. } | Op::TakeEffect { .. } | Op::Merge { .. } | Op::Undef => {
            panic!("lowering writes no {op:?}: the core writes the order chain's (effects §3, §6), and a pass an `Undef`")
        }
    }
}

/// Whether a body of this effect issues effects, and so takes an order
/// token (effects §3). A body's effect holds the contexts it reads and
/// writes, whose loads and commits are effects (context §3).
pub(super) fn ordered(effect: &Effect) -> bool {
    effect.reissue() != Reissue::Pure || !effect.reads().is_empty() || !effect.writes().is_empty()
}
