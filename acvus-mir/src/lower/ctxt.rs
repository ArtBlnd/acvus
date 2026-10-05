//! What lowering one body holds: the Builder, what typeck decided, the home
//! of each binding, the targets of the loops it is inside, and the home of
//! each context it names. A lambda's body is lowered with its own `Lcx` over
//! a nested Builder.
//!
//! A form, and a value built from a form's by a coercion or by a structure
//! tree, yields `Lowered`: its value where a run gets past it, the block it
//! ends in open, and `Unreached` where none does, the block it ends in
//! terminated and no block open (types §4, control-flow).

use rustc_hash::FxHashMap;

use acvus_ast::{AstId, Literal, Span};
use acvus_utils::{Astr, QualifiedRef};

use crate::decl::{Declarations, PairMember};
use crate::ir::cfg::{BlockId, Builder, Op, Terminator};
use crate::ir::{Callee, Chosen as IrChosen, ValueId};
use crate::ty::{Effect, IntTy, Mutability, Reissue, Ty, TyKind};
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
    pub builder: Builder<'m>,
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
        self.builder.intern(TyKind::Bool)
    }

    pub fn unit_ty(&mut self) -> Ty {
        self.builder.intern(TyKind::Unit)
    }

    pub fn u64_ty(&mut self) -> Ty {
        self.builder.intern(TyKind::Int(IntTy::U64))
    }

    pub fn ref_ty(&mut self, mutability: Mutability, target: Ty) -> Ty {
        self.builder.intern(TyKind::Ref(mutability, target))
    }

    pub fn constant(&mut self, value: Literal, ty: Ty, span: Span) -> ValueId {
        self.builder.push(Op::Const { value }, ty, span)
    }

    pub fn bool_const(&mut self, value: bool, span: Span) -> ValueId {
        let ty = self.bool_ty();
        self.constant(Literal::Bool(value), ty, span)
    }

    pub fn unit(&mut self, span: Span) -> ValueId {
        let ty = self.unit_ty();
        self.constant(Literal::Unit, ty, span)
    }

    /// A value of `ty`, a type no value has, standing where no run reaches:
    /// the coercion of a value of an enum of no variant.
    pub fn unreached_value(&mut self, ty: Ty, span: Span) -> ValueId {
        self.builder.push(Op::Undef, ty, span)
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

/// Whether a body of this effect issues effects, and so takes an order
/// token (effects §3). A body's effect holds the contexts it reads and
/// writes (D69), whose loads and commits are effects (context §3).
pub(super) fn ordered(effect: &Effect) -> bool {
    effect.reissue() != Reissue::Pure || !effect.reads().is_empty() || !effect.writes().is_empty()
}
