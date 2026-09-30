//! What lowering one body holds: the Builder, what typeck decided, the home
//! of each binding, the targets of the loops it is inside, the home of each
//! context it names, and whether the block being appended to is reached. A
//! lambda's body is lowered with its own `Lcx` over a nested Builder.

use rustc_hash::FxHashMap;

use acvus_ast::{AstId, Literal, Span};
use acvus_utils::Astr;

use crate::decl::Declarations;
use crate::ir::cfg::{BlockId, Builder, Op, Terminator};
use crate::ir::{Callee, Chosen as IrChosen, ExternImpl, RefTarget, ValueId};
use crate::ty::{Effect, IntTy, Mutability, Reissue, Ty, TyKind, TypeArg};
use crate::typeck::{Chosen, Impl, Typeck};

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
    /// Whether a run reaches the current block: after a `return`, a
    /// `break`, a `continue` or a `?`'s exit, what follows is appended to
    /// a block no edge enters, and its own edges out become `Diverge`.
    pub reachable: bool,
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
            reachable: true,
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
        self.builder.intern(TyKind::Ref(mutability, TypeArg::Uniform(target)))
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

    /// A value of `ty` standing where no run reaches: a form after a
    /// diverging one still yields a value of its recorded type.
    pub fn unreached_value(&mut self, ty: Ty, span: Span) -> ValueId {
        self.builder.push(Op::Undef, ty, span)
    }

    /// Ends the current block with a terminator no run passes, such as a
    /// `Return`, and goes on in a block no edge enters.
    pub fn end_block(&mut self, terminator: Terminator, span: Span) {
        self.builder.terminate(terminator, span);
        let dead = self.builder.new_block(&[]);
        self.builder.switch_to(dead);
        self.reachable = false;
    }

    /// Ends the current block with an edge to `to`, or with `Diverge` where
    /// no run reaches the block.
    pub fn jump(&mut self, to: BlockId, args: Vec<ValueId>, span: Span) {
        let terminator = match self.reachable {
            true => Terminator::Jump { to, args },
            false => Terminator::Diverge,
        };
        self.builder.terminate(terminator, span);
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

    /// Appends `op`, a `Take` or a `Ref`, typed `ty`, and returns its
    /// result: on the order chain where it names a volatile context, whose
    /// every place operation is an effect (context §3).
    pub fn push_place(&mut self, op: Op, ty: Ty, span: Span) -> ValueId {
        match self.volatile_place(&op) {
            true => self.builder.push_effect(op, ty, span),
            false => self.builder.push(op, ty, span),
        }
    }

    /// Appends `op`, an `Assign`: on the order chain where it names a
    /// volatile context, as `push_place` does.
    pub fn push_store(&mut self, op: Op, span: Span) {
        match self.volatile_place(&op) {
            true => self.builder.push_effect_unit(op, span),
            false => self.builder.push_unit(op, span),
        }
    }

    /// Whether `op` names a context declared volatile.
    ///
    /// # Panics
    /// It names a context the declarations do not declare.
    fn volatile_place(&self, op: &Op) -> bool {
        let Some((RefTarget::Context(context), _)) = op.place() else {
            return false;
        };
        self.decls
            .context_is_volatile(*context)
            .unwrap_or_else(|| panic!("the context {context:?} an admitted program names is declared"))
    }

    /// The impl `chosen` reached, numbered as the IR numbers a generic
    /// function's impls: a concrete impl by its index, the generic impl
    /// after them. A declaration without impls is itself the callee, at no
    /// impl.
    pub fn impl_index(&self, chosen: &Chosen) -> Option<usize> {
        let at = match chosen.r#impl {
            None => return None,
            Some(Impl::Concrete(at)) => at,
            Some(Impl::Generic) => {
                let scheme = self
                    .decls
                    .scheme(chosen.declaration)
                    .unwrap_or_else(|| panic!("a reached declaration {:?} is declared", chosen.declaration));
                let impls = scheme.impls.as_ref().expect("a declaration reached at its generic impl has impls");
                impls.concrete.len()
            }
            // A call answered structurally is lowered by its tree
            // (`part_tree::lifted`), and a requirement so answered is refused
            // by typeck (`Unimplemented::LiftedImplAsRequirement`), so no
            // chosen impl numbered here is one.
            Some(Impl::Structural) => unreachable!(
                "a lifted impl {:?} is no callee: typeck refuses one as a requirement, and lower calls its parts",
                chosen.declaration
            ),
        };
        Some(at)
    }

    /// # Panics
    /// A required impl is at no impl: a requirement names a generic
    /// function, which is reached only at one of its impls.
    pub fn ir_chosen(&self, chosen: &Chosen) -> IrChosen {
        IrChosen {
            generic_function: chosen.declaration,
            r#impl: self.impl_index(chosen).unwrap_or_else(|| {
                panic!("a requirement names a generic function reached at one of its impls, not {:?}", chosen.declaration)
            }),
            required: chosen.required.iter().map(|required| self.ir_chosen(required)).collect(),
        }
    }

    pub fn callee(&self, chosen: &Chosen) -> Callee {
        Callee::Extern {
            id: chosen.declaration,
            r#impl: self.impl_index(chosen),
            required: chosen.required.iter().map(|required| self.ir_chosen(required)).collect(),
        }
    }

    pub fn extern_impl(&self, chosen: &Chosen) -> ExternImpl {
        ExternImpl { id: chosen.declaration, r#impl: self.impl_index(chosen) }
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
