//! What the transforms' tests share: a module over a small type table,
//! bodies built by `ir::cfg::Builder`, a run of one transform through the
//! analysis manager, and the invariants every edit keeps, checked again on
//! the output.

use std::fmt::Write;

use acvus_ast::{Literal, Span};
use acvus_utils::{Astr, Interner, QualifiedRef};

use crate::decl::{Core, Declarations, LoopMarkers, RawDeclarations};
use crate::ir::cfg::edit::{DefDominatesUses, Invariants, OrderChainWhole, StructureKept, TypesAgree};
use crate::ir::cfg::{BlockId, BodyId, Changes, Module, Op, Terminator};
use crate::ir::{BinOp, Callee, Overflow, RefTarget, ValueId};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::{BodyTransform, ModuleTransform};
use crate::ty::{
    Calls, Concrete, Contexts, Cost, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Reissue, SplitCost, Ty,
    TyKind, TypeArg, Types,
};

pub(in crate::passes) struct Fixture {
    pub module: Module,
    pub interner: Interner,
    pub int: Ty,
    pub i8: Ty,
    pub bool: Ty,
    /// `String`, which does not copy.
    pub string: Ty,
    /// `&i64`.
    pub ref_int: Ty,
    /// `&mut i64`.
    pub mut_ref_int: Ty,
    /// `fn() -> i64`, pure.
    pub pure: Ty,
    /// `fn() -> i64`, opaque.
    pub opaque: Ty,
    /// `fn(i64) -> i64`, pure.
    pub pure_unary: Ty,
    /// `fn(i64) -> i64`, opaque.
    pub opaque_unary: Ty,
    /// `fn() -> i64`, opaque and declared commutative.
    pub commuting: Ty,
    /// `fn(i64) -> i64`, opaque and declared commutative.
    pub commuting_unary: Ty,
    /// `fn() -> i64`, opaque and splittable.
    pub splittable: Ty,
    /// `(i64, i64)`.
    pub pair: Ty,
    /// `(i64, i64, i64)`.
    pub triple: Ty,
    /// `(i64, i64, i64, i64)`.
    pub quad: Ty,
}

impl Fixture {
    pub fn new() -> Fixture {
        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let i8 = types.intern(TyKind::Int(IntTy::I8));
        let bool = types.intern(TyKind::Bool);
        let string = types.intern(TyKind::String);
        let ref_int = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
        let mut_ref_int = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(int)));
        let mut call_ty = |params: &[Ty], effect: Effect| {
            types.intern(TyKind::Fn {
                params: params.into(),
                ret: int,
                effect: EffectTerm::Known(effect),
                flows: FlowTerm::Known(Flows::none()),
                calls: Calls::Many,
                holds: Holds::Nothing,
            })
        };
        let pure_effect = Effect::new(Reissue::Pure, false, Contexts::new(), Contexts::new(), SplitCost::Unavailable);
        let commuting_effect = Effect::new(Reissue::Opaque, true, Contexts::new(), Contexts::new(), SplitCost::Unavailable);
        let splittable_effect =
            Effect::new(Reissue::Opaque, false, Contexts::new(), Contexts::new(), SplitCost::WithCost(Cost::stated(1)));
        let pure = call_ty(&[], pure_effect.clone());
        let opaque = call_ty(&[], Effect::opaque());
        let pure_unary = call_ty(&[int], pure_effect);
        let opaque_unary = call_ty(&[int], Effect::opaque());
        let commuting = call_ty(&[], commuting_effect.clone());
        let commuting_unary = call_ty(&[int], commuting_effect);
        let splittable = call_ty(&[], splittable_effect);
        let pair = types.intern(TyKind::Tuple(Box::new([int, int])));
        let triple = types.intern(TyKind::Tuple(Box::new([int, int, int])));
        let quad = types.intern(TyKind::Tuple(Box::new([int, int, int, int])));
        Fixture {
            module: Module::new(types),
            interner: Interner::new(),
            int,
            i8,
            bool,
            string,
            ref_int,
            mut_ref_int,
            pure,
            opaque,
            pure_unary,
            opaque_unary,
            commuting,
            commuting_unary,
            splittable,
            pair,
            triple,
            quad,
        }
    }

    pub fn name(&self, name: &str) -> Astr {
        self.interner.intern(name)
    }

    /// A direct call of `f`, typed `callee_ty`, with no arguments.
    pub fn call(&self, callee_ty: Ty) -> Op {
        Op::FunctionCall { callee: Callee::Direct(QualifiedRef::root(self.name("f"))), callee_ty, args: Vec::new() }
    }

    /// Runs `T` over `body` with a fresh analysis manager.
    pub fn run<T>(&mut self, body: BodyId) -> Changes
    where
        T: BodyTransform,
    {
        let decls = declarations(&self.interner);
        let mut am = AnalysisManager::new(&decls);
        T::run(&mut self.module, body, &mut am)
    }

    /// Runs `T` over the module with a fresh analysis manager.
    pub fn run_module<T>(&mut self) -> Vec<(BodyId, Changes)>
    where
        T: ModuleTransform,
    {
        let decls = declarations(&self.interner);
        let mut am = AnalysisManager::new(&decls);
        T::run(&mut self.module, &mut am)
    }

    /// Every block's parameters, instructions and terminator, one per line.
    pub fn snapshot(&self, body: BodyId) -> String {
        let mut out = String::new();
        for (id, block) in self.module.body(body).blocks() {
            writeln!(out, "{id:?} {:?}", block.params()).expect("a String takes every write");
            for inst in block.insts() {
                writeln!(out, "  {inst:?}").expect("a String takes every write");
            }
            writeln!(out, "  {:?}", block.terminator()).expect("a String takes every write");
        }
        out
    }

    /// The ops of `block`, in order.
    pub fn ops(&self, body: BodyId, block: BlockId) -> Vec<&Op> {
        self.module.body(body).block(block).insts().iter().map(|inst| inst.op()).collect()
    }
}

pub(in crate::passes) fn declarations(interner: &Interner) -> Declarations {
    Declarations::verify(RawDeclarations {
        types: Types::new(),
        functions: Vec::new(),
        lang_impls: Vec::new(),
        user_types: Vec::new(),
        casts: Vec::new(),
        conversions: Vec::new(),
        contexts: Vec::new(),
        inputs: Vec::new(),
        loop_markers: LoopMarkers::default(),
        core: Core::of(interner),
    })
    .expect("empty declarations verify")
}

/// The invariants every scope keeps hold on the transform's output.
pub(in crate::passes) fn assert_whole(module: &Module, body: BodyId) {
    type Base = (TypesAgree, DefDominatesUses, OrderChainWhole, StructureKept);
    if let Err(violation) = Base::check(module.body(body), module.types()) {
        panic!("the output breaks {}: {}", violation.invariant, violation.detail);
    }
}

/// A direct call of `name`, typed `callee_ty`, with `args`.
pub(in crate::passes) fn direct_call(name: Astr, callee_ty: Ty, args: Vec<ValueId>) -> Op {
    Op::FunctionCall { callee: Callee::Direct(QualifiedRef::root(name)), callee_ty, args }
}

pub(in crate::passes) fn int_const(value: i128) -> Op {
    Op::Const { value: Literal::Int(value) }
}

pub(in crate::passes) fn bool_const(value: bool) -> Op {
    Op::Const { value: Literal::Bool(value) }
}

pub(in crate::passes) fn trapping_add(left: ValueId, right: ValueId) -> Op {
    Op::BinOp { op: BinOp::Add(Overflow::Trap), left, right }
}

pub(in crate::passes) fn wrapping_add(left: ValueId, right: ValueId) -> Op {
    Op::BinOp { op: BinOp::Add(Overflow::Wrap), left, right }
}

pub(in crate::passes) fn take(slot: ValueId) -> Op {
    Op::Take { target: RefTarget::Var(slot), path: Vec::new(), taken_out: false }
}

pub(in crate::passes) fn assign(slot: ValueId, value: ValueId) -> Op {
    Op::Assign { target: RefTarget::Var(slot), path: Vec::new(), value, restores: false }
}

pub(in crate::passes) fn ret(value: ValueId) -> Terminator {
    Terminator::Return { value, order: None, span: Span::ZERO }
}

pub(in crate::passes) fn jump(to: BlockId) -> Terminator {
    Terminator::Jump { to, args: Vec::new() }
}
