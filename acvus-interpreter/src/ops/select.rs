//! A diamond of one pure node and a pass-through arm, as one operation
//! (RFC-0074).
//!
//! Obligation across artifacts: the node runs whichever way the condition
//! goes, and what makes that sound is the shape `prepare::select_shape`
//! admits — one arithmetic or comparison node whose operator is neither `/`
//! nor `%`, which RFC-0037 names as the only integer operations that can
//! raise, and a join of one word register, which is the only register this
//! operation writes.

use std::marker::PhantomData;

use acvus_mir::ty::IntTy;

use crate::code::{Exit, Off, Op, Where, successor};
use crate::machine::Machine;
use crate::ops::arith::for_int_ty;
use crate::ops::chain::{ChainTy, Node, Num, Operands, Plan, Rooted, pick_root, tree1};
use crate::ops::place::{self, Place};

/// `prepare` read off the lowering which side of the test computes, so this
/// `run` compares the condition against a constant rather than holding an arm
/// per side.
pub struct Select<T, C, D, const R: u8, const COMPUTES_ON_TRUE: bool>
where
    T: Num,
    C: Place,
    D: Place,
{
    pub cond: C::At,
    pub plan: Plan,
    pub passed: Off,
    pub dst: D::At,
    pub next: Box<dyn Op>,
    pub at: PhantomData<fn() -> (T, C, D)>,
}

impl<T, C, D, const R: u8, const COMPUTES_ON_TRUE: bool> Op for Select<T, C, D, R, COMPUTES_ON_TRUE>
where
    T: Num,
    C: Place,
    D: Place,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let taken = C::read(m.regs(), self.cond, r0) != 0;
        let computed = tree1::<T, R, true>(&self.plan, Operands::of_frame(m.regs()));
        let passed = m.regs().word(self.passed);
        let bits = chosen(taken == COMPUTES_ON_TRUE, computed, passed);
        let carried = D::write(m.regs(), self.dst, bits);
        self.next.run(m, carried)
    }
}

/// The word the condition picked. Which machine instruction that becomes is
/// LLVM's choice and not this function's: a `match`, the arithmetic mask
/// `(a & m) | (b & !m)`, and `hint::select_unpredictable` were each built at
/// `opt-level = 3` with fat LTO and each produced the **byte-identical**
/// `accum` bench binary, in which `Select::<i64, Slot, Slot, Add, true>::run`
/// sinks the two operand loads into the two sides of a `jne` and joins them
/// with a phi, where a `cmov` would need them hoisted. RFC-0074's measurement
/// of this shape is against that code, so the gain it records is the region
/// entry and the two arm-chain dispatches this operation removes, not a
/// branch it does not.
#[inline(always)]
fn chosen(first: bool, first_bits: u64, second_bits: u64) -> u64 {
    match first {
        true => first_bits,
        false => second_bits,
    }
}

pub struct Places {
    pub cond: Where,
    pub dst: Where,
}

pub struct Arms {
    pub passed: Off,
    pub computes_on_true: bool,
}

struct Chosen<C, D>
where
    C: Place,
    D: Place,
{
    cond: C::At,
    dst: D::At,
}

pub fn select_op(
    ty: ChainTy,
    places: Places,
    plan: Plan,
    arms: Arms,
    next: Box<dyn Op>,
) -> Box<dyn Op> {
    match (places.cond, places.dst) {
        (Where::Frame(cond), Where::Frame(dst)) => at_ty(
            ty,
            Chosen::<place::Slot, place::Slot> { cond, dst },
            plan,
            arms,
            next,
        ),
        (Where::Frame(cond), Where::Register) => at_ty(
            ty,
            Chosen::<place::Slot, place::R0> { cond, dst: () },
            plan,
            arms,
            next,
        ),
        (Where::Register, Where::Frame(dst)) => at_ty(
            ty,
            Chosen::<place::R0, place::Slot> { cond: (), dst },
            plan,
            arms,
            next,
        ),
        (Where::Register, Where::Register) => at_ty(
            ty,
            Chosen::<place::R0, place::R0> { cond: (), dst: () },
            plan,
            arms,
            next,
        ),
    }
}

fn at_ty<C, D>(
    ty: ChainTy,
    chosen: Chosen<C, D>,
    plan: Plan,
    arms: Arms,
    next: Box<dyn Op>,
) -> Box<dyn Op>
where
    C: Place,
    D: Place,
{
    match ty {
        ChainTy::Int(k) => for_int_ty!(k, |T| at_side::<T, C, D>(chosen, plan, arms, next)),
        ChainTy::Float => at_side::<f64, C, D>(chosen, plan, arms, next),
    }
}

fn at_side<T, C, D>(chosen: Chosen<C, D>, plan: Plan, arms: Arms, next: Box<dyn Op>) -> Box<dyn Op>
where
    T: Num,
    C: Place,
    D: Place,
{
    let root = Node::of_root(plan.root);
    match arms.computes_on_true {
        true => {
            let mut make = BuildSelect::<T, C, D, true>::of(chosen, plan, arms.passed, next);
            pick_root::<T, BuildSelect<T, C, D, true>>(root, &mut make)
        }
        false => {
            let mut make = BuildSelect::<T, C, D, false>::of(chosen, plan, arms.passed, next);
            pick_root::<T, BuildSelect<T, C, D, false>>(root, &mut make)
        }
    }
}

struct BuildSelect<T, C, D, const COMPUTES_ON_TRUE: bool>
where
    T: Num,
    C: Place,
    D: Place,
{
    chosen: Chosen<C, D>,
    plan: Option<Plan>,
    passed: Off,
    next: Option<Box<dyn Op>>,
    at: PhantomData<fn() -> T>,
}

impl<T, C, D, const COMPUTES_ON_TRUE: bool> BuildSelect<T, C, D, COMPUTES_ON_TRUE>
where
    T: Num,
    C: Place,
    D: Place,
{
    fn of(
        chosen: Chosen<C, D>,
        plan: Plan,
        passed: Off,
        next: Box<dyn Op>,
    ) -> BuildSelect<T, C, D, COMPUTES_ON_TRUE> {
        BuildSelect {
            chosen,
            plan: Some(plan),
            passed,
            next: Some(next),
            at: PhantomData,
        }
    }
}

impl<T, C, D, const COMPUTES_ON_TRUE: bool> Rooted<T> for BuildSelect<T, C, D, COMPUTES_ON_TRUE>
where
    T: Num,
    C: Place,
    D: Place,
{
    type Out = Box<dyn Op>;

    fn of<const R: u8>(&mut self) -> Box<dyn Op> {
        let plan = self
            .plan
            .take()
            .expect("a select instance is built once from its plan");
        let next = self
            .next
            .take()
            .expect("a select instance is built once from its plan");
        Box::new(Select::<T, C, D, R, COMPUTES_ON_TRUE> {
            cond: self.chosen.cond,
            plan,
            passed: self.passed,
            dst: self.chosen.dst,
            next,
            at: PhantomData,
        })
    }
}
