//! An arithmetic chain as one operation (RFC-0044, stage 4).
//!
//! Every instance is named by its shape and by one `Slot` per operator
//! node: a concrete operator compiles to the one machine instruction, and
//! `Any` reads that node's operator from the payload. The alphabet the
//! `instances!` invocation lists is the knob, and it is deliberately
//! small: every operator outside it still runs, through `Any`, at one
//! predicted branch for that node.

use std::mem::size_of;

use acvus_mir::ty::IntTy;

use crate::code::{
    Arith, Arity, Chain, Compare, ExprFn, Flow, Op, OpFn, Payload, Root, Shape, Shape2, Shape3,
};
use crate::machine::Machine;
use crate::ops::arith::{Int, for_int_ty};
use crate::value::{Kind, Value};

/// One numeric type a chain runs at: the integer widths and `f64`.
///
/// Integer arithmetic wraps and `/` and `%` keep their two panics, exactly
/// as the one-operator-per-operation form does — the chain calls the same
/// `Int` methods, so the panic texts are the same constants.
pub trait Num: Copy + 'static {
    const KIND: Kind;

    fn read(bits: u64) -> Self;
    fn word(self) -> u64;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn rem(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn compare(self, other: Self, how: Compare) -> bool;
}

macro_rules! impl_num_for_int {
    ($($t:ty),* $(,)?) => {
        $(impl Num for $t {
            const KIND: Kind = <$t as Int>::KIND;

            #[inline(always)]
            fn read(bits: u64) -> Self {
                <$t as Int>::read(bits)
            }
            #[inline(always)]
            fn word(self) -> u64 {
                <$t as Int>::word(self)
            }
            #[inline(always)]
            fn add(self, other: Self) -> Self {
                <$t as Int>::wrapping_add(self, other)
            }
            #[inline(always)]
            fn sub(self, other: Self) -> Self {
                <$t as Int>::wrapping_sub(self, other)
            }
            #[inline(always)]
            fn mul(self, other: Self) -> Self {
                <$t as Int>::wrapping_mul(self, other)
            }
            #[inline(always)]
            fn div(self, other: Self) -> Self {
                <$t as Int>::read(
                    crate::ops::arith::word::div::<$t>(
                        <$t as Int>::word(self),
                        <$t as Int>::word(other),
                    )
                    .bits(),
                )
            }
            #[inline(always)]
            fn rem(self, other: Self) -> Self {
                <$t as Int>::read(
                    crate::ops::arith::word::rem::<$t>(
                        <$t as Int>::word(self),
                        <$t as Int>::word(other),
                    )
                    .bits(),
                )
            }
            #[inline(always)]
            fn neg(self) -> Self {
                <$t as Int>::wrapping_neg(self)
            }
            #[inline(always)]
            fn compare(self, other: Self, how: Compare) -> bool {
                match how {
                    Compare::Lt => self < other,
                    Compare::Le => self <= other,
                    Compare::Gt => self > other,
                    Compare::Ge => self >= other,
                    Compare::Eq => self == other,
                    Compare::Ne => self != other,
                }
            }
        })*
    };
}

impl_num_for_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl Num for f64 {
    const KIND: Kind = Kind::F64;

    #[inline(always)]
    fn read(bits: u64) -> Self {
        f64::from_bits(bits)
    }
    #[inline(always)]
    fn word(self) -> u64 {
        self.to_bits()
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self + other
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self - other
    }
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        self * other
    }
    #[inline(always)]
    fn div(self, other: Self) -> Self {
        self / other
    }
    #[inline(always)]
    fn rem(self, other: Self) -> Self {
        self % other
    }
    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }

    /// The same total order the one-operator form uses, so a chain and the
    /// operations it replaced order `NaN` and `-0.0` alike.
    #[inline(always)]
    fn compare(self, other: Self, how: Compare) -> bool {
        match how {
            Compare::Lt => self.total_cmp(&other).is_lt(),
            Compare::Le => self.total_cmp(&other).is_le(),
            Compare::Gt => self.total_cmp(&other).is_gt(),
            Compare::Ge => self.total_cmp(&other).is_ge(),
            Compare::Eq => self.to_bits() == other.to_bits(),
            Compare::Ne => self.to_bits() != other.to_bits(),
        }
    }
}

pub type Operands<'a> = &'a [Value];

#[inline(always)]
fn leaf<T>(operands: Operands<'_>, offset: u16) -> T
where
    T: Num,
{
    let at = offset as usize;
    debug_assert!(
        at % size_of::<Value>() == Value::WORD_OFFSET,
        "a chain leaf offset does not reach the word of a Value"
    );
    debug_assert!(
        at + size_of::<u64>() <= operands.len() * size_of::<Value>(),
        "a chain leaf reads past the operand space its preparation sized"
    );
    debug_assert!(
        operands[at / size_of::<Value>()].kind() == T::KIND,
        "a chain leaf reads a register of another kind than the chain's"
    );
    // SAFETY: `prepare::Prepare::check_chain` states that every slot a
    // chain reads is below the operand space's length and carries the
    // chain's own type, and `Chain::offset` is the only writer of these
    // offsets, so the address is inside the space and aligned to a `Value`
    // word. The three assertions above are that statement, executed.
    unsafe { T::read(operands.as_ptr().cast::<u8>().add(at).cast::<u64>().read()) }
}

#[inline(always)]
fn any<T>(op: Arith, left: T, right: T) -> T
where
    T: Num,
{
    match op {
        Arith::Add => left.add(right),
        Arith::Sub => left.sub(right),
        Arith::Mul => left.mul(right),
        Arith::Div => left.div(right),
        Arith::Rem => left.rem(right),
        Arith::Neg => left.neg(),
    }
}

pub struct Slots {
    pub shape: Shape,
    pub ops: [Slot; Chain::MAX_INTERIOR],
    pub root: Slot,
}

pub struct SlotCount {
    pub concrete: usize,
    pub total: usize,
}

macro_rules! instances {
    (concrete: [$($v:ident => $m:ident),* $(,)?], generic: [$($g:ident),* $(,)?] $(,)?) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        #[repr(u8)]
        pub enum Slot {
            Any,
            $($v),*
        }

        impl Slot {
            pub fn of(op: Arith) -> Slot {
                match op {
                    $(Arith::$v => Slot::$v,)*
                    $(Arith::$g => Slot::Any,)*
                }
            }

            pub fn of_root(root: Root) -> Slot {
                match root {
                    Root::Num(op) => Slot::of(op),
                    Root::Cmp(_) => Slot::Any,
                }
            }

            const fn read(raw: u8) -> Slot {
                match raw {
                    0 => Slot::Any,
                    $(x if x == Slot::$v as u8 => Slot::$v,)*
                    _ => panic!("a chain instance was parameterized by no slot"),
                }
            }
        }

        #[inline(always)]
        fn apply<T, const O: u8>(op: Arith, left: T, right: T) -> T
        where
            T: Num,
        {
            match Slot::read(O) {
                Slot::Any => any(op, left, right),
                $(Slot::$v => {
                    debug_assert_eq!(
                        op,
                        Arith::$v,
                        "a chain instance ran a node whose operator is not its slot's"
                    );
                    left.$m(right)
                })*
            }
        }

        #[inline(always)]
        fn finish<T, const R: u8>(root: Root, left: T, right: T) -> Value
        where
            T: Num,
        {
            match Slot::read(R) {
                Slot::Any => match root {
                    Root::Num(op) => Value::inline(T::KIND, any(op, left, right).word()),
                    Root::Cmp(how) => Value::bool_(left.compare(right, how)),
                },
                $(Slot::$v => {
                    debug_assert_eq!(
                        root,
                        Root::Num(Arith::$v),
                        "a chain instance ran a root whose operator is not its slot's"
                    );
                    Value::inline(T::KIND, left.$m(right).word())
                })*
            }
        }

        fn pick1<T>(slots: &Slots) -> Instance
        where
            T: Num,
        {
            match slots.root {
                Slot::Any => one_of::<T, { Slot::Any as u8 }>(),
                $(Slot::$v => one_of::<T, { Slot::$v as u8 }>(),)*
            }
        }

        fn pick2<T>(shape: Shape2, slots: &Slots) -> Instance
        where
            T: Num,
        {
            match shape {
                Shape2::NNLLL => pick2_op::<T, { Shape2::NNLLL as u8 }>(slots),
                Shape2::NLNLL => pick2_op::<T, { Shape2::NLNLL as u8 }>(slots),
            }
        }

        fn pick2_op<T, const S: u8>(slots: &Slots) -> Instance
        where
            T: Num,
        {
            match slots.ops[0] {
                Slot::Any => pick2_root::<T, S, { Slot::Any as u8 }>(slots),
                $(Slot::$v => pick2_root::<T, S, { Slot::$v as u8 }>(slots),)*
            }
        }

        fn pick2_root<T, const S: u8, const O0: u8>(slots: &Slots) -> Instance
        where
            T: Num,
        {
            match slots.root {
                Slot::Any => two_of::<T, S, O0, { Slot::Any as u8 }>(),
                $(Slot::$v => two_of::<T, S, O0, { Slot::$v as u8 }>(),)*
            }
        }

        fn pick3<T>(shape: Shape3, slots: &Slots) -> Instance
        where
            T: Num,
        {
            match shape {
                Shape3::NNNLLLL => pick3_op0::<T, { Shape3::NNNLLLL as u8 }>(slots),
                Shape3::NNLNLLL => pick3_op0::<T, { Shape3::NNLNLLL as u8 }>(slots),
                Shape3::NNLLNLL => pick3_op0::<T, { Shape3::NNLLNLL as u8 }>(slots),
                Shape3::NLNNLLL => pick3_op0::<T, { Shape3::NLNNLLL as u8 }>(slots),
                Shape3::NLNLNLL => pick3_op0::<T, { Shape3::NLNLNLL as u8 }>(slots),
            }
        }

        fn pick3_op0<T, const S: u8>(slots: &Slots) -> Instance
        where
            T: Num,
        {
            match slots.ops[0] {
                Slot::Any => pick3_op1::<T, S, { Slot::Any as u8 }>(slots),
                $(Slot::$v => pick3_op1::<T, S, { Slot::$v as u8 }>(slots),)*
            }
        }

        fn pick3_op1<T, const S: u8, const O0: u8>(slots: &Slots) -> Instance
        where
            T: Num,
        {
            match slots.ops[1] {
                Slot::Any => pick3_root::<T, S, O0, { Slot::Any as u8 }>(slots),
                $(Slot::$v => pick3_root::<T, S, O0, { Slot::$v as u8 }>(slots),)*
            }
        }

        fn pick3_root<T, const S: u8, const O0: u8, const O1: u8>(slots: &Slots) -> Instance
        where
            T: Num,
        {
            match slots.root {
                Slot::Any => three_of::<T, S, O0, O1, { Slot::Any as u8 }>(),
                $(Slot::$v => three_of::<T, S, O0, O1, { Slot::$v as u8 }>(),)*
            }
        }
    };
}

#[cfg(not(feature = "chain-alphabet-add-sub-mul"))]
instances!(
    concrete: [Add => add, Mul => mul],
    generic: [Sub, Div, Rem, Neg],
);

#[cfg(feature = "chain-alphabet-add-sub-mul")]
instances!(
    concrete: [Add => add, Sub => sub, Mul => mul],
    generic: [Div, Rem, Neg],
);

#[inline(always)]
fn tree1<T, const R: u8>(chain: &Chain, operands: Operands<'_>) -> Value
where
    T: Num,
{
    let offsets = chain.leaf_offsets;
    let a: T = leaf(operands, offsets[0]);
    let b: T = leaf(operands, offsets[1]);
    finish::<T, R>(chain.root, a, b)
}

#[inline(always)]
fn tree2<T, const S: u8, const O0: u8, const R: u8>(chain: &Chain, operands: Operands<'_>) -> Value
where
    T: Num,
{
    let offsets = chain.leaf_offsets;
    let a: T = leaf(operands, offsets[0]);
    let b: T = leaf(operands, offsets[1]);
    let c: T = leaf(operands, offsets[2]);
    let ops = chain.post_order_ops;
    match Shape2::read(S) {
        Shape2::NNLLL => {
            let n0 = apply::<T, O0>(ops[0], a, b);
            finish::<T, R>(chain.root, n0, c)
        }
        Shape2::NLNLL => {
            let n0 = apply::<T, O0>(ops[0], b, c);
            finish::<T, R>(chain.root, a, n0)
        }
    }
}

#[inline(always)]
fn tree3<T, const S: u8, const O0: u8, const O1: u8, const R: u8>(
    chain: &Chain,
    operands: Operands<'_>,
) -> Value
where
    T: Num,
{
    let offsets = chain.leaf_offsets;
    let a: T = leaf(operands, offsets[0]);
    let b: T = leaf(operands, offsets[1]);
    let c: T = leaf(operands, offsets[2]);
    let d: T = leaf(operands, offsets[3]);
    let ops = chain.post_order_ops;
    match Shape3::read(S) {
        Shape3::NNNLLLL => {
            let n0 = apply::<T, O0>(ops[0], a, b);
            let n1 = apply::<T, O1>(ops[1], n0, c);
            finish::<T, R>(chain.root, n1, d)
        }
        Shape3::NNLNLLL => {
            let n0 = apply::<T, O0>(ops[0], b, c);
            let n1 = apply::<T, O1>(ops[1], a, n0);
            finish::<T, R>(chain.root, n1, d)
        }
        Shape3::NNLLNLL => {
            let n0 = apply::<T, O0>(ops[0], a, b);
            let n1 = apply::<T, O1>(ops[1], c, d);
            finish::<T, R>(chain.root, n0, n1)
        }
        Shape3::NLNNLLL => {
            let n0 = apply::<T, O0>(ops[0], b, c);
            let n1 = apply::<T, O1>(ops[1], n0, d);
            finish::<T, R>(chain.root, a, n1)
        }
        Shape3::NLNLNLL => {
            let n0 = apply::<T, O0>(ops[0], c, d);
            let n1 = apply::<T, O1>(ops[1], b, n0);
            finish::<T, R>(chain.root, a, n1)
        }
    }
}

#[derive(Clone, Copy)]
pub struct Instance {
    pub op: OpFn,
    pub expr: ExprFn,
}

#[inline(always)]
fn chain_of<'c>(machine: &Machine<'c>, op: &Op) -> &'c Chain {
    // SAFETY: `prepare::Prepare::chain_op` writes the address of the
    // `Chain` that the payload at `op.b` owns through a `Box`, so the
    // address is fixed when the payload table is built, and the `Body`
    // that owns the table outlives every run of these operations. The
    // table is never mutated after preparation.
    let chain = unsafe { &*(op.p as *const Chain) };
    debug_assert!(
        matches!(
            &machine.code().payloads[op.b as usize],
            Payload::Chain(owned) if std::ptr::eq(&**owned, chain)
        ),
        "a chain operation's payload pointer is not the chain its payload owns"
    );
    chain
}

macro_rules! entry_points {
    ($($tree:ident => ($run:ident, $eval:ident, $of:ident) [$($c:ident),*]),* $(,)?) => {
        $(
            fn $run<T, $(const $c: u8),*>(machine: &mut Machine<'_>, op: &Op) -> Flow
            where
                T: Num,
            {
                let chain = chain_of(machine, op);
                let value = $tree::<T, $($c),*>(chain, machine.regs());
                machine.store(op.a, value);
                Flow::Next
            }

            fn $eval<T, $(const $c: u8),*>(chain: &Chain, operands: Operands<'_>) -> Value
            where
                T: Num,
            {
                $tree::<T, $($c),*>(chain, operands)
            }

            fn $of<T, $(const $c: u8),*>() -> Instance
            where
                T: Num,
            {
                Instance {
                    op: $run::<T, $($c),*>,
                    expr: $eval::<T, $($c),*>,
                }
            }
        )*
    };
}

entry_points!(
    tree1 => (run1, eval1, one_of) [R],
    tree2 => (run2, eval2, two_of) [S, O0, R],
    tree3 => (run3, eval3, three_of) [S, O0, O1, R],
);

/// The numeric type a chain runs at, decided at preparation from the
/// types of the operations it collapses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChainTy {
    Int(IntTy),
    Float,
}

impl Slots {
    pub fn of(chain: &Chain) -> Slots {
        let mut ops = [Slot::Any; Chain::MAX_INTERIOR];
        let used = chain.shape.interior();
        for (slot, op) in ops[..used].iter_mut().zip(&chain.post_order_ops[..used]) {
            *slot = Slot::of(*op);
        }
        Slots {
            shape: chain.shape,
            ops,
            root: Slot::of_root(chain.root),
        }
    }

    pub fn count(&self) -> SlotCount {
        let ops = self.ops[..self.shape.interior()]
            .iter()
            .filter(|slot| **slot != Slot::Any)
            .count();
        SlotCount {
            concrete: ops + usize::from(self.root != Slot::Any),
            total: self.shape.slots(),
        }
    }
}

pub fn instance(ty: ChainTy, chain: &Chain) -> Instance {
    match ty {
        ChainTy::Int(k) => for_int_ty!(k, |T| instance_at::<T>(chain)),
        ChainTy::Float => instance_at::<f64>(chain),
    }
}

fn instance_at<T>(chain: &Chain) -> Instance
where
    T: Num,
{
    let slots = Slots::of(chain);
    match chain.shape.arity() {
        Arity::One => pick1::<T>(&slots),
        Arity::Two(shape) => pick2::<T>(shape, &slots),
        Arity::Three(shape) => pick3::<T>(shape, &slots),
    }
}
