//! An arithmetic chain as one operation (RFC-0044 rule 5; RFC-0052).
//!
//! An instance is named by its operand type and by one `Node` per operator
//! it fixed in the type. Its shape is a field, not a parameter.

use std::marker::PhantomData;
use std::mem::{MaybeUninit, size_of};
use std::slice;

use acvus_mir::ty::IntTy;

use crate::code::{
    Arith, ChainBounds, Code, Compare, Entry, Exit, ExprChain, Op, Root, Shape, Where, successor,
};
use crate::machine::Machine;
use crate::ops::arith::{Int, for_int_ty};
use crate::ops::cast::AsNum;
use crate::ops::place::Place;
use crate::regs::{FrameState, Regs};
use crate::runtime::AcvusRuntime;
use crate::value::Value;

/// One numeric type a chain runs at: the integer widths and `f64`.
///
/// Integer arithmetic wraps and `/` and `%` keep their two panics, exactly
/// as the one-operator-per-operation form does — the chain calls the same
/// `Int` methods, so the panic texts are the same constants.
pub trait Num: Copy + 'static {
    fn read(bits: u64) -> Self;
    fn word(self) -> u64;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn rem(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn compare(self, other: Self, how: Compare) -> bool;

    fn of_i8(v: i8) -> Self;
    fn of_i16(v: i16) -> Self;
    fn of_i32(v: i32) -> Self;
    fn of_i64(v: i64) -> Self;
    fn of_u8(v: u8) -> Self;
    fn of_u16(v: u16) -> Self;
    fn of_u32(v: u32) -> Self;
    fn of_u64(v: u64) -> Self;
    fn of_f64(v: f64) -> Self;
}

/// `Num`'s nine constructors at one target type, each the Rust `as`
/// expression for its pair (RFC-0049).
macro_rules! num_of {
    ($t:ty) => {
        #[inline(always)]
        fn of_i8(v: i8) -> Self {
            v as $t
        }
        #[inline(always)]
        fn of_i16(v: i16) -> Self {
            v as $t
        }
        #[inline(always)]
        fn of_i32(v: i32) -> Self {
            v as $t
        }
        #[inline(always)]
        fn of_i64(v: i64) -> Self {
            v as $t
        }
        #[inline(always)]
        fn of_u8(v: u8) -> Self {
            v as $t
        }
        #[inline(always)]
        fn of_u16(v: u16) -> Self {
            v as $t
        }
        #[inline(always)]
        fn of_u32(v: u32) -> Self {
            v as $t
        }
        #[inline(always)]
        fn of_u64(v: u64) -> Self {
            v as $t
        }
        #[inline(always)]
        fn of_f64(v: f64) -> Self {
            v as $t
        }
    };
}

macro_rules! impl_num_for_int {
    ($($t:ty),* $(,)?) => {
        $(impl Num for $t {
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
                    ),
                )
            }
            #[inline(always)]
            fn rem(self, other: Self) -> Self {
                <$t as Int>::read(
                    crate::ops::arith::word::rem::<$t>(
                        <$t as Int>::word(self),
                        <$t as Int>::word(other),
                    ),
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

            num_of!($t);
        })*
    };
}

impl_num_for_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl Num for f64 {
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

    num_of!(f64);
}

/// The run a chain reads. Its offsets are byte displacements produced by
/// `code::ChainBounds::byte_offset_of_word`, which is what fixes the stride
/// and the word position this indexes by.
#[derive(Clone, Copy)]
pub struct Operands<'a> {
    base: *const Value,
    len: usize,
    borrow: PhantomData<&'a Value>,
}

impl<'a> Operands<'a> {
    #[inline(always)]
    pub fn of(values: &'a [Value]) -> Operands<'a> {
        Operands {
            base: values.as_ptr(),
            len: values.len(),
            borrow: PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) fn of_frame(regs: &'a Regs<'_>) -> Operands<'a> {
        Operands {
            base: regs.as_ptr(),
            len: regs.len(),
            borrow: PhantomData,
        }
    }
}

/// The type one leaf reads its slot at: the chain's own, or the type a
/// cast the chain absorbed read (RFC-0049). A cast is a leaf and never a
/// node, so absorbing one leaves `Chain{1,2,3}<T>` at one `T`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LeafRead {
    Own,
    Cast(ChainTy),
}

/// How a chain reads its leaves. `Own` is every leaf at the chain's own
/// type, and it is a variant rather than an array of `LeafRead::Own`
/// because it is the one a chain that absorbed no cast runs: the test is
/// one branch for the whole chain instead of one per leaf, which is what
/// `range | sum` measured (RFC-0049).
#[derive(Clone, Copy)]
pub enum Reads {
    Own,
    Leafwise([LeafRead; ChainBounds::MAX_LEAVES]),
}

impl Reads {
    pub fn of(leafwise: [LeafRead; ChainBounds::MAX_LEAVES]) -> Reads {
        match leafwise.iter().all(|read| *read == LeafRead::Own) {
            true => Reads::Own,
            false => Reads::Leafwise(leafwise),
        }
    }

    pub fn leafwise(self) -> [LeafRead; ChainBounds::MAX_LEAVES] {
        match self {
            Reads::Own => [LeafRead::Own; ChainBounds::MAX_LEAVES],
            Reads::Leafwise(held) => held,
        }
    }
}

#[inline(always)]
fn word(operands: Operands<'_>, offset: u16) -> u64 {
    let at = offset as usize;
    debug_assert!(
        at % size_of::<Value>() == Value::WORD_OFFSET,
        "a chain leaf offset does not reach the word of a Value"
    );
    debug_assert!(
        at + size_of::<u64>() <= operands.len * size_of::<Value>(),
        "a chain leaf reads past the operand space its preparation sized"
    );
    // SAFETY: `prepare::Prepare::check_chain` states that every slot a
    // chain reads is below the operand space's length and carries the type
    // its `LeafRead` names, and `ChainBounds::byte_offset_of_word` is the
    // only writer of these offsets, so the address is inside the space and
    // aligned to a `Value` word. The two assertions above are that
    // statement, executed.
    unsafe { operands.base.cast::<u8>().add(at).cast::<u64>().read() }
}

#[inline(always)]
fn converted<T>(bits: u64, read: LeafRead) -> T
where
    T: Num,
{
    match read {
        LeafRead::Own => T::read(bits),
        LeafRead::Cast(ChainTy::Int(k)) => for_int_ty!(k, |S| <S as Num>::read(bits).as_num()),
        LeafRead::Cast(ChainTy::Float) => f64::from_bits(bits).as_num(),
    }
}

#[inline(always)]
fn own<T>(plan: &Plan, operands: Operands<'_>, at: usize) -> T
where
    T: Num,
{
    T::read(word(operands, plan.leaves[at]))
}

#[inline(always)]
fn cast<T>(plan: &Plan, operands: Operands<'_>, at: usize, read: LeafRead) -> T
where
    T: Num,
{
    converted(word(operands, plan.leaves[at]), read)
}

/// Two leaves, in leaf order.
///
/// `PLAIN` is the caller's knowledge that this chain absorbed no cast.
/// Where the caller has it — a body that is one chain, whose evaluator
/// `chain_eval` chooses once at preparation — the reads compile to what
/// they were before `as` existed. Where it does not, the chain reads
/// `plan.reads` at one branch for the whole chain rather than one per
/// leaf; a `range | sum` measurement decided between those two
/// (RFC-0049).
#[inline(always)]
fn pair<T, const PLAIN: bool>(plan: &Plan, operands: Operands<'_>) -> (T, T)
where
    T: Num,
{
    if PLAIN {
        return (own(plan, operands, 0), own(plan, operands, 1));
    }
    match plan.reads {
        Reads::Own => (own(plan, operands, 0), own(plan, operands, 1)),
        Reads::Leafwise(r) => (cast(plan, operands, 0, r[0]), cast(plan, operands, 1, r[1])),
    }
}

#[inline(always)]
fn triple<T, const PLAIN: bool>(plan: &Plan, operands: Operands<'_>) -> (T, T, T)
where
    T: Num,
{
    if PLAIN {
        return (
            own(plan, operands, 0),
            own(plan, operands, 1),
            own(plan, operands, 2),
        );
    }
    match plan.reads {
        Reads::Own => (
            own(plan, operands, 0),
            own(plan, operands, 1),
            own(plan, operands, 2),
        ),
        Reads::Leafwise(r) => (
            cast(plan, operands, 0, r[0]),
            cast(plan, operands, 1, r[1]),
            cast(plan, operands, 2, r[2]),
        ),
    }
}

#[inline(always)]
fn quad<T, const PLAIN: bool>(plan: &Plan, operands: Operands<'_>) -> (T, T, T, T)
where
    T: Num,
{
    if PLAIN {
        return (
            own(plan, operands, 0),
            own(plan, operands, 1),
            own(plan, operands, 2),
            own(plan, operands, 3),
        );
    }
    match plan.reads {
        Reads::Own => (
            own(plan, operands, 0),
            own(plan, operands, 1),
            own(plan, operands, 2),
            own(plan, operands, 3),
        ),
        Reads::Leafwise(r) => (
            cast(plan, operands, 0, r[0]),
            cast(plan, operands, 1, r[1]),
            cast(plan, operands, 2, r[2]),
            cast(plan, operands, 3, r[3]),
        ),
    }
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

pub struct Nodes {
    pub ops: [Node; ChainBounds::MAX_INTERIOR],
    pub root: Node,
}

macro_rules! instances {
    (concrete: [$($v:ident => $m:ident),* $(,)?], generic: [$($g:ident),* $(,)?] $(,)?) => {
        /// The operator a chain instance carries in its type at one node.
        /// `Any` reads that node's operator from the instance's `ops` field
        /// instead, at one predicted branch; the alphabet the `instances!`
        /// invocation lists is the knob, and it is deliberately small
        /// because every operator outside it still runs through `Any`.
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        #[repr(u8)]
        pub enum Node {
            Any,
            $($v),*
        }

        impl Node {
            pub fn of(op: Arith) -> Node {
                match op {
                    $(Arith::$v => Node::$v,)*
                    $(Arith::$g => Node::Any,)*
                }
            }

            pub fn of_root(root: Root) -> Node {
                match root {
                    Root::Num(op) => Node::of(op),
                    Root::Cmp(_) => Node::Any,
                }
            }

            const fn read(raw: u8) -> Node {
                match raw {
                    0 => Node::Any,
                    $(x if x == Node::$v as u8 => Node::$v,)*
                    _ => panic!("a chain instance was parameterized by no node"),
                }
            }
        }

        #[inline(always)]
        fn apply<T, const O: u8>(op: Arith, left: T, right: T) -> T
        where
            T: Num,
        {
            match Node::read(O) {
                Node::Any => any(op, left, right),
                $(Node::$v => {
                    debug_assert_eq!(
                        op,
                        Arith::$v,
                        "a chain instance ran a node whose operator is not its own"
                    );
                    left.$m(right)
                })*
            }
        }

        /// The destination slot's kind was written when the frame was made,
        /// so the caller stores these eight bytes and nothing else
        /// (RFC-0052 rule 5).
        #[inline(always)]
        fn finish<T, const R: u8>(root: Root, left: T, right: T) -> u64
        where
            T: Num,
        {
            match Node::read(R) {
                Node::Any => match root {
                    Root::Num(op) => any(op, left, right).word(),
                    Root::Cmp(how) => left.compare(right, how) as u64,
                },
                $(Node::$v => {
                    debug_assert_eq!(
                        root,
                        Root::Num(Arith::$v),
                        "a chain instance ran a root whose operator is not its own"
                    );
                    left.$m(right).word()
                })*
            }
        }

        fn pick1<T, B>(nodes: &Nodes, make: &mut B) -> B::Out
        where
            T: Num,
            B: Build<T>,
        {
            match nodes.root {
                Node::Any => make.one::<{ Node::Any as u8 }>(),
                $(Node::$v => make.one::<{ Node::$v as u8 }>(),)*
            }
        }

        /// The root operator as a const parameter, for an operation whose
        /// shape is one node and which therefore has no interior to walk.
        pub(crate) fn pick_root<T, B>(root: Node, make: &mut B) -> B::Out
        where
            T: Num,
            B: Rooted<T>,
        {
            match root {
                Node::Any => make.of::<{ Node::Any as u8 }>(),
                $(Node::$v => make.of::<{ Node::$v as u8 }>(),)*
            }
        }

        fn pick2<T, B>(nodes: &Nodes, make: &mut B) -> B::Out
        where
            T: Num,
            B: Build<T>,
        {
            match nodes.ops[0] {
                Node::Any => pick2_root::<T, B, { Node::Any as u8 }>(nodes, make),
                $(Node::$v => pick2_root::<T, B, { Node::$v as u8 }>(nodes, make),)*
            }
        }

        fn pick2_root<T, B, const O0: u8>(nodes: &Nodes, make: &mut B) -> B::Out
        where
            T: Num,
            B: Build<T>,
        {
            match nodes.root {
                Node::Any => make.two::<O0, { Node::Any as u8 }>(),
                $(Node::$v => make.two::<O0, { Node::$v as u8 }>(),)*
            }
        }

        fn pick3<T, B>(nodes: &Nodes, make: &mut B) -> B::Out
        where
            T: Num,
            B: Build<T>,
        {
            match nodes.ops[0] {
                Node::Any => pick3_op1::<T, B, { Node::Any as u8 }>(nodes, make),
                $(Node::$v => pick3_op1::<T, B, { Node::$v as u8 }>(nodes, make),)*
            }
        }

        fn pick3_op1<T, B, const O0: u8>(nodes: &Nodes, make: &mut B) -> B::Out
        where
            T: Num,
            B: Build<T>,
        {
            match nodes.ops[1] {
                Node::Any => pick3_root::<T, B, O0, { Node::Any as u8 }>(nodes, make),
                $(Node::$v => pick3_root::<T, B, O0, { Node::$v as u8 }>(nodes, make),)*
            }
        }

        fn pick3_root<T, B, const O0: u8, const O1: u8>(nodes: &Nodes, make: &mut B) -> B::Out
        where
            T: Num,
            B: Build<T>,
        {
            match nodes.root {
                Node::Any => make.three::<O0, O1, { Node::Any as u8 }>(),
                $(Node::$v => make.three::<O0, O1, { Node::$v as u8 }>(),)*
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

fn wrong_shape(shape: Shape, nodes: usize) -> ! {
    panic!(
        "a chain instance of {nodes} nodes ran shape {shape:?}, which has {}",
        shape.nodes()
    )
}

#[inline(always)]
pub(crate) fn tree1<T, const R: u8, const PLAIN: bool>(plan: &Plan, operands: Operands<'_>) -> u64
where
    T: Num,
{
    let (a, b) = pair::<T, PLAIN>(plan, operands);
    finish::<T, R>(plan.root, a, b)
}

#[inline(always)]
fn tree2<T, const O0: u8, const R: u8, const PLAIN: bool>(
    plan: &Plan,
    operands: Operands<'_>,
) -> u64
where
    T: Num,
{
    let (a, b, c) = triple::<T, PLAIN>(plan, operands);
    match plan.shape {
        Shape::NNLLL => {
            let n0 = apply::<T, O0>(plan.ops[0], a, b);
            finish::<T, R>(plan.root, n0, c)
        }
        Shape::NLNLL => {
            let n0 = apply::<T, O0>(plan.ops[0], b, c);
            finish::<T, R>(plan.root, a, n0)
        }
        shape => wrong_shape(shape, 2),
    }
}

#[inline(always)]
fn tree3<T, const O0: u8, const O1: u8, const R: u8, const PLAIN: bool>(
    plan: &Plan,
    operands: Operands<'_>,
) -> u64
where
    T: Num,
{
    let (a, b, c, d) = quad::<T, PLAIN>(plan, operands);
    let ops = plan.ops;
    match plan.shape {
        Shape::NNNLLLL => {
            let n0 = apply::<T, O0>(ops[0], a, b);
            let n1 = apply::<T, O1>(ops[1], n0, c);
            finish::<T, R>(plan.root, n1, d)
        }
        Shape::NNLNLLL => {
            let n0 = apply::<T, O0>(ops[0], b, c);
            let n1 = apply::<T, O1>(ops[1], a, n0);
            finish::<T, R>(plan.root, n1, d)
        }
        Shape::NNLLNLL => {
            let n0 = apply::<T, O0>(ops[0], a, b);
            let n1 = apply::<T, O1>(ops[1], c, d);
            finish::<T, R>(plan.root, n0, n1)
        }
        Shape::NLNNLLL => {
            let n0 = apply::<T, O0>(ops[0], b, c);
            let n1 = apply::<T, O1>(ops[1], n0, d);
            finish::<T, R>(plan.root, a, n1)
        }
        Shape::NLNLNLL => {
            let n0 = apply::<T, O0>(ops[0], c, d);
            let n1 = apply::<T, O1>(ops[1], b, n0);
            finish::<T, R>(plan.root, a, n1)
        }
        shape => wrong_shape(shape, 3),
    }
}

/// RFC-0044 made the shape a type parameter of the chain operation. Under
/// RFC-0052 it is a field, and the chain family fell from 1404 instances
/// per entry to 351.
pub struct Plan {
    pub shape: Shape,
    pub root: Root,
    pub ops: [Arith; ChainBounds::MAX_INTERIOR],
    pub leaves: [u16; ChainBounds::MAX_LEAVES],
    pub reads: Reads,
}

pub struct Chain1<T, D, const R: u8>
where
    T: Num,
    D: Place,
{
    pub dst: D::At,
    pub plan: Plan,
    pub next: Box<dyn Op>,
    pub at: PhantomData<fn() -> (T, D)>,
}

pub struct Chain2<T, D, const O0: u8, const R: u8>
where
    T: Num,
    D: Place,
{
    pub dst: D::At,
    pub plan: Plan,
    pub next: Box<dyn Op>,
    pub at: PhantomData<fn() -> (T, D)>,
}

pub struct Chain3<T, D, const O0: u8, const O1: u8, const R: u8>
where
    T: Num,
    D: Place,
{
    pub dst: D::At,
    pub plan: Plan,
    pub next: Box<dyn Op>,
    pub at: PhantomData<fn() -> (T, D)>,
}

impl<T, D, const R: u8> Op for Chain1<T, D, R>
where
    T: Num,
    D: Place,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, _r0: u64) -> Exit {
        let bits = tree1::<T, R, false>(&self.plan, Operands::of_frame(m.regs()));
        let carried = D::write(m.regs(), self.dst, bits);
        self.next.run(m, carried)
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn chain(&self) -> Option<crate::code::ChainProbe<'_>> {
        Some(crate::code::ChainProbe {
            dst: D::whence(self.dst),
            plan: &self.plan,
        })
    }
}

impl<T, D, const O0: u8, const R: u8> Op for Chain2<T, D, O0, R>
where
    T: Num,
    D: Place,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, _r0: u64) -> Exit {
        let bits = tree2::<T, O0, R, false>(&self.plan, Operands::of_frame(m.regs()));
        let carried = D::write(m.regs(), self.dst, bits);
        self.next.run(m, carried)
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn chain(&self) -> Option<crate::code::ChainProbe<'_>> {
        Some(crate::code::ChainProbe {
            dst: D::whence(self.dst),
            plan: &self.plan,
        })
    }
}

impl<T, D, const O0: u8, const O1: u8, const R: u8> Op for Chain3<T, D, O0, O1, R>
where
    T: Num,
    D: Place,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, _r0: u64) -> Exit {
        let bits = tree3::<T, O0, O1, R, false>(&self.plan, Operands::of_frame(m.regs()));
        let carried = D::write(m.regs(), self.dst, bits);
        self.next.run(m, carried)
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn chain(&self) -> Option<crate::code::ChainProbe<'_>> {
        Some(crate::code::ChainProbe {
            dst: D::whence(self.dst),
            plan: &self.plan,
        })
    }
}

/// The chain at the head of `code`, and the assertion that the call brought
/// the arity its body was prepared with.
///
/// # Safety
/// `code` is one this module's entries was written into, which `Code::expr`
/// does for an `Expr` whose body is a `Chain`.
#[inline(always)]
unsafe fn chain_of(code: &Code, arity: u16) -> &ExprChain {
    // SAFETY: the caller's contract.
    let expr = unsafe { code.body.expr_unchecked() };
    debug_assert_eq!(
        usize::from(arity),
        expr.arity as usize,
        "an expression body is called with the arguments it reads"
    );
    // SAFETY: as above.
    unsafe { expr.chain_unchecked() }
}

/// The three entries of a body that is one chain (RFC-0069 rule 5). Each reads
/// the argument run the caller laid, builds the operand space over it and
/// evaluates the chain in one body, so a closure call reaches the arithmetic
/// through the one indirect call at the code's head and none inside.
macro_rules! chain_entry {
    ($name:ident, $tree:ident $(, const $node:ident: u8)*) => {
        /// # Safety
        /// As `Entry`, and `code`'s body is the `Expr` this entry was picked
        /// for.
        unsafe fn $name<T $(, const $node: u8)*, const R: u8, const PLAIN: bool>(
            code: &Code,
            _f: Value,
            _rt: &AcvusRuntime,
            window: &mut FrameState,
            arity: u16,
        ) -> Value
        where
            T: Num,
        {
            // SAFETY: the caller's contract.
            let chain = unsafe { chain_of(code, arity) };
            let space = OperandSpace::of(window.laid(arity), &chain.konsts);
            let word = $tree::<T $(, $node)*, R, PLAIN>(
                &chain.plan,
                Operands::of(space.as_slice()),
            );
            Value::inline(chain.kind, word)
        }
    };
}

chain_entry!(entry1, tree1);
chain_entry!(entry2, tree2, const O0: u8);
chain_entry!(entry3, tree3, const O0: u8, const O1: u8);

/// The operands a frameless chain reads: the arguments, then the constants,
/// which is the order `prepare::expression_body` assigned the chain's leaf
/// offsets in.
struct OperandSpace {
    values: [MaybeUninit<Value>; ExprChain::MAX_OPERANDS],
    len: usize,
}

impl OperandSpace {
    /// The `#[inline]` is a measurement, not a taste. Without it LLVM
    /// outlines this across the entries' monomorphizations, and an entry
    /// then spends a frame and a call on the way to arithmetic it already
    /// holds: `map add fil | sum` measured 13.8–14.5 ns per element that way
    /// against 13.4–13.5 with it. It costs 535 KB of text in the `accum`
    /// bench, the copy loop unrolled once per instance.
    #[inline]
    fn of(args: &[Value], konsts: &[Value]) -> OperandSpace {
        let len = args.len() + konsts.len();
        debug_assert!(
            len <= ExprChain::MAX_OPERANDS,
            "an expression body reads {len} operands, past the {} a frameless call builds",
            ExprChain::MAX_OPERANDS
        );
        let mut values = [const { MaybeUninit::uninit() }; ExprChain::MAX_OPERANDS];
        for (slot, value) in values.iter_mut().zip(args.iter().chain(konsts)) {
            slot.write(*value);
        }
        OperandSpace { values, len }
    }

    fn as_slice(&self) -> &[Value] {
        // SAFETY: `prepare::expression_body` refuses a body whose operands
        // outnumber `ExprChain::MAX_OPERANDS`, and `chain_of` asserts the
        // call brought the arity that body was prepared with, so `of` wrote
        // exactly `len` values into an array that holds them.
        unsafe { slice::from_raw_parts(self.values.as_ptr().cast::<Value>(), self.len) }
    }
}

/// What the picker's walk ends in. The walk is one `match` per node, and
/// its leaf is the only place the node operators are const generics, so
/// each caller reaches that leaf with its own builder and takes its own
/// result type out of it.
trait Build<T>
where
    T: Num,
{
    type Out;

    fn one<const R: u8>(&mut self) -> Self::Out;
    fn two<const O0: u8, const R: u8>(&mut self) -> Self::Out;
    fn three<const O0: u8, const O1: u8, const R: u8>(&mut self) -> Self::Out;
}

/// What `pick_root`'s walk ends in: one node's operator as a const parameter.
pub(crate) trait Rooted<T>
where
    T: Num,
{
    type Out;

    fn of<const R: u8>(&mut self) -> Self::Out;
}

struct BuildOp<T, D>
where
    T: Num,
    D: Place,
{
    dst: D::At,
    plan: Option<Plan>,
    next: Option<Box<dyn Op>>,
    at: PhantomData<fn() -> (T, D)>,
}

impl<T, D> BuildOp<T, D>
where
    T: Num,
    D: Place,
{
    fn take(&mut self) -> (Plan, Box<dyn Op>) {
        let plan = self
            .plan
            .take()
            .expect("a chain instance is built once from its plan");
        let next = self
            .next
            .take()
            .expect("a chain instance is built once from its plan");
        (plan, next)
    }
}

impl<T, D> Build<T> for BuildOp<T, D>
where
    T: Num,
    D: Place,
{
    type Out = Box<dyn Op>;

    fn one<const R: u8>(&mut self) -> Box<dyn Op> {
        let (plan, next) = self.take();
        Box::new(Chain1::<T, D, R> {
            dst: self.dst,
            plan,
            next,
            at: PhantomData,
        })
    }

    fn two<const O0: u8, const R: u8>(&mut self) -> Box<dyn Op> {
        let (plan, next) = self.take();
        Box::new(Chain2::<T, D, O0, R> {
            dst: self.dst,
            plan,
            next,
            at: PhantomData,
        })
    }

    fn three<const O0: u8, const O1: u8, const R: u8>(&mut self) -> Box<dyn Op> {
        let (plan, next) = self.take();
        Box::new(Chain3::<T, D, O0, O1, R> {
            dst: self.dst,
            plan,
            next,
            at: PhantomData,
        })
    }
}

/// A body that is one chain is entered through this function pointer and
/// does nothing else, so whether its leaves hold a cast is settled here,
/// once, rather than read on every call.
struct BuildExpr<T>
where
    T: Num,
{
    plain: bool,
    at: PhantomData<fn() -> T>,
}

impl<T> Build<T> for BuildExpr<T>
where
    T: Num,
{
    type Out = Entry;

    fn one<const R: u8>(&mut self) -> Entry {
        match self.plain {
            true => entry1::<T, R, true>,
            false => entry1::<T, R, false>,
        }
    }

    fn two<const O0: u8, const R: u8>(&mut self) -> Entry {
        match self.plain {
            true => entry2::<T, O0, R, true>,
            false => entry2::<T, O0, R, false>,
        }
    }

    fn three<const O0: u8, const O1: u8, const R: u8>(&mut self) -> Entry {
        match self.plain {
            true => entry3::<T, O0, O1, R, true>,
            false => entry3::<T, O0, O1, R, false>,
        }
    }
}

/// The numeric type a chain runs at, decided at preparation from the
/// types of the operations it collapses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChainTy {
    Int(IntTy),
    Float,
}

impl Nodes {
    pub fn of(plan: &Plan) -> Nodes {
        let mut ops = [Node::Any; ChainBounds::MAX_INTERIOR];
        let used = plan.shape.interior();
        for (node, op) in ops[..used].iter_mut().zip(&plan.ops[..used]) {
            *node = Node::of(*op);
        }
        Nodes {
            ops,
            root: Node::of_root(plan.root),
        }
    }

    pub fn concrete(&self, shape: Shape) -> usize {
        let ops = self.ops[..shape.interior()]
            .iter()
            .filter(|node| **node != Node::Any)
            .count();
        ops + usize::from(self.root != Node::Any)
    }
}

fn pick<T, B>(shape: Shape, nodes: &Nodes, make: &mut B) -> B::Out
where
    T: Num,
    B: Build<T>,
{
    match shape.nodes() {
        1 => pick1::<T, B>(nodes, make),
        2 => pick2::<T, B>(nodes, make),
        _ => pick3::<T, B>(nodes, make),
    }
}

/// Decision not to build: a leaf has no place of its own. A chain's leaves
/// read registers the chain did not produce — a one-use value feeding a leaf
/// was absorbed into the chain instead of reaching it — so no leaf rides.
fn chain_at<D>(ty: ChainTy, dst: D::At, plan: Plan, next: Box<dyn Op>) -> Box<dyn Op>
where
    D: Place,
{
    let nodes = Nodes::of(&plan);
    let shape = plan.shape;
    match ty {
        ChainTy::Int(k) => for_int_ty!(k, |T| {
            let mut make = BuildOp::<T, D> {
                dst,
                plan: Some(plan),
                next: Some(next),
                at: PhantomData,
            };
            pick::<T, BuildOp<T, D>>(shape, &nodes, &mut make)
        }),
        ChainTy::Float => {
            let mut make = BuildOp::<f64, D> {
                dst,
                plan: Some(plan),
                next: Some(next),
                at: PhantomData,
            };
            pick::<f64, BuildOp<f64, D>>(shape, &nodes, &mut make)
        }
    }
}

pub fn chain_op(ty: ChainTy, dst: Where, plan: Plan, next: Box<dyn Op>) -> Box<dyn Op> {
    match dst {
        Where::Frame(off) => chain_at::<crate::ops::place::Slot>(ty, off, plan, next),
        Where::Register => chain_at::<crate::ops::place::R0>(ty, (), plan, next),
    }
}

/// The entry a frameless chain body is called through (RFC-0044 rule 5;
/// RFC-0069 rule 5).
pub fn chain_eval(ty: ChainTy, plan: &Plan) -> Entry {
    let nodes = Nodes::of(plan);
    let shape = plan.shape;
    let plain = matches!(plan.reads, Reads::Own);
    match ty {
        ChainTy::Int(k) => for_int_ty!(k, |T| {
            let mut make = BuildExpr::<T> {
                plain,
                at: PhantomData,
            };
            pick::<T, BuildExpr<T>>(shape, &nodes, &mut make)
        }),
        ChainTy::Float => {
            let mut make = BuildExpr::<f64> {
                plain,
                at: PhantomData,
            };
            pick::<f64, BuildExpr<f64>>(shape, &nodes, &mut make)
        }
    }
}
