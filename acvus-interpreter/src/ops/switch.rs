//! A `match` is one dispatch (RFC-0051 §5): the key is read once and the
//! block it names is returned, where the chain used to run one
//! `TestVariant` or `TestLiteral` and one `JumpIf` per arm.
//!
//! Obligation across artifacts: `prepare::switch_op` and its region form
//! `prepare::switch_region_op` hand every one of these operations a
//! `default` that is a real successor — the catch-all where the `match`
//! wrote one, and otherwise the last arm, which is then the one tag left
//! untested. So no `run` here has to decide that no arm holds;
//! `validate::exhaustive` (RFC-0051 §3) decided that already.
//!
//! Decision not to build: the table RFC-0051 §5 names. A tag word is the
//! program's number for an interned name, which is sparse, so no table can be
//! indexed by it; the nearest form is a hashed table whose lookup is a multiply
//! and a dependent load, and at the seven arms of `benches/programs.rs`'s
//! `bf table` that measured 24.4 ns a step against the scan's 23.3 (2026-09-19,
//! three pinned reps, ranges apart). `ops::run::SwitchRun` records the same
//! effect for a table a dense ordinal would have allowed.

use std::marker::PhantomData;

#[cfg(any(debug_assertions, feature = "probe"))]
use crate::code::OwnedOps;
use crate::code::{BlockId, Exit, Off, Op, successor};
use crate::machine::Machine;
use crate::ops::arith::Int;
use crate::ops::variant::scrutinee;

/// One tested arm: the word it names — a variant's tag or a literal
/// (RFC-0051) — and the block the machine enters for it.
pub struct Arm {
    pub key: u64,
    pub target: BlockId,
}

/// A heap variant's dispatch: one tag read and a scan of the arms, which
/// `prepare` ordered as the `match` wrote them.
pub struct Switch<const THROUGH: bool> {
    pub src: Off,
    pub arms: Box<[Arm]>,
    pub default: BlockId,
}

impl<const THROUGH: bool> Op for Switch<THROUGH> {
    #[inline]
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        let source = scrutinee::<THROUGH>(m.regs().peek(self.src));
        // SAFETY: the preparation read an enum from the source's type.
        let tag = unsafe { source.as_variant() }.tag().bits();
        self.arms
            .iter()
            .find(|arm| arm.key == tag)
            .map_or(self.default, |arm| arm.target)
            .into()
    }
}

/// An `Option`'s dispatch. Its tag is the value's own kind (RFC-0039), so
/// the read and the test are one operation; `prepare` resolved each side's
/// block, the catch-all included, and no name is compared here.
pub struct SwitchOption<const THROUGH: bool> {
    pub src: Off,
    pub on_some: BlockId,
    pub on_none: BlockId,
}

impl<const THROUGH: bool> Op for SwitchOption<THROUGH> {
    #[inline]
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        match scrutinee::<THROUGH>(m.regs().peek(self.src)).is_none() {
            true => self.on_none.into(),
            false => self.on_some.into(),
        }
    }
}

/// One tested arm of a `match` whose arms all rejoin: the word it names and
/// the chain the machine runs for it, ended by `Yield` as every region part
/// is (RFC-0052 §3).
pub struct RegionArm {
    pub key: u64,
    pub head: Box<dyn Op>,
}

/// The region form of `Switch`: the arms are chains of this operation rather
/// than blocks of the stream, so the dispatch chooses no `BlockId` and the
/// `match` is one operation of the chain it sits in.
pub struct SwitchRegion<const THROUGH: bool> {
    pub src: Off,
    pub arms: Box<[RegionArm]>,
    pub default: Box<dyn Op>,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool> Op for SwitchRegion<THROUGH> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let source = scrutinee::<THROUGH>(m.regs().peek(self.src));
        // SAFETY: the preparation read an enum from the source's type.
        let tag = unsafe { source.as_variant() }.tag().bits();
        let arm = self
            .arms
            .iter()
            .find(|arm| arm.key == tag)
            .map_or(self.default.as_ref(), |arm| arm.head.as_ref());
        let word = arm.run(m, r0);
        self.next.run(m, word)
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        self.arms
            .iter()
            .map(|arm| OwnedOps {
                part: "arm",
                head: arm.head.as_ref(),
            })
            .chain([OwnedOps {
                part: "default",
                head: self.default.as_ref(),
            }])
            .collect()
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns_mut(&mut self) -> Vec<&mut Box<dyn Op>> {
        self.arms
            .iter_mut()
            .map(|arm| &mut arm.head)
            .chain([&mut self.default])
            .collect()
    }
}

/// The region form of `SwitchOption`.
pub struct SwitchOptionRegion<const THROUGH: bool> {
    pub src: Off,
    pub on_some: Box<dyn Op>,
    pub on_none: Box<dyn Op>,
    pub next: Box<dyn Op>,
}

impl<const THROUGH: bool> Op for SwitchOptionRegion<THROUGH> {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let arm = match scrutinee::<THROUGH>(m.regs().peek(self.src)).is_none() {
            true => self.on_none.as_ref(),
            false => self.on_some.as_ref(),
        };
        let word = arm.run(m, r0);
        self.next.run(m, word)
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        vec![
            OwnedOps {
                part: "on_some",
                head: self.on_some.as_ref(),
            },
            OwnedOps {
                part: "on_none",
                head: self.on_none.as_ref(),
            },
        ]
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns_mut(&mut self) -> Vec<&mut Box<dyn Op>> {
        vec![&mut self.on_some, &mut self.on_none]
    }
}

/// A `match` on integer or char literals (RFC-0051): the scrutinee is an
/// inline word, so the read is the register itself where a variant's is
/// `as_variant().tag()`.
///
/// `T` is the scrutinee's width, so the word is normalized exactly as
/// `pattern::TestInt` normalizes the one it compares, and `prepare`
/// normalized the keys through the same function: a dispatch decides what
/// the chain of tests decided.
pub struct SwitchWord<T>
where
    T: Int,
{
    src: Off,
    arms: Box<[Arm]>,
    default: BlockId,
    width: PhantomData<fn() -> T>,
}

impl<T> SwitchWord<T>
where
    T: Int,
{
    pub fn new(src: Off, arms: Box<[Arm]>, default: BlockId) -> SwitchWord<T> {
        SwitchWord {
            src,
            arms,
            default,
            width: PhantomData,
        }
    }
}

impl<T> Op for SwitchWord<T>
where
    T: Int,
{
    #[inline]
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        let word = T::read(m.regs().word(self.src)).word();
        self.arms
            .iter()
            .find(|arm| arm.key == word)
            .map_or(self.default, |arm| arm.target)
            .into()
    }
}

/// The region form of [`SwitchWord`].
pub struct SwitchWordRegion<T>
where
    T: Int,
{
    src: Off,
    arms: Box<[RegionArm]>,
    default: Box<dyn Op>,
    next: Box<dyn Op>,
    width: PhantomData<fn() -> T>,
}

impl<T> SwitchWordRegion<T>
where
    T: Int,
{
    pub fn new(
        src: Off,
        arms: Box<[RegionArm]>,
        default: Box<dyn Op>,
        next: Box<dyn Op>,
    ) -> SwitchWordRegion<T> {
        SwitchWordRegion {
            src,
            arms,
            default,
            next,
            width: PhantomData,
        }
    }
}

impl<T> Op for SwitchWordRegion<T>
where
    T: Int,
{
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let word = T::read(m.regs().word(self.src)).word();
        let arm = self
            .arms
            .iter()
            .find(|arm| arm.key == word)
            .map_or(self.default.as_ref(), |arm| arm.head.as_ref());
        let taken = arm.run(m, r0);
        self.next.run(m, taken)
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns(&self) -> Vec<OwnedOps<'_>> {
        self.arms
            .iter()
            .map(|arm| OwnedOps {
                part: "arm",
                head: arm.head.as_ref(),
            })
            .chain([OwnedOps {
                part: "default",
                head: self.default.as_ref(),
            }])
            .collect()
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    fn owns_mut(&mut self) -> Vec<&mut Box<dyn Op>> {
        self.arms
            .iter_mut()
            .map(|arm| &mut arm.head)
            .chain([&mut self.default])
            .collect()
    }
}
