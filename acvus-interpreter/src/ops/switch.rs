//! A `match` is one dispatch (RFC-0051 §5): the tag is read once and the
//! block it names is returned, where the chain used to run one
//! `TestVariant` and one `JumpIf` per arm.
//!
//! Obligation across artifacts: `prepare::switch_op` hands every one of
//! these operations a `default` that is a real edge — the catch-all where
//! the `match` wrote one, and otherwise the last arm, which is then the one
//! tag left untested. So no `run` here has to decide that no arm holds;
//! `validate::exhaustive` (RFC-0051 §3) decided that already.
//!
//! Decision not to build: the table RFC-0051 §5 names. A value's tag today
//! is an `Astr`, an interned name and not an ordinal, so no table can be
//! indexed by the tag itself; the nearest form is a hashed table whose
//! lookup is a multiply and a dependent load, and at the seven arms of
//! `benches/programs.rs`'s `bf table` that measured 24.4 ns a step against
//! the scan's 23.3 (2026-09-19, three pinned reps, ranges apart). RFC-0050
//! gives the value a tag word; the table is that RFC's, with an ordinal to
//! index by and a variant count to bound it.

use acvus_utils::Astr;

use crate::code::{BlockId, Exit, Off, Op};
use crate::machine::Machine;
use crate::ops::variant::scrutinee;

/// One tested arm: the tag it names and the block the machine enters for it.
pub struct Arm {
    pub key: Astr,
    pub target: BlockId,
}

/// A boxed variant's dispatch: one tag read and a scan of the arms, which
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
        let tag = unsafe { source.as_variant() }.tag;
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

/// A `Result`'s dispatch, reading the tag where `TestResult` reads it.
pub struct SwitchResult<const THROUGH: bool> {
    pub src: Off,
    pub on_ok: BlockId,
    pub on_err: BlockId,
}

impl<const THROUGH: bool> Op for SwitchResult<THROUGH> {
    #[inline]
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        let source = scrutinee::<THROUGH>(m.regs().peek(self.src));
        // SAFETY: the preparation read `Result` from the source's type.
        match unsafe { source.as_result() }.is_ok() {
            true => self.on_ok.into(),
            false => self.on_err.into(),
        }
    }
}
