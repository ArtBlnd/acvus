//! An aggregate in a run of registers (RFC-0050 rules 2, 3, 8, 9).
//!
//! A run's register is addressed by `Off` like every other, so there is no
//! operation here for a field read, a field write or a payload move. That is a
//! decision rather than an omission.

#[cfg(any(debug_assertions, feature = "probe"))]
use crate::code::OwnedOps;
use crate::code::{BlockId, Exit, Marked, Off, Op, successor};
use crate::machine::Machine;
use crate::value::Value;

pub struct LaidKonst {
    pub at: Marked,
    pub value: Value,
}

pub struct LaidMove {
    pub at: Marked,
    pub src: Marked,
    pub large: bool,
}

pub struct LayRun {
    pub konsts: Box<[LaidKonst]>,
    pub moved: Box<[LaidMove]>,
    pub next: Box<dyn Op>,
}

impl Op for LayRun {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        for laid in &self.moved {
            match laid.large {
                true => {
                    let value = regs.take::<true>(laid.src);
                    regs.assign::<true>(laid.at, value);
                }
                false => {
                    let value = regs.take::<false>(laid.src);
                    regs.assign::<false>(laid.at, value);
                }
            }
        }
        for laid in &self.konsts {
            regs.assign::<false>(laid.at, laid.value);
        }
        self.next.run(m, r0)
    }
}

pub struct Project {
    pub dst: Off,
    pub at: Off,
    pub next: Box<dyn Op>,
}

impl Op for Project {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let projection = regs.projection(self.at);
        regs.put(self.dst, projection);
        self.next.run(m, r0)
    }
}

pub struct DropRun {
    pub registers: Box<[Marked]>,
    pub next: Box<dyn Op>,
}

impl Op for DropRun {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        for at in &self.registers {
            regs.assign::<false>(*at, Value::UNDEF);
        }
        self.next.run(m, r0)
    }
}

pub struct TestRun {
    pub dst: Off,
    pub src: Off,
    pub tag: u64,
    pub next: Box<dyn Op>,
}

impl Op for TestRun {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let matches = regs.word(self.src) == self.tag;
        regs.set_word(self.dst, matches as u64);
        self.next.run(m, r0)
    }
}

/// One tested arm: the tag word `prepare::runs::Member::word` gives its name,
/// and the block the machine enters for it.
pub struct RunArm {
    pub tag: u64,
    pub target: BlockId,
}

/// Obligation across artifacts: `arms` carry the words `Member::word` gives
/// their names, which is what `value::Value::tag` writes into a heap variant's tag
/// register and what `prepare::lay_variant` writes into a run's. `default` is
/// the edge RFC-0051's `switch_op` guarantees.
///
/// Decision not to index a table of blocks by the tag, which a dense ordinal
/// would have made possible. Measured on `benches/shapes.rs`'s `enum match held` at three arms,
/// four alternating pinned reps, min of each: the table ran 46.2 ns an
/// iteration, this scan 38.2, and the heap form it replaces 42.6 — so the table
/// was slower than the heap form it was meant to beat. It executed 1.36 G fewer
/// instructions than the heap form and spent more cycles doing it, at equal
/// cache misses: a second data-dependent indirect branch beside the machine's
/// own dispatch costs more than three compares save. `ops::switch` records the
/// same effect at seven arms. The ordinal that table needed is withdrawn for a
/// second reason in RFC-0050's Consequences: a heap construction does not hold
/// the type it would be a position in.
pub struct SwitchRun {
    pub src: Off,
    pub arms: Box<[RunArm]>,
    pub default: BlockId,
}

impl Op for SwitchRun {
    #[inline]
    fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
        let tag = m.regs().word(self.src);
        self.arms
            .iter()
            .find(|arm| arm.tag == tag)
            .map_or(self.default, |arm| arm.target)
            .into()
    }
}

/// One tested arm of a run-resident `match` whose arms all rejoin.
pub struct RunRegionArm {
    pub tag: u64,
    pub head: Box<dyn Op>,
}

/// The region form of `SwitchRun`.
pub struct SwitchRunRegion {
    pub src: Off,
    pub arms: Box<[RunRegionArm]>,
    pub default: Box<dyn Op>,
    pub next: Box<dyn Op>,
}

impl Op for SwitchRunRegion {
    successor!();

    #[inline]
    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let tag = m.regs().word(self.src);
        let arm = self
            .arms
            .iter()
            .find(|arm| arm.tag == tag)
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
