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

/// The fixed forms are the shapes `prepare` emitted over the interpreter test
/// corpus and the bench scripts when they were added, and they covered every
/// construction counted there. No form is built for a shape that was not seen;
/// such a shape runs `LayRun`.
pub fn lay_run(konsts: Box<[LaidKonst]>, moved: Box<[LaidMove]>, next: Box<dyn Op>) -> Box<dyn Op> {
    match (&*konsts, &*moved) {
        ([tag], [payload @ LaidMove { large: false, .. }]) => Box::new(LayRunOne::<1, false> {
            moved: FixedMove::of(payload),
            konsts: [tag.copied()],
            next,
        }),
        ([tag], [payload @ LaidMove { large: true, .. }]) => Box::new(LayRunOne::<1, true> {
            moved: FixedMove::of(payload),
            konsts: [tag.copied()],
            next,
        }),
        ([], [field @ LaidMove { large: false, .. }]) => Box::new(LayRunOne::<0, false> {
            moved: FixedMove::of(field),
            konsts: [],
            next,
        }),
        (
            [],
            [
                first @ LaidMove { large: false, .. },
                second @ LaidMove { large: true, .. },
            ],
        ) => Box::new(LayRunTwo::<0, false, true> {
            moved: [FixedMove::of(first), FixedMove::of(second)],
            konsts: [],
            next,
        }),
        _ => Box::new(LayRun {
            konsts,
            moved,
            next,
        }),
    }
}

impl LaidKonst {
    fn copied(&self) -> LaidKonst {
        LaidKonst {
            at: self.at,
            value: self.value,
        }
    }
}

#[derive(Clone, Copy)]
pub struct FixedMove {
    at: Marked,
    src: Marked,
}

impl FixedMove {
    fn of(laid: &LaidMove) -> FixedMove {
        FixedMove {
            at: laid.at,
            src: laid.src,
        }
    }

    #[inline(always)]
    fn lay<const LARGE: bool>(self, regs: &mut crate::regs::Regs<'_>) {
        let value = regs.take::<LARGE>(self.src);
        regs.assign::<LARGE>(self.at, value);
    }
}

pub struct LayRunOne<const K: usize, const LARGE: bool> {
    pub moved: FixedMove,
    pub konsts: [LaidKonst; K],
    pub next: Box<dyn Op>,
}

impl<const K: usize, const LARGE: bool> Op for LayRunOne<K, LARGE> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        self.moved.lay::<LARGE>(regs);
        for laid in &self.konsts {
            regs.assign::<false>(laid.at, laid.value);
        }
        self.next.run(m, r0)
    }
}

pub struct LayRunTwo<const K: usize, const FIRST: bool, const SECOND: bool> {
    pub moved: [FixedMove; 2],
    pub konsts: [LaidKonst; K],
    pub next: Box<dyn Op>,
}

impl<const K: usize, const FIRST: bool, const SECOND: bool> Op for LayRunTwo<K, FIRST, SECOND> {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let [first, second] = self.moved;
        first.lay::<FIRST>(regs);
        second.lay::<SECOND>(regs);
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

/// The fixed forms are the counts `prepare` emitted over the interpreter test
/// corpus and the bench scripts when they were added, and they covered every
/// release counted there. No form is built for a count that was not seen; such
/// a count runs `DropRun`.
pub fn drop_run(registers: Box<[Marked]>, next: Box<dyn Op>) -> Box<dyn Op> {
    match *registers {
        [] => Box::new(DropRunN::<0> {
            registers: [],
            next,
        }),
        [at] => Box::new(DropRunN::<1> {
            registers: [at],
            next,
        }),
        _ => Box::new(DropRun { registers, next }),
    }
}

pub struct DropRunN<const N: usize> {
    pub registers: [Marked; N],
    pub next: Box<dyn Op>,
}

impl<const N: usize> Op for DropRunN<N> {
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
/// second reason in RFC-0050: a heap construction does not hold
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use acvus_ast::Span;
    use acvus_utils::Interner;
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::code::{Body, FALL, Literals};
    use crate::executor::SequentialExecutor;
    use crate::interpreter::InterpreterContext;
    use crate::ops::control::Fall;
    use crate::regs::{FrameSlot, MarkWords, Store, cells_for};
    use crate::value::Kind;

    const FRAME: u16 = 8;

    #[derive(Clone, Copy)]
    enum Before {
        Undef,
        Word(i64),
        ClaimedText(&'static str),
    }

    /// Two runs allocate their strings apart, so a `Large` is compared by the
    /// register it was placed in rather than by its address.
    #[derive(Debug, PartialEq, Eq)]
    enum Word {
        Inline(Kind, u64),
        Large { placed_in: usize },
    }

    #[derive(Debug, PartialEq, Eq)]
    struct After {
        registers: Vec<Word>,
        mark: u64,
    }

    fn body() -> Body {
        Body {
            heads: Box::new([]),
            entry: 0,
            frame_len: FRAME,
            frame_cells: u16::try_from(cells_for(FRAME)).expect("a small frame"),
            mark_words: MarkWords::of(FRAME),
            entry_konsts: Box::new([]),
            literals: Arc::new(Literals::of(std::iter::empty())),
            slot_kinds: Box::new([]),
            may_suspend: false,
            returns_a_view: false,
            params: Box::new([]),
            param_run: 0,
            param_marks: 0,
            captures: Box::new([]),
            order_param: None,
            span: Span::ZERO,
        }
    }

    fn reg(index: u16) -> Marked {
        Marked::of(FrameSlot::of(index))
    }

    fn konst(at: u16, value: Value) -> LaidKonst {
        LaidKonst { at: reg(at), value }
    }

    fn laid(at: u16, src: u16, large: bool) -> LaidMove {
        LaidMove {
            at: reg(at),
            src: reg(src),
            large,
        }
    }

    fn after(op: Box<dyn Op>, before: &[Before; FRAME as usize]) -> After {
        let interner = Interner::new();
        let ctx = InterpreterContext::new(
            &interner,
            FxHashMap::default(),
            Arc::new(SequentialExecutor),
        );
        let rt = ctx.runtime_over_an_empty_page();
        let body = body();
        let mut store = Store::new();
        let (mut regs, _) = store.bind(&body);
        regs.open_marks(body.mark_words);
        for slot in FrameSlot::first(usize::from(FRAME)) {
            regs.open(Off::bounded(slot), Value::UNDEF);
        }
        let mut placed: Vec<(u64, usize)> = Vec::new();
        for (index, before) in (0..FRAME).zip(before) {
            match *before {
                Before::Undef => {}
                Before::Word(n) => regs.assign::<false>(reg(index), Value::int(n)),
                Before::ClaimedText(text) => {
                    let value = Value::string(text);
                    placed.push((value.word_of_any_kind(), usize::from(index)));
                    regs.assign::<true>(reg(index), value);
                }
            }
        }
        let mut m = Machine::new(&body, regs, &rt);
        assert_eq!(
            op.run(&mut m, 0),
            FALL,
            "the operation runs on to its successor"
        );
        let regs = m.regs();
        let registers = FrameSlot::first(usize::from(FRAME))
            .map(|slot| {
                let value = regs.read(Off::bounded(slot));
                match value.kind() {
                    Kind::Large => Word::Large {
                        placed_in: placed
                            .iter()
                            .find(|(word, _)| *word == value.word_of_any_kind())
                            .map(|(_, at)| *at)
                            .expect("every Large in the frame was placed there before the run"),
                    },
                    kind => Word::Inline(kind, value.word_of_any_kind()),
                }
            })
            .collect();
        let mark = regs.mark_word(0);
        regs.sweep(body.mark_words);
        After { registers, mark }
    }

    fn fall() -> Box<dyn Op> {
        Box::new(Fall)
    }

    fn general_lay(konsts: &[LaidKonst], moved: &[LaidMove]) -> Box<dyn Op> {
        Box::new(LayRun {
            konsts: konsts.iter().map(LaidKonst::copied).collect(),
            moved: moved
                .iter()
                .map(|laid| LaidMove {
                    at: laid.at,
                    src: laid.src,
                    large: laid.large,
                })
                .collect(),
            next: fall(),
        })
    }

    fn general_drop(registers: &[Marked]) -> Box<dyn Op> {
        Box::new(DropRun {
            registers: registers.into(),
            next: fall(),
        })
    }

    use Before::{ClaimedText, Undef, Word as W};

    fn assert_lays_as_general<F>(
        konsts: &[LaidKonst],
        moved: &[LaidMove],
        fixed: F,
        frames: &[[Before; FRAME as usize]],
    ) where
        F: Fn() -> Box<dyn Op>,
    {
        for before in frames {
            assert_eq!(
                after(fixed(), before),
                after(general_lay(konsts, moved), before)
            );
        }
    }

    #[test]
    fn lay_one_konst_one_word_writes_as_lay_run() {
        let frames = [
            [
                W(10),
                W(11),
                W(12),
                ClaimedText("a"),
                Undef,
                Undef,
                Undef,
                Undef,
            ],
            [
                W(10),
                W(11),
                ClaimedText("b"),
                ClaimedText("a"),
                Undef,
                Undef,
                Undef,
                Undef,
            ],
        ];
        for (tag_at, payload_at, src) in [(2, 3, 1), (1, 3, 1), (3, 3, 1)] {
            let konsts = [konst(tag_at, Value::int(7))];
            let moved = [laid(payload_at, src, false)];
            assert_lays_as_general(
                &konsts,
                &moved,
                || {
                    Box::new(LayRunOne::<1, false> {
                        moved: FixedMove::of(&moved[0]),
                        konsts: [konsts[0].copied()],
                        next: fall(),
                    })
                },
                &frames,
            );
        }
    }

    #[test]
    fn lay_one_konst_one_large_writes_as_lay_run() {
        let frames = [
            [
                W(10),
                ClaimedText("p"),
                W(12),
                ClaimedText("a"),
                Undef,
                Undef,
                Undef,
                Undef,
            ],
            [
                W(10),
                ClaimedText("p"),
                ClaimedText("b"),
                W(13),
                Undef,
                Undef,
                Undef,
                Undef,
            ],
        ];
        for (tag_at, payload_at, src) in [(2, 3, 1), (1, 3, 1), (3, 3, 1), (2, 1, 1)] {
            let konsts = [konst(tag_at, Value::int(7))];
            let moved = [laid(payload_at, src, true)];
            assert_lays_as_general(
                &konsts,
                &moved,
                || {
                    Box::new(LayRunOne::<1, true> {
                        moved: FixedMove::of(&moved[0]),
                        konsts: [konsts[0].copied()],
                        next: fall(),
                    })
                },
                &frames,
            );
        }
    }

    #[test]
    fn lay_one_word_writes_as_lay_run() {
        let frames = [
            [
                W(10),
                W(11),
                ClaimedText("a"),
                Undef,
                Undef,
                Undef,
                Undef,
                Undef,
            ],
            [W(10), W(11), W(12), Undef, Undef, Undef, Undef, Undef],
        ];
        for (at, src) in [(2, 1), (1, 1)] {
            let moved = [laid(at, src, false)];
            assert_lays_as_general(
                &[],
                &moved,
                || {
                    Box::new(LayRunOne::<0, false> {
                        moved: FixedMove::of(&moved[0]),
                        konsts: [],
                        next: fall(),
                    })
                },
                &frames,
            );
        }
    }

    #[test]
    fn lay_a_word_then_a_large_writes_as_lay_run() {
        let frames = [
            [
                W(10),
                W(11),
                ClaimedText("q"),
                ClaimedText("a"),
                ClaimedText("b"),
                Undef,
                Undef,
                Undef,
            ],
            [
                W(10),
                W(11),
                ClaimedText("q"),
                W(13),
                W(14),
                Undef,
                Undef,
                Undef,
            ],
        ];
        for (first, second) in [((3, 1), (4, 2)), ((2, 1), (4, 2)), ((3, 1), (3, 2))] {
            let moved = [
                laid(first.0, first.1, false),
                laid(second.0, second.1, true),
            ];
            assert_lays_as_general(
                &[],
                &moved,
                || {
                    Box::new(LayRunTwo::<0, false, true> {
                        moved: [FixedMove::of(&moved[0]), FixedMove::of(&moved[1])],
                        konsts: [],
                        next: fall(),
                    })
                },
                &frames,
            );
        }
    }

    struct KonstsFirst {
        konst: LaidKonst,
        moved: FixedMove,
    }

    impl Op for KonstsFirst {
        fn run(&self, m: &mut Machine<'_>, _: u64) -> Exit {
            let regs = m.regs();
            regs.assign::<false>(self.konst.at, self.konst.value);
            self.moved.lay::<true>(regs);
            FALL
        }
    }

    #[test]
    fn the_comparison_tells_a_reordered_lay_apart() {
        let before = [
            W(10),
            ClaimedText("p"),
            W(12),
            ClaimedText("a"),
            Undef,
            Undef,
            Undef,
            Undef,
        ];
        let konsts = [konst(1, Value::int(7))];
        let moved = [laid(3, 1, true)];
        let reordered = Box::new(KonstsFirst {
            konst: konsts[0].copied(),
            moved: FixedMove::of(&moved[0]),
        });
        assert_ne!(
            after(reordered, &before),
            after(general_lay(&konsts, &moved), &before)
        );
    }

    #[test]
    fn drop_none_writes_as_drop_run() {
        let before = [
            ClaimedText("a"),
            W(11),
            Undef,
            Undef,
            Undef,
            Undef,
            Undef,
            Undef,
        ];
        let fixed = Box::new(DropRunN::<0> {
            registers: [],
            next: fall(),
        });
        assert_eq!(after(fixed, &before), after(general_drop(&[]), &before));
    }

    #[test]
    fn drop_one_writes_as_drop_run() {
        let frames = [
            [
                ClaimedText("a"),
                W(11),
                Undef,
                Undef,
                Undef,
                Undef,
                Undef,
                Undef,
            ],
            [W(10), W(11), Undef, Undef, Undef, Undef, Undef, Undef],
        ];
        for before in &frames {
            for at in [0, 1] {
                let fixed = Box::new(DropRunN::<1> {
                    registers: [reg(at)],
                    next: fall(),
                });
                assert_eq!(
                    after(fixed, before),
                    after(general_drop(&[reg(at)]), before)
                );
            }
        }
    }

    #[cfg(any(debug_assertions, feature = "probe"))]
    #[test]
    fn a_shape_with_no_fixed_form_runs_the_general_op() {
        let name = |op: Box<dyn Op>| crate::listing::last_path_segment(op.as_ref());
        let lay = |konsts: &[LaidKonst], moved: &[LaidMove]| {
            name(lay_run(
                konsts.iter().map(LaidKonst::copied).collect(),
                moved
                    .iter()
                    .map(|laid| LaidMove {
                        at: laid.at,
                        src: laid.src,
                        large: laid.large,
                    })
                    .collect(),
                fall(),
            ))
        };
        let tag = konst(0, Value::int(7));
        assert_eq!(
            lay(&[tag.copied()], &[laid(1, 2, false)]),
            "LayRunOne<1, false>"
        );
        assert_eq!(
            lay(&[tag.copied()], &[laid(1, 2, true)]),
            "LayRunOne<1, true>"
        );
        assert_eq!(lay(&[], &[laid(1, 2, false)]), "LayRunOne<0, false>");
        assert_eq!(
            lay(&[], &[laid(1, 2, false), laid(3, 4, true)]),
            "LayRunTwo<0, false, true>"
        );
        assert_eq!(lay(&[], &[laid(1, 2, true)]), "LayRun");
        assert_eq!(lay(&[tag.copied()], &[]), "LayRun");
        assert_eq!(lay(&[], &[laid(1, 2, true), laid(3, 4, false)]), "LayRun");
        assert_eq!(name(drop_run(Box::new([]), fall())), "DropRunN<0>");
        assert_eq!(name(drop_run(Box::new([reg(1)]), fall())), "DropRunN<1>");
        assert_eq!(
            name(drop_run(Box::new([reg(1), reg(2)]), fall())),
            "DropRun"
        );
    }
}
