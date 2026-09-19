//! RFC-0026: a `String` value used before its last use is copied first.

use acvus_ast::Span;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::{inst_info, liveness};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Inst, InstKind, ValueId};
use crate::optimize::drop_insertion::ends_ownership;
use crate::optimize::ssa_pass::map_uses;
use crate::ty::Ty;

pub fn run(cfg: &mut CfgBody) {
    if cfg.blocks.is_empty() {
        return;
    }
    let live = liveness::analyze(cfg);
    for bi in 0..cfg.blocks.len() {
        let live_after = live_after_each(cfg, bi, &live);
        let old = std::mem::take(&mut cfg.blocks[bi].insts);
        let mut new = Vec::with_capacity(old.len());
        for (ii, mut inst) in old.into_iter().enumerate() {
            let consumed: Vec<ValueId> = inst_info::uses(&inst.kind)
                .iter()
                .copied()
                .filter(|v| ends_ownership(&inst.kind, *v, &cfg.val_types))
                .collect();
            let copies = copies_for(cfg, &live_after[ii], consumed.into_iter());
            for (src, dst) in &copies.clones {
                new.push(clone_inst(inst.span, *src, *dst));
            }
            let mut remaining = copies.clone();
            map_uses(&mut inst.kind, &mut |v| remaining.redirect(v));
            new.push(inst);
        }
        let mut terminator =
            std::mem::replace(&mut cfg.blocks[bi].terminator, Terminator::Fallthrough);
        let span = new.last().map(|i| i.span).unwrap_or(Span::ZERO);
        let live_by_own_name: FxHashSet<ValueId> = cfg
            .successors(BlockIdx(bi))
            .into_iter()
            .flat_map(|succ| {
                let params: FxHashSet<ValueId> =
                    cfg.blocks[succ.0].params.iter().copied().collect();
                live.live_in[succ.0]
                    .iter()
                    .copied()
                    .filter(move |v| !params.contains(v))
                    .collect::<Vec<_>>()
            })
            .collect();
        let copies = copies_for(
            cfg,
            &live_by_own_name,
            terminator_args(&terminator).into_iter(),
        );
        for (src, dst) in &copies.clones {
            new.push(clone_inst(span, *src, *dst));
        }
        let mut remaining = copies;
        for arg in terminator_args_mut(&mut terminator) {
            remaining.redirect(arg);
        }
        cfg.blocks[bi].terminator = terminator;
        cfg.blocks[bi].insts = new;
    }
}

fn live_after_each(
    cfg: &CfgBody,
    bi: usize,
    live: &liveness::LivenessResult,
) -> Vec<FxHashSet<ValueId>> {
    let block = &cfg.blocks[bi];
    let mut current: FxHashSet<ValueId> = live.live_out[bi].clone();
    current.extend(terminator_args(&block.terminator));
    let mut after = vec![FxHashSet::default(); block.insts.len()];
    for (ii, inst) in block.insts.iter().enumerate().rev() {
        after[ii] = current.clone();
        for d in inst_info::defs(&inst.kind) {
            current.remove(&d);
        }
        current.extend(inst_info::uses(&inst.kind));
    }
    after
}

#[derive(Clone)]
struct Copies {
    clones: Vec<(ValueId, ValueId)>,
    pending: FxHashMap<ValueId, Vec<ValueId>>,
}

impl Copies {
    fn redirect(&mut self, v: &mut ValueId) {
        if let Some(queue) = self.pending.get_mut(v)
            && let Some(copy) = queue.pop()
        {
            *v = copy;
        }
    }
}

fn copies_for(
    cfg: &mut CfgBody,
    live_after: &FxHashSet<ValueId>,
    uses: impl Iterator<Item = ValueId>,
) -> Copies {
    let mut occurrences: FxHashMap<ValueId, usize> = FxHashMap::default();
    let mut order: Vec<ValueId> = Vec::new();
    for v in uses {
        if !matches!(cfg.val_types.get(&v), Some(Ty::String)) {
            continue;
        }
        let n = occurrences.entry(v).or_insert(0);
        if *n == 0 {
            order.push(v);
        }
        *n += 1;
    }
    let mut copies = Copies {
        clones: Vec::new(),
        pending: FxHashMap::default(),
    };
    for v in order {
        let k = occurrences[&v];
        let needed = if live_after.contains(&v) { k } else { k - 1 };
        let mut queue: Vec<ValueId> = (0..needed)
            .map(|_| {
                let c = cfg.val_factory.next();
                cfg.val_types.insert(c, Ty::String);
                copies.clones.push((v, c));
                c
            })
            .collect();
        queue.reverse();
        copies.pending.insert(v, queue);
    }
    copies
}

fn clone_inst(span: Span, src: ValueId, dst: ValueId) -> Inst {
    Inst {
        span,
        kind: InstKind::StringClone { dst, src },
    }
}

fn terminator_args(t: &Terminator) -> Vec<ValueId> {
    match t {
        Terminator::Jump { args, .. } => args.clone(),
        Terminator::JumpIf {
            then_args,
            else_args,
            ..
        } => then_args.iter().chain(else_args).copied().collect(),
        Terminator::For {
            body_args,
            exit_args,
            ..
        } => body_args.iter().chain(exit_args).copied().collect(),
        Terminator::Switch { arms, default, .. } => arms
            .iter()
            .flat_map(|(_, _, args)| args.iter())
            .chain(default.iter().flat_map(|(_, args)| args.iter()))
            .copied()
            .collect(),
        Terminator::Return { .. } | Terminator::Fallthrough | Terminator::Diverge => Vec::new(),
    }
}

fn terminator_args_mut(t: &mut Terminator) -> Vec<&mut ValueId> {
    match t {
        Terminator::Jump { args, .. } => args.iter_mut().collect(),
        Terminator::JumpIf {
            then_args,
            else_args,
            ..
        } => then_args.iter_mut().chain(else_args.iter_mut()).collect(),
        Terminator::For {
            body_args,
            exit_args,
            ..
        } => body_args.iter_mut().chain(exit_args.iter_mut()).collect(),
        Terminator::Switch { arms, default, .. } => arms
            .iter_mut()
            .flat_map(|(_, _, args)| args.iter_mut())
            .chain(default.iter_mut().flat_map(|(_, args)| args.iter_mut()))
            .collect(),
        Terminator::Return { .. } | Terminator::Fallthrough | Terminator::Diverge => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ir::{DebugInfo, Label, MirBody};
    use acvus_ast::Literal;
    use acvus_utils::{LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn body(insts: Vec<InstKind>, types: Vec<(ValueId, Ty)>) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..16 {
            factory.next();
        }
        promote(MirBody {
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: Span::ZERO,
                    kind,
                })
                .collect(),
            val_types: types.into_iter().collect(),
            params: vec![],
            captures: vec![],
            order_param: None,
            task: crate::ty::Task::Sync,
            debug: DebugInfo::new(),
            label_count: 0,
            val_factory: factory,
        })
    }

    fn clones(cfg: &CfgBody) -> usize {
        cfg.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.kind, InstKind::StringClone { .. }))
            .count()
    }

    #[test]
    fn a_string_used_twice_is_copied_once() {
        let mut cfg = body(
            vec![
                InstKind::Const {
                    dst: v(1),
                    value: Literal::String("a".into()),
                },
                InstKind::StringConcat {
                    dst: v(2),
                    parts: vec![v(1)],
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            vec![(v(1), Ty::String), (v(2), Ty::String)],
        );
        run(&mut cfg);
        assert_eq!(clones(&cfg), 1);
        let InstKind::StringConcat { parts, .. } = &cfg.blocks[0].insts[2].kind else {
            panic!("the concat stays third, after the copy");
        };
        assert_ne!(parts[0], v(1), "the earlier use takes the copy");
    }

    #[test]
    fn a_string_used_twice_in_one_instruction_is_copied_for_each_but_the_last() {
        let mut cfg = body(
            vec![
                InstKind::Const {
                    dst: v(1),
                    value: Literal::String("a".into()),
                },
                InstKind::StringConcat {
                    dst: v(2),
                    parts: vec![v(1), v(1)],
                },
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            vec![(v(1), Ty::String), (v(2), Ty::String)],
        );
        run(&mut cfg);
        assert_eq!(clones(&cfg), 1);
        let InstKind::StringConcat { parts, .. } = &cfg.blocks[0].insts[2].kind else {
            panic!("the concat stays third");
        };
        assert_ne!(parts[0], parts[1]);
        assert_eq!(parts[1], v(1));
    }

    #[test]
    fn a_jump_argument_alive_only_as_the_successor_parameter_is_not_copied() {
        let mut cfg = body(
            vec![
                InstKind::Const {
                    dst: v(1),
                    value: Literal::String("a".into()),
                },
                InstKind::Jump {
                    label: Label(0),
                    args: vec![v(1)],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![v(2)],
                },
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            vec![(v(1), Ty::String), (v(2), Ty::String)],
        );
        run(&mut cfg);
        assert_eq!(clones(&cfg), 0);
    }

    #[test]
    fn a_string_used_once_is_not_copied() {
        let mut cfg = body(
            vec![
                InstKind::Const {
                    dst: v(1),
                    value: Literal::String("a".into()),
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            vec![(v(1), Ty::String)],
        );
        run(&mut cfg);
        assert_eq!(clones(&cfg), 0);
    }
}
