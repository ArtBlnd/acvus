//! RFC-0089 rules 1, 3 and 5, asked of every `For` from what
//! `analysis::loop_deps` computes: the chain's shape, no dependence cycle
//! through a token crossing a boundary, no stage reading a header
//! parameter's state before the stage its cycle lies in, and no stage
//! before the exiting one holding an operation rule 5 holds back.
//!
//! The exits themselves need no check of their own. They are the control
//! token's cycle, which `loop_deps` joins with every cycle of the stage it
//! lies in, and a cycle through several tokens is `InOrder`; the exits
//! therefore leave from an `InOrder` stage unless they lie in two stages,
//! which is that cycle crossing a boundary.
//!
//! It is asked of the module the pipeline hands on, after every pass and
//! after the drops, which is the body the machine runs.

use crate::analysis::loop_deps::{BodyDeps, HeaderDeps};
use crate::cfg::promote;
use crate::ir::{InstKind, Label, MirBody, MirModule};
use crate::laws::LawTable;
use crate::validate::{ValidationError, ValidationErrorKind};

pub fn check(module: &MirModule, laws: &LawTable) -> Vec<ValidationError> {
    let mut errors = check_body("main", &module.main, laws);
    for (label, closure) in &module.closures {
        errors.extend(check_body(&format!("closure({label:?})"), closure, laws));
    }
    errors
}

fn check_body(scope: &str, body: &MirBody, laws: &LawTable) -> Vec<ValidationError> {
    let mut stated_at: Vec<StatedAt> = Vec::new();
    let mut block = crate::cfg::ENTRY_LABEL;
    for (at, inst) in body.insts.iter().enumerate() {
        match inst.kind {
            InstKind::BlockLabel { label, .. } => block = label,
            InstKind::For { .. } => stated_at.push(StatedAt { header: block, at }),
            _ => {}
        }
    }
    if stated_at.is_empty() {
        return Vec::new();
    }
    let cfg = promote(body.clone());
    let mut errors: Vec<ValidationError> = Vec::new();
    for HeaderDeps { header, deps } in BodyDeps::of(&cfg, laws).loops {
        let header = cfg.blocks[header.0].label;
        let inst_index = stated_at
            .iter()
            .find(|stated| stated.header == header)
            .expect("promoting a body keeps each `For` at the end of its block")
            .at;
        let refusals: Vec<ValidationErrorKind> = match deps {
            Err(fault) => vec![ValidationErrorKind::StageShape { header, fault }],
            Ok(deps) => deps
                .crossing()
                .map(|crossing| ValidationErrorKind::CycleCrossesBoundary {
                    header,
                    tokens: crossing.tokens.to_vec(),
                    stages: crossing.stages.to_vec(),
                })
                .chain(deps.early_reads().iter().map(|read| {
                    ValidationErrorKind::StateReadBeforeItsCycle {
                        header,
                        token: read.token,
                        stage: read.stage,
                        cycle_stage: read.cycle_stage,
                    }
                }))
                .chain(deps.ahead_of_exit().iter().map(|ahead| {
                    ValidationErrorKind::WorkAheadOfExit {
                        header,
                        stage: ahead.stage,
                        exit_stage: ahead.exit_stage,
                        held_back: ahead.held_back,
                    }
                }))
                .collect(),
        };
        errors.extend(refusals.into_iter().map(|kind| ValidationError {
            scope: scope.to_string(),
            inst_index,
            span: body.insts[inst_index].span,
            kind,
        }));
    }
    errors
}

struct StatedAt {
    header: Label,
    at: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::inst_info;
    use crate::analysis::loop_deps::{LoopDeps, Order, Token};
    use crate::cfg::{BlockIdx, Terminator};
    use crate::ir::{BinOp, DebugInfo, ExitTrip, ForSource, Inst, Overflow, Stages, ValueId};
    use crate::ty::{Task, Ty};
    use acvus_ast::Literal;
    use acvus_utils::{LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    const AT: usize = 0;
    const HI: usize = 1;
    const S_INIT: usize = 2;
    const P_INIT: usize = 3;
    const H_S: usize = 4;
    const H_P: usize = 5;
    const I: usize = 6;
    const T: usize = 7;
    const COND: usize = 8;
    const S_NEXT: usize = 9;
    const P_NEXT: usize = 10;
    const SUM: usize = 11;
    const SPARE: usize = 12;
    const COND_2: usize = 13;

    const HEADER: Label = Label(0);
    const PURE: Label = Label(1);
    const EXIT: Label = Label(2);
    const S_JOIN: Label = Label(3);
    const P_JOIN: Label = Label(4);
    const LEAVE: Label = Label(5);
    const STAY: Label = Label(6);
    const LEAVE_2: Label = Label(7);
    const STAY_2: Label = Label(8);

    fn binop(dst: usize, left: usize, right: usize, op: BinOp) -> InstKind {
        InstKind::BinOp {
            dst: v(dst),
            op,
            left: v(left),
            right: v(right),
        }
    }

    /// `for i in 0..10 { let t = i + i; s = s + t; p = p * i }` cut after
    /// `t` and after `s`'s update, `edit` applied to the body before the
    /// chain is checked.
    fn body(edit: impl FnOnce(&mut Vec<InstKind>)) -> MirModule {
        let mut insts = vec![
            InstKind::Const {
                dst: v(AT),
                value: Literal::Int(0),
            },
            InstKind::Const {
                dst: v(HI),
                value: Literal::Int(10),
            },
            InstKind::Const {
                dst: v(S_INIT),
                value: Literal::Int(0),
            },
            InstKind::Const {
                dst: v(P_INIT),
                value: Literal::Int(1),
            },
            InstKind::Jump {
                label: HEADER,
                args: vec![v(S_INIT), v(P_INIT)],
            },
            InstKind::BlockLabel {
                label: HEADER,
                params: vec![v(H_S), v(H_P)],
            },
            InstKind::For {
                source: ForSource::Range {
                    at: v(AT),
                    hi: v(HI),
                },
                stages: Stages::new(PURE, vec![S_JOIN, P_JOIN]),
                exit: EXIT,
                exit_trip: ExitTrip::Absent,
                exit_args: vec![],
            },
            InstKind::BlockLabel {
                label: PURE,
                params: vec![v(I)],
            },
            binop(T, I, I, BinOp::Add(Overflow::Trap)),
            InstKind::Jump {
                label: S_JOIN,
                args: vec![],
            },
            InstKind::BlockLabel {
                label: S_JOIN,
                params: vec![],
            },
            binop(S_NEXT, H_S, T, BinOp::Add(Overflow::Trap)),
            InstKind::Jump {
                label: P_JOIN,
                args: vec![],
            },
            InstKind::BlockLabel {
                label: P_JOIN,
                params: vec![],
            },
            binop(P_NEXT, H_P, I, BinOp::Mul(Overflow::Trap)),
            InstKind::Jump {
                label: HEADER,
                args: vec![v(S_NEXT), v(P_NEXT)],
            },
            InstKind::BlockLabel {
                label: EXIT,
                params: vec![],
            },
            binop(SUM, H_S, H_P, BinOp::Add(Overflow::Trap)),
            InstKind::Return {
                value: v(SUM),
                order: None,
            },
        ];
        edit(&mut insts);
        let mut val_types: FxHashMap<ValueId, Ty> = (0..=COND_2).map(|n| (v(n), Ty::I64)).collect();
        val_types.insert(v(COND), Ty::Bool);
        val_types.insert(v(COND_2), Ty::Bool);
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..=COND_2 {
            factory.next();
        }
        MirModule {
            declared_params: 0,
            main: MirBody {
                demoted_diamonds: Default::default(),
                insts: insts
                    .into_iter()
                    .map(|kind| Inst {
                        span: acvus_ast::Span::ZERO,
                        kind,
                    })
                    .collect(),
                val_types,
                params: Vec::new(),
                captures: Vec::new(),
                order_param: None,
                task: Task::Sync,
                debug: DebugInfo::new(),
                val_factory: factory,
                label_count: 9,
            },
            closures: FxHashMap::default(),
            ret: Ty::I64,
            flows: crate::ty::Flows::Every,
        }
    }

    fn position(insts: &[InstKind], wanted: impl Fn(&InstKind) -> bool) -> usize {
        insts
            .iter()
            .position(wanted)
            .expect("the baseline body holds the instruction")
    }

    fn defines(dst: usize) -> impl Fn(&InstKind) -> bool {
        move |kind| inst_info::defs(kind).contains(&v(dst))
    }

    fn labelled(label: Label) -> impl Fn(&InstKind) -> bool {
        move |kind| matches!(kind, InstKind::BlockLabel { label: at, .. } if *at == label)
    }

    /// A branch at the top of block `from` whose one arm leaves for the exit.
    fn leave_from(insts: &mut Vec<InstKind>, from: Label, cond: usize, leave: Label, stay: Label) {
        let at = position(insts, labelled(from));
        insts.splice(
            at + 1..at + 1,
            [
                InstKind::Const {
                    dst: v(cond),
                    value: Literal::Bool(true),
                },
                InstKind::JumpIf {
                    cond: v(cond),
                    then_label: leave,
                    then_args: vec![],
                    else_label: stay,
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: leave,
                    params: vec![],
                },
                InstKind::Jump {
                    label: EXIT,
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: stay,
                    params: vec![],
                },
            ],
        );
    }

    #[derive(Debug, PartialEq)]
    struct Ordered {
        tokens: Vec<Token>,
        order: Order,
    }

    fn refusals(module: &MirModule) -> Vec<ValidationErrorKind> {
        check(module, &LawTable::default()).into_iter().map(|error| error.kind).collect()
    }

    struct OnlyLoop {
        cfg: crate::cfg::CfgBody,
        deps: LoopDeps,
    }

    fn only_loop(module: &MirModule) -> OnlyLoop {
        let cfg = promote(module.main.clone());
        let header = (0..cfg.blocks.len())
            .map(BlockIdx)
            .find(|at| matches!(cfg.blocks[at.0].terminator, Terminator::For { .. }))
            .expect("the body holds one `For`");
        let deps = LoopDeps::of(&cfg, &LawTable::default(), header).expect("the chain's shape holds");
        OnlyLoop { cfg, deps }
    }

    #[test]
    fn a_chain_no_cycle_crosses_is_admitted() {
        let module = body(|_| {});
        assert!(refusals(&module).is_empty(), "{:?}", refusals(&module));
        let OnlyLoop { cfg, deps } = only_loop(&module);
        assert!(deps.is_free(0));
        let judged = deps.judge(&cfg, &LawTable::default());
        let orders: Vec<Ordered> = deps
            .cycles
            .iter()
            .zip(&judged)
            .map(|(cycle, judged)| Ordered {
                tokens: cycle.tokens.clone(),
                order: judged.order,
            })
            .collect();
        assert_eq!(
            orders,
            [
                Ordered {
                    tokens: vec![Token::Carried(v(H_S))],
                    order: Order::AnyOrder
                },
                Ordered {
                    tokens: vec![Token::Carried(v(H_P))],
                    order: Order::AnyOrder
                },
            ]
        );
    }

    // -- Rule 1 -----------------------------------------------------------

    #[test]
    fn a_body_block_parameter_after_the_counter_is_refused() {
        let module = body(|insts| {
            let at = position(insts, labelled(PURE));
            if let InstKind::BlockLabel { params, .. } = &mut insts[at] {
                params.push(v(SPARE));
            }
        });
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::StageShape {
                fault: crate::analysis::loop_deps::ShapeFault::BodyParams,
                ..
            }]
        ));
    }

    #[test]
    fn a_stage_that_skips_the_next_is_refused() {
        let module = body(|insts| {
            let at = position(
                insts,
                |kind| matches!(kind, InstKind::Jump { label, .. } if *label == S_JOIN),
            );
            insts.splice(
                at..=at,
                [
                    InstKind::Const {
                        dst: v(COND),
                        value: Literal::Bool(true),
                    },
                    InstKind::JumpIf {
                        cond: v(COND),
                        then_label: S_JOIN,
                        then_args: vec![],
                        else_label: P_JOIN,
                        else_args: vec![],
                    },
                ],
            );
        });
        let found = refusals(&module);
        assert!(
            matches!(
                found[..],
                [ValidationErrorKind::StageShape {
                    fault: crate::analysis::loop_deps::ShapeFault::StageDoesNotReachNext,
                    ..
                }]
            ),
            "{found:?}"
        );
    }

    // -- Rule 3 -----------------------------------------------------------

    /// `s`'s update split in two, `u = s + t` in the first stage and
    /// `s' = u + i` in the second: its cycle lies in both.
    #[test]
    fn a_cycle_split_across_two_stages_is_refused() {
        let module = body(|insts| {
            let t = position(insts, defines(T));
            insts.insert(t + 1, binop(SPARE, H_S, T, BinOp::Add(Overflow::Trap)));
            let next = position(insts, defines(S_NEXT));
            insts[next] = binop(S_NEXT, SPARE, I, BinOp::Add(Overflow::Trap));
        });
        let found = refusals(&module);
        assert!(
            matches!(
                &found[..],
                [ValidationErrorKind::CycleCrossesBoundary { tokens, stages, .. }]
                    if *tokens == [Token::Carried(v(H_S))] && *stages == [0, 1]
            ),
            "{found:?}"
        );
    }

    /// The first stage reads `p`, whose cycle lies in the third: that stage
    /// would wait on the third stage of the iteration before.
    #[test]
    fn a_stage_reading_a_carried_value_before_its_cycle_is_refused() {
        let module = body(|insts| {
            let t = position(insts, defines(T));
            insts.insert(t + 1, binop(SPARE, H_P, I, BinOp::Add(Overflow::Trap)));
        });
        let found = refusals(&module);
        assert!(
            matches!(
                found[..],
                [ValidationErrorKind::StateReadBeforeItsCycle {
                    token: Token::Carried(p),
                    stage: 0,
                    cycle_stage: 2,
                    ..
                }] if p == v(H_P)
            ),
            "{found:?}"
        );
    }

    // -- Rule 5 -----------------------------------------------------------

    /// The exits are the control token's cycle; two stages that each leave
    /// are that cycle crossing a boundary.
    #[test]
    fn exits_from_two_stages_are_refused() {
        let module = body(|insts| {
            leave_from(insts, PURE, COND, LEAVE, STAY);
            leave_from(insts, P_JOIN, COND_2, LEAVE_2, STAY_2);
        });
        let found = refusals(&module);
        assert!(
            matches!(
                &found[..],
                [ValidationErrorKind::CycleCrossesBoundary { tokens, stages, .. }]
                    if *tokens == [Token::Control] && *stages == [0, 2]
            ),
            "{found:?}"
        );
    }

    /// A stage an exit leaves from holds the control token's cycle, so it is
    /// no free stage: the exit is admitted, and its stage is `InOrder`.
    #[test]
    fn an_exit_makes_its_stage_hold_the_control_token() {
        let module = body(|insts| leave_from(insts, PURE, COND, LEAVE, STAY));
        assert!(refusals(&module).is_empty(), "{:?}", refusals(&module));
        let OnlyLoop { cfg, deps } = only_loop(&module);
        assert!(!deps.is_free(0));
        let judged = deps.judge(&cfg, &LawTable::default());
        let control = deps
            .cycles
            .iter()
            .position(|cycle| cycle.tokens == [Token::Control])
            .expect("the exit is a cycle of its own");
        assert_eq!(deps.cycles[control].stage(), Some(0));
        assert_eq!(judged[control].order, Order::InOrder);
    }

    /// Forcing a handle waits for work nothing states will end, so rule 5
    /// holds it back: in a stage before the one the loop leaves from, it is
    /// refused.
    #[test]
    fn work_not_known_to_finish_ahead_of_the_exit_is_refused() {
        let module = body(|insts| {
            let t = position(insts, defines(T));
            insts.insert(
                t + 1,
                InstKind::Eval {
                    dst: v(SPARE),
                    src: v(I),
                    order: None,
                },
            );
            leave_from(insts, S_JOIN, COND, LEAVE, STAY);
        });
        let found = refusals(&module);
        assert!(
            matches!(
                found[..],
                [ValidationErrorKind::WorkAheadOfExit {
                    stage: 0,
                    exit_stage: 1,
                    held_back: crate::analysis::loop_deps::HeldBack::MayNotFinish,
                    ..
                }]
            ),
            "{found:?}"
        );
    }

    /// The control token passes with the exiting stage's tokens (RFC-0066
    /// rule 10): `s`'s cycle, `AnyOrder` alone, is one `InOrder` join with
    /// the exit in the stage both lie in.
    #[test]
    fn an_exit_joins_the_cycles_of_its_stage_in_order() {
        let module = body(|insts| leave_from(insts, S_JOIN, COND, LEAVE, STAY));
        assert!(refusals(&module).is_empty(), "{:?}", refusals(&module));
        let OnlyLoop { cfg, deps } = only_loop(&module);
        let judged = deps.judge(&cfg, &LawTable::default());
        let in_stage: Vec<Ordered> = deps
            .cycles
            .iter()
            .zip(&judged)
            .filter(|(cycle, _)| cycle.stage() == Some(1))
            .map(|(cycle, judged)| Ordered {
                tokens: cycle.tokens.clone(),
                order: judged.order,
            })
            .collect();
        assert_eq!(
            in_stage,
            [Ordered {
                tokens: vec![Token::Carried(v(H_S)), Token::Control],
                order: Order::InOrder
            }]
        );
    }
}
