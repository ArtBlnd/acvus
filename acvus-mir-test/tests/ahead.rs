//! RFC-0103 at `analysis::ahead`: which loops run their first stage's
//! heavy or io spawns ahead, what the plan names, and the first condition
//! of rule 1 each other loop fails, as `acvus mir` prints it.

use acvus_extern::{Registry, TypesOnly, extern_fn, extern_registry};
use acvus_mir::analysis::ahead::{Lowerer, Lowering, Plan, Refused, RefusedSource, RestStart};
use acvus_mir::analysis::inst_info::defs;
use acvus_mir::analysis::loop_deps::{BodyDeps, Member};
use acvus_mir::cfg::{BlockIdx, CfgBody, Terminator, promote};
use acvus_mir::ir::{Callee, InstKind};
use acvus_mir::laws::LawTable;
use acvus_mir::printer::dump_with_facts;
use acvus_mir::ty::Task;
use acvus_mir_test::{LoweredScript, optimized_script};
use acvus_utils::Interner;

/// Type-only fixtures: a heavy parser and a heavy arithmetic kernel, an io
/// call declared as an `async fn` with and without an effect, and the same
/// effectful io declared as a plain `fn`.
mod fx {
    use super::*;

    #[extern_fn(heavy, effect = pure, returns)]
    pub fn parse_heavy(s: String) -> Option<i64> {
        let _ = s;
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(heavy, effect = pure, returns)]
    pub fn crunch(x: i64) -> i64 {
        let _ = x;
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = idempotent)]
    pub async fn fetch_io(url: String) -> String {
        let _ = url;
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = pure, returns)]
    pub async fn fetch_pure(url: String) -> String {
        let _ = url;
        unreachable!("a type-only fixture is never run")
    }

    #[extern_fn(effect = idempotent)]
    pub fn fetch_plain(url: String) -> String {
        let _ = url;
        unreachable!("a type-only fixture is never run")
    }

    pub fn registry() -> Registry<TypesOnly> {
        extern_registry! {
            ns: "fx",
            fns: [parse_heavy, crunch, fetch_io, fetch_pure, fetch_plain],
        }
    }
}

struct Decided {
    interner: Interner,
    listing: String,
    cfg: CfgBody,
    laws: LawTable,
}

impl Decided {
    fn of(source: &str) -> Self {
        let interner = Interner::new();
        let LoweredScript { module, laws } =
            optimized_script(&interner, source, &[], vec![fx::registry()])
                .unwrap_or_else(|e| panic!("{source}\n{e}"));
        Self {
            listing: dump_with_facts(&interner, &module, &laws),
            cfg: promote(module.main),
            laws,
            interner,
        }
    }

    /// Each loop's lowering, in the order of its header block.
    fn lowerings(&self) -> Vec<Lowering> {
        let lowerer = Lowerer::of(&self.cfg, &self.laws);
        BodyDeps::of(&self.cfg, &self.laws)
            .loops
            .iter()
            .map(|found| lowerer.lowering(found))
            .collect()
    }

    fn sole(&self) -> Lowering {
        match &self.lowerings()[..] {
            [only] => only.clone(),
            other => panic!("one loop, found {}:\n{}", other.len(), self.listing),
        }
    }

    fn plan(&self) -> Plan {
        match self.sole() {
            Lowering::Ahead(plan) => plan,
            Lowering::InPlace(refused) => {
                panic!("in place ({refused:?}), expected ahead:\n{}", self.listing)
            }
        }
    }

    fn refused(&self) -> Refused {
        match self.sole() {
            Lowering::InPlace(refused) => refused,
            Lowering::Ahead(plan) => panic!("ahead ({plan:?}), expected in place:\n{}", self.listing),
        }
    }

    fn refuses(&self, expected: Refused) {
        assert_eq!(self.refused(), expected, "{}", self.listing);
    }

    fn kind(&self, member: Member) -> Option<&InstKind> {
        match member {
            Member::Inst(at) => Some(&self.cfg.blocks[at.block.0].insts[at.at].kind),
            Member::Term(_) => None,
        }
    }

    fn callee_name(&self, callee: &Callee) -> &str {
        match callee {
            Callee::Direct(id) | Callee::Extern { id, .. } => self.interner.resolve(id.name),
            Callee::Indirect(_) => panic!("a call through a value names no callee"),
        }
    }

    fn prints(&self, line: &str) {
        let wanted = format!("// {line}");
        assert!(
            self.listing.lines().any(|printed| printed.trim_start_matches([' ', '|']).trim() == wanted),
            "no `{wanted}`:\n{}",
            self.listing
        );
    }
}

const TEXTS: &str = r#"let xs = vec(["12".to_string(), "7".to_string(), "40".to_string()]);"#;

fn with_texts(rest: &str) -> String {
    format!("{TEXTS}\n{rest}")
}

fn spawn_tasks(plan: &Plan) -> Vec<Task> {
    plan.spawns.iter().map(|spawn| spawn.task).collect()
}

// -- Accepted --------------------------------------------------------------

#[test]
fn a_heavy_call_per_element_runs_ahead() {
    let decided = Decided::of(&with_texts(
        "let out = vec([]);
         for s in &xs { out.push(fx::parse_heavy(s.to_string()).unwrap()); }
         out.len()",
    ));
    let plan = decided.plan();
    assert_eq!(spawn_tasks(&plan), [Task::Heavy], "{}", decided.listing);
    assert_eq!(plan.task, Task::Heavy);
    let spawn = plan.spawns[0];
    assert!(plan.written.contains(&spawn.handle));
    assert!(!plan.written.contains(&plan.laid.element));
    assert!(
        plan.prefix
            .iter()
            .all(|member| !matches!(decided.kind(*member), Some(InstKind::Eval { .. }))),
        "the prefix ends before the evaluation"
    );
    assert_eq!(
        plan.rest,
        RestStart::Inst(acvus_mir::analysis::loop_deps::InstAt {
            block: spawn.at.block,
            at: spawn.at.at + 1,
        }),
        "the rest begins after the last spawn"
    );
    decided.prints("lower ahead {spawn parse_heavy}");
}

/// Under `anyorder` an effectful `async fn` is free, and its spawn runs
/// ahead: the overlap is what `anyorder` gives up (RFC-0103, Cost).
#[test]
fn an_io_call_per_element_under_anyorder_runs_ahead() {
    let decided = Decided::of(&with_texts(
        "let out = vec([]);
         anyorder { for u in &xs { out.push(fx::fetch_io(u.to_string())); } }
         out.len()",
    ));
    let plan = decided.plan();
    assert_eq!(spawn_tasks(&plan), [Task::Async], "{}", decided.listing);
    assert_eq!(plan.task, Task::Async);
    decided.prints("lower ahead {spawn fetch_io}");
}

#[test]
fn a_pure_io_call_per_element_runs_ahead_without_anyorder() {
    let decided = Decided::of(&with_texts(
        "let out = vec([]);
         for u in &xs { out.push(fx::fetch_pure(u.to_string())); }
         out.len()",
    ));
    assert_eq!(spawn_tasks(&decided.plan()), [Task::Async], "{}", decided.listing);
    decided.prints("lower ahead {spawn fetch_pure}");
}

/// The request is formatted in the prefix: a `let` slot, its borrow and
/// the concatenations run ahead with the spawn, and the plan writes the
/// slot.
#[test]
fn a_request_formatted_before_the_call_runs_ahead_with_it() {
    let decided = Decided::of(&with_texts(
        r#"let out = vec([]);
           for id in &xs {
               let url = "https://api.example/items/" + id.to_string() + "?fields=name";
               out.push(fx::fetch_pure(url.to_string()));
           }
           out.len()"#,
    ));
    let plan = decided.plan();
    assert_eq!(spawn_tasks(&plan), [Task::Async], "{}", decided.listing);
    let concats = plan
        .prefix
        .iter()
        .filter(|member| matches!(decided.kind(**member), Some(InstKind::StringConcat { .. })))
        .count();
    assert_eq!(concats, 2, "{}", decided.listing);
    let slots: Vec<_> = plan
        .prefix
        .iter()
        .filter_map(|member| match decided.kind(*member) {
            Some(assign @ InstKind::Assign { .. }) => Some(defs(assign)),
            _ => None,
        })
        .flatten()
        .collect();
    assert!(!slots.is_empty(), "the `let` slot is assigned in the prefix");
    assert!(slots.iter().all(|slot| plan.written.contains(slot)));
}

/// Unwrapping, scaling and summing the result run in place after the
/// evaluation; only the spawn and what it reads run ahead.
#[test]
fn processing_after_the_call_runs_in_place() {
    let decided = Decided::of(&with_texts(
        "let total = 0;
         for s in &xs { let v = fx::parse_heavy(s.to_string()).unwrap(); total = total + v * 2; }
         total",
    ));
    let plan = decided.plan();
    assert_eq!(spawn_tasks(&plan), [Task::Heavy], "{}", decided.listing);
    let called_ahead: Vec<&str> = plan
        .prefix
        .iter()
        .filter_map(|member| match decided.kind(*member) {
            Some(InstKind::FunctionCall { callee, .. } | InstKind::Spawn { callee, .. }) => {
                Some(decided.callee_name(callee))
            }
            _ => None,
        })
        .collect();
    assert_eq!(called_ahead, ["to_string", "parse_heavy"], "{}", decided.listing);
    let computed_ahead = plan.prefix.iter().any(|member| {
        matches!(
            decided.kind(*member),
            Some(InstKind::BinOp { .. } | InstKind::Eval { .. })
        )
    });
    assert!(!computed_ahead, "{}", decided.listing);
}

#[test]
fn a_heavy_call_over_a_range_runs_ahead() {
    let decided = Decided::of(
        "let total = 0;
         for i in 0..1000 { total = total + fx::crunch(i * 3 + 1); }
         total",
    );
    let plan = decided.plan();
    assert_eq!(spawn_tasks(&plan), [Task::Heavy], "{}", decided.listing);
    assert_eq!(plan.laid.element, plan.laid.counter, "a range's element is its counter");
    decided.prints("lower ahead {spawn crunch}");
}

/// An exit after the call leaves the prefix, which has no effect, ahead:
/// what runs past the exit is discarded (RFC-0103 rule 4).
#[test]
fn a_pure_prefix_runs_ahead_of_a_later_exit() {
    let decided = Decided::of(&with_texts(
        "let total = 0;
         for s in &xs { let v = fx::parse_heavy(s.to_string()).unwrap(); if v > 20 { break; } total = total + v; }
         total",
    ));
    assert_eq!(spawn_tasks(&decided.plan()), [Task::Heavy], "{}", decided.listing);
}

// -- Refused, each by its condition ---------------------------------------

/// C1: the inner loop lies in the outer one; the outer loop's first stage
/// spawns nothing.
#[test]
fn a_nested_loop_runs_in_place() {
    let decided = Decided::of(
        "let total = 0;
         for i in 0..3 { for j in 0..4 { total = total + fx::crunch(i * j); } }
         total",
    );
    let lowerings = decided.lowerings();
    let enclosed: Vec<BlockIdx> = lowerings
        .iter()
        .filter_map(|lowering| match lowering {
            Lowering::InPlace(Refused::Enclosed { by }) => Some(*by),
            _ => None,
        })
        .collect();
    let [outer] = enclosed[..] else {
        panic!("one loop is enclosed:\n{:?}\n{}", lowerings, decided.listing)
    };
    assert!(matches!(
        decided.cfg.blocks[outer.0].terminator,
        Terminator::For { .. }
    ));
    assert!(
        lowerings.contains(&Lowering::InPlace(Refused::NoAheadSpawn)),
        "{lowerings:?}\n{}",
        decided.listing
    );
    decided.prints(&format!(
        "lower in place: inside the loop at L{}",
        decided.cfg.blocks[outer.0].label.0
    ));
}

/// C2: a `&mut` source is written at its counter's slot.
#[test]
fn a_mut_slice_source_runs_in_place() {
    let decided = Decided::of(&with_texts(
        "let total = 0;
         for s in &mut xs { total = total + fx::parse_heavy(s.to_string()).unwrap(); }
         total",
    ));
    decided.refuses(Refused::Source(RefusedSource::SliceMut));
    decided.prints("lower in place: its source is a `&mut` slice");
}

/// C2: an array source releases its elements in place (RFC-0089 rule 5).
#[test]
fn an_array_source_runs_in_place() {
    let decided = Decided::of(
        r#"let total = 0;
           for s in ["12".to_string(), "7".to_string()] { total = total + fx::parse_heavy(s).unwrap(); }
           total"#,
    );
    decided.refuses(Refused::Source(RefusedSource::Array));
    decided.prints("lower in place: its source is an array");
}

/// C3: the heavy call reads the storage a later iteration writes, so it
/// lies in that storage's cycle.
#[test]
fn a_call_reading_what_the_loop_writes_runs_in_place() {
    let decided = Decided::of(
        r#"let log = vec(["0".to_string()]);
           for i in 0..3 {
               let v = fx::parse_heavy(log.len().to_string()).unwrap();
               if v > i { log.push(v.to_string()); }
           }
           log.len()"#,
    );
    decided.refuses(Refused::FirstStageNotFree);
    decided.prints("lower in place: its first stage is not free");
}

/// C4: a plain `fn` with an effect has the callee type of an `async fn`
/// (RFC-0046 rule 3), and its declaration runs it at `Sync`.
#[test]
fn a_plain_effectful_fn_runs_in_place() {
    let source = |call: &str| {
        with_texts(&format!(
            "let out = vec([]);
             anyorder {{ for u in &xs {{ out.push(fx::{call}(u.to_string())); }} }}
             out.len()"
        ))
    };
    let plain = Decided::of(&source("fetch_plain"));
    plain.refuses(Refused::NoAheadSpawn);
    plain.prints("lower in place: its first stage spawns no heavy or io call before it waits");

    let spawned = |decided: &Decided| {
        decided
            .cfg
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .find_map(|inst| match &inst.kind {
                InstKind::Spawn {
                    callee: callee @ Callee::Extern { .. },
                    callee_ty,
                    ..
                } => Some((
                    decided.laws.task_of(callee),
                    callee_ty.effect().map(|effect| effect.task),
                )),
                _ => None,
            })
            .unwrap_or_else(|| panic!("a spawn:\n{}", decided.listing))
    };
    let io = Decided::of(&source("fetch_io"));
    assert_eq!(spawned(&plain), (Some(Task::Sync), Some(Task::Async)));
    assert_eq!(spawned(&io), (Some(Task::Async), Some(Task::Async)));
    assert!(matches!(io.sole(), Lowering::Ahead(_)));
}

/// C5: `ys[i * 2]` is a checked index the intervals do not clear, and it
/// would trap at the header of an earlier iteration.
#[test]
fn a_prefix_that_can_trap_runs_in_place() {
    let decided = Decided::of(
        r#"let ys = vec(["1".to_string(), "2".to_string(), "3".to_string(), "4".to_string()]);
           let t = 0;
           for i in 0..3 { let v = fx::parse_heavy(ys[i * 2].to_string()).unwrap(); t = t + v; }
           t"#,
    );
    let Refused::MayTrap(at) = decided.refused() else {
        panic!("{:?}:\n{}", decided.refused(), decided.listing)
    };
    assert!(matches!(
        decided.cfg.blocks[at.block.0].insts[at.at].kind,
        InstKind::Index { .. }
    ));
    decided.prints("lower in place: the prefix may trap at index");
}

/// C6: the prefix branches on a header parameter.
#[test]
fn a_prefix_reading_a_carried_flag_runs_in_place() {
    let decided = Decided::of(&with_texts(
        r#"let first = true;
           let t = 0;
           for s in &xs {
               let text = if first { "100".to_string() } else { s.to_string() };
               t = t + fx::parse_heavy(text).unwrap();
               first = false;
           }
           t"#,
    ));
    let Refused::ReadsLoopValue { value, .. } = decided.refused() else {
        panic!("{:?}:\n{}", decided.refused(), decided.listing)
    };
    let header = (0..decided.cfg.blocks.len())
        .map(BlockIdx)
        .find(|at| matches!(decided.cfg.blocks[at.0].terminator, Terminator::For { .. }))
        .expect("a `for`");
    assert!(decided.cfg.blocks[header.0].params.contains(&value));
}

/// C7: a validated body holds no effect before its exit stage (RFC-0089
/// rule 5), so the effect is written into an accepted body's spawn.
#[test]
fn a_prefix_with_an_effect_in_a_loop_that_leaves_runs_in_place() {
    let mut decided = Decided::of(&with_texts(
        "let total = 0;
         for s in &xs { let v = fx::parse_heavy(s.to_string()).unwrap(); if v > 20 { break; } total = total + v; }
         total",
    ));
    let spawn = decided.plan().spawns[0].at;
    let ordered = decided.cfg.val_factory.next();
    let InstKind::Spawn { order, .. } = &mut decided.cfg.blocks[spawn.block.0].insts[spawn.at].kind
    else {
        panic!("the plan's spawn")
    };
    *order = Some(ordered);
    decided.refuses(Refused::EffectBeforeExit(spawn));
}
