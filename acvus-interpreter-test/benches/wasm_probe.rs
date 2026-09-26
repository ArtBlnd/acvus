//! The `wasm32` module this workspace builds holds every `impl Op`'s `run`
//! ending in the tail call to its successor, `return_call_indirect`, as
//! `asm_probe` holds the tail `jmp` natively, or ending a chain, or of a
//! family listed below for the linear-stack address it holds across the
//! call; and it links a 16 MiB linear stack (RFC-0105 rules 1–3). Where node
//! is installed, a straight body of more than 40 000 operations runs in the
//! module at the engine depth and linear stack pointer of a body of one
//! step, and is dropped without running the engine's stack out (rule 4).
//!
//! `cargo bench -p acvus-interpreter-test --bench wasm_probe`. What it reads
//! is the module, which it builds with the `wasm` profile, the deployed one
//! (fat LTO, `opt-level = "z"`), in a cargo of its own through the
//! workspace's `.cargo/config.toml`; the profile this binary runs under does
//! not reach the module. A bench and not a test, so that `cargo test` does
//! not build a fat-LTO module.

use std::path::{Path, PathBuf};
use std::process::Command;

use acvus_ext::std_registries;
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter_test::Context;
use acvus_interpreter_test::listing::{ops_of, script_listing_with_externs};
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use wasmparser::{KnownCustom, Name, Operator, Parser, Payload, TypeRef};

#[path = "../benches/common/no_successor.rs"]
mod no_successor;

use no_successor::NO_SUCCESSOR;

/// One family that may call its successor, and the linear-stack address that
/// is why, as `asm_probe`'s list names its own natively.
struct Exception {
    /// The `module::Type` `op_of` produces, which is what the probe matches.
    family: &'static str,
    /// The local whose address reaches a callee: on `wasm32` such a local
    /// lives in the `run`'s linear-stack frame, and LLVM does not tail-call
    /// out of a function whose frame a callee may reach.
    stack_address: &'static str,
}

impl Exception {
    fn matches(&self, run: &Run) -> bool {
        self.family == run.of
    }
}

/// A handler's `Handler::call` takes its argument run and its result slot
/// by reference: the signature `acvus-extern` gives every handler.
const HANDLER_RUN_AND_SLOT: &str = "the argument run and the result slot `call::one` lays, \
                                    `&[a, …]` and `&mut out` into the handler's `Handler::call`";

/// The closed list of families that hold the address of a linear-stack local
/// across a callee (RFC-0105 rule 3). LLVM does not tail-call out of a
/// function whose frame a callee may still reach, so an instance calls its
/// successor where LLVM does not prove the callee leaves the address behind;
/// an instance whose callee it does prove tail-calls, and the entry does not
/// count it. None is a region: an escaping region's call is the one call its
/// `run` ends in (`ops::control::PAST`).
///
/// "Returns through a slot in the frame": the wasm32 ABI returns a value
/// wider than one word, a `Value` among them, through a slot in the caller's
/// frame whose address it passes the callee. The `multivalue` target
/// feature, on by default, leaves that ABI as it is: the module holds no
/// function type with two results either way.
const HOLDS_A_STACK_ADDRESS: &[Exception] = &[
    Exception {
        family: "call::CallExtern1",
        stack_address: HANDLER_RUN_AND_SLOT,
    },
    Exception {
        family: "call::CallExtern2",
        stack_address: HANDLER_RUN_AND_SLOT,
    },
    Exception {
        family: "call::CallExtern3",
        stack_address: HANDLER_RUN_AND_SLOT,
    },
    Exception {
        family: "call::CallExtern4",
        stack_address: HANDLER_RUN_AND_SLOT,
    },
    Exception {
        family: "call::CallPair1",
        stack_address: "the argument run and the result pair `call::pair` lays, into the \
                        handler's `Handler::call`, and `&out` into `slice_from_run`",
    },
    Exception {
        family: "call::CallPair2",
        stack_address: "the argument run and the result pair `call::pair` lays, into the \
                        handler's `Handler::call`, and `&out` into `slice_from_run`",
    },
    Exception {
        family: "call::CallPair4",
        stack_address: "the argument run and the result pair `call::pair` lays, into the \
                        handler's `Handler::call`, and `&out` into `slice_from_run`",
    },
    Exception {
        family: "call::CallIndirect",
        stack_address: "the `Value` the closure's `Code` entry returns, which returns \
                        through a slot in the frame",
    },
    Exception {
        family: "call::CallDirect",
        stack_address: "the `Value` `call_module_sync` returns, which returns through a slot \
                        in the frame, and the `Arc<Prepared>` it borrows, `&prepared`",
    },
    Exception {
        family: "call::Fused",
        stack_address: "the `Value` each call's `Invoke::invoke` returns, which returns \
                        through a slot in the frame, and the held `Value`, `&held` into the \
                        run's tail `Deref`",
    },
    Exception {
        family: "call::MakeClosure",
        stack_address: "the capture iterator, `&mut captures` into `Value::closure`",
    },
    Exception {
        family: "call::SpawnModule",
        stack_address: "the argument `Vec` `staged` returns, which returns through a slot in \
                        the frame, handed on to the spawned frame",
    },
    Exception {
        family: "call::SpawnExternAsync",
        stack_address: "the argument window lent to the handler, `&args` into `call_async`",
    },
    Exception {
        family: "storage::Fetch",
        stack_address: "the `Held` `fetch_at` returns, which returns through a slot in the \
                        frame, by address into `Held::into_word`",
    },
    Exception {
        family: "storage::Commit",
        stack_address: "the `Held` `committed` returns, which returns through a slot in the \
                        frame, by address into `Port::store` or `Port::restore`",
    },
    Exception {
        family: "storage::SetStep",
        stack_address: "`&mut object`, into the step's `Segment::at_mut`",
    },
    Exception {
        family: "storage::SetPath",
        stack_address: "`&mut object`, into `walk_mut`",
    },
    Exception {
        family: "string::Concat",
        stack_address: "`&mut out`, the `String` the parts are pushed into, and the `&str` \
                        `lent` returns, which returns through a slot in the frame",
    },
];

/// The `-zstack-size` `.cargo/config.toml` links (RFC-0105 rule 2).
const LINEAR_STACK: u32 = 16 << 20;

const RUNS_AT_LEAST: usize = 40;

/// The straight body's steps, each an arithmetic chain and an extern call.
/// Without the tail call a step nests two engine frames, and node 26's
/// default stack ran out between 5 000 and 10 000 steps; dropping the body
/// through each successor ran it out from 8 000 steps (2026-09-26).
const STEPS: u32 = 20_000;

/// RFC-0105 rule 4's body: at least this many operations are dropped.
const OPS_AT_LEAST: u32 = 40_000;

const NODE: &str = "/usr/bin/node";

struct Run {
    of: String,
    symbol: String,
    returns: usize,
    tail_calls: usize,
    nesting_calls: usize,
    /// The globals the body writes: a `run` that writes `__stack_pointer`
    /// sets up a linear-stack frame, which is where an address it hands a
    /// callee lives.
    sets_globals: Vec<u32>,
    frame: bool,
}

impl Run {
    fn ends_a_chain(&self) -> bool {
        NO_SUCCESSOR.contains(&self.of.as_str())
    }

    /// No path leaves by a return: each ends in the tail call or traps. A
    /// handler that diverges (`panic`) holds no tail call and still lands.
    fn never_returns(&self) -> bool {
        self.returns == 0
    }

    /// A listed family's instance that calls, and holds the frame its entry
    /// names an address in: an instance with no linear-stack frame holds no
    /// such address, so the entry does not cover it.
    fn excepted(&self) -> Option<&'static Exception> {
        match self.frame {
            true => HOLDS_A_STACK_ADDRESS
                .iter()
                .find(|listed| listed.matches(self)),
            false => None,
        }
    }

    fn lands_right(&self) -> bool {
        self.ends_a_chain() || self.never_returns() || self.excepted().is_some()
    }

    fn why(&self) -> String {
        match self.tail_calls {
            0 => format!(
                "no return_call at all; {} returns, {} call_indirect",
                self.returns, self.nesting_calls
            ),
            tails => format!("{} returns beside {tails} return_call", self.returns),
        }
    }
}

fn op_of(symbol: &str) -> Option<String> {
    let ty = symbol
        .strip_suffix(" as acvus_interpreter::code::Op>::run")?
        .trim_start_matches('<');
    let head = match ty.find('<') {
        Some(angle) => &ty[..angle],
        None => ty,
    };
    let mut parts: Vec<&str> = head.rsplit("::").take(2).collect();
    parts.reverse();
    Some(parts.join("::"))
}

/// One open block of a body, as the scan reads it.
struct Frame {
    /// A `loop`'s label is its start, so a branch to it never reaches its
    /// `end`.
    is_loop: bool,
    /// Some branch to the label was reached.
    targeted: bool,
    /// An `if` whose `else` has not been read: its `end` is reached when the
    /// `if` was, through the arm the module does not hold.
    open_if: Option<bool>,
    /// The `then` arm of an `if` fell through to its `else`.
    then_falls: bool,
}

/// Reads a `run` as wasm validation does: code after a branch, a return, a
/// tail call or `unreachable` is dead until its block ends, and a block's
/// `end` is reached by falling into it or by a branch to it. A reached
/// branch to the function's own label, a reached `return` and a reached
/// function `end` are the paths that return.
fn run_of(of: String, symbol: String, body: &wasmparser::FunctionBody<'_>) -> Run {
    let mut run = Run {
        of,
        symbol,
        returns: 0,
        tail_calls: 0,
        nesting_calls: 0,
        sets_globals: Vec::new(),
        frame: false,
    };
    let mut frames: Vec<Frame> = Vec::new();
    let mut live = true;
    let mut reader = body
        .get_operators_reader()
        .expect("a function body holds its operators");
    // Marks the label `depth` blocks out as branched to; the function's own
    // label is a return.
    let branch = |frames: &mut Vec<Frame>, run: &mut Run, depth: u32| {
        let depth = usize::try_from(depth).expect("a label depth fits usize");
        match frames.len().checked_sub(depth + 1) {
            Some(at) => frames[at].targeted = true,
            None => run.returns += 1,
        }
    };
    while !reader.eof() {
        let operator = reader.read().expect("the module's operators parse");
        match operator {
            Operator::Block { .. } | Operator::Try { .. } | Operator::TryTable { .. } => frames
                .push(Frame {
                    is_loop: false,
                    targeted: false,
                    open_if: None,
                    then_falls: false,
                }),
            Operator::Loop { .. } => frames.push(Frame {
                is_loop: true,
                targeted: false,
                open_if: None,
                then_falls: false,
            }),
            Operator::If { .. } => frames.push(Frame {
                is_loop: false,
                targeted: false,
                open_if: Some(live),
                then_falls: false,
            }),
            Operator::Else => {
                let frame = frames.last_mut().expect("an `else` closes an `if`");
                let entered = frame.open_if.take().expect("an `else` follows its `if`");
                frame.then_falls = live;
                live = entered;
            }
            Operator::End => match frames.pop() {
                Some(frame) => {
                    live = live
                        || frame.then_falls
                        || frame.open_if == Some(true)
                        || (!frame.is_loop && frame.targeted);
                }
                None => {
                    if live {
                        run.returns += 1;
                    }
                    break;
                }
            },
            _ if !live => {}
            Operator::Return => {
                run.returns += 1;
                live = false;
            }
            Operator::Br { relative_depth } => {
                branch(&mut frames, &mut run, relative_depth);
                live = false;
            }
            Operator::BrIf { relative_depth } => branch(&mut frames, &mut run, relative_depth),
            Operator::BrTable { ref targets } => {
                for target in targets.targets().chain([Ok(targets.default())]) {
                    let target = target.expect("a branch table's targets parse");
                    branch(&mut frames, &mut run, target);
                }
                live = false;
            }
            Operator::ReturnCall { .. }
            | Operator::ReturnCallIndirect { .. }
            | Operator::ReturnCallRef { .. } => {
                run.tail_calls += 1;
                live = false;
            }
            Operator::Unreachable => live = false,
            Operator::CallIndirect { .. } => run.nesting_calls += 1,
            Operator::GlobalSet { global_index } => run.sets_globals.push(global_index),
            _ => {}
        }
    }
    run
}

struct Module {
    runs: Vec<Run>,
    linear_stack_top: u32,
}

fn read(bytes: &[u8]) -> Module {
    let mut imported_functions = 0;
    let mut bodies = Vec::new();
    let mut globals = Vec::new();
    let mut function_names = std::collections::HashMap::new();
    let mut global_names = std::collections::HashMap::new();
    for payload in Parser::new(0).parse_all(bytes) {
        match payload.expect("the module parses") {
            Payload::ImportSection(imports) => {
                for import in imports.into_imports() {
                    let import = import.expect("an import parses");
                    if matches!(import.ty, TypeRef::Func(_)) {
                        imported_functions += 1;
                    }
                }
            }
            Payload::GlobalSection(section) => {
                for global in section {
                    let global = global.expect("a global parses");
                    let init = global
                        .init_expr
                        .get_operators_reader()
                        .read()
                        .expect("a global's initializer parses");
                    let constant = match init {
                        Operator::I32Const { value } => Some(value),
                        _ => None,
                    };
                    globals.push(constant);
                }
            }
            Payload::CodeSectionEntry(body) => bodies.push(body),
            Payload::CustomSection(section) => {
                if let KnownCustom::Name(names) = section.as_known() {
                    for subsection in names {
                        match subsection.expect("the name section parses") {
                            Name::Function(map) => {
                                for naming in map {
                                    let naming = naming.expect("a function name parses");
                                    function_names.insert(
                                        naming.index,
                                        format!("{:#}", rustc_demangle::demangle(naming.name)),
                                    );
                                }
                            }
                            Name::Global(map) => {
                                for naming in map {
                                    let naming = naming.expect("a global name parses");
                                    global_names.insert(naming.name.to_string(), naming.index);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    assert!(
        !function_names.is_empty(),
        "the module has no function names; the probe reads `Op::run` by its symbol"
    );

    let stack_pointer = *global_names
        .get("__stack_pointer")
        .expect("the module names its `__stack_pointer`");
    let runs = bodies
        .iter()
        .zip(imported_functions..)
        .filter_map(|(body, index)| {
            let symbol = function_names
                .get(&index)
                .unwrap_or_else(|| panic!("function {index} has no name"));
            let of = op_of(symbol)?;
            let mut run = run_of(of, symbol.clone(), body);
            run.frame = run.sets_globals.contains(&stack_pointer);
            Some(run)
        })
        .collect();
    let initial = globals[usize::try_from(stack_pointer).expect("a global index fits usize")]
        .expect("`__stack_pointer` starts at an `i32.const`");
    Module {
        runs,
        linear_stack_top: initial.cast_unsigned(),
    }
}

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the test crate sits in the workspace")
        .to_path_buf()
}

/// The profile the module is built with: the one an embedder deploys.
const PROFILE: &str = "wasm";

/// Builds the module into a target directory of its own beside this bench's:
/// the `cargo bench` that runs this holds the lock on its own.
fn build_module() -> PathBuf {
    let exe = std::env::current_exe().expect("this bench's own path");
    let target = exe
        .ancestors()
        .nth(3)
        .expect("the bench runs from <target>/<profile>/deps")
        .join("wasm-probe");
    let cargo = std::env::var_os("CARGO").expect("cargo sets CARGO for the bench it runs");
    let status = Command::new(cargo)
        .current_dir(workspace())
        .args([
            "build",
            "--profile",
            PROFILE,
            "--package",
            "acvus-wasm-probe",
            "--target",
            "wasm32-unknown-unknown",
            "--target-dir",
        ])
        .arg(&target)
        .status()
        .expect("cargo runs");
    assert!(status.success(), "the wasm32 build of acvus-wasm-probe failed");
    target.join(format!(
        "wasm32-unknown-unknown/{PROFILE}/acvus_wasm_probe.wasm"
    ))
}

fn body_ops() -> usize {
    let interner = Interner::new();
    let mut registries = std_registries::<AcvusRuntime>();
    registries.push(acvus_wasm_probe::registry());
    let blocks = script_listing_with_externs(
        &interner,
        &acvus_wasm_probe::straight_body(STEPS),
        Context::default(),
        registries,
        Ty::I64,
    );
    ops_of(&blocks).len()
}

/// Prints every `run` that calls its successor outside the list, and each
/// listed family's count, and answers how many land wrong; `main` asserts
/// none after the other checks, so a build that fails several of them names
/// its operations first.
fn check_runs(module: &Module) -> Landing {
    assert!(
        module.runs.len() >= RUNS_AT_LEAST,
        "the probe found {} `Op::run` functions, fewer than the {RUNS_AT_LEAST} the module \
         holds — the name section was not read",
        module.runs.len()
    );
    let failed: Vec<&Run> = module.runs.iter().filter(|run| !run.lands_right()).collect();
    for run in &failed {
        let frame = match run.frame {
            true => "linear-stack frame",
            false => "no frame",
        };
        println!("{}: {}; {frame} — {}", run.of, run.why(), run.symbol);
    }
    let mut families = std::collections::BTreeMap::<&str, usize>::new();
    for run in &failed {
        *families.entry(run.of.as_str()).or_default() += 1;
    }
    for (family, instances) in &families {
        println!("wasm_probe: {family}: {instances} instances call their successor unlisted");
    }

    let mut listed = 0;
    let mut listed_families = 0;
    for entry in HOLDS_A_STACK_ADDRESS {
        let instances: Vec<&Run> = module
            .runs
            .iter()
            .filter(|run| entry.matches(run))
            .collect();
        let calling = instances
            .iter()
            .filter(|run| !run.never_returns() && run.excepted().is_some())
            .count();
        listed += calling;
        match (instances.len(), calling) {
            (0, _) => println!("wasm_probe: {}: not instantiated here", entry.family),
            (_, 0) => println!(
                "wasm_probe: {}: list entry no longer needed — every instance tail-calls",
                entry.family
            ),
            (all, calling) => {
                listed_families += 1;
                println!(
                    "wasm_probe: {}: {calling} of {all} call their successor — {}",
                    entry.family, entry.stack_address
                )
            }
        }
    }

    let ends = module.runs.iter().filter(|run| run.ends_a_chain()).count();
    let tail = module
        .runs
        .iter()
        .filter(|run| !run.ends_a_chain() && run.never_returns() && run.tail_calls > 0)
        .count();
    let traps = module
        .runs
        .iter()
        .filter(|run| !run.ends_a_chain() && run.never_returns() && run.tail_calls == 0)
        .count();
    println!(
        "wasm_probe: {tail} tail-call, {ends} end a chain, {listed} listed in \
         {listed_families} families, {traps} only trap, {} call unlisted",
        failed.len()
    );
    Landing {
        calling: failed.len(),
        families: families.len(),
    }
}

struct Landing {
    calling: usize,
    families: usize,
}

fn check_linear_stack(module: &Module) {
    let top = module.linear_stack_top;
    // `wasm32-unknown-unknown` links the stack first, so its top is its size.
    assert_eq!(
        top, LINEAR_STACK,
        "the linear stack starts at {top}, not the {LINEAR_STACK} RFC-0105 links — the \
         module was built without `-C link-arg=-zstack-size={LINEAR_STACK}`"
    );
    println!("wasm_probe: __stack_pointer starts at {top}");
}

#[derive(Debug, PartialEq)]
struct Mark {
    engine_frames: u64,
    linear_sp: u64,
}

/// One run of the straight body in node, as `acvus-wasm-probe/run.mjs`
/// prints it.
struct Observed {
    steps: u64,
    marks: Vec<Mark>,
    sp_before: u64,
    sp_after: u64,
    release: String,
}

fn observed(run: &serde_json::Value) -> Observed {
    let field = |name: &str| {
        run.get(name)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_else(|| panic!("the runner reported no `{name}`: {run}"))
    };
    let steps = field("steps");
    if let Some(trap) = run.get("trap") {
        panic!("the {steps}-step body trapped in node: {trap} {}", run["panic"]);
    }
    let marks = run["marks"]
        .as_array()
        .expect("a finished run has marks")
        .iter()
        .map(|mark| {
            let at = |name: &str| mark[name].as_u64().expect("a mark's fields are numbers");
            Mark {
                engine_frames: at("engine_frames"),
                linear_sp: at("linear_sp"),
            }
        })
        .collect();
    Observed {
        steps,
        marks,
        sp_before: field("sp_before"),
        sp_after: field("sp_after"),
        release: run["release"]
            .as_str()
            .expect("the runner reports the release")
            .to_string(),
    }
}

fn check_engine(module: &Path) {
    if !Path::new(NODE).exists() {
        println!("wasm_probe: {NODE} does not exist; the straight body is not run");
        return;
    }
    let runner = workspace().join("acvus-wasm-probe/run.mjs");
    let out = Command::new(NODE)
        .arg(&runner)
        .arg(module)
        .args(["1", &STEPS.to_string()])
        .output()
        .expect("node runs");
    assert!(
        out.status.success(),
        "node failed: {}",
        String::from_utf8(out.stderr).expect("node's diagnostics are UTF-8")
    );
    let runs: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("the runner prints one JSON line");
    let [one, many] = runs
        .as_array()
        .expect("the runner prints an array")
        .as_slice()
    else {
        panic!("the runner reported other than two runs: {runs}");
    };
    let (one, many) = (observed(one), observed(many));
    for run in [&one, &many] {
        assert_eq!(
            run.sp_before, run.sp_after,
            "the {}-step run left the linear stack pointer moved",
            run.steps
        );
    }
    assert_eq!(
        one.marks, many.marks,
        "the {}-step body marks the engine frames and linear stack pointer {:?}, the \
         1-step body {:?}: the chain nests",
        many.steps, many.marks, one.marks
    );
    println!(
        "wasm_probe: {} and 1 steps mark (engine frames, linear sp) {:?}",
        many.steps, many.marks
    );
    // RFC-0105 rule 4: the body that ran drops.
    for run in [&one, &many] {
        assert_eq!(
            run.release, "ok",
            "dropping the prepared {}-step body failed in node",
            run.steps
        );
    }
    println!(
        "wasm_probe: the prepared {}-step and 1-step bodies drop in node",
        many.steps
    );
}

fn main() {
    let ops = body_ops();
    assert!(
        ops >= usize::try_from(OPS_AT_LEAST).expect("an operation count fits usize"),
        "the {STEPS}-step body prepares to {ops} operations, fewer than {OPS_AT_LEAST}"
    );
    println!("wasm_probe: the {STEPS}-step body is {ops} operations");

    let path = build_module();
    let bytes = std::fs::read(&path).expect("the module the build wrote");
    let module = read(&bytes);
    let landing = check_runs(&module);
    check_linear_stack(&module);
    check_engine(&path);
    assert!(
        landing.calling == 0,
        "{} operations of {} families call their successor where the return_call belongs, \
         and no listed family holds a linear-stack address for them",
        landing.calling,
        landing.families
    );
}
