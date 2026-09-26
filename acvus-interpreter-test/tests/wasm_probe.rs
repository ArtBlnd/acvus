//! The `wasm32` module this workspace builds holds every `impl Op`'s `run`
//! ending in the tail call to its successor, `return_call_indirect`, as
//! `asm_probe` holds the tail `jmp` natively, and links a 16 MiB linear stack
//! (RFC-0105). Where node is installed, a straight body of many thousand
//! operations runs in the module at the engine depth and linear stack pointer
//! of a body of one step.
//!
//! A test and not a bench, unlike `asm_probe`: what it reads is the module,
//! which it builds with `--release` in a cargo of its own through the
//! workspace's `.cargo/config.toml`, whatever profile this binary has. A
//! build without the flags that file names fails here, naming the
//! operations: `RUSTFLAGS= cargo test -p acvus-interpreter-test --test
//! wasm_probe` is that build.

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

/// The `-zstack-size` `.cargo/config.toml` links (RFC-0105 rule 2).
const LINEAR_STACK: u32 = 16 << 20;

const RUNS_AT_LEAST: usize = 40;

/// The straight body's steps, each an arithmetic chain and an extern call.
/// Without the tail call a step nests two engine frames, and node 26's
/// default stack ran out between 5 000 and 10 000 steps (2026-09-26).
const STEPS: u32 = 10_000;

const OPS_AT_LEAST: u32 = 2 * STEPS;

const NODE: &str = "/usr/bin/node";

struct Run {
    of: String,
    symbol: String,
    returns: usize,
    tail_calls: usize,
    nesting_calls: usize,
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

    fn lands_right(&self) -> bool {
        self.ends_a_chain() || self.never_returns()
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

fn run_of(of: String, symbol: String, body: &wasmparser::FunctionBody<'_>) -> Run {
    let mut run = Run {
        of,
        symbol,
        returns: 0,
        tail_calls: 0,
        nesting_calls: 0,
    };
    // The blocks open around the operator read, so a branch whose label is
    // `depth` leaves the function.
    let mut depth: u32 = 0;
    // The operator before the function's own closing `end`: an inner block's
    // `end` there falls through to the function's end too.
    let mut last = None;
    let mut reader = body
        .get_operators_reader()
        .expect("a function body holds its operators");
    while !reader.eof() {
        let operator = reader.read().expect("the module's operators parse");
        match operator {
            Operator::Block { .. }
            | Operator::Loop { .. }
            | Operator::If { .. }
            | Operator::Try { .. }
            | Operator::TryTable { .. } => depth += 1,
            Operator::End if depth == 0 => break,
            Operator::End => depth -= 1,
            Operator::Return => run.returns += 1,
            Operator::Br { relative_depth } | Operator::BrIf { relative_depth }
                if relative_depth == depth =>
            {
                run.returns += 1
            }
            Operator::BrTable { ref targets } => {
                let to_the_end = targets
                    .targets()
                    .chain([Ok(targets.default())])
                    .map(|target| target.expect("a branch table's targets parse"))
                    .filter(|target| *target == depth)
                    .count();
                run.returns += to_the_end;
            }
            Operator::ReturnCall { .. }
            | Operator::ReturnCallIndirect { .. }
            | Operator::ReturnCallRef { .. } => run.tail_calls += 1,
            Operator::CallIndirect { .. } => run.nesting_calls += 1,
            _ => {}
        }
        last = Some(operator);
    }
    let falls_off_the_end = !matches!(
        last,
        Some(
            Operator::Return
                | Operator::ReturnCall { .. }
                | Operator::ReturnCallIndirect { .. }
                | Operator::ReturnCallRef { .. }
                | Operator::Unreachable
                | Operator::Br { .. }
                | Operator::BrTable { .. }
        )
    );
    if falls_off_the_end {
        run.returns += 1;
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

    let runs = bodies
        .iter()
        .zip(imported_functions..)
        .filter_map(|(body, index)| {
            let symbol = function_names
                .get(&index)
                .unwrap_or_else(|| panic!("function {index} has no name"));
            let of = op_of(symbol)?;
            Some(run_of(of, symbol.clone(), body))
        })
        .collect();
    let stack_pointer = *global_names
        .get("__stack_pointer")
        .expect("the module names its `__stack_pointer`");
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
            "--release",
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
    target.join("wasm32-unknown-unknown/release/acvus_wasm_probe.wasm")
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

/// Prints every `run` that calls its successor, by family, and answers how
/// many there are; `main` asserts none after the other checks, so a build
/// that fails several of them names its operations first.
fn check_runs(module: &Module) -> Landing {
    assert!(
        module.runs.len() >= RUNS_AT_LEAST,
        "the probe found {} `Op::run` functions, fewer than the {RUNS_AT_LEAST} the module \
         holds — the name section was not read",
        module.runs.len()
    );
    let failed: Vec<&Run> = module.runs.iter().filter(|run| !run.lands_right()).collect();
    for run in &failed {
        println!("{}: {} — {}", run.of, run.why(), run.symbol);
    }
    let mut families = std::collections::BTreeMap::<&str, usize>::new();
    for run in &failed {
        *families.entry(run.of.as_str()).or_default() += 1;
    }
    for (family, instances) in &families {
        println!("wasm_probe: {family}: {instances} instances call their successor");
    }
    let ends = module.runs.iter().filter(|run| run.ends_a_chain()).count();
    println!(
        "wasm_probe: {} operations never return, {ends} end a chain, {} call their successor",
        module.runs.len() - ends - failed.len(),
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
    // Not RFC-0105's rule: dropping a prepared body is not running it. It is
    // printed because a drop that nests per operation runs the same engine
    // stack out.
    println!(
        "wasm_probe: dropping the prepared bodies (not asserted): 1 step {}, {} steps {}",
        one.release, many.steps, many.release
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
        "{} operations of {} families call their successor where the return_call belongs",
        landing.calling,
        landing.families
    );
}
