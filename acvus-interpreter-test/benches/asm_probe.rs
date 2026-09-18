//! Every `impl Op`'s `run` ends in the tail call to its successor.
//!
//! A bench and not a test: what it reads is the release machine, and a debug
//! build has no tail call in it, so a `cargo test` copy would pass on an
//! artifact nobody runs. `cargo bench --bench asm_probe` builds the profile
//! the numbers come from and disassembles itself.

use std::hint::black_box;
use std::process::Command;

use acvus_interpreter_test::Context;
use acvus_interpreter_test::listing::prepared_script;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// The operations that hold no successor — `code.rs`'s terminators, the node
/// that ends a region's part, and the calls that leave through the driver.
/// A `ret` ends one of these; every other `Op::run` ends in a `jmp`.
const NO_SUCCESSOR: &[&str] = &[
    "control::Goto",
    "control::JumpIf",
    "control::Return",
    "control::Diverge",
    "control::Poison",
    "control::Yield",
    "call::CallExternAsync",
    "call::CallStateAsync",
    "call::CallHeavy",
    "call::CallDirectAsync",
    "call::CallIndirectAsync",
    "call::Eval",
];

/// One family that may end in `call` + `ret`, and the stack address that is
/// why. Both fields are text and neither reads as the other, so each is
/// named.
struct Exception {
    /// The `module::Type` `op_of` produces, which is what the probe matches.
    family: &'static str,
    /// The local whose address escapes into a callee.
    stack_address: &'static str,
}

/// The closed list of families that hold the address of a stack local across
/// a callee. LLVM's sibling-call rule refuses a tail call out of any function
/// an alloca's address escapes, because the callee may reach the caller's
/// frame.
///
/// Decision not to build: every one is a boundary operation costing 50–250
/// instructions, where one `call`/`ret` pair is a fraction. Staging the
/// argument run in the frame window rather than on the stack would remove
/// them, and that is RFC-0052's queued Consequences entry, not this run's.
const HOLDS_A_STACK_ADDRESS: &[Exception] = &[
    Exception {
        family: "call::Fused",
        stack_address: "the SmallVec<[Value; 2]> of staged arguments, handed to the callee",
    },
    Exception {
        family: "call::CallIndirect",
        stack_address: "the staged argument array, `&mut args` to the callee",
    },
    Exception {
        family: "call::CallDirect",
        stack_address: "the staged argument array, `&mut args` to the callee",
    },
    Exception {
        family: "storage::SetStep",
        stack_address: "`&mut object`, into `at_mut`",
    },
    Exception {
        family: "storage::SetPath",
        stack_address: "`&mut object`, into `walk_mut`",
    },
    Exception {
        family: "composite::MakeObject",
        stack_address: "the field buffer the object is built in",
    },
    Exception {
        family: "call::SpawnModule",
        stack_address: "the argument array the spawned frame is filled from",
    },
    Exception {
        family: "call::SpawnExternAsync",
        stack_address: "the argument window lent to the handler",
    },
];

/// What the linker leaves after a function, which is not its last
/// instruction.
const PADDING: &[&str] = &["nop", "nopw", "nopl", "int3", "xchg", "cs"];

/// The fewest `Op::run` symbols this binary instantiates.
const AT_LEAST: usize = 40;

/// One `Op::run`, read as the two facts that separate a tail call from a
/// call: a `run` that tail-calls its successor never returns, so it holds no
/// `ret` at all, and the `jmp` it leaves by is the last one in it.
struct Run {
    of: String,
    rets: usize,
    jumps: Option<String>,
}

impl Run {
    fn ends_a_chain(&self) -> bool {
        NO_SUCCESSOR.contains(&self.of.as_str())
    }

    fn excepted(&self) -> Option<&'static Exception> {
        HOLDS_A_STACK_ADDRESS
            .iter()
            .find(|listed| listed.family == self.of)
    }

    fn tail_jumps(&self) -> bool {
        self.rets == 0 && self.jumps.is_some()
    }

    /// The obligation `code::Op`'s doc names, held here: an operation that
    /// holds a successor ends by calling it, and that call is a tail call.
    fn lands_right(&self) -> bool {
        self.ends_a_chain() || self.tail_jumps() || self.excepted().is_some()
    }

    fn why(&self) -> String {
        match self.jumps {
            Some(ref jump) if self.rets > 0 => {
                format!("{} `ret`s beside the tail `{jump}`", self.rets)
            }
            Some(_) => "no tail jmp at all".to_string(),
            None => format!("{} `ret`s and no `jmp`", self.rets),
        }
    }
}

fn disassembly() -> String {
    let exe = std::env::current_exe().expect("this bench's own path");
    let out = Command::new("objdump")
        .arg("-d")
        .arg("--demangle=rust")
        .arg(&exe)
        .output()
        .expect("objdump is on the path and ran");
    assert!(
        out.status.success(),
        "objdump failed on {}: {}",
        exe.display(),
        String::from_utf8(out.stderr).expect("objdump's diagnostics are UTF-8")
    );
    String::from_utf8(out.stdout).expect("objdump's disassembly is UTF-8")
}

/// The `module::Type` a `<T as acvus_interpreter::code::Op>::run` symbol
/// belongs to; `None` where the symbol is not one.
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

/// The symbol a `0000… <name>:` line names.
fn symbol_line(line: &str) -> Option<&str> {
    let open = line.find(" <")?;
    if !line.ends_with(">:") || !line[..open].chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(&line[open + 2..line.len() - 2])
}

/// The instruction an `  addr:\tbytes\ttext` line holds, padding excluded.
fn instruction(line: &str) -> Option<&str> {
    let (_address, rest) = line.split_once('\t')?;
    let text = match rest.split_once('\t') {
        Some((_bytes, text)) => text,
        None => rest,
    };
    let text = text.trim();
    let mnemonic = text.split_whitespace().next()?;
    match PADDING.contains(&mnemonic) {
        true => None,
        false => Some(text),
    }
}

fn runs(text: &str) -> Vec<Run> {
    let mut found = Vec::new();
    let mut open: Option<Run> = None;
    for line in text.lines() {
        if let Some(symbol) = symbol_line(line) {
            found.extend(open.take());
            open = op_of(symbol).map(|of| Run {
                of,
                rets: 0,
                jumps: None,
            });
            continue;
        }
        let (Some(run), Some(text)) = (open.as_mut(), instruction(line)) else {
            continue;
        };
        let mnemonic = text
            .split_whitespace()
            .next()
            .expect("`instruction` returns a line that holds a mnemonic");
        if mnemonic == "ret" {
            run.rets += 1;
        }
        if mnemonic == "jmp" {
            run.jumps = Some(text.to_string());
        }
    }
    found.extend(open);
    found
}

/// Preparing one body links `prepare`, which names every operation's
/// constructor, so every `Op` vtable — and so every `run` — is in this
/// binary for the disassembly to find.
fn link_the_machine() {
    let interner = Interner::new();
    let prepared = prepared_script(
        &interner,
        "let acc = 0; let i = 0; while i < 10 { acc = acc + i; i = i + 1; } acc",
        Context::default(),
        Ty::I64,
    );
    black_box(&prepared);
}

fn main() {
    link_the_machine();
    let found = runs(&disassembly());
    assert!(
        found.len() >= AT_LEAST,
        "the probe found {} `Op::run` symbols, fewer than the {AT_LEAST} this \
         binary instantiates — the disassembly was not read",
        found.len()
    );

    let failed: Vec<&Run> = found.iter().filter(|run| !run.lands_right()).collect();
    for run in &failed {
        println!("{}: {}", run.of, run.why());
    }
    assert!(
        failed.is_empty(),
        "{} operations land with a call where the tail jmp belongs, and no listed \
         family holds a stack address for them — a ninth family does not join the \
         list silently",
        failed.len()
    );

    let mut still_needed = 0;
    for listed in HOLDS_A_STACK_ADDRESS {
        let instances: Vec<&Run> = found.iter().filter(|run| run.of == listed.family).collect();
        let calling = instances.iter().filter(|run| !run.tail_jumps()).count();
        still_needed += calling;
        match (instances.len(), calling) {
            (0, _) => println!("{}: not instantiated here", listed.family),
            (_, 0) => println!(
                "{}: list entry no longer needed — every instance tail-jumps",
                listed.family
            ),
            (all, calling) => println!(
                "{}: {calling} of {all} end in call+ret — {}",
                listed.family, listed.stack_address
            ),
        }
    }

    let ends = found.iter().filter(|run| run.ends_a_chain()).count();
    println!(
        "asm_probe: {} operations tail-call their successor, {ends} end a chain, \
         {still_needed} hold a stack address across a listed callee",
        found.len() - ends - still_needed
    );
}
