//! Every `impl Op`'s `run` ends in the tail call to its successor.
//!
//! A bench and not a test: what it reads is the release machine, and a debug
//! build has no tail call in it, so a `cargo test` copy would pass on an
//! artifact nobody runs. `cargo bench --bench asm_probe` builds the profile
//! the numbers come from and disassembles itself.
//!
//! `PADDING` below is the mnemonic set `benches/README.md` names for dropping
//! inter-function alignment padding out of an `Op::run` body diff.

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
    "control::ForAt",
    "switch::Switch",
    "switch::SwitchOption",
    "run::SwitchRun",
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
    /// The handler the operation was monomorphized over, as the substring of
    /// the demangled symbol that names it (`__extern_fn_next`); `None` where
    /// every instance of the family holds the address.
    handler: Option<&'static str>,
    /// The local whose address escapes into a callee.
    stack_address: &'static str,
}

impl Exception {
    fn matches(&self, run: &Run) -> bool {
        self.family == run.of
            && self
                .handler
                .is_none_or(|handler| run.symbol.contains(handler))
    }

    fn name(&self) -> String {
        match self.handler {
            Some(handler) => format!("{}<{handler}>", self.family),
            None => self.family.to_string(),
        }
    }
}

/// The closed list of families that hold the address of a stack local across
/// a callee. LLVM's sibling-call rule refuses a tail call out of any function
/// an alloca's address escapes, because the callee may reach the caller's
/// frame.
///
/// Decision not to build: every one is a boundary operation costing 50–250
/// instructions, where one `call`/`ret` pair is a fraction.
const HOLDS_A_STACK_ADDRESS: &[Exception] = &[
    Exception {
        family: "call::Fused",
        handler: None,
        stack_address: "the held `Value`, `&held` into the run's tail `Deref`",
    },
    Exception {
        family: "call::CallIndirect",
        handler: None,
        stack_address: "the `FnValue` materialized out of the callee register, `&closure` into \
                        `call_fn_sync`",
    },
    Exception {
        family: "call::CallDirect",
        handler: None,
        stack_address: "the `Arc<Prepared>` the module table hands back, `&prepared` into \
                        `call_module_sync`",
    },
    Exception {
        family: "storage::SetPath",
        handler: None,
        stack_address: "`&mut object`, into `walk_mut`",
    },
    Exception {
        family: "call::SpawnModule",
        handler: None,
        stack_address: "the argument array the spawned frame is filled from",
    },
    Exception {
        family: "call::SpawnExternAsync",
        handler: None,
        stack_address: "the argument window lent to the handler",
    },
    Exception {
        family: "call::CallExtern1",
        handler: Some("__extern_fn_next"),
        stack_address: "the iterator the handler drains, `&mut it` into its stage's `next` \
                        through the stage's vtable",
    },
    Exception {
        family: "call::CallWindow",
        handler: Some("__extern_fn_next"),
        stack_address: "the iterator the handler drains, `&mut it` into its stage's `next` \
                        through the stage's vtable",
    },
    Exception {
        family: "call::CallExtern2",
        handler: Some("__extern_fn_max_by_key"),
        stack_address: "the iterator the handler drains, `&mut it` into its stage's `next` \
                        through the stage's vtable",
    },
    Exception {
        family: "call::CallExtern2",
        handler: Some("__extern_fn_min_by_key"),
        stack_address: "the iterator the handler drains, `&mut it` into its stage's `next` \
                        through the stage's vtable",
    },
    Exception {
        family: "call::CallExtern1",
        handler: Some("__extern_fn_panic"),
        stack_address: "the message `String` the handler materializes, by address into the \
                        panic's formatting",
    },
    Exception {
        family: "call::CallWindow",
        handler: Some("__extern_fn_panic"),
        stack_address: "the message `String` the handler materializes, by address into the \
                        panic's formatting",
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
    /// The demangled symbol, which names the handler an operation was
    /// monomorphized over.
    symbol: String,
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
            .find(|listed| listed.matches(self))
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
                symbol: symbol.to_string(),
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
        let instances: Vec<&Run> = found.iter().filter(|run| listed.matches(run)).collect();
        let calling = instances.iter().filter(|run| !run.tail_jumps()).count();
        still_needed += calling;
        match (instances.len(), calling) {
            (0, _) => println!("{}: not instantiated here", listed.name()),
            (_, 0) => println!(
                "{}: list entry no longer needed — every instance tail-jumps",
                listed.name()
            ),
            (all, calling) => println!(
                "{}: {calling} of {all} end in call+ret — {}",
                listed.name(),
                listed.stack_address
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
