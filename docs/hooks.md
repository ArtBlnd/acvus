# Hooks: a script calling into its host

An extern's body is a Rust function the macro declares, so it captures
nothing. A host that must reach its own state from a script (a model
client a manifest names, another program it keeps, a counter) declares a
*hook* instead: a dynamic extern (RFC-0097) whose body is a closure the host
binds after compiling (RFC-0101). A hook is this interpreter's host
feature; the extern contract every runtime shares has nothing of it.

- **Declaring.** `Host::hook(name, arity, effect)` declares
  `name(a0, …) -> Option<T>` before compiling. The script writer passes the
  arguments. Each argument's type, and `T`, is settled where the script
  calls the hook, so one hook serves sites of different types. `effect` is
  `HookEffect::Pure`, `Idempotent` or `Opaque`, and the caller reads the
  call's effect as declared. A hook call can wait, so the script that calls
  one suspends at the call. A hook takes at most eight arguments.
- **Binding.** `Program::bind::<N, _>(name, closure)` binds a hook
  declared with `N` arguments. The closure is `Fn(args, out) ->
  BoxFuture<'c, HookFinished<'c>>`, and it may capture the host's state. A
  closure bound to a hook of another arity is refused as
  `Cause::Hook { part: HookPart::Arity, .. }`, and a second binding as
  `HookPart::Bound`. Binding an undeclared name is `HostError::NotInGraph`.
- **In a host graph.** A hook a host of a `HostGraph` declares is a hook of
  the graph's program in that host's scope, as its entries are (RFC-0095
  rule 1). Only that host's scripts reach it, so two hosts may each
  declare a hook of one name, and they are two hooks.
  `Program::bind_in::<N, _>(host, name, closure)` binds it; `bind` by the
  name alone reaches no hook of a graph. A hook an exposed entry calls
  runs in the caller's run, and its effect is the caller's as the
  entry's other effects are.
- **Arguments.** `args` is RFC-0097 rule 1's `Args` view over the call's
  arguments (`()` for a hook of none). `args.with(i, |x: &T| …)` and
  `args.with_mut(i, |x: &mut T| …)` lend argument `i` to a closure and give
  `Some` exactly when `T` is the type the site settled for it, and
  `args.len()` counts them. A write through `with_mut` to a `&mut` argument
  lands in the caller's place. A write to a by-value argument stays in the
  view and is released with it. Nothing lent outlives the call, which the
  borrow checker enforces.
- **Result.** `out` is RFC-0097 rule 3's `Output`, typed by the site's
  `T`: `out.write(v)` fills it whole, `out.field(name, |f| f.write(v))`
  fills an object field by field, and `out.finish()` seals it. The script
  sees `Some` of what was filled, or `None` where a write's type is not the
  site's, a place is written twice, or a field is missing.
- **No context.** The closure is handed the arguments and the output, and
  nothing else. No context or storage of the caller, or of any program,
  reaches it, so a context promoted to a register stays there across the
  call as across any extern's. Running another program is the host's, with
  what the closure captures.
- **Running.** A program with an unbound hook does not run: its runs end
  with `HostError::Unbound`, naming the hook and, in a graph, its host
  (`HookName`).

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use acvus_interpreter::{HookEffect, Host, HostError, MemoryStorage, SequentialExecutor, Source};

fn main() -> Result<(), HostError> {
    let program = Host::new(acvus_ext::std_registries())
        .hook("complete", 2, HookEffect::Opaque)
        .entry::<(), String>(
            "main",
            Source::Script(
                r#"let name = "ann".to_string();
let asked = 0;
let greeting = match complete(&name, &mut asked) {
    Some(text) => text,
    None => "no answer".to_string(),
};
greeting + "; asked " + asked.to_string()"#,
            ),
        )
        .compile(SequentialExecutor)?;

    // The host's state: how many prompts its model client was sent.
    let sent = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&sent);
    program.bind::<2, _>("complete", move |mut args, mut out| {
        let counted = Arc::clone(&counted);
        Box::pin(async move {
            let prompt = args.with(0, |prompt: &String| prompt.clone());
            // Stands for a model call: it waits before it answers.
            tokio::time::sleep(Duration::from_millis(5)).await;
            let n = counted.fetch_add(1, Ordering::SeqCst) + 1;
            args.with_mut(1, |asked: &mut i64| *asked = n as i64);
            // A prompt of another type leaves `out` empty: the site sees `None`.
            if let Some(prompt) = prompt {
                out.write(format!("hello, {prompt}"));
            }
            out.finish()
        })
    })?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("a runtime");
    let text = runtime.block_on(program.scope(async |s| {
        let mut storage = MemoryStorage::new();
        let mut page = s.open(&mut storage);
        let main = s.entry::<(), String>("main")?;
        main.run(&mut page, ()).await?.with(|text: &String| text.clone())
    }))?;
    assert_eq!(text, "hello, ann; asked 1");
    assert_eq!(sent.load(Ordering::SeqCst), 1);
    Ok(())
}
```

Two hosts of a graph, each with its own `ask`:

```rust
use acvus_interpreter::{
    AcvusRuntime, HookEffect, HostError, HostGraph, MemoryStorage, SequentialExecutor, Source,
};

fn main() -> Result<(), HostError> {
    let asks = "match ask() { Some(n) => n, None => -1 }";
    let program = HostGraph::new(acvus_ext::std_registries::<AcvusRuntime>())
        .host("a", |host| {
            Ok(host.hook("ask", 0, HookEffect::Opaque).entry::<(), i64>("main", Source::Script(asks)))
        })?
        .host("b", |host| {
            Ok(host.hook("ask", 0, HookEffect::Opaque).entry::<(), i64>("main", Source::Script(asks)))
        })?
        .entry("a", "main")
        .entry("b", "main")
        .compile(SequentialExecutor)?;

    for (host, answer) in [("a", 1i64), ("b", 2)] {
        program.bind_in::<0, _>(host, "ask", move |(), mut out| {
            Box::pin(async move {
                out.write(answer);
                out.finish()
            })
        })?;
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime");
    let run = |entry: &'static str| {
        runtime.block_on(program.scope(async |s| {
            let mut storage = MemoryStorage::new();
            let mut page = s.open(&mut storage);
            let main = s.entry::<(), i64>(entry)?;
            main.run(&mut page, ()).await?.with(|n: &i64| *n)
        }))
    };
    assert_eq!(run("a/main")?, 1);
    assert_eq!(run("b/main")?, 2);
    Ok(())
}
```
