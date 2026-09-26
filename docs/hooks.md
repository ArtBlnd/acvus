# Hooks: one program calling another

A host that loads programs separately (a manifest names them at run time)
still lets one call another. The calling program declares a *hook*, a
function its scripts call. The called program declares a *lent entry*. Once
both are compiled, the host binds the hook to the entry (RFC-0101).

- **Names.** Compile both programs with `Host::with_names` over one `Names`.
  A value carries its program's interned names, so binding refuses programs
  compiled over two tables.
- **Declaring.** `Host::hook(name, arity, effect)` declares
  `name(a0, …) -> Option<T>` in the calling program. Each argument's type,
  and `T`, is settled where the script calls it. A program with an unbound
  hook does not run: its runs end with `HostError::Unbound`.
- **Lent entry.** `Host::lent_entry::<R>(name, inputs, source)` declares an
  entry whose inputs, in argument order, are `field::<T>` (`$x: T`),
  `field_ref::<T>` (`&T`) and `field_mut::<T>` (`&mut T`). It reads every
  input in place and never takes one. A body that moves or writes a
  by-value input is refused when it compiles, naming the input. A write
  through a `&mut` input lands in the caller's storage. A lent entry is not
  run by `Scope::entry`, and no script calls it.
- **Binding.** `Program::bind(hook, &callee.lent(entry)?, run)` compares
  every call site of the hook with the entry: each argument with the input at
  its position, and the site's `T` with the entry's result. The comparison
  is by structure: an object by its field names and types, an enum by its
  path, variant names and payloads. Only the language's own types cross: an
  extension type, a function value, a task handle or a view is refused, and
  so is a reference anywhere but a whole argument. The entry's effect must be
  within the hook's. The hook's caller cannot see the entry's contexts, so it
  counts a context read as idempotent and a context write as opaque. The
  entry must not wait. Each refusal is a `Refusal` whose cause is
  `Cause::Hook { hook, part }`, and a site's refusal points at that call.
- **Calling.** A call of the hook calls `run` with a `Call`. `call.run(&mut
  storage)` runs the entry over the storage that holds the entry's program's
  contexts, and fills the call's result. A `Call` cannot leave the closure
  or its thread. Nothing the entry is lent outlives the call. The entry's
  result becomes the caller's value. A call that would run an entry of a
  program already running on the same call stack traps.

```rust
use std::sync::{Arc, Mutex};

use acvus_interpreter::{
    Host, HookEffect, HostError, LentInputs, MemoryStorage, Names, SequentialExecutor, Source,
};

fn main() -> Result<(), HostError> {
    let names = Names::new();

    // The called program: it greets a name it is lent, and counts its visits
    // in its own context.
    let greeter = Host::with_names(&names, acvus_ext::std_registries())
        .init("visits", Source::Expr("0"))
        .lent_entry::<String>(
            "greet",
            LentInputs::new().field::<String>("name").field_mut::<i64>("seen"),
            Source::Script(
                r#"@visits = @visits + 1;
*$seen = *$seen + @visits;
"hello, " + $name"#,
            ),
        )
        .compile(SequentialExecutor)?;

    // The calling program: `greet` is a hook it declares.
    let caller = Host::with_names(&names, acvus_ext::std_registries())
        .hook("greet", 2, HookEffect::Opaque)
        .entry::<(), String>(
            "main",
            Source::Script(
                r#"let seen = 10;
let greeting = match greet("ann".to_string(), &mut seen) {
    Some(text) => text,
    None => "no greeting".to_string(),
};
greeting + "; seen " + if seen == 11 { "11" } else { "?" }"#,
            ),
        )
        .compile(SequentialExecutor)?;

    // The greeter's contexts live in a storage the host keeps for it.
    let greeter_storage = Arc::new(Mutex::new(MemoryStorage::new()));
    let kept = Arc::clone(&greeter_storage);
    caller.bind("greet", &greeter.lent("greet")?, move |call| {
        let mut storage = kept.lock().expect("no call panicked holding the storage");
        call.run(&mut *storage)
    })?;

    let text = futures::executor::block_on(caller.scope(async |s| {
        let mut storage = MemoryStorage::new();
        let mut page = s.open(&mut storage);
        let main = s.entry::<(), String>("main")?;
        main.run(&mut page, ()).await?.with(|text: &String| text.clone())
    }))?;
    assert_eq!(text, "hello, ann; seen 11");
    Ok(())
}
```
