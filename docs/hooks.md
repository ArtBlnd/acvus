# Hooks: one program calling another

A host that loads programs separately (a manifest names them at run time)
still lets one call another. The calling program declares a *hook*, a
function its scripts call. The called program declares a *lent entry*. Once
both are compiled, the host binds the hook to the entry (RFC-0101). The
entry may wait, on a model's answer for one, and the call waits with it.

- **Names.** Compile both programs with `Host::with_names` over one `Names`.
  A value carries its program's interned names, so binding refuses programs
  compiled over two tables.
- **Declaring.** `Host::hook(name, arity, effect)` declares
  `name(a0, …) -> Option<T>` in the calling program. Each argument's type,
  and `T`, is settled where the script calls it. A call of a hook can wait,
  so the script that calls one is a suspending one. A program with an
  unbound hook does not run: its runs end with `HostError::Unbound`.
- **Lent entry.** `Host::lent_entry::<R>(name, inputs, source)` declares an
  entry whose inputs, in argument order, are `field::<T>` (`$x: T`),
  `field_ref::<T>` (`&T`) and `field_mut::<T>` (`&mut T`). It reads every
  input in place and never takes one. A body that moves or writes a
  by-value input is refused when it compiles, naming the input. A write
  through a `&mut` input lands in the caller's storage. A lent entry is not
  run by `Scope::entry`, and no script calls it.
- **Binding.** `Program::bind(hook, &callee.lent(entry)?, run)` compares
  every call site of the hook with the entry: each argument must be within
  the input at its position, and the entry's result within the site's `T`.
  The comparison is by structure. An enum is within one of its path with
  more variants, where both are laid out at one width: a site that only
  builds `Mode::Busy` passes it to an entry taking `Mode { Busy, Idle }`.
  An object is within only one of the same fields, and behind a `&mut` both
  sides are the same type. Only the language's own types cross: an
  extension type, a function value, a task handle or a view is refused, and
  so is a reference anywhere but a whole argument. The entry's effect must be
  within the hook's. The hook's caller cannot see the entry's contexts, so it
  counts a context read as idempotent and a context write as opaque. Each
  refusal is a `Refusal` whose cause is `Cause::Hook { hook, part }`, and a
  site's refusal points at that call.
- **Calling.** A call of the hook calls `run` with a `Call`.
  `call.run(&storage)` runs the entry over `storage`, an
  `Arc<futures::lock::Mutex<S>>` holding the entry's program's contexts,
  and fills the call's result. The call waits for the storage and holds it
  until the entry ends. The calling run is suspended until then. Nothing the
  entry is lent outlives the call, and the entry's result becomes the
  caller's value. A call that would run an entry of a program already
  running on the same call stack traps, a waiting one included.
- **Releasing.** A binding does not keep the entry's program alive: two
  programs bound to each other are both released when the host drops them.
  A hook whose entry's program was released is unbound again.

```rust
use std::sync::Arc;
use std::time::Duration;

use acvus_extern::{extern_fn, extern_registry};
use acvus_interpreter::{
    AcvusRuntime, Host, HookEffect, HostError, LentInputs, MemoryStorage, Names, SequentialExecutor, Source,
};
use futures::lock::Mutex;

/// Stands for a model call: it waits before it answers.
#[extern_fn(effect = opaque)]
async fn complete(prompt: String) -> String {
    tokio::time::sleep(Duration::from_millis(5)).await;
    format!("hello, {prompt}")
}

fn main() -> Result<(), HostError> {
    let names = Names::new();
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(extern_registry! { ns: "llm", fns: [complete], });

    // The called program: it asks the model about a name it is lent, and
    // counts its visits in its own context.
    let greeter = Host::with_names(&names, registries)
        .init("visits", Source::Expr("0"))
        .lent_entry::<String>(
            "greet",
            LentInputs::new().field::<String>("name").field_mut::<i64>("seen"),
            Source::Script(
                r#"@visits = @visits + 1;
*$seen = *$seen + @visits;
complete(clone(&$name))"#,
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
    caller.bind("greet", &greeter.lent("greet")?, move |call| call.run(&kept))?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("a runtime");
    let text = runtime.block_on(caller.scope(async |s| {
        let mut storage = MemoryStorage::new();
        let mut page = s.open(&mut storage);
        let main = s.entry::<(), String>("main")?;
        main.run(&mut page, ()).await?.with(|text: &String| text.clone())
    }))?;
    assert_eq!(text, "hello, ann; seen 11");
    Ok(())
}
```
