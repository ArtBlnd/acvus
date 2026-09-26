//! The wasm32 half of a check whose driver is `clockless_host.mjs`: build
//! this with
//! `cargo build -p acvus-interpreter --example clockless_host --target wasm32-unknown-unknown --release`
//! and run `node clockless_host.mjs <the built .wasm>`. On wasm32 a panic
//! aborts the instance, so the driver reads a refused compile or a failed
//! run as a trap out of `main`.

use acvus_interpreter::{Host, MemoryStorage, SequentialExecutor, Source};

const SCRIPT: &str = "fn f(n) { if n == 0 { 0 } else { n + f(n - 1) } }\nf(100)";

fn main() {
    let program = match Host::new(Vec::new())
        .entry::<(), i64>("main", Source::Script(SCRIPT))
        .compile(SequentialExecutor)
    {
        Ok(program) => program,
        Err(error) => panic!("the script is refused: {error}"),
    };
    let ran = futures::executor::block_on(program.scope(async |scope| {
        let mut storage = MemoryStorage::new();
        let mut page = scope.open(&mut storage);
        let entry = scope.entry::<(), i64>("main")?;
        let out = entry.run(&mut page, ()).await?;
        out.with(|n: &i64| *n)
    }));
    match ran {
        Ok(5050) => {}
        Ok(other) => panic!("the script answers {other}, not 5050"),
        Err(error) => panic!("the run fails: {error}"),
    }
}
