//! A closure that stores what `with` lends it into the host's state.
use std::sync::{Arc, Mutex};

use acvus_interpreter::{HookEffect, Host, SequentialExecutor, Source};

fn main() {
    let program = Host::new(acvus_ext::std_registries())
        .hook("keep", 1, HookEffect::Opaque)
        .entry::<(), i64>("main", Source::Script("match keep(\"a\".to_string()) { Some(v) => v, None => 0 }"))
        .compile(SequentialExecutor)
        .unwrap();
    let kept: Arc<Mutex<Vec<&'static String>>> = Arc::new(Mutex::new(Vec::new()));
    program
        .bind::<1, _>("keep", move |args, out| {
            let kept = Arc::clone(&kept);
            Box::pin(async move {
                args.with(0, |text: &String| kept.lock().unwrap().push(text));
                out.finish()
            })
        })
        .unwrap();
}
