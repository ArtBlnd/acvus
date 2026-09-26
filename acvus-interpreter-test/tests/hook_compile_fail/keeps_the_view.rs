//! A closure that sends the call's `Args` view itself out of the call.
use std::sync::mpsc;

use acvus_interpreter::{HookArgs, HookEffect, Host, SequentialExecutor, Source};

fn main() {
    let program = Host::new(acvus_ext::std_registries())
        .hook("keep", 1, HookEffect::Opaque)
        .entry::<(), i64>("main", Source::Script("match keep(1) { Some(v) => v, None => 0 }"))
        .compile(SequentialExecutor)
        .unwrap();
    let (send, _receive) = mpsc::sync_channel::<HookArgs<'static, 1>>(1);
    program
        .bind::<1, _>("keep", move |args, out| {
            send.send(args).unwrap();
            Box::pin(async move { out.finish() })
        })
        .unwrap();
}
