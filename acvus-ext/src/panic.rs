//! `std::panic`: the script's own trap (RFC-0038).

use acvus_extern::{Never, Registry, Runtime, Trap, extern_fn, extern_registry};

#[extern_fn(effect = pure)]
fn panic(message: String) -> Result<Never, Trap> {
    Err(Trap::call("panic", message))
}

pub fn panic_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        fns: [panic],
    }
}
