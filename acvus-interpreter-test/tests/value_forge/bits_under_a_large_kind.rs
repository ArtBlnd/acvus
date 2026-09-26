//! A write through `bits_mut` keeps the kind, so it would put any bits under
//! a `Large`'s: it is the runtime's alone.
use acvus_interpreter::Value;

fn main() {
    let mut value = Value::string("x");
    *value.bits_mut() = 8;
}
