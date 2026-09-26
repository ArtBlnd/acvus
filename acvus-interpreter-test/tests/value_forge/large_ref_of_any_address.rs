//! `Value::large_ref` names whatever address it is given: it is the
//! runtime's alone.
use acvus_interpreter::Value;

fn main() {
    let _ = Value::large_ref(std::ptr::dangling_mut());
}
