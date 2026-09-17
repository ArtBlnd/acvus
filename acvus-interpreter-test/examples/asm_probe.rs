//! Disassembly probe: the erase/materialize path of a word and of a reference.
use std::hint::black_box;

use acvus_extern::{Cross, Runtime};
use acvus_interpreter::{AcvusRuntime, Value};

#[inline(never)]
pub fn probe_erase_i64(rt: &AcvusRuntime, v: i64) -> Value {
    unsafe { rt.erase(v) }
}

#[inline(never)]
pub fn probe_materialize_i64(rt: &AcvusRuntime, v: Value) -> i64 {
    unsafe { rt.materialize::<i64>(v) }
}

#[inline(never)]
pub fn probe_erase_f64(rt: &AcvusRuntime, v: f64) -> Value {
    unsafe { rt.erase(v) }
}

#[inline(never)]
pub fn probe_deref_vec(rt: &AcvusRuntime, r: &Value) -> usize {
    let v: &acvus_extern::Arr<Value, ()> = unsafe { rt.deref(r) };
    v.0.len()
}

#[inline(never)]
pub fn probe_reference(rt: &AcvusRuntime, target: &Value) -> Value {
    unsafe { rt.reference(target) }
}

fn main() {
    let rt = black_box(None::<&AcvusRuntime>);
    if let Some(rt) = rt {
        let a = probe_erase_i64(rt, black_box(1));
        let b = probe_materialize_i64(rt, a);
        let c = probe_erase_f64(rt, black_box(1.0));
        let d = probe_reference(rt, &c);
        println!("{b} {}", probe_deref_vec(rt, &d));
    }
}
