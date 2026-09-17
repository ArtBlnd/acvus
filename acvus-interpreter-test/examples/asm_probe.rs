//! Disassembly probe: the erase/materialize path of a word and of a reference.
use std::hint::black_box;

use acvus_extern::{Cross, Runtime};
use acvus_interpreter::{AcvusRuntime, Value};

#[inline(never)]
#[unsafe(no_mangle)]
pub fn probe_erase_i64(rt: &AcvusRuntime, v: i64) -> Value {
    unsafe { rt.erase(v) }
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn probe_materialize_i64(rt: &AcvusRuntime, v: Value) -> i64 {
    unsafe { rt.materialize::<i64>(v) }
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn probe_erase_f64(rt: &AcvusRuntime, v: f64) -> Value {
    unsafe { rt.erase(v) }
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn probe_deref_vec(rt: &AcvusRuntime, r: &Value) -> usize {
    let v: &acvus_extern::Arr<Value, ()> = unsafe { rt.deref(r) };
    v.0.len()
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn probe_reference(rt: &AcvusRuntime, target: &Value) -> Value {
    unsafe { rt.reference(target) }
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn probe_some_i64(rt: &AcvusRuntime, v: i64) -> Option<Value> {
    Some(unsafe { rt.erase(v) })
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn probe_erase_some_i64(rt: &AcvusRuntime, v: Option<i64>) -> Value {
    <Option<i64> as Cross<AcvusRuntime>>::erase(v, rt)
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn probe_materialize_some_i64(rt: &AcvusRuntime, v: Value) -> Option<i64> {
    unsafe { <Option<i64> as Cross<AcvusRuntime>>::materialize(rt, v) }
}

#[inline(never)]
#[unsafe(no_mangle)]
pub fn probe_erase_some_some_i64(rt: &AcvusRuntime, v: Option<Option<i64>>) -> Value {
    <Option<Option<i64>> as Cross<AcvusRuntime>>::erase(v, rt)
}

fn main() {
    let rt = black_box(None::<&AcvusRuntime>);
    let erase_i64 = black_box(probe_erase_i64 as fn(&AcvusRuntime, i64) -> Value);
    let materialize_i64 = black_box(probe_materialize_i64 as fn(&AcvusRuntime, Value) -> i64);
    let erase_f64 = black_box(probe_erase_f64 as fn(&AcvusRuntime, f64) -> Value);
    let reference = black_box(probe_reference as fn(&AcvusRuntime, &Value) -> Value);
    let deref_vec = black_box(probe_deref_vec as fn(&AcvusRuntime, &Value) -> usize);
    let some_i64 = black_box(probe_some_i64 as fn(&AcvusRuntime, i64) -> Option<Value>);
    let erase_some_i64 = black_box(probe_erase_some_i64 as fn(&AcvusRuntime, Option<i64>) -> Value);
    let materialize_some_i64 =
        black_box(probe_materialize_some_i64 as fn(&AcvusRuntime, Value) -> Option<i64>);
    let erase_some_some_i64 =
        black_box(probe_erase_some_some_i64 as fn(&AcvusRuntime, Option<Option<i64>>) -> Value);
    if let Some(rt) = rt {
        let a = erase_i64(rt, black_box(1));
        let b = materialize_i64(rt, a);
        let c = erase_f64(rt, black_box(1.0));
        let d = reference(rt, &c);
        let e = erase_some_i64(rt, black_box(Some(3)));
        let g = erase_some_some_i64(rt, black_box(Some(None)));
        println!(
            "{b} {:?} {} {:?} {g:?}",
            some_i64(rt, black_box(2)).is_some(),
            deref_vec(rt, &d),
            materialize_some_i64(rt, e)
        );
    }
}
