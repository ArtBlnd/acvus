//! This interpreter as a `Runtime`: its `Value` enum is the erased
//! representation every ExternFn sees.

use std::sync::Arc;

use acvus_extern::{BoxFuture, ExternValue, FromValue, IntoValue, Runtime};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::error::{RuntimeError, ValueKind};
use crate::value::{FnValue, Value};

pub type ExternHandler = acvus_extern::ExternHandler<AcvusRuntime>;
pub type ExternEntry = acvus_extern::ExternEntry<AcvusRuntime>;

pub struct AcvusRuntime;

fn shape(operation: &'static str, expected: ValueKind, got: &Value) -> RuntimeError {
    let expected: &'static [ValueKind] = match expected {
        ValueKind::Int => &[ValueKind::Int],
        ValueKind::Float => &[ValueKind::Float],
        ValueKind::String => &[ValueKind::String],
        ValueKind::Bool => &[ValueKind::Bool],
        ValueKind::Unit => &[ValueKind::Unit],
        ValueKind::Byte => &[ValueKind::Byte],
        ValueKind::Array => &[ValueKind::Array],
        ValueKind::Object => &[ValueKind::Object],
        ValueKind::Tuple => &[ValueKind::Tuple],
        ValueKind::Variant => &[ValueKind::Variant],
        ValueKind::Fn => &[ValueKind::Fn],
        ValueKind::ExternFn => &[ValueKind::ExternFn],
        ValueKind::Handle => &[ValueKind::Handle],
        ValueKind::Extern => &[ValueKind::Extern],
    };
    RuntimeError::unexpected_type(operation, expected, got.kind())
}

fn unshare<T: Clone>(arc: Arc<T>) -> T {
    Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone())
}

impl FromValue<AcvusRuntime> for Value {
    fn from_value(value: Value, _: &Interner) -> Result<Self, RuntimeError> {
        Ok(value)
    }
    fn from_value_seq(values: Vec<Value>, _: &Interner) -> Result<Vec<Value>, RuntimeError> {
        Ok(values)
    }
}

impl IntoValue<AcvusRuntime> for Value {
    fn into_value(self, _: &Interner) -> Value {
        self
    }
    fn into_value_seq(items: Vec<Value>, _: &Interner) -> Vec<Value> {
        items
    }
}

impl Runtime for AcvusRuntime {
    type Value = Value;
    type Closure = FnValue;
    type Error = RuntimeError;

    fn equals(a: &Value, b: &Value) -> bool {
        a.structural_eq(b)
    }
    fn unit() -> Value {
        Value::Unit
    }
    fn into_unit(value: Value) -> Result<(), RuntimeError> {
        match value {
            Value::Unit => Ok(()),
            other => Err(shape("into_unit", ValueKind::Unit, &other)),
        }
    }
    fn int(n: i64) -> Value {
        Value::Int(n)
    }
    fn float(f: f64) -> Value {
        Value::Float(f)
    }
    fn bool(b: bool) -> Value {
        Value::Bool(b)
    }
    fn byte(b: u8) -> Value {
        Value::Byte(b)
    }
    fn small_bits(value: Value) -> u64 {
        match value {
            Value::Int(n) => n as u64,
            Value::Float(f) => f.to_bits(),
            Value::Bool(b) => b as u64,
            Value::Byte(b) => b as u64,
            other => panic!("small_bits on a non-scalar value: {other:?}"),
        }
    }
    fn string(s: String) -> Value {
        Value::string(s)
    }
    fn into_string(value: Value) -> Result<String, RuntimeError> {
        match value {
            Value::String(s) => Ok(unshare(s)),
            other => Err(shape("into_string", ValueKind::String, &other)),
        }
    }

    fn array(items: Vec<Value>) -> Value {
        Value::array(items)
    }
    fn into_array(value: Value) -> Result<Vec<Value>, RuntimeError> {
        match value {
            Value::Array(items) => Ok(unshare(items)),
            other => Err(shape("into_array", ValueKind::Array, &other)),
        }
    }
    fn tuple(items: Vec<Value>) -> Value {
        Value::tuple(items)
    }
    fn into_tuple(value: Value) -> Result<Vec<Value>, RuntimeError> {
        match value {
            Value::Tuple(items) => Ok(unshare(items)),
            other => Err(shape("into_tuple", ValueKind::Tuple, &other)),
        }
    }
    fn object(fields: FxHashMap<Astr, Value>) -> Value {
        Value::object(fields)
    }
    fn into_object(value: Value) -> Result<FxHashMap<Astr, Value>, RuntimeError> {
        match value {
            Value::Object(fields) => Ok(unshare(fields)),
            other => Err(shape("into_object", ValueKind::Object, &other)),
        }
    }
    fn some(interner: &Interner, value: Value) -> Value {
        Value::some(interner, value)
    }
    fn none(interner: &Interner) -> Value {
        Value::none(interner)
    }
    fn into_option(interner: &Interner, value: Value) -> Result<Option<Value>, RuntimeError> {
        match value {
            Value::Variant(v) if v.tag == interner.intern("Some") => match v.payload {
                Some(payload) => Ok(Some(unshare(payload))),
                None => Err(RuntimeError::internal("Some without payload")),
            },
            Value::Variant(v) if v.tag == interner.intern("None") => Ok(None),
            other => Err(shape("into_option", ValueKind::Variant, &other)),
        }
    }

    fn extern_value(value: ExternValue) -> Value {
        Value::extern_value(value)
    }
    fn into_extern(value: Value) -> Result<ExternValue, RuntimeError> {
        match value {
            Value::Extern(o) => Ok(*o),
            other => Err(shape("into_extern", ValueKind::Extern, &other)),
        }
    }

    fn closure(closure: FnValue) -> Value {
        Value::closure(closure)
    }
    fn into_closure(value: Value) -> Result<FnValue, RuntimeError> {
        match value {
            Value::Fn(f) => Ok(*f),
            other => Err(shape("into_closure", ValueKind::Fn, &other)),
        }
    }
    fn call(closure: &FnValue, args: Vec<Value>) -> BoxFuture<'_, Result<Value, RuntimeError>> {
        Box::pin(crate::interpreter::fn_value_call(closure, args))
    }
}
