use crate::error::RuntimeError;
use crate::value::{LazyValue, PureValue, Value};

// -- FromValue / IntoValue ------------------------------------------------

pub trait FromValue: Sized {
    fn from_value(v: Value) -> Self;
}

pub trait IntoValue {
    fn into_value(self) -> Value;
}

impl FromValue for i64 {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Pure(PureValue::Int(n)) => n,
            _ => unreachable!("FromValue<i64>: expected Int, got {v:?}"),
        }
    }
}

impl FromValue for f64 {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Pure(PureValue::Float(f)) => f,
            _ => unreachable!("FromValue<f64>: expected Float, got {v:?}"),
        }
    }
}

impl FromValue for String {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Pure(PureValue::String(s)) => s,
            _ => unreachable!("FromValue<String>: expected String, got {v:?}"),
        }
    }
}

impl FromValue for bool {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Pure(PureValue::Bool(b)) => b,
            _ => unreachable!("FromValue<bool>: expected Bool, got {v:?}"),
        }
    }
}

impl FromValue for u8 {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Pure(PureValue::Byte(b)) => b,
            _ => unreachable!("FromValue<u8>: expected Byte, got {v:?}"),
        }
    }
}

impl<T> FromValue for Vec<T>
where
    T: FromValue,
{
    fn from_value(v: Value) -> Self {
        match v {
            Value::Lazy(LazyValue::List(items)) => items.into_iter().map(T::from_value).collect(),
            Value::Lazy(LazyValue::Deque(deque)) => {
                deque.into_vec().into_iter().map(|v| T::from_value(v)).collect()
            }
            _ => unreachable!("FromValue<Vec<T>>: expected List, got {v:?}"),
        }
    }
}

impl FromValue for Value {
    fn from_value(v: Value) -> Self {
        v
    }
}

impl IntoValue for i64 {
    fn into_value(self) -> Value {
        Value::int(self)
    }
}

impl IntoValue for f64 {
    fn into_value(self) -> Value {
        Value::float(self)
    }
}

impl IntoValue for String {
    fn into_value(self) -> Value {
        Value::string(self)
    }
}

impl IntoValue for bool {
    fn into_value(self) -> Value {
        Value::bool_(self)
    }
}

impl IntoValue for u8 {
    fn into_value(self) -> Value {
        Value::byte(self)
    }
}

impl IntoValue for Value {
    fn into_value(self) -> Value {
        self
    }
}

impl<T> IntoValue for Option<T>
where
    T: IntoValue,
{
    fn into_value(self) -> Value {
        let interner = crate::interner_ctx::get_interner()
            .expect("IntoValue<Option>: requires interner context");
        match self {
            Some(v) => Value::variant(
                interner.intern("Some"),
                Some(Box::new(v.into_value())),
            ),
            None => Value::variant(
                interner.intern("None"),
                None,
            ),
        }
    }
}

impl<T> FromValue for Option<T>
where
    T: FromValue,
{
    fn from_value(v: Value) -> Self {
        let interner = crate::interner_ctx::get_interner()
            .expect("FromValue<Option>: requires interner context");
        let some_tag = interner.intern("Some");
        let none_tag = interner.intern("None");
        match v {
            Value::Lazy(LazyValue::Variant {
                tag,
                payload: Some(inner),
            }) if tag == some_tag => Some(T::from_value(*inner)),
            Value::Lazy(LazyValue::Variant { tag, .. }) if tag == none_tag => None,
            _ => unreachable!("FromValue<Option<T>>: expected Variant, got {v:?}"),
        }
    }
}

// -- PureBuiltin trait (Axum Handler pattern) -----------------------------

pub trait PureBuiltin<Args> {
    fn call(self, args: Vec<Value>) -> Result<Value, RuntimeError>;
}

impl<F, R, A> PureBuiltin<(A,)> for F
where
    F: Fn(A) -> R,
    A: FromValue,
    R: IntoValue,
{
    fn call(self, args: Vec<Value>) -> Result<Value, RuntimeError> {
        let mut it = args.into_iter();
        let a = A::from_value(it.next().unwrap());
        Ok(self(a).into_value())
    }
}

impl<F, R, A, B> PureBuiltin<(A, B)> for F
where
    F: Fn(A, B) -> R,
    A: FromValue,
    B: FromValue,
    R: IntoValue,
{
    fn call(self, args: Vec<Value>) -> Result<Value, RuntimeError> {
        let mut it = args.into_iter();
        let a = A::from_value(it.next().unwrap());
        let b = B::from_value(it.next().unwrap());
        Ok(self(a, b).into_value())
    }
}

impl<F, R, A, B, C> PureBuiltin<(A, B, C)> for F
where
    F: Fn(A, B, C) -> R,
    A: FromValue,
    B: FromValue,
    C: FromValue,
    R: IntoValue,
{
    fn call(self, args: Vec<Value>) -> Result<Value, RuntimeError> {
        let mut it = args.into_iter();
        let a = A::from_value(it.next().unwrap());
        let b = B::from_value(it.next().unwrap());
        let c = C::from_value(it.next().unwrap());
        Ok(self(a, b, c).into_value())
    }
}
