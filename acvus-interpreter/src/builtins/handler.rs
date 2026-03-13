use crate::error::RuntimeError;
use crate::value::Value;

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
            Value::Int(n) => n,
            _ => unreachable!("FromValue<i64>: expected Int, got {v:?}"),
        }
    }
}

impl FromValue for f64 {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Float(f) => f,
            _ => unreachable!("FromValue<f64>: expected Float, got {v:?}"),
        }
    }
}

impl FromValue for String {
    fn from_value(v: Value) -> Self {
        match v {
            Value::String(s) => s,
            _ => unreachable!("FromValue<String>: expected String, got {v:?}"),
        }
    }
}

impl FromValue for bool {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Bool(b) => b,
            _ => unreachable!("FromValue<bool>: expected Bool, got {v:?}"),
        }
    }
}

impl FromValue for u8 {
    fn from_value(v: Value) -> Self {
        match v {
            Value::Byte(b) => b,
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
            Value::List(items) => items.into_iter().map(T::from_value).collect(),
            Value::Deque(deque) => {
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
        Value::Int(self)
    }
}

impl IntoValue for f64 {
    fn into_value(self) -> Value {
        Value::Float(self)
    }
}

impl IntoValue for String {
    fn into_value(self) -> Value {
        Value::String(self)
    }
}

impl IntoValue for bool {
    fn into_value(self) -> Value {
        Value::Bool(self)
    }
}

impl IntoValue for u8 {
    fn into_value(self) -> Value {
        Value::Byte(self)
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
            Some(v) => Value::Variant {
                tag: interner.intern("Some"),
                payload: Some(Box::new(v.into_value())),
            },
            None => Value::Variant {
                tag: interner.intern("None"),
                payload: None,
            },
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
            Value::Variant {
                tag,
                payload: Some(inner),
            } if tag == some_tag => Some(T::from_value(*inner)),
            Value::Variant { tag, .. } if tag == none_tag => None,
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
