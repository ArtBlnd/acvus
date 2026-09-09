//! Type-erased handlers for ExternFns.

use std::pin::Pin;
use std::sync::Arc;

use acvus_utils::Interner;

use crate::error::RuntimeError;
use crate::value::{FromValues, IntoValue, Value};

/// Type-erased extern handler. Closure-based - can capture environment.
///
/// `Sync` may run on a blocking thread pool; `Async` runs on the async
/// runtime and owns its interner because it lives across await points.
#[derive(Clone)]
pub enum ExternHandler {
    Sync(Arc<dyn Fn(Vec<Value>, &Interner) -> Result<Value, RuntimeError> + Send + Sync>),
    Async(
        Arc<
            dyn Fn(Vec<Value>, Interner) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + Send>>
                + Send
                + Sync,
        >,
    ),
}

impl ExternHandler {
    pub fn is_sync(&self) -> bool {
        matches!(self, Self::Sync(_))
    }
}

pub fn into_sync_extern_handler<A, R, F>(f: F) -> ExternHandler
where
    F: Fn(&Interner, A) -> Result<R, RuntimeError> + Send + Sync + 'static,
    A: FromValues + 'static,
    R: IntoValue + 'static,
{
    ExternHandler::Sync(Arc::new(move |args, interner| {
        let a = A::from_values(args, interner)?;
        Ok(f(interner, a)?.into_value(interner))
    }))
}

pub fn into_async_extern_handler<A, R, F, Fut>(f: F) -> ExternHandler
where
    F: Fn(Interner, A) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<R, RuntimeError>> + Send + 'static,
    A: FromValues + 'static,
    R: IntoValue + 'static,
{
    ExternHandler::Async(Arc::new(move |args, interner| {
        let a = match A::from_values(args, &interner) {
            Ok(v) => v,
            Err(e) => return Box::pin(std::future::ready(Err(e))),
        };
        let fut = f(interner.clone(), a);
        Box::pin(async move { Ok(fut.await?.into_value(&interner)) })
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_handler_pure_add() {
        let handler = into_sync_extern_handler(|_: &Interner, (a, b): (i64, i64)| Ok(a + b));
        let interner = Interner::new();
        let output = match &handler {
            ExternHandler::Sync(f) => f(vec![Value::Int(10), Value::Int(32)], &interner).unwrap(),
            _ => panic!("expected sync"),
        };
        assert_eq!(output, Value::Int(42));
    }

    #[test]
    fn from_value_type_mismatch() {
        let handler = into_sync_extern_handler(|_: &Interner, (x,): (i64,)| Ok(x));
        let interner = Interner::new();
        let result = match &handler {
            ExternHandler::Sync(f) => f(vec![Value::string("not a number")], &interner),
            _ => panic!("expected sync"),
        };
        assert!(result.is_err());
    }

    #[test]
    fn handler_captures_environment() {
        let multiplier = 7i64;
        let handler =
            into_sync_extern_handler(move |_: &Interner, (x,): (i64,)| Ok(x * multiplier));
        let interner = Interner::new();
        let output = match &handler {
            ExternHandler::Sync(f) => f(vec![Value::Int(6)], &interner).unwrap(),
            _ => panic!("expected sync"),
        };
        assert_eq!(output, Value::Int(42));
    }
}
