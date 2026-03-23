use acvus_interpreter::{RuntimeError, TypedValue, ValueKind};
use acvus_mir::graph::ContextId;
use rustc_hash::FxHashMap;

use super::Unit;

pub struct AssertNode {
    check_expr_id: ContextId,
    value_id: ContextId,
    retry: u32,
}

impl AssertNode {
    pub fn new(check_expr_id: ContextId, value_id: ContextId, retry: u32) -> Self {
        Self { check_expr_id, value_id, retry }
    }
}

impl Unit for AssertNode {
    fn spawn(
        &self,
        _local_context: FxHashMap<ContextId, TypedValue>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError, ContextId> {
        let check_expr_id = self.check_expr_id;
        let value_id = self.value_id;
        let retry = self.retry;

        acvus_utils::coroutine(move |handle| async move {
            for _ in 0..=retry {
                let value: TypedValue = handle.request_context(value_id).await;
                let check: TypedValue = handle.request_context(check_expr_id).await;
                let passed = check.value().try_expect_ref::<bool>()
                    .ok_or_else(|| RuntimeError::unexpected_type(
                        "assert",
                        &[ValueKind::Bool],
                        check.value().kind(),
                    ))?;

                if *passed {
                    handle.yield_val(value).await;
                    return Ok(());
                }
            }
            Err(RuntimeError::assert_failed())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_interpreter::Stepped;

    #[tokio::test]
    async fn pass_through_on_true() {
        let check_id = ContextId::alloc();
        let value_id = ContextId::alloc();
        let node = AssertNode::new(check_id, value_id, 0);
        let co = node.spawn(FxHashMap::default());

        let (co, stepped) = co.step().await;
        let request = match stepped {
            Stepped::NeedContext(r) => { assert_eq!(r.key(), value_id); r }
            _ => panic!("expected NeedContext for value"),
        };
        request.resolve(TypedValue::int(42));

        let (co, stepped) = co.step().await;
        let request = match stepped {
            Stepped::NeedContext(r) => { assert_eq!(r.key(), check_id); r }
            _ => panic!("expected NeedContext for check"),
        };
        request.resolve(TypedValue::bool_(true));

        let (_co, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => {
                assert_eq!(*value.value().expect_ref::<i64>("test"), 42);
            }
            _ => panic!("expected Emit"),
        }
    }

    #[tokio::test]
    async fn error_on_false_no_retry() {
        let check_id = ContextId::alloc();
        let value_id = ContextId::alloc();
        let node = AssertNode::new(check_id, value_id, 0);
        let co = node.spawn(FxHashMap::default());

        let (co, stepped) = co.step().await;
        match stepped {
            Stepped::NeedContext(r) => r.resolve(TypedValue::int(42)),
            _ => panic!("expected NeedContext"),
        };

        let (co, stepped) = co.step().await;
        match stepped {
            Stepped::NeedContext(r) => r.resolve(TypedValue::bool_(false)),
            _ => panic!("expected NeedContext"),
        };

        let (_co, stepped) = co.step().await;
        assert!(matches!(stepped, Stepped::Error(_)), "expected Error on false assert");
    }

    #[tokio::test]
    async fn retry_then_pass() {
        let check_id = ContextId::alloc();
        let value_id = ContextId::alloc();
        let node = AssertNode::new(check_id, value_id, 1);
        let co = node.spawn(FxHashMap::default());

        // Attempt 1: false
        let (co, stepped) = co.step().await;
        match stepped { Stepped::NeedContext(r) => r.resolve(TypedValue::int(1)), _ => panic!() };
        let (co, stepped) = co.step().await;
        match stepped { Stepped::NeedContext(r) => r.resolve(TypedValue::bool_(false)), _ => panic!() };

        // Attempt 2 (retry): true
        let (co, stepped) = co.step().await;
        match stepped { Stepped::NeedContext(r) => r.resolve(TypedValue::int(2)), _ => panic!() };
        let (co, stepped) = co.step().await;
        match stepped { Stepped::NeedContext(r) => r.resolve(TypedValue::bool_(true)), _ => panic!() };

        let (_co, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => assert_eq!(*value.value().expect_ref::<i64>("test"), 2),
            _ => panic!("expected Emit"),
        }
    }
}
