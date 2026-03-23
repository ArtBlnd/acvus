use acvus_interpreter::{RuntimeError, TypedValue};
use acvus_mir::graph::ContextId;
use rustc_hash::FxHashMap;

use super::Unit;

/// Init gate: tries storage first, falls back to initial_value.
///
/// Mirrors assert pattern:
///   assert: check(Bool) + value → gate
///   init:   try_storage → if found, pass through; if not, evaluate init_value
pub struct InitNode {
    /// ContextId to try reading from storage (Context entity → context_provider).
    storage_read_id: ContextId,
    /// ContextId of the init_value unit (LocalUnit, evaluates initial_value expression).
    init_value_id: ContextId,
}

impl InitNode {
    pub fn new(storage_read_id: ContextId, init_value_id: ContextId) -> Self {
        Self { storage_read_id, init_value_id }
    }
}

impl Unit for InitNode {
    fn spawn(
        &self,
        _local_context: FxHashMap<ContextId, TypedValue>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError, ContextId> {
        let storage_read_id = self.storage_read_id;
        let init_value_id = self.init_value_id;

        acvus_utils::coroutine(move |handle| async move {
            // Try storage first.
            if let Some(val) = handle.try_request_context(storage_read_id).await {
                handle.yield_val(val).await;
            } else {
                // Storage empty → evaluate initial_value expression.
                let val = handle.request_context(init_value_id).await;
                handle.yield_val(val).await;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_interpreter::Stepped;

    #[tokio::test]
    async fn storage_has_value_passes_through() {
        let storage_id = ContextId::alloc();
        let init_val_id = ContextId::alloc();
        let node = InitNode::new(storage_id, init_val_id);
        let co = node.spawn(FxHashMap::default());

        // First step: NeedContext for storage_read_id (try)
        let (co, stepped) = co.step().await;
        let request = match stepped {
            Stepped::NeedContext(r) => { assert_eq!(r.key(), storage_id); r }
            _ => panic!("expected NeedContext for storage"),
        };
        // Storage has value → resolve with it
        request.resolve(TypedValue::int(42));

        // Should emit the stored value
        let (_co, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => assert_eq!(*value.value().expect_ref::<i64>("test"), 42),
            _ => panic!("expected Emit"),
        }
    }

    #[tokio::test]
    async fn storage_empty_falls_back_to_init_value() {
        let storage_id = ContextId::alloc();
        let init_val_id = ContextId::alloc();
        let node = InitNode::new(storage_id, init_val_id);
        let co = node.spawn(FxHashMap::default());

        // First step: NeedContext for storage_read_id (try)
        let (co, stepped) = co.step().await;
        let request = match stepped {
            Stepped::NeedContext(r) => { assert_eq!(r.key(), storage_id); r }
            _ => panic!("expected NeedContext for storage"),
        };
        // Storage empty → not found
        request.resolve_not_found();

        // Should request init_value_id
        let (co, stepped) = co.step().await;
        let request = match stepped {
            Stepped::NeedContext(r) => { assert_eq!(r.key(), init_val_id); r }
            _ => panic!("expected NeedContext for init_value"),
        };
        request.resolve(TypedValue::int(0));

        // Should emit the initial value
        let (_co, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => assert_eq!(*value.value().expect_ref::<i64>("test"), 0),
            _ => panic!("expected Emit"),
        }
    }
}
