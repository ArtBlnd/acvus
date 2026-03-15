use std::sync::Arc;

use acvus_interpreter::{Interpreter, PureValue, RuntimeError, TypedValue, Value};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use super::Node;
use super::helpers::{eval_script_in_coroutine, render_block_in_coroutine};
use crate::compile::CompiledScript;

// ---------------------------------------------------------------------------
// DisplayNode — iterable display (next-based, per-item yield)
// ---------------------------------------------------------------------------

/// Iterable display node.
///
/// Expects `@start: Int` in local context.
/// Evaluates the iterator script, skips `start` items via `next`, then for
/// each remaining item: renders the template and yields a String.
pub struct DisplayNode {
    iterator: CompiledScript,
    template: CompiledScript,
    item_ty: Ty,
    interner: Interner,
}

impl DisplayNode {
    pub fn new(
        iterator: CompiledScript,
        template: CompiledScript,
        item_ty: Ty,
        interner: &Interner,
    ) -> Self {
        Self {
            iterator,
            template,
            item_ty,
            interner: interner.clone(),
        }
    }
}

impl Node for DisplayNode {
    fn spawn(
        &self,
        local: FxHashMap<Astr, Arc<TypedValue>>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError> {
        let iterator = self.iterator.clone();
        let template = self.template.clone();
        let item_ty = self.item_ty.clone();
        let interner = self.interner.clone();

        acvus_utils::coroutine(move |handle| async move {
            let start_key = interner.intern("start");
            let start: usize = match local
                .get(&start_key)
                .map(|tv| tv.value())
            {
                Some(Value::Pure(PureValue::Int(n))) => *n as usize,
                _ => panic!("DisplayNode: @start must be Int"),
            };

            // Evaluate iterator script → get an IterHandle
            let iter_val = eval_script_in_coroutine(
                &interner,
                &iterator.module,
                &local,
                &handle,
            )
            .await?;

            let ih = Arc::unwrap_or_clone(iter_val.into_value())
                .into_iter_handle(acvus_mir::ty::Effect::Pure);

            // Skip `start` items via next
            let mut interp = Interpreter::new(&interner, iterator.module.clone());
            let mut current = ih;
            for _ in 0..start {
                let result;
                (interp, result) = Interpreter::exec_next(interp, current, &handle).await?;
                match result {
                    Some((_, rest)) => current = rest,
                    None => return Ok(()), // exhausted before start
                }
            }

            // Pull items one by one, render template, yield
            let item_key = interner.intern("item");
            let index_key = interner.intern("index");
            let mut idx = start;

            loop {
                let result;
                (interp, result) = Interpreter::exec_next(interp, current, &handle).await?;
                match result {
                    Some((item, rest)) => {
                        current = rest;

                        let mut entry_local = local.clone();
                        entry_local.insert(
                            item_key,
                            Arc::new(TypedValue::new(Arc::new(item), item_ty.clone())),
                        );
                        entry_local.insert(
                            index_key,
                            Arc::new(TypedValue::int(idx as i64)),
                        );

                        let content = render_block_in_coroutine(
                            &interner,
                            &template.module,
                            &entry_local,
                            &handle,
                        )
                        .await?;

                        handle.yield_val(TypedValue::string(content)).await;
                        idx += 1;
                    }
                    None => break,
                }
            }

            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// DisplayNodeStatic — static display (single template render)
// ---------------------------------------------------------------------------

/// Static display node.
///
/// Evaluates a single template and yields the rendered String.
pub struct DisplayNodeStatic {
    template: CompiledScript,
    interner: Interner,
}

impl DisplayNodeStatic {
    pub fn new(template: CompiledScript, interner: &Interner) -> Self {
        Self {
            template,
            interner: interner.clone(),
        }
    }
}

impl Node for DisplayNodeStatic {
    fn spawn(
        &self,
        local: FxHashMap<Astr, Arc<TypedValue>>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError> {
        let template = self.template.clone();
        let interner = self.interner.clone();

        acvus_utils::coroutine(move |handle| async move {
            let content = render_block_in_coroutine(
                &interner,
                &template.module,
                &local,
                &handle,
            )
            .await?;

            handle.yield_val(TypedValue::string(content)).await;
            Ok(())
        })
    }
}
