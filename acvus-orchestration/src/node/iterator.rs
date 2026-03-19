use acvus_interpreter::{Interpreter, LazyValue, PureValue, RuntimeError, TypedValue, Value, ValueKind};
use acvus_mir::ir::MirModule;
use acvus_mir::ty::{Effect, Ty};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use super::Node;
use super::helpers::{eval_script_in_coroutine, render_block_in_coroutine};
use crate::dsl::{KEY_ITEM, KEY_INDEX};
use crate::spec::{CompiledIteratorEntry, CompiledIteratorSource, CompiledSourceTransform};

/// Composite iterator node: pulls from multiple sources, yields items one by one.
///
/// Each source evaluates `expr` → gets a value, then:
/// - If Iterator: `exec_next` per item
/// - If List/Deque: iterate elements (converted to IterHandle)
/// - If scalar: yield once
///
/// Then applies pagination (start/end) and per-item transform.
///
/// `unordered=false`: sequential — exhaust source A, then B, etc.
/// `unordered=true`: concurrent — yield from whichever source is ready first.
pub struct IteratorNode {
    sources: Vec<CompiledIteratorSource>,
    unordered: bool,
    /// The element type yielded by this iterator (from compiled output_ty).
    /// For `Iterator(elem, _)` output_ty, this is `elem`.
    yield_ty: Ty,
    interner: Interner,
}

impl IteratorNode {
    pub fn new(
        sources: Vec<CompiledIteratorSource>,
        unordered: bool,
        output_ty: &Ty,
        interner: &Interner,
    ) -> Self {
        let yield_ty = match output_ty {
            Ty::Iterator(elem, _) => (**elem).clone(),
            // Fallback: if not Iterator type, use the output_ty itself.
            other => other.clone(),
        };
        Self {
            sources,
            unordered,
            yield_ty,
            interner: interner.clone(),
        }
    }
}

impl Node for IteratorNode {
    fn spawn(
        &self,
        local: FxHashMap<Astr, TypedValue>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError> {
        let sources = self.sources.clone();
        let unordered = self.unordered;
        let yield_ty = self.yield_ty.clone();
        let interner = self.interner.clone();

        acvus_utils::coroutine(move |handle| async move {
            if unordered {
                // TODO: FuturesUnordered concurrent execution
                // For now, fall back to sequential
                sequential_iterate(&interner, &sources, &local, &yield_ty, &handle).await?;
            } else {
                sequential_iterate(&interner, &sources, &local, &yield_ty, &handle).await?;
            }
            Ok(())
        })
    }
}

/// Sequential iteration: exhaust each source in order.
async fn sequential_iterate(
    interner: &Interner,
    sources: &[CompiledIteratorSource],
    local: &FxHashMap<Astr, TypedValue>,
    yield_ty: &Ty,
    handle: &acvus_utils::YieldHandle<TypedValue>,
) -> Result<(), RuntimeError> {
    let name_key = interner.intern("name");
    let item_key = interner.intern(KEY_ITEM);

    for source in sources {
        // Evaluate pagination bounds
        let start: usize = match &source.start {
            Some(start_script) => {
                let val = eval_script_in_coroutine(
                    interner, &start_script.module, local, handle,
                ).await?;
                match val.value() {
                    Value::Pure(PureValue::Int(n)) if *n >= 0 => *n as usize,
                    Value::Pure(PureValue::Int(_)) => 0, // negative start clamped to 0
                    other => return Err(RuntimeError::unexpected_type(
                        "iterator start",
                        &[ValueKind::Int],
                        other.kind(),
                    )),
                }
            }
            None => 0,
        };

        let end: Option<usize> = match &source.end {
            Some(end_script) => {
                let some_tag = interner.intern("Some");
                let val = eval_script_in_coroutine(
                    interner, &end_script.module, local, handle,
                ).await?;
                match val.value() {
                    Value::Lazy(acvus_interpreter::LazyValue::Variant { tag, payload })
                        if *tag == some_tag =>
                    {
                        match payload.as_deref() {
                            Some(Value::Pure(PureValue::Int(v))) if *v >= 0 => Some(*v as usize),
                            Some(Value::Pure(PureValue::Int(_))) => Some(0), // negative end clamped to 0
                            Some(other) => return Err(RuntimeError::unexpected_type(
                                "iterator end (Some payload)",
                                &[ValueKind::Int],
                                other.kind(),
                            )),
                            None => None, // Some(Unit) — treat as no end bound
                        }
                    }
                    Value::Pure(PureValue::Int(v)) if *v >= 0 => Some(*v as usize),
                    Value::Pure(PureValue::Int(_)) => Some(0), // negative end clamped to 0
                    // None variant — no end bound
                    Value::Lazy(LazyValue::Variant { tag, .. })
                        if interner.resolve(*tag) == "None" => None,
                    other => return Err(RuntimeError::unexpected_type(
                        "iterator end",
                        &[ValueKind::Int, ValueKind::Variant],
                        other.kind(),
                    )),
                }
            }
            None => None,
        };

        // Evaluate source expression
        let source_val = eval_script_in_coroutine(
            interner, &source.expr.module, local, handle,
        ).await?;

        // Extract element type from the source value's type.
        let elem_ty = match source_val.ty() {
            Ty::Iterator(elem, _)
            | Ty::Sequence(elem, ..)
            | Ty::List(elem)
            | Ty::Deque(elem, _) => (**elem).clone(),
            other => other.clone(), // scalar source — the item IS the value
        };

        let inner = source_val.into_inner();

        // Determine if iterable or scalar
        let is_iterator = inner.try_expect_ref::<acvus_interpreter::IterHandle>().is_some();
        if is_iterator {
            let ih = inner.expect::<acvus_interpreter::IterHandle>("iter_node");
            let empty_module = MirModule::default();
            let mut interp = Interpreter::new(interner, empty_module);
            let mut current = Some(ih);

            // Skip `start` items
            for _ in 0..start {
                let Some(ih) = current.take() else { break };
                let result;
                (interp, result) = Interpreter::exec_next(interp, ih, handle).await?;
                match result {
                    Some((_, rest)) => current = Some(rest),
                    None => break,
                }
            }

            // Yield items, stopping at `end`
            let take = end.map(|e| e.saturating_sub(start));
            let mut count = 0usize;
            while let Some(ih) = current.take() {
                if let Some(limit) = take {
                    if count >= limit {
                        break;
                    }
                }

                let result;
                (interp, result) = Interpreter::exec_next(interp, ih, handle).await?;
                let Some((item, rest)) = result else {
                    break;
                };
                current = Some(rest);

                let transformed = apply_entries(
                    interner, &source.entries, item, start + count, &elem_ty, local, handle,
                ).await?;

                if let Some(val) = transformed {
                    yield_tagged(handle, name_key, item_key, &source.name, val, yield_ty).await;
                }
                count += 1;
            }
        } else {
            // Scalar: apply entries and yield once
            let transformed = apply_entries(
                interner, &source.entries, inner, 0, &elem_ty, local, handle,
            ).await?;

            if let Some(val) = transformed {
                yield_tagged(handle, name_key, item_key, &source.name, val, yield_ty).await;
            }
        }
    }
    Ok(())
}

/// Apply first-match entry processing.
///
/// - If `entries` is empty: pass-through (yield item as-is).
/// - Otherwise: evaluate each entry's condition in order.
///   - `None` condition always matches.
///   - `Some` condition: evaluate with `@item` + `@index`; match if `true`.
///   - First match: apply its transform and return `Some(transformed)`.
///   - No match: return `None` (skip this item).
async fn apply_entries(
    interner: &Interner,
    entries: &[CompiledIteratorEntry],
    item: Value,
    index: usize,
    elem_ty: &Ty,
    local: &FxHashMap<Astr, TypedValue>,
    handle: &acvus_utils::YieldHandle<TypedValue>,
) -> Result<Option<Value>, RuntimeError> {
    if entries.is_empty() {
        return Ok(Some(item));
    }

    let item_key = interner.intern(KEY_ITEM);
    let index_key = interner.intern(KEY_INDEX);

    let mut entry_local = local.clone();
    entry_local.insert(
        item_key,
        TypedValue::new(item.clone(), elem_ty.clone()),
    );
    entry_local.insert(
        index_key,
        TypedValue::int(index as i64),
    );

    for entry in entries {
        // Evaluate condition (None = always matches)
        let matched = match &entry.condition {
            None => true,
            Some(cond_script) => {
                let result = eval_script_in_coroutine(
                    interner, &cond_script.module, &entry_local, handle,
                ).await?;
                matches!(result.value(), Value::Pure(PureValue::Bool(true)))
            }
        };

        if matched {
            let transformed = apply_transform(
                interner, &entry.transform, &entry_local, handle,
            ).await?;
            return Ok(Some(transformed));
        }
    }

    // No entry matched — skip this item
    Ok(None)
}

/// Apply a single transform with the pre-built local context (already contains `@item` + `@index`).
async fn apply_transform(
    interner: &Interner,
    transform: &CompiledSourceTransform,
    local: &FxHashMap<Astr, TypedValue>,
    handle: &acvus_utils::YieldHandle<TypedValue>,
) -> Result<Value, RuntimeError> {
    match transform {
        CompiledSourceTransform::Script(script) => {
            let result = eval_script_in_coroutine(
                interner, &script.module, local, handle,
            ).await?;
            Ok(result.into_inner())
        }
        CompiledSourceTransform::Template(template) => {
            let rendered = render_block_in_coroutine(
                interner, &template.module, local, handle,
            ).await?;
            Ok(Value::string(rendered))
        }
    }
}

/// Yield a tagged item: `{name: String, item: T}`.
async fn yield_tagged(
    handle: &acvus_utils::YieldHandle<TypedValue>,
    name_key: Astr,
    item_key: Astr,
    name_tag: &str,
    item: Value,
    yield_ty: &Ty,
) {
    let mut fields = FxHashMap::default();
    fields.insert(name_key, Value::string(name_tag));
    fields.insert(item_key, item);
    let obj = Value::object(fields);
    handle.yield_val(TypedValue::new(obj, yield_ty.clone())).await;
}
