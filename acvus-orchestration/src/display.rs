use std::sync::Arc;

use acvus_interpreter::{Interpreter, LazyValue, PureValue, RuntimeError, Value};
use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::compile::{
    CompiledScript, compile_script, compile_script_with_hint, compile_template, expect_list,
};
use crate::error::OrchError;
use crate::storage::EntryRef;

// ---------------------------------------------------------------------------
// Spec (source strings — compilation input)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DisplayEntrySpec {
    pub name: String,
    pub condition: String,
    pub template: String,
}

#[derive(Debug, Clone)]
pub struct StaticDisplaySpec {
    pub template: String,
}

#[derive(Debug, Clone)]
pub struct IterableDisplaySpec {
    pub iterator: String,
    pub entries: Vec<DisplayEntrySpec>,
}

// ---------------------------------------------------------------------------
// Compiled
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompiledDisplayEntry {
    pub name: String,
    pub condition: Option<CompiledScript>,
    pub template: CompiledScript,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RenderedDisplayEntry {
    pub name: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct CompiledStaticDisplay {
    pub template: CompiledScript,
}

#[derive(Debug, Clone)]
pub struct CompiledIterableDisplay {
    pub iterator: CompiledScript,
    pub entries: Vec<CompiledDisplayEntry>,
    pub item_ty: Ty,
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

fn compile_template_as_script(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
) -> Result<CompiledScript, OrchError> {
    let block = compile_template(interner, source, 0, registry)?;
    Ok(CompiledScript {
        module: block.module,
        context_keys: block.context_keys,
        val_def: block.val_def,
    })
}

fn compile_entries(
    interner: &Interner,
    entries: &[DisplayEntrySpec],
    registry: &ContextTypeRegistry,
) -> Result<Vec<CompiledDisplayEntry>, Vec<OrchError>> {
    let mut compiled = Vec::new();
    let mut errors = Vec::new();

    for spec in entries {
        let condition = if spec.condition.trim().is_empty() {
            None
        } else {
            match compile_script_with_hint(
                interner,
                &spec.condition,
                registry,
                Some(&Ty::Bool),
            ) {
                Ok((script, _)) => Some(script),
                Err(e) => {
                    errors.push(e);
                    continue;
                }
            }
        };
        let template =
            match compile_template_as_script(interner, &spec.template, registry) {
                Ok(t) => t,
                Err(e) => {
                    errors.push(e);
                    continue;
                }
            };
        compiled.push(CompiledDisplayEntry {
            name: spec.name.clone(),
            condition,
            template,
        });
    }

    if errors.is_empty() {
        Ok(compiled)
    } else {
        Err(errors)
    }
}

pub fn compile_static_display(
    interner: &Interner,
    spec: &StaticDisplaySpec,
    registry: &ContextTypeRegistry,
) -> Result<CompiledStaticDisplay, Vec<OrchError>> {
    let template = compile_template_as_script(interner, &spec.template, registry)
        .map_err(|e| vec![e])?;
    Ok(CompiledStaticDisplay { template })
}

pub fn compile_iterable_display(
    interner: &Interner,
    spec: &IterableDisplaySpec,
    registry: &ContextTypeRegistry,
) -> Result<CompiledIterableDisplay, Vec<OrchError>> {
    let (iterator, iter_ty) =
        compile_script(interner, &spec.iterator, registry).map_err(|e| vec![e])?;
    let item_ty = expect_list("display iterator", iter_ty).map_err(|e| vec![e])?;

    let entry_reg = registry.with_extra_scoped([
        (interner.intern("item"), item_ty.clone()),
        (interner.intern("index"), Ty::Int),
    ]).map_err(|e| {
        vec![OrchError::new(crate::error::OrchErrorKind::RegistryConflict {
            key: e.key,
            tier_a: e.tier_a,
            tier_b: e.tier_b,
        })]
    })?;

    let entries = compile_entries(interner, &spec.entries, &entry_reg)?;

    Ok(CompiledIterableDisplay {
        iterator,
        entries,
        item_ty,
    })
}

// ---------------------------------------------------------------------------
// Rendering — storage read-only, no resolve chain
// ---------------------------------------------------------------------------

async fn drive_from_entry<EH>(
    interner: &Interner,
    script: &CompiledScript,
    entry: &(impl EntryRef<'_> + Sync),
    local: &FxHashMap<String, Arc<Value>>,
    extern_handler: &EH,
) -> Value
where
    EH: AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync,
{
    let interp = Interpreter::new(interner, script.module.clone());
    let mut coroutine = interp.execute();
    loop {
        match coroutine.resume().await {
            acvus_interpreter::Stepped::Emit(value) => {
                return value;
            }
            acvus_interpreter::Stepped::NeedContext(request) => {
                let name = interner.resolve(request.name()).to_string();
                let Some(value) = local.get(&name).cloned().or_else(|| entry.get(&name))
                else {
                    return Value::unit();
                };
                request.resolve(value);
            }
            acvus_interpreter::Stepped::NeedExternCall(request) => {
                let name = request.name();
                let args = request.args().to_vec();
                match extern_handler(name, args).await {
                    Ok(value) => request.resolve(Arc::new(value)),
                    Err(e) => panic!("display extern call error: {e}"),
                }
            }
            acvus_interpreter::Stepped::Done => return Value::unit(),
            acvus_interpreter::Stepped::Error(e) => panic!("display runtime error: {e}"),
        }
    }
}

async fn drive_template_from_entry<EH>(
    interner: &Interner,
    script: &CompiledScript,
    entry: &(impl EntryRef<'_> + Sync),
    local: &FxHashMap<String, Arc<Value>>,
    extern_handler: &EH,
) -> String
where
    EH: AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync,
{
    let interp = Interpreter::new(interner, script.module.clone());
    let mut coroutine = interp.execute();
    let mut output = String::new();
    loop {
        match coroutine.resume().await {
            acvus_interpreter::Stepped::Emit(value) => match value {
                Value::Pure(PureValue::String(s)) => output.push_str(&s),
                other => panic!("display template: expected String, got {other:?}"),
            },
            acvus_interpreter::Stepped::NeedContext(request) => {
                let name = interner.resolve(request.name()).to_string();
                let Some(value) = local.get(&name).cloned().or_else(|| entry.get(&name))
                else {
                    break;
                };
                request.resolve(value);
            }
            acvus_interpreter::Stepped::NeedExternCall(request) => {
                let name = request.name();
                let args = request.args().to_vec();
                match extern_handler(name, args).await {
                    Ok(value) => request.resolve(Arc::new(value)),
                    Err(e) => panic!("display extern call error: {e}"),
                }
            }
            acvus_interpreter::Stepped::Done => break,
            acvus_interpreter::Stepped::Error(e) => panic!("display runtime error: {e}"),
        }
    }
    output
}

async fn eval_entries<EH>(
    interner: &Interner,
    entries: &[CompiledDisplayEntry],
    entry: &(impl EntryRef<'_> + Sync),
    local: &FxHashMap<String, Arc<Value>>,
    extern_handler: &EH,
) -> Vec<RenderedDisplayEntry>
where
    EH: AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync,
{
    let mut result = Vec::new();
    for e in entries {
        if let Some(ref cond) = e.condition {
            let val = drive_from_entry(interner, cond, entry, local, extern_handler).await;
            let Value::Pure(PureValue::Bool(true)) = val else {
                continue;
            };
        }
        let content =
            drive_template_from_entry(interner, &e.template, entry, local, extern_handler).await;
        result.push(RenderedDisplayEntry {
            name: e.name.clone(),
            content,
        });
    }
    result
}

/// Render a static display region.
pub async fn render_display<EH>(
    interner: &Interner,
    display: &CompiledStaticDisplay,
    entry: &(impl EntryRef<'_> + Sync),
    extern_handler: &EH,
) -> Vec<RenderedDisplayEntry>
where
    EH: AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync,
{
    let local = FxHashMap::default();
    let content =
        drive_template_from_entry(interner, &display.template, entry, &local, extern_handler).await;
    vec![RenderedDisplayEntry {
        name: String::new(),
        content,
    }]
}

/// Render one index of an iterable display region.
///
/// Evaluates the iterator to get the list, indexes into it,
/// then evaluates each entry's condition/template with `@item` and `@index` injected.
/// Returns rendered HTML strings (entries whose condition is false are skipped).
pub async fn render_display_with_idx<EH>(
    interner: &Interner,
    display: &CompiledIterableDisplay,
    entry: &(impl EntryRef<'_> + Sync),
    index: usize,
    extern_handler: &EH,
) -> Vec<RenderedDisplayEntry>
where
    EH: AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync,
{
    let empty_local = FxHashMap::default();
    let list = drive_from_entry(
        interner,
        &display.iterator,
        entry,
        &empty_local,
        extern_handler,
    )
    .await;

    let items = match list {
        Value::Lazy(LazyValue::List(items)) => items,
        Value::Lazy(LazyValue::Deque(deque)) => deque.into_vec(),
        _ => return Vec::new(),
    };
    let Some(item) = items.into_iter().nth(index) else {
        return Vec::new();
    };

    let mut local = FxHashMap::default();
    local.insert("item".into(), Arc::new(item));
    local.insert("index".into(), Arc::new(Value::int(index as i64)));

    eval_entries(interner, &display.entries, entry, &local, extern_handler).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_mir::context_registry::ContextTypeRegistry;
    use crate::storage::{EntryMut, Journal, TreeJournal};
    use acvus_utils::Interner;

    async fn noop_extern(_: Astr, _: Vec<Value>) -> Result<Value, RuntimeError> {
        Ok(Value::unit())
    }

    async fn journal_with(entries: Vec<(&str, Value)>) -> (TreeJournal, uuid::Uuid) {
        let (mut j, root) = TreeJournal::new();
        let mut e = j.entry_mut(root).await.next().await;
        let uuid = e.uuid();
        for (k, v) in entries {
            e.apply(k, crate::StorageDiff::Snapshot(v));
        }
        drop(e);
        (j, uuid)
    }

    #[test]
    fn compile_static_display_ok() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("name"), Ty::String);
        let reg = ContextTypeRegistry::all_system(ctx);
        let spec = StaticDisplaySpec {
            template: "hello {{ @name }}".into(),
        };
        let result = compile_static_display(&interner, &spec, &reg);
        assert!(result.is_ok());
    }

    #[test]
    fn compile_iterable_display_ok() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("items"), Ty::List(Box::new(Ty::String)));
        let reg = ContextTypeRegistry::all_system(ctx);
        let spec = IterableDisplaySpec {
            iterator: "@items".into(),
            entries: vec![DisplayEntrySpec {
                name: String::new(),
                condition: String::new(),
                template: "{{ @item }}".into(),
            }],
        };
        let result = compile_iterable_display(&interner, &spec, &reg);
        assert!(result.is_ok());
    }

    #[test]
    fn compile_iterable_display_type_error() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("items"), Ty::String);
        let reg = ContextTypeRegistry::all_system(ctx);
        let spec = IterableDisplaySpec {
            iterator: "@items".into(),
            entries: vec![],
        };
        let result = compile_iterable_display(&interner, &spec, &reg);
        assert!(result.is_err(), "iterator must be List<_>");
    }

    #[tokio::test]
    async fn render_static() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("greeting"), Ty::String);
        let reg = ContextTypeRegistry::all_system(ctx);
        let spec = StaticDisplaySpec {
            template: "hello {{ @greeting }}".into(),
        };
        let compiled = compile_static_display(&interner, &spec, &reg).unwrap();
        let (journal, uuid) = journal_with(vec![("greeting", Value::string("world".into()))]).await;
        let result = render_display(&interner, &compiled, &journal.entry(uuid).await, &noop_extern).await;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "hello world");
        assert_eq!(result[0].name, "");
    }

    #[tokio::test]
    async fn render_iterable_with_idx() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("messages"), Ty::List(Box::new(Ty::String)));
        let reg = ContextTypeRegistry::all_system(ctx);
        let spec = IterableDisplaySpec {
            iterator: "@messages".into(),
            entries: vec![DisplayEntrySpec {
                name: "msg".into(),
                condition: String::new(),
                template: "msg: {{ @item }}".into(),
            }],
        };
        let compiled = compile_iterable_display(&interner, &spec, &reg).unwrap();
        let (journal, uuid) = journal_with(vec![(
            "messages",
            Value::list(vec![
                Value::string("a".into()),
                Value::string("b".into()),
                Value::string("c".into()),
            ]),
        )])
        .await;
        let r0 =
            render_display_with_idx(&interner, &compiled, &journal.entry(uuid).await, 0, &noop_extern).await;
        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].name, "msg");
        assert_eq!(r0[0].content, "msg: a");
        let r2 =
            render_display_with_idx(&interner, &compiled, &journal.entry(uuid).await, 2, &noop_extern).await;
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].content, "msg: c");
    }

    #[tokio::test]
    async fn render_iterable_condition_filters() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("nums"), Ty::List(Box::new(Ty::Int)));
        let reg = ContextTypeRegistry::all_system(ctx);
        let spec = IterableDisplaySpec {
            iterator: "@nums".into(),
            entries: vec![
                DisplayEntrySpec {
                    name: "big".into(),
                    condition: "@item > 5".into(),
                    template: "big: {{ @item | to_string }}".into(),
                },
                DisplayEntrySpec {
                    name: "all".into(),
                    condition: String::new(),
                    template: "all: {{ @item | to_string }}".into(),
                },
            ],
        };
        let compiled = compile_iterable_display(&interner, &spec, &reg).unwrap();
        let (journal, uuid) = journal_with(vec![(
            "nums",
            Value::list(vec![Value::int(3), Value::int(10)]),
        )])
        .await;

        let r0 =
            render_display_with_idx(&interner, &compiled, &journal.entry(uuid).await, 0, &noop_extern).await;
        assert_eq!(r0.len(), 1, "3 <= 5, first entry skipped");
        assert_eq!(r0[0].content, "all: 3");

        let r1 =
            render_display_with_idx(&interner, &compiled, &journal.entry(uuid).await, 1, &noop_extern).await;
        assert_eq!(r1.len(), 2, "10 > 5, both entries pass");
        assert_eq!(r1[0].content, "big: 10");
        assert_eq!(r1[1].content, "all: 10");
    }

    #[tokio::test]
    async fn render_iterable_out_of_bounds() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("items"), Ty::List(Box::new(Ty::String)));
        let reg = ContextTypeRegistry::all_system(ctx);
        let spec = IterableDisplaySpec {
            iterator: "@items".into(),
            entries: vec![DisplayEntrySpec {
                name: String::new(),
                condition: String::new(),
                template: "{{ @item }}".into(),
            }],
        };
        let compiled = compile_iterable_display(&interner, &spec, &reg).unwrap();
        let (journal, uuid) = journal_with(vec![(
            "items",
            Value::list(vec![Value::string("only".into())]),
        )])
        .await;
        let result =
            render_display_with_idx(&interner, &compiled, &journal.entry(uuid).await, 99, &noop_extern).await;
        assert!(result.is_empty(), "out of bounds returns empty");
    }
}
