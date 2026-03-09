use std::sync::Arc;

use acvus_interpreter::{ExternFnRegistry, Interpreter, Value};
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::compile::{
    CompiledScript, compile_script, compile_script_with_hint, compile_template, expect_list,
};
use crate::error::OrchError;
use crate::storage::Storage;

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
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
) -> Result<CompiledScript, OrchError> {
    let block = compile_template(interner, source, 0, context_types, registry)?;
    Ok(CompiledScript {
        module: block.module,
        context_keys: block.context_keys,
        val_def: block.val_def,
    })
}

fn compile_entries(
    interner: &Interner,
    entries: &[DisplayEntrySpec],
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
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
                context_types,
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
            match compile_template_as_script(interner, &spec.template, context_types, registry) {
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
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
) -> Result<CompiledStaticDisplay, Vec<OrchError>> {
    let template = compile_template_as_script(interner, &spec.template, context_types, registry)
        .map_err(|e| vec![e])?;
    Ok(CompiledStaticDisplay { template })
}

pub fn compile_iterable_display(
    interner: &Interner,
    spec: &IterableDisplaySpec,
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
) -> Result<CompiledIterableDisplay, Vec<OrchError>> {
    let (iterator, iter_ty) =
        compile_script(interner, &spec.iterator, context_types, registry).map_err(|e| vec![e])?;
    let item_ty = expect_list("display iterator", iter_ty).map_err(|e| vec![e])?;

    let mut entry_ctx = context_types.clone();
    entry_ctx.insert(interner.intern("item"), item_ty.clone());
    entry_ctx.insert(interner.intern("index"), Ty::Int);

    let entries = compile_entries(interner, &spec.entries, &entry_ctx, registry)?;

    Ok(CompiledIterableDisplay {
        iterator,
        entries,
        item_ty,
    })
}

// ---------------------------------------------------------------------------
// Rendering — storage read-only, no resolve chain
// ---------------------------------------------------------------------------

async fn drive_from_storage<S>(
    interner: &Interner,
    script: &CompiledScript,
    storage: &S,
    local: &FxHashMap<String, Arc<Value>>,
    extern_fns: &ExternFnRegistry,
) -> Value
where
    S: Storage,
{
    let interp = Interpreter::new(interner, script.module.clone(), extern_fns);
    let (mut coroutine, mut key) = interp.execute();
    loop {
        match coroutine.resume(key).await {
            acvus_interpreter::Stepped::Emit(emit) => {
                let (value, _) = emit.into_parts();
                return value;
            }
            acvus_interpreter::Stepped::NeedContext(need) => {
                let name = interner.resolve(need.name()).to_string();
                let Some(value) = local.get(&name).cloned().or_else(|| storage.get(&name)) else {
                    return Value::Unit;
                };
                key = need.into_key(value);
            }
            acvus_interpreter::Stepped::Done => return Value::Unit,
            acvus_interpreter::Stepped::Error(e) => panic!("display runtime error: {e}"),
        }
    }
}

async fn drive_template_from_storage<S>(
    interner: &Interner,
    script: &CompiledScript,
    storage: &S,
    local: &FxHashMap<String, Arc<Value>>,
    extern_fns: &ExternFnRegistry,
) -> String
where
    S: Storage,
{
    let interp = Interpreter::new(interner, script.module.clone(), extern_fns);
    let (mut coroutine, mut key) = interp.execute();
    let mut output = String::new();
    loop {
        match coroutine.resume(key).await {
            acvus_interpreter::Stepped::Emit(emit) => {
                let (value, next_key) = emit.into_parts();
                match value {
                    Value::String(s) => output.push_str(&s),
                    other => panic!("display template: expected String, got {other:?}"),
                }
                key = next_key;
            }
            acvus_interpreter::Stepped::NeedContext(need) => {
                let name = interner.resolve(need.name()).to_string();
                let Some(value) = local.get(&name).cloned().or_else(|| storage.get(&name)) else {
                    break;
                };
                key = need.into_key(value);
            }
            acvus_interpreter::Stepped::Done => break,
            acvus_interpreter::Stepped::Error(e) => panic!("display runtime error: {e}"),
        }
    }
    output
}

async fn eval_entries<S>(
    interner: &Interner,
    entries: &[CompiledDisplayEntry],
    storage: &S,
    local: &FxHashMap<String, Arc<Value>>,
    extern_fns: &ExternFnRegistry,
) -> Vec<RenderedDisplayEntry>
where
    S: Storage,
{
    let mut result = Vec::new();
    for entry in entries {
        if let Some(ref cond) = entry.condition {
            let val = drive_from_storage(interner, cond, storage, local, extern_fns).await;
            let Value::Bool(true) = val else {
                continue;
            };
        }
        let content =
            drive_template_from_storage(interner, &entry.template, storage, local, extern_fns)
                .await;
        result.push(RenderedDisplayEntry {
            name: entry.name.clone(),
            content,
        });
    }
    result
}

/// Render a static display region.
pub async fn render_display<S>(
    interner: &Interner,
    display: &CompiledStaticDisplay,
    storage: &S,
    extern_fns: &ExternFnRegistry,
) -> Vec<RenderedDisplayEntry>
where
    S: Storage,
{
    let local = FxHashMap::default();
    let content =
        drive_template_from_storage(interner, &display.template, storage, &local, extern_fns).await;
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
pub async fn render_display_with_idx<S>(
    interner: &Interner,
    display: &CompiledIterableDisplay,
    storage: &S,
    extern_fns: &ExternFnRegistry,
    index: usize,
) -> Vec<RenderedDisplayEntry>
where
    S: Storage,
{
    let empty_local = FxHashMap::default();
    let list = drive_from_storage(
        interner,
        &display.iterator,
        storage,
        &empty_local,
        extern_fns,
    )
    .await;

    let Value::List(items) = list else {
        return Vec::new();
    };
    let Some(item) = items.into_iter().nth(index) else {
        return Vec::new();
    };

    let mut local = FxHashMap::default();
    local.insert("item".into(), Arc::new(item));
    local.insert("index".into(), Arc::new(Value::Int(index as i64)));

    eval_entries(interner, &display.entries, storage, &local, extern_fns).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::HashMapStorage;
    use acvus_utils::Interner;

    fn empty_registry() -> ExternRegistry {
        ExternRegistry::default()
    }

    fn empty_extern_fns() -> ExternFnRegistry {
        let interner = Interner::new();
        ExternFnRegistry::new(&interner)
    }

    fn storage_with(entries: Vec<(&str, Value)>) -> HashMapStorage {
        let mut s = HashMapStorage::new();
        for (k, v) in entries {
            s.set(k.into(), v);
        }
        s
    }

    #[test]
    fn compile_static_display_ok() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("name"), Ty::String);
        let spec = StaticDisplaySpec {
            template: "hello {{ @name }}".into(),
        };
        let result = compile_static_display(&interner, &spec, &ctx, &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn compile_iterable_display_ok() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("items"), Ty::List(Box::new(Ty::String)));
        let spec = IterableDisplaySpec {
            iterator: "@items".into(),
            entries: vec![DisplayEntrySpec {
                name: String::new(),
                condition: String::new(),
                template: "{{ @item }}".into(),
            }],
        };
        let result = compile_iterable_display(&interner, &spec, &ctx, &empty_registry());
        assert!(result.is_ok());
    }

    #[test]
    fn compile_iterable_display_type_error() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("items"), Ty::String);
        let spec = IterableDisplaySpec {
            iterator: "@items".into(),
            entries: vec![],
        };
        let result = compile_iterable_display(&interner, &spec, &ctx, &empty_registry());
        assert!(result.is_err(), "iterator must be List<_>");
    }

    #[tokio::test]
    async fn render_static() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("greeting"), Ty::String);
        let spec = StaticDisplaySpec {
            template: "hello {{ @greeting }}".into(),
        };
        let compiled = compile_static_display(&interner, &spec, &ctx, &empty_registry()).unwrap();
        let storage = storage_with(vec![("greeting", Value::String("world".into()))]);
        let result = render_display(&interner, &compiled, &storage, &empty_extern_fns()).await;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "hello world");
        assert_eq!(result[0].name, "");
    }

    #[tokio::test]
    async fn render_iterable_with_idx() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("messages"), Ty::List(Box::new(Ty::String)));
        let spec = IterableDisplaySpec {
            iterator: "@messages".into(),
            entries: vec![DisplayEntrySpec {
                name: "msg".into(),
                condition: String::new(),
                template: "msg: {{ @item }}".into(),
            }],
        };
        let compiled = compile_iterable_display(&interner, &spec, &ctx, &empty_registry()).unwrap();
        let storage = storage_with(vec![(
            "messages",
            Value::List(vec![
                Value::String("a".into()),
                Value::String("b".into()),
                Value::String("c".into()),
            ]),
        )]);
        let r0 =
            render_display_with_idx(&interner, &compiled, &storage, &empty_extern_fns(), 0).await;
        assert_eq!(r0.len(), 1);
        assert_eq!(r0[0].name, "msg");
        assert_eq!(r0[0].content, "msg: a");
        let r2 =
            render_display_with_idx(&interner, &compiled, &storage, &empty_extern_fns(), 2).await;
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].content, "msg: c");
    }

    #[tokio::test]
    async fn render_iterable_condition_filters() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("nums"), Ty::List(Box::new(Ty::Int)));
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
        let compiled = compile_iterable_display(&interner, &spec, &ctx, &empty_registry()).unwrap();
        let storage = storage_with(vec![(
            "nums",
            Value::List(vec![Value::Int(3), Value::Int(10)]),
        )]);

        let r0 =
            render_display_with_idx(&interner, &compiled, &storage, &empty_extern_fns(), 0).await;
        assert_eq!(r0.len(), 1, "3 <= 5, first entry skipped");
        assert_eq!(r0[0].content, "all: 3");

        let r1 =
            render_display_with_idx(&interner, &compiled, &storage, &empty_extern_fns(), 1).await;
        assert_eq!(r1.len(), 2, "10 > 5, both entries pass");
        assert_eq!(r1[0].content, "big: 10");
        assert_eq!(r1[1].content, "all: 10");
    }

    #[tokio::test]
    async fn render_iterable_out_of_bounds() {
        let interner = Interner::new();
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("items"), Ty::List(Box::new(Ty::String)));
        let spec = IterableDisplaySpec {
            iterator: "@items".into(),
            entries: vec![DisplayEntrySpec {
                name: String::new(),
                condition: String::new(),
                template: "{{ @item }}".into(),
            }],
        };
        let compiled = compile_iterable_display(&interner, &spec, &ctx, &empty_registry()).unwrap();
        let storage = storage_with(vec![(
            "items",
            Value::List(vec![Value::String("only".into())]),
        )]);
        let result =
            render_display_with_idx(&interner, &compiled, &storage, &empty_extern_fns(), 99).await;
        assert!(result.is_empty(), "out of bounds returns empty");
    }
}
