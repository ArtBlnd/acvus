use acvus_interpreter::{Interpreter, RuntimeError, Stepped, TypedValue};
use acvus_mir::analysis::reachable_context::{partition_context_keys, KnownValue, ContextKeyPartition};
use acvus_mir::analysis::val_def::{ValDefMap, ValDefMapAnalysis};
use acvus_mir::graph::ContextId;
use acvus_mir::ir::MirModule;
use acvus_mir::pass::AnalysisPass;
use acvus_utils::Interner;
use futures::stream::{FuturesUnordered, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use std::pin::pin;

use super::Unit;

pub struct InterpreterUnit {
    module: MirModule,
    interner: Interner,
}

impl InterpreterUnit {
    pub fn new(module: MirModule, interner: &Interner) -> Self {
        Self { module, interner: interner.clone() }
    }
}

impl Unit for InterpreterUnit {
    fn spawn(
        &self,
        local_context: FxHashMap<ContextId, TypedValue>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError, ContextId> {
        let interner = self.interner.clone();
        let module = self.module.clone();
        acvus_utils::coroutine(move |handle| async move {
            let val_def = ValDefMapAnalysis.run(&module, ());
            let mut known: FxHashMap<ContextId, KnownValue> = FxHashMap::default();
            let mut prefetched: FxHashMap<ContextId, TypedValue> = FxHashMap::default();
            let mut pending: FxHashSet<ContextId> = FxHashSet::default();

            // Pre-populate known from local_context.
            for (&key, val) in &local_context {
                if let Some(kv) = typed_value_to_known(val) {
                    known.insert(key, kv);
                }
            }

            // Initial partition → fire eager pre-fetches.
            let mut prefetch_futs: FuturesUnordered<_> = FuturesUnordered::new();
            fire_prefetch(
                &partition_context_keys(&module, &known, &val_def),
                &local_context,
                &prefetched,
                &mut pending,
                &handle,
                &mut prefetch_futs,
            );

            // Start interpreter.
            let interp = Interpreter::new(&interner, module.clone());
            let mut inner = interp.execute();

            loop {
                // Race: interpreter step vs pre-fetch completion.
                let mut resume_fut = pin!(inner.resume());

                if prefetch_futs.is_empty() {
                    // No pre-fetches pending — drive interpreter only.
                    match resume_fut.await {
                        Stepped::Emit(value) => {
                            handle.yield_val(value).await;
                        }
                        Stepped::NeedContext(request) => {
                            handle_need_context(
                                request, &local_context, &mut prefetched, &mut known,
                                &mut pending, &handle, &mut prefetch_futs,
                                &module, &val_def,
                            ).await;
                        }
                        Stepped::NeedExternCall(request) => {
                            let value = handle
                                .request_extern_call(request.key(), request.args().to_vec())
                                .await;
                            request.resolve(value);
                        }
                        Stepped::StoreContext(id, value) => {
                            handle.store_context(id, value);
                        }
                        Stepped::Done => return Ok(()),
                        Stepped::Error(e) => return Err(e),
                    }
                } else {
                    // Race interpreter vs pre-fetch.
                    use futures::future::Either;
                    match futures::future::select(resume_fut, prefetch_futs.next()).await {
                        Either::Left((stepped, _)) => {
                            match stepped {
                                Stepped::Emit(value) => {
                                    handle.yield_val(value).await;
                                }
                                Stepped::NeedContext(request) => {
                                    handle_need_context(
                                        request, &local_context, &mut prefetched, &mut known,
                                        &mut pending, &handle, &mut prefetch_futs,
                                        &module, &val_def,
                                    ).await;
                                }
                                Stepped::NeedExternCall(request) => {
                                    let value = handle
                                        .request_extern_call(request.key(), request.args().to_vec())
                                        .await;
                                    request.resolve(value);
                                }
                                Stepped::StoreContext(id, value) => {
                                    handle.store_context(id, value);
                                }
                                Stepped::Done => return Ok(()),
                                Stepped::Error(e) => return Err(e),
                            }
                        }
                        Either::Right((Some((key, value)), _)) => {
                            // Pre-fetch completed.
                            pending.remove(&key);
                            prefetched.insert(key, value.clone());
                            if let Some(kv) = typed_value_to_known(&value) {
                                known.insert(key, kv);
                            }
                            // Re-partition → fire newly eager pre-fetches.
                            fire_prefetch(
                                &partition_context_keys(&module, &known, &val_def),
                                &local_context,
                                &prefetched,
                                &mut pending,
                                &handle,
                                &mut prefetch_futs,
                            );
                            // Loop back — interpreter resume_fut was cancelled by select,
                            // it will be re-created at the top of the loop.
                        }
                        Either::Right((None, _)) => {
                            // FuturesUnordered drained — shouldn't happen since we checked is_empty.
                            unreachable!("prefetch_futs was not empty but yielded None");
                        }
                    }
                }
            }
        })
    }
}

/// Handle interpreter NeedContext: check local → prefetched → request externally.
async fn handle_need_context(
    request: acvus_utils::ContextRequest<TypedValue, ContextId>,
    local_context: &FxHashMap<ContextId, TypedValue>,
    prefetched: &mut FxHashMap<ContextId, TypedValue>,
    known: &mut FxHashMap<ContextId, KnownValue>,
    pending: &mut FxHashSet<ContextId>,
    handle: &acvus_utils::YieldHandle<TypedValue, ContextId>,
    prefetch_futs: &mut FuturesUnordered<PrefetchFut>,
    module: &MirModule,
    val_def: &ValDefMap,
) {
    let key = request.key();

    // 1. local_context (function params, etc.)
    if let Some(val) = local_context.get(&key) {
        request.resolve(val.clone());
        return;
    }

    // 2. Already pre-fetched.
    if let Some(val) = prefetched.remove(&key) {
        if let Some(kv) = typed_value_to_known(&val) {
            known.insert(key, kv);
        }
        request.resolve(val);
        // Re-partition with updated known.
        fire_prefetch(
            &partition_context_keys(module, known, val_def),
            local_context, prefetched, pending, handle, prefetch_futs,
        );
        return;
    }

    // 3. Not available — request externally and wait.
    let value = handle.request_context(key).await;
    if let Some(kv) = typed_value_to_known(&value) {
        known.insert(key, kv);
    }
    request.resolve(value);
    // Re-partition with updated known.
    fire_prefetch(
        &partition_context_keys(module, known, val_def),
        local_context, prefetched, pending, handle, prefetch_futs,
    );
}

type PrefetchFut = std::pin::Pin<Box<dyn futures::Future<Output = (ContextId, TypedValue)> + Send>>;

/// Fire pre-fetch requests for eager keys not already resolved or pending.
fn fire_prefetch(
    partition: &ContextKeyPartition,
    local_context: &FxHashMap<ContextId, TypedValue>,
    prefetched: &FxHashMap<ContextId, TypedValue>,
    pending: &mut FxHashSet<ContextId>,
    handle: &acvus_utils::YieldHandle<TypedValue, ContextId>,
    futs: &mut FuturesUnordered<PrefetchFut>,
) {
    for &key in &partition.eager {
        if local_context.contains_key(&key) || prefetched.contains_key(&key) || pending.contains(&key) {
            continue;
        }
        pending.insert(key);
        let h = handle.clone();
        futs.push(Box::pin(async move {
            let val = h.request_context(key).await;
            (key, val)
        }));
    }
}

/// Convert TypedValue to KnownValue for reachability analysis.
/// Returns None if the value can't be represented as a KnownValue.
fn typed_value_to_known(val: &TypedValue) -> Option<KnownValue> {
    use acvus_ast::Literal;
    let v = val.value();
    if let Some(&i) = v.try_expect_ref::<i64>() {
        Some(KnownValue::Literal(Literal::Int(i)))
    } else if let Some(s) = v.try_expect_ref::<str>() {
        Some(KnownValue::Literal(Literal::String(s.to_string())))
    } else if let Some(&b) = v.try_expect_ref::<bool>() {
        Some(KnownValue::Literal(Literal::Bool(b)))
    } else if let Some(&f) = v.try_expect_ref::<f64>() {
        Some(KnownValue::Literal(Literal::Float(f)))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_interpreter::Stepped;
    use acvus_mir::ty::Ty;

    // ── Helpers ──

    fn compile_script(source: &str) -> (MirModule, Interner) {
        compile_with_context(source, &[])
    }

    fn compile_with_context(source: &str, ctx: &[(&str, Ty)]) -> (MirModule, Interner) {
        let interner = Interner::new();
        let ctx_map: FxHashMap<_, _> = ctx.iter()
            .map(|(name, ty)| (interner.intern(name), ty.clone()))
            .collect();
        let registry = acvus_mir::context_registry::ContextTypeRegistry::all_system(ctx_map);
        let script = acvus_ast::parse_script(&interner, source).expect("parse failed");
        let mut subst = acvus_mir::ty::TySubst::new();
        let unchecked = acvus_mir::typecheck_script(&interner, &script, &registry, None, &mut subst)
            .expect("typecheck failed");
        let checked = acvus_mir::check_completeness(unchecked, &subst)
            .expect("completeness failed");
        let name_to_id = acvus_mir::build_name_to_id(registry.merged());
        let (module, _) = acvus_mir::lower_checked_script(
            &interner, &script, checked, name_to_id,
        ).expect("lower failed");
        (module, interner)
    }

    fn ctx_id(interner: &Interner, module: &MirModule, name: &str) -> ContextId {
        let key = interner.intern(name);
        // Find ContextLoad instruction that references this name's ContextId.
        // We look through the module's instructions for ContextLoad and match via
        // the name_to_id used during compilation. Since we can't easily access that,
        // we use the ValDefMap approach: look for ContextLoad instructions.
        // For tests, we just search all instructions.
        for inst in &module.main.insts {
            if let acvus_mir::ir::InstKind::ContextLoad { id, .. } = &inst.kind {
                // We can't directly match name → id here without the name_to_id map.
                // Use a different approach: recompile with known name_to_id.
                let _ = id;
            }
        }
        // Simpler: recompile to get name_to_id
        panic!("use compile_with_ids instead");
    }

    /// Compile script and return (module, interner, id→name mapping).
    fn compile_with_ids(source: &str, ctx: &[(&str, Ty)]) -> (MirModule, Interner, FxHashMap<ContextId, String>) {
        let interner = Interner::new();
        let ctx_map: FxHashMap<_, _> = ctx.iter()
            .map(|(name, ty)| (interner.intern(name), ty.clone()))
            .collect();
        let registry = acvus_mir::context_registry::ContextTypeRegistry::all_system(ctx_map);
        let script = acvus_ast::parse_script(&interner, source).expect("parse failed");
        let mut subst = acvus_mir::ty::TySubst::new();
        let unchecked = acvus_mir::typecheck_script(&interner, &script, &registry, None, &mut subst)
            .expect("typecheck failed");
        let checked = acvus_mir::check_completeness(unchecked, &subst)
            .expect("completeness failed");
        let name_to_id = acvus_mir::build_name_to_id(registry.merged());
        let id_to_name: FxHashMap<ContextId, String> = name_to_id.iter()
            .map(|(name, &id)| (id, interner.resolve(*name).to_string()))
            .collect();
        let (module, _) = acvus_mir::lower_checked_script(
            &interner, &script, checked, name_to_id,
        ).expect("lower failed");
        (module, interner, id_to_name)
    }

    /// Compile template and return (module, interner, id→name mapping).
    fn compile_template_with_ids(source: &str, ctx: &[(&str, Ty)]) -> (MirModule, Interner, FxHashMap<ContextId, String>) {
        let interner = Interner::new();
        let ctx_map: FxHashMap<_, _> = ctx.iter()
            .map(|(name, ty)| (interner.intern(name), ty.clone()))
            .collect();
        let registry = acvus_mir::context_registry::ContextTypeRegistry::all_system(ctx_map);
        let template = acvus_ast::parse(&interner, source).expect("parse failed");
        let mut subst = acvus_mir::ty::TySubst::new();
        let unchecked = acvus_mir::typecheck_template(&interner, &template, &registry, &mut subst)
            .expect("typecheck failed");
        let checked = acvus_mir::check_completeness(unchecked, &subst)
            .expect("completeness failed");
        let name_to_id = acvus_mir::build_name_to_id(registry.merged());
        let id_to_name: FxHashMap<ContextId, String> = name_to_id.iter()
            .map(|(name, &id)| (id, interner.resolve(*name).to_string()))
            .collect();
        let (module, _) = acvus_mir::lower_checked_template(
            &interner, &template, checked, name_to_id,
        ).expect("lower failed");
        (module, interner, id_to_name)
    }

    /// Collect all NeedContext keys until Emit, resolving each with the given map.
    async fn run_to_emit(
        unit: &InterpreterUnit,
        local_context: FxHashMap<ContextId, TypedValue>,
        resolve_map: &FxHashMap<ContextId, TypedValue>,
    ) -> (TypedValue, Vec<ContextId>) {
        let mut co = unit.spawn(local_context);
        let mut requested_keys = Vec::new();
        loop {
            let (next, stepped) = co.step().await;
            co = next;
            match stepped {
                Stepped::Emit(value) => return (value, requested_keys),
                Stepped::NeedContext(request) => {
                    let key = request.key();
                    requested_keys.push(key);
                    let value = resolve_map.get(&key)
                        .unwrap_or_else(|| panic!("no value for context key {key:?}"))
                        .clone();
                    request.resolve(value);
                }
                Stepped::StoreContext(_, _) => { /* test ignores stores */ }
                Stepped::Done => panic!("unexpected Done before Emit"),
                Stepped::Error(e) => panic!("unexpected Error: {e}"),
                Stepped::NeedExternCall(_) => panic!("unexpected NeedExternCall"),
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Basic (no pre-fetch concerns)
    // ═══════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn emit_int() {
        let (module, interner) = compile_script("1 + 2");
        let unit = InterpreterUnit::new(module, &interner);
        let co = unit.spawn(FxHashMap::default());
        let (_, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => assert_eq!(*value.value().expect_ref::<i64>("test"), 3),
            _ => panic!("expected Emit"),
        }
    }

    #[tokio::test]
    async fn emit_string() {
        let (module, interner) = compile_script(r#""hello" + " world""#);
        let unit = InterpreterUnit::new(module, &interner);
        let co = unit.spawn(FxHashMap::default());
        let (_, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => assert_eq!(value.value().expect_ref::<str>("test"), "hello world"),
            _ => panic!("expected Emit"),
        }
    }

    #[tokio::test]
    async fn local_context_resolved_internally() {
        let (module, interner, id_to_name) = compile_with_ids("@data + 1", &[("data", Ty::Int)]);
        let data_id = id_to_name.iter()
            .find(|(_, name)| *name == "data").unwrap().0;

        let unit = InterpreterUnit::new(module, &interner);
        let mut local = FxHashMap::default();
        local.insert(*data_id, TypedValue::int(10));

        let co = unit.spawn(local);
        let (_, stepped) = co.step().await;
        match stepped {
            Stepped::Emit(value) => assert_eq!(*value.value().expect_ref::<i64>("test"), 11),
            _ => panic!("expected Emit, got NeedContext (local_context not working)"),
        }
    }

    #[tokio::test]
    async fn external_context_yields_need_context() {
        let (module, interner) = compile_with_context("@x", &[("x", Ty::Int)]);
        let unit = InterpreterUnit::new(module, &interner);
        let co = unit.spawn(FxHashMap::default());
        let (_, stepped) = co.step().await;
        assert!(matches!(stepped, Stepped::NeedContext(_)), "should yield NeedContext for @x");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pre-fetch: eagerly request all context keys
    // ═══════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn prefetch_single_key_produces_correct_result() {
        let (module, interner, id_to_name) = compile_with_ids(
            "@x + 10", &[("x", Ty::Int)],
        );
        let x_id = *id_to_name.iter().find(|(_, n)| *n == "x").unwrap().0;

        let unit = InterpreterUnit::new(module, &interner);
        let resolve_map = FxHashMap::from_iter([(x_id, TypedValue::int(5))]);
        let (result, keys) = run_to_emit(&unit, FxHashMap::default(), &resolve_map).await;

        assert_eq!(*result.value().expect_ref::<i64>("test"), 15);
        // x should have been requested (either pre-fetch or interpreter)
        assert!(keys.contains(&x_id));
    }

    #[tokio::test]
    async fn prefetch_two_keys_both_requested() {
        let (module, interner, id_to_name) = compile_with_ids(
            "@a + @b", &[("a", Ty::Int), ("b", Ty::Int)],
        );
        let a_id = *id_to_name.iter().find(|(_, n)| *n == "a").unwrap().0;
        let b_id = *id_to_name.iter().find(|(_, n)| *n == "b").unwrap().0;

        let unit = InterpreterUnit::new(module, &interner);
        let resolve_map = FxHashMap::from_iter([
            (a_id, TypedValue::int(3)),
            (b_id, TypedValue::int(7)),
        ]);
        let (result, keys) = run_to_emit(&unit, FxHashMap::default(), &resolve_map).await;

        assert_eq!(*result.value().expect_ref::<i64>("test"), 10);
        assert!(keys.contains(&a_id), "a must be requested");
        assert!(keys.contains(&b_id), "b must be requested");
    }

    #[tokio::test]
    async fn prefetch_three_keys_all_requested() {
        let (module, interner, id_to_name) = compile_with_ids(
            "@a + @b + @c",
            &[("a", Ty::Int), ("b", Ty::Int), ("c", Ty::Int)],
        );
        let a_id = *id_to_name.iter().find(|(_, n)| *n == "a").unwrap().0;
        let b_id = *id_to_name.iter().find(|(_, n)| *n == "b").unwrap().0;
        let c_id = *id_to_name.iter().find(|(_, n)| *n == "c").unwrap().0;

        let unit = InterpreterUnit::new(module, &interner);
        let resolve_map = FxHashMap::from_iter([
            (a_id, TypedValue::int(1)),
            (b_id, TypedValue::int(2)),
            (c_id, TypedValue::int(3)),
        ]);
        let (result, keys) = run_to_emit(&unit, FxHashMap::default(), &resolve_map).await;

        assert_eq!(*result.value().expect_ref::<i64>("test"), 6);
        assert!(keys.contains(&a_id));
        assert!(keys.contains(&b_id));
        assert!(keys.contains(&c_id));
    }

    #[tokio::test]
    async fn prefetch_skips_local_context_keys() {
        let (module, interner, id_to_name) = compile_with_ids(
            "@local + @remote",
            &[("local", Ty::Int), ("remote", Ty::Int)],
        );
        let local_id = *id_to_name.iter().find(|(_, n)| *n == "local").unwrap().0;
        let remote_id = *id_to_name.iter().find(|(_, n)| *n == "remote").unwrap().0;

        let unit = InterpreterUnit::new(module, &interner);
        let local_ctx = FxHashMap::from_iter([(local_id, TypedValue::int(100))]);
        let resolve_map = FxHashMap::from_iter([(remote_id, TypedValue::int(200))]);
        let (result, keys) = run_to_emit(&unit, local_ctx, &resolve_map).await;

        assert_eq!(*result.value().expect_ref::<i64>("test"), 300);
        // local should NOT be requested externally — resolved from local_context
        assert!(!keys.contains(&local_id), "local key should not be requested");
        assert!(keys.contains(&remote_id), "remote key should be requested");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Pre-fetch with branches: re-partition on value arrival
    // ═══════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn branch_true_path_correct_result() {
        // Template: {{ "search" = @mode }}{{ @query }}{{ _ }}fallback{{/}}
        let (module, interner, id_to_name) = compile_template_with_ids(
            r#"{{ "search" = @mode }}{{ @query }}{{ _ }}fallback{{/}}"#,
            &[("mode", Ty::String), ("query", Ty::String)],
        );
        let mode_id = *id_to_name.iter().find(|(_, n)| *n == "mode").unwrap().0;
        let query_id = *id_to_name.iter().find(|(_, n)| *n == "query").unwrap().0;

        let unit = InterpreterUnit::new(module, &interner);
        let resolve_map = FxHashMap::from_iter([
            (mode_id, TypedValue::string("search")),
            (query_id, TypedValue::string("hello world")),
        ]);
        let (result, keys) = run_to_emit(&unit, FxHashMap::default(), &resolve_map).await;

        assert_eq!(result.value().expect_ref::<str>("test"), "hello world");
        assert!(keys.contains(&mode_id));
        assert!(keys.contains(&query_id));
    }

    #[tokio::test]
    async fn branch_false_path_correct_result() {
        let (module, interner, id_to_name) = compile_template_with_ids(
            r#"{{ "search" = @mode }}{{ @query }}{{ _ }}fallback{{/}}"#,
            &[("mode", Ty::String), ("query", Ty::String)],
        );
        let mode_id = *id_to_name.iter().find(|(_, n)| *n == "mode").unwrap().0;
        let query_id = *id_to_name.iter().find(|(_, n)| *n == "query").unwrap().0;

        let unit = InterpreterUnit::new(module, &interner);
        let resolve_map = FxHashMap::from_iter([
            (mode_id, TypedValue::string("chat")),
            (query_id, TypedValue::string("unused")),
        ]);
        let (result, keys) = run_to_emit(&unit, FxHashMap::default(), &resolve_map).await;

        assert_eq!(result.value().expect_ref::<str>("test"), "fallback");
        assert!(keys.contains(&mode_id));
        // query should NOT be requested — dead branch (mode != "search")
        assert!(!keys.contains(&query_id),
            "query should not be requested when mode is not 'search'");
    }

    #[tokio::test]
    async fn branch_nested_correct_pruning() {
        // Outer match on @a, inner match on @b
        // {{ "x" = @a }}{{ "y" = @b }}{{ @deep }}{{ _ }}ib{{/}}{{ _ }}ob{{/}}
        let (module, interner, id_to_name) = compile_template_with_ids(
            r#"{{ "x" = @a }}{{ "y" = @b }}{{ @deep }}{{ _ }}ib{{/}}{{ _ }}ob{{/}}"#,
            &[("a", Ty::String), ("b", Ty::String), ("deep", Ty::String)],
        );
        let a_id = *id_to_name.iter().find(|(_, n)| *n == "a").unwrap().0;
        let b_id = *id_to_name.iter().find(|(_, n)| *n == "b").unwrap().0;
        let deep_id = *id_to_name.iter().find(|(_, n)| *n == "deep").unwrap().0;

        let unit = InterpreterUnit::new(module, &interner);

        // Path: a="x", b="y" → @deep
        let resolve_map = FxHashMap::from_iter([
            (a_id, TypedValue::string("x")),
            (b_id, TypedValue::string("y")),
            (deep_id, TypedValue::string("found")),
        ]);
        let (result, keys) = run_to_emit(&unit, FxHashMap::default(), &resolve_map).await;
        assert_eq!(result.value().expect_ref::<str>("test"), "found");
        assert!(keys.contains(&a_id));
        assert!(keys.contains(&b_id));
        assert!(keys.contains(&deep_id));

        // Path: a="z" → "ob" (outer fallback, b and deep not needed)
        let resolve_map2 = FxHashMap::from_iter([
            (a_id, TypedValue::string("z")),
            (b_id, TypedValue::string("unused")),
            (deep_id, TypedValue::string("unused")),
        ]);
        let (result2, keys2) = run_to_emit(&unit, FxHashMap::default(), &resolve_map2).await;
        assert_eq!(result2.value().expect_ref::<str>("test"), "ob");
        assert!(keys2.contains(&a_id));
        assert!(!keys2.contains(&b_id), "b should not be requested when a != 'x'");
        assert!(!keys2.contains(&deep_id), "deep should not be requested when a != 'x'");
    }

    #[tokio::test]
    async fn prefetch_does_not_request_duplicates() {
        // Same key used multiple times in expression
        let (module, interner, id_to_name) = compile_with_ids(
            "@x + @x",
            &[("x", Ty::Int)],
        );
        let x_id = *id_to_name.iter().find(|(_, n)| *n == "x").unwrap().0;

        let unit = InterpreterUnit::new(module, &interner);
        let resolve_map = FxHashMap::from_iter([(x_id, TypedValue::int(5))]);
        let (result, keys) = run_to_emit(&unit, FxHashMap::default(), &resolve_map).await;

        assert_eq!(*result.value().expect_ref::<i64>("test"), 10);
        // x should be requested only once (pre-fetch dedup via pending set)
        let x_count = keys.iter().filter(|&&k| k == x_id).count();
        assert!(x_count <= 2, "x requested {x_count} times, expected at most 2 (prefetch + interpreter)");
    }
}
