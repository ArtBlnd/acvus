

use std::collections::VecDeque;

use acvus_mir::context_registry::{ContextTypeRegistry, PartialContextTypeRegistry, RegistryConflictError};
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::{Effect, FnKind, Ty};
use acvus_mir_pass::AnalysisPass;
use acvus_mir_pass::analysis::reachable_context::{
    ContextKeyPartition, KnownValue, partition_context_keys, reachable_context_keys,
};
use acvus_mir_pass::analysis::val_def::{ValDefMap, ValDefMapAnalysis};
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::TokenBudget;
use crate::convert::value_to_known;
use crate::dsl::{ContextScope, Execution, FnParam, MessageSpec, NodeLocalTypes, NodeSpec, Persistency, KEY_SELF, KEY_RAW, KEY_TURN_INDEX, KEY_ITEM, KEY_INDEX};
use crate::error::{OrchError, OrchErrorKind};
use crate::spec::{
    CompiledExpression, CompiledNodeKind, NodeKind,
    compile_anthropic, compile_expression, compile_google, compile_google_cache,
    compile_openai, compile_plain, parse_type_name,
};
use crate::storage::EntryRef;

/// Compiled execution strategy.
#[derive(Debug, Clone)]
pub enum CompiledExecution {
    Always,
    OncePerTurn,
    IfModified { key: CompiledScript },
}

/// Persist mode for non-ephemeral nodes.
#[derive(Debug, Clone)]
pub enum PersistMode {
    Sequence,
    Patch,
}

/// Compiled strategy — groups execution, initial_value, retry, and assert.
///
/// Persistency information (mode + bind script) lives in `NodeRole::Persistent`,
/// not here. Only Persistent nodes have that data; Body and Standalone nodes
/// never carry it.
#[derive(Debug, Clone)]
pub struct CompiledStrategy {
    pub execution: CompiledExecution,
    pub initial_value: Option<CompiledScript>,
    pub retry: u32,
    pub assert: Option<CompiledScript>,
}

/// Unique identifier for a node in the compiled graph.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct NodeId(pub usize);

impl NodeId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// Role of a compiled node in the execution graph.
///
/// Determines how the resolver prepares, executes, and finalizes the node.
#[derive(Debug, Clone)]
pub enum NodeRole {
    /// Lazy body node — spawned on-demand when its bind node requests @raw.
    /// Does NOT appear in the DAG. Has no initial_value or IfModified of its own.
    Body { bind_id: NodeId },
    /// Persistent node with storage (Sequence or Patch).
    /// Always the bind half of a body+bind split pair.
    /// Carries the persist mode and compiled bind script.
    Persistent {
        body_id: NodeId,
        mode: PersistMode,
        bind: CompiledScript,
    },
    /// Standalone ephemeral node — no bind accumulation.
    Standalone,
}

/// A compiled orchestration node.
#[derive(Debug, Clone)]
pub struct CompiledNode {
    pub id: NodeId,
    pub name: Astr,
    pub kind: CompiledNodeKind,
    pub all_context_keys: FxHashSet<Astr>,
    pub strategy: CompiledStrategy,
    pub is_function: bool,
    pub fn_params: Vec<FnParam>,
    /// The type of values this node produces (stored or ephemeral).
    /// Used at storage boundaries to construct TypedValue and verify storability.
    pub output_ty: Ty,
    /// Role of this node in the execution graph.
    pub role: NodeRole,
}

/// The compiled node graph with Id-based identification.
///
/// For Sequence/Patch nodes, the DSL-level node is split into two:
/// - **Body node**: executes the computation (LLM/Expr/etc), produces @raw
/// - **Bind node**: executes the bind script, produces the accumulated value
///
/// External consumers see the bind node as the primary (via `name_to_primary`).
/// The body node is an internal dependency resolved via scope-aware NeedContext.
#[derive(Debug, Clone)]
pub struct CompiledNodeGraph {
    pub nodes: Vec<CompiledNode>,
    pub name_to_primary: FxHashMap<Astr, NodeId>,
}

impl CompiledNodeGraph {
    pub fn node(&self, id: NodeId) -> &CompiledNode {
        &self.nodes[id.0]
    }
}

/// Compiled expression (Script → MIR).
#[derive(Debug, Clone)]
pub struct CompiledScript {
    pub module: MirModule,
    pub context_keys: FxHashSet<Astr>,
    pub val_def: ValDefMap,
}

/// A compiled message entry.
#[derive(Debug, Clone)]
pub enum CompiledMessage {
    Block(CompiledBlock),
    Iterator {
        expr: CompiledScript,
        slice: Option<Vec<i64>>,
        role: Option<Astr>,
        token_budget: Option<TokenBudget>,
    },
}

/// A compiled message block within a node.
#[derive(Debug, Clone)]
pub struct CompiledBlock {
    pub role: Astr,
    pub module: MirModule,
    pub context_keys: FxHashSet<Astr>,
    pub val_def: ValDefMap,
}

impl CompiledBlock {
    /// Context keys still needed on live execution paths, given known values.
    ///
    /// Uses dead branch pruning: if a known value resolves a branch condition,
    /// context loads in the dead branch are excluded.
    pub fn required_context_keys(&self, known: &FxHashMap<Astr, KnownValue>) -> FxHashSet<Astr> {
        reachable_context_keys(&self.module, known, &self.val_def)
    }
}

impl CompiledNode {
    /// Context keys still needed across all blocks in this node, given known values.
    ///
    /// Aggregates results from all message blocks and the key module (if present).
    /// Keys in `resolvable` (e.g. dependency node names) are excluded.
    pub fn required_context_keys(
        &self,
        known: &FxHashMap<Astr, KnownValue>,
        resolvable: &FxHashSet<Astr>,
    ) -> FxHashSet<Astr> {
        let mut needed = FxHashSet::default();
        for msg in self.kind.messages() {
            if let CompiledMessage::Block(block) = msg {
                needed.extend(block.required_context_keys(known));
            }
        }
        needed.retain(|k| !resolvable.contains(k));
        needed
    }

    /// Context keys that must be provided externally.
    ///
    /// Reads already-resolved values from `storage` for dead branch pruning,
    /// and excludes keys in `resolvable` (dependency nodes that auto-resolve).
    pub fn required_external_keys(
        &self,
        interner: &Interner,
        entry: &dyn EntryRef<'_>,
        resolvable: &FxHashSet<Astr>,
    ) -> FxHashSet<Astr> {
        let known = self.known_from_entry(interner, entry);
        self.required_context_keys(&known, resolvable)
    }

    /// Partition context keys into eager (definitely needed) and lazy
    /// (conditionally needed), excluding resolvable dependency nodes.
    pub fn partition_external_keys(
        &self,
        interner: &Interner,
        entry: &dyn EntryRef<'_>,
        resolvable: &FxHashSet<Astr>,
    ) -> ContextKeyPartition {
        let known = self.known_from_entry(interner, entry);
        let mut merged = ContextKeyPartition::default();
        for msg in self.kind.messages() {
            if let CompiledMessage::Block(block) = msg {
                let p = partition_context_keys(&block.module, &known, &block.val_def);
                merged.eager.extend(p.eager);
                merged.lazy.extend(p.lazy);
            }
        }
        merged.eager.retain(|k| !resolvable.contains(k));
        merged
            .lazy
            .retain(|k| !resolvable.contains(k) && !merged.eager.contains(k));
        merged
    }

    pub(crate) fn known_from_entry(
        &self,
        interner: &Interner,
        entry: &dyn EntryRef<'_>,
    ) -> FxHashMap<Astr, KnownValue> {
        self.all_context_keys
            .iter()
            .filter_map(|k| {
                let arc = entry.get(interner.resolve(*k))?;
                let known = value_to_known(&arc)?;
                Some((*k, known))
            })
            .collect()
    }
}

/// Compile an expression string (script syntax) with type checking.
/// Returns the compiled script and its tail expression type.
pub fn compile_script(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
) -> Result<(CompiledScript, Ty), OrchError> {
    compile_script_with_hint(interner, source, registry, None)
}

/// Compile a script with an optional expected tail type hint for unification.
pub fn compile_script_with_hint(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
) -> Result<(CompiledScript, Ty), OrchError> {
    let mut subst = acvus_mir::ty::TySubst::new();
    compile_script_with_hint_subst(interner, source, registry, expected_tail, &mut subst)
}

/// Like `compile_script_with_hint`, but uses an externally provided `TySubst`.
/// Allows sharing origin/type-variable state across multiple compilations.
pub(crate) fn compile_script_with_hint_subst(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
    subst: &mut acvus_mir::ty::TySubst,
) -> Result<(CompiledScript, Ty), OrchError> {
    let script = acvus_ast::parse_script(interner, source).map_err(|e| {
        OrchError::new(OrchErrorKind::ScriptParse {
            error: format!("{e}"),
        })
    })?;
    let (module, _hints, tail_ty) = acvus_mir::compile_script_with_hint_subst(
        interner,
        &script,
        registry,
        expected_tail,
        subst,
    )
    .map_err(|errs| {
        OrchError::new(OrchErrorKind::ScriptCompile {
            context: source.to_string(),
            errors: errs,
        })
    })?;
    let context_keys = extract_context_keys(&module);
    let val_def = ValDefMapAnalysis.run(&module, ());
    Ok((
        CompiledScript {
            module,
            context_keys,
            val_def,
        },
        tail_ty,
    ))
}

// ── Script output type expectations ──────────────────────────────────
//
//   Field               Expected type            Notes
//   ──────────────────  ───────────────────────  ──────────────────────────
//   iterator + body     List<T>                  T bound to context for body
//   iterator (no body)  List<MESSAGE_ELEM_TY>    elements used as messages directly
//   cache_key           String
//   bind script         (any)
//

/// Compile a template source string into a `CompiledBlock`.
pub(crate) fn compile_template(
    interner: &Interner,
    source: &str,
    block_idx: usize,
    registry: &ContextTypeRegistry,
) -> Result<CompiledBlock, OrchError> {
    let ast = acvus_ast::parse(interner, source).map_err(|e| {
        OrchError::new(OrchErrorKind::TemplateParse {
            block: block_idx,
            error: format!("{e}"),
        })
    })?;

    let (module, _hints) = acvus_mir::compile(
        interner,
        &ast,
        registry,
    )
    .map_err(|errs| {
        OrchError::new(OrchErrorKind::TemplateCompile {
            block: block_idx,
            errors: errs,
        })
    })?;

    let context_keys = extract_context_keys(&module);
    let val_def = ValDefMapAnalysis.run(&module, ());

    Ok(CompiledBlock {
        role: interner.intern(""),
        module,
        context_keys,
        val_def,
    })
}

/// Compile messages from a message spec list.
pub(crate) fn compile_messages(
    interner: &Interner,
    messages: &[MessageSpec],
    registry: &ContextTypeRegistry,
    iterator_elem_ty: &Ty,
) -> Result<(Vec<CompiledMessage>, FxHashSet<Astr>), Vec<OrchError>> {
    let mut compiled_messages = Vec::new();
    let mut all_context_keys = FxHashSet::default();
    let mut errors = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        match msg {
            MessageSpec::Block { role, source } => {
                let block = match compile_template(interner, source, i, registry) {
                    Ok(b) => b,
                    Err(e) => {
                        errors.push(e);
                        continue;
                    }
                };
                all_context_keys.extend(block.context_keys.iter().copied());
                compiled_messages.push(CompiledMessage::Block(CompiledBlock {
                    role: *role,
                    ..block
                }));
            }
            MessageSpec::Iterator {
                key,
                slice,
                role,
                token_budget,
            } => {
                let _ctx = format!("iterator (block {i})");
                // Hint: Iterator<T> — List/Deque/Sequence all coerce to Iterator.
                // If it doesn't coerce, compile itself fails — no post-check needed.
                let iter_hint = Ty::Iterator(Box::new(iterator_elem_ty.clone()), acvus_mir::ty::Effect::Pure);
                let (expr, _tail_ty) =
                    match compile_script_with_hint(interner, interner.resolve(*key), registry, Some(&iter_hint)) {
                        Ok(v) => v,
                        Err(e) => {
                            errors.push(e);
                            continue;
                        }
                    };

                all_context_keys.extend(expr.context_keys.iter().copied());
                compiled_messages.push(CompiledMessage::Iterator {
                    expr,
                    slice: slice.clone(),
                    role: *role,
                    token_budget: token_budget.clone(),
                });
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    Ok((compiled_messages, all_context_keys))
}

/// Compile a node spec into a `CompiledNode`.
///
/// `registry` must already include @self (if applicable) via `build_node_context`.
/// `initial_value` is the compiled initial_value script.
/// Each message's `source` field is compiled directly — no file I/O.
pub fn compile_node(
    interner: &Interner,
    spec: &NodeSpec,
    registry: &ContextTypeRegistry,
    id: NodeId,
    initial_value: Option<CompiledScript>,
    compiled_execution: CompiledExecution,
    compiled_assert: Option<CompiledScript>,
    output_ty: Ty,
    role: NodeRole,
) -> Result<CompiledNode, Vec<OrchError>> {
    let (kind, mut all_context_keys): (_, FxHashSet<_>) = match &spec.kind {
        NodeKind::Plain(plain_spec) => {
            let (compiled, keys) = compile_plain(interner, plain_spec, registry)?;
            (CompiledNodeKind::Plain(compiled), keys)
        }
        NodeKind::OpenAICompatible(spec) => {
            let (compiled, keys) = compile_openai(interner, spec, registry)?;
            (CompiledNodeKind::OpenAICompatible(compiled), keys)
        }
        NodeKind::Anthropic(spec) => {
            let (compiled, keys) = compile_anthropic(interner, spec, registry)?;
            (CompiledNodeKind::Anthropic(compiled), keys)
        }
        NodeKind::GoogleAI(spec) => {
            let (compiled, keys) = compile_google(interner, spec, registry)?;
            (CompiledNodeKind::GoogleAI(compiled), keys)
        }
        NodeKind::GoogleAICache(spec) => {
            let (compiled, keys) = compile_google_cache(interner, spec, registry)?;
            (CompiledNodeKind::GoogleAICache(compiled), keys)
        }
        NodeKind::Expression(expr_spec) => {
            let (compiled, keys) = compile_expression(interner, expr_spec, registry)?;
            (CompiledNodeKind::Expression(compiled), keys)
        }
        NodeKind::Iterator(iter_spec) => {
            let mut keys = FxHashSet::default();
            let mut compiled_sources = Vec::with_capacity(iter_spec.sources.len());

            for source in &iter_spec.sources {
                // Compile the source expression (evaluated in node's body context)
                let (expr_script, expr_ty) = compile_script(interner, interner.resolve(source.expr), registry)
                    .map_err(|e| vec![e])?;
                keys.extend(expr_script.context_keys.iter().copied());

                // Determine item type from the expr's result type
                let item_ty = match &expr_ty {
                    Ty::Iterator(elem, _)
                    | Ty::Sequence(elem, _, _)
                    | Ty::List(elem)
                    | Ty::Deque(elem, _) => (**elem).clone(),
                    Ty::Error(_) => Ty::error(),
                    other => {
                        return Err(vec![OrchError::new(OrchErrorKind::ScriptTypeMismatch {
                            context: format!("iterator source expr '{}'", source.name),
                            expected: Ty::Iterator(Box::new(Ty::error()), acvus_mir::ty::Effect::Pure),
                            got: other.clone(),
                        })]);
                    }
                };

                // Entry scripts need @item and @index in scope
                let transform_registry = registry.with_extra_scoped([
                    (interner.intern(KEY_ITEM), item_ty.clone()),
                    (interner.intern(KEY_INDEX), Ty::Int),
                ]).map_err(|e| vec![OrchError::new(OrchErrorKind::RegistryConflict {
                    key: e.key, tier_a: e.tier_a, tier_b: e.tier_b,
                })])?;

                let mut compiled_entries = Vec::with_capacity(source.entries.len());
                for entry in &source.entries {
                    // Compile optional condition as Bool script
                    let condition = match &entry.condition {
                        Some(src) => {
                            let (script, _) = compile_script_with_hint(
                                interner, interner.resolve(*src), &transform_registry, Some(&Ty::Bool),
                            ).map_err(|e| vec![e])?;
                            keys.extend(script.context_keys.iter().copied());
                            Some(script)
                        }
                        None => None,
                    };

                    // Compile transform (Template or Script)
                    let transform = match &entry.transform {
                        crate::spec::SourceTransform::Template(src) => {
                            let tmpl_block = compile_template(interner, interner.resolve(*src), 0, &transform_registry)
                                .map_err(|e| vec![e])?;
                            keys.extend(tmpl_block.context_keys.iter().copied());
                            crate::spec::CompiledSourceTransform::Template(CompiledScript {
                                module: tmpl_block.module,
                                context_keys: tmpl_block.context_keys,
                                val_def: tmpl_block.val_def,
                            })
                        }
                        crate::spec::SourceTransform::Script(src) => {
                            let (script, _) = compile_script(interner, interner.resolve(*src), &transform_registry)
                                .map_err(|e| vec![e])?;
                            keys.extend(script.context_keys.iter().copied());
                            crate::spec::CompiledSourceTransform::Script(script)
                        }
                    };

                    compiled_entries.push(crate::spec::CompiledIteratorEntry {
                        condition,
                        transform,
                    });
                }

                // start: script → Int
                let start = match &source.start {
                    Some(src) => {
                        let (s, _) = compile_script_with_hint(
                            interner, interner.resolve(*src), registry, Some(&Ty::Int),
                        ).map_err(|e| vec![e])?;
                        keys.extend(s.context_keys.iter().copied());
                        Some(s)
                    }
                    None => None,
                };

                // end: script → Option<Int>
                let end = match &source.end {
                    Some(src) => {
                        let opt_int = Ty::Option(Box::new(Ty::Int));
                        let (s, _) = compile_script_with_hint(
                            interner, interner.resolve(*src), registry, Some(&opt_int),
                        ).map_err(|e| vec![e])?;
                        keys.extend(s.context_keys.iter().copied());
                        Some(s)
                    }
                    None => None,
                };

                compiled_sources.push(crate::spec::CompiledIteratorSource {
                    name: source.name.clone(),
                    expr: expr_script,
                    entries: compiled_entries,
                    start,
                    end,
                });
            }

            (CompiledNodeKind::Iterator {
                sources: compiled_sources,
                unordered: iter_spec.unordered,
            }, keys)
        }
    };

    // initial_value context keys contribute to dependencies
    if let Some(ref iv) = initial_value {
        all_context_keys.extend(iv.context_keys.iter().copied());
    }

    // assert context keys contribute
    if let Some(ref compiled_assert) = compiled_assert {
        all_context_keys.extend(compiled_assert.context_keys.iter().copied());
    }

    // execution context keys contribute
    match &compiled_execution {
        CompiledExecution::Always | CompiledExecution::OncePerTurn => {}
        CompiledExecution::IfModified { key } => {
            all_context_keys.extend(key.context_keys.iter().copied());
        }
    }

    // persistency context keys contribute (bind script lives in NodeRole::Persistent)
    if let NodeRole::Persistent { bind, .. } = &role {
        all_context_keys.extend(bind.context_keys.iter().copied());
    }

    Ok(CompiledNode {
        id,
        name: spec.name,
        kind,
        all_context_keys,
        strategy: CompiledStrategy {
            execution: compiled_execution,
            initial_value,
            retry: spec.strategy.retry,
            assert: compiled_assert,
        },
        is_function: spec.is_function,
        fn_params: spec.fn_params.clone(),
        output_ty,
        role,
    })
}

/// Result of computing external context types from node specs.
///
/// `context_types` contains all externally-visible types:
/// - injected types (from project.toml / bindings)
/// - `@nodeName` → stored type (from self_bind tail)
/// - `@turn_index` → Int (system context)
///
/// Local types (`@self`, `@raw`) are NOT included — use `NodeLocalTypes` from dsl.

pub struct ExternalContextEnv {
    pub registry: PartialContextTypeRegistry,
    /// Types of values stored in storage (node self types + @turn_index).
    /// Does not include injected types (those come from the resolver, not storage).
    pub storage_types: FxHashMap<Astr, Ty>,
    /// Per-node local types, indexed by node name.
    pub node_locals: FxHashMap<Astr, NodeLocalTypes>,
    pub(crate) stored_types: Vec<Ty>,
}

/// Resolve `NodeLocalTypes` for a single node — the SINGLE source of truth for @self and @raw types.
///
/// `raw_ty` is the node's raw output type — passed explicitly because Expr nodes may have
/// their type resolved externally (via topo sort) rather than from the spec.
///
/// - `@raw` = `raw_ty` (what the node produces each execution)
/// - `@self`:
///   - Sequence/Patch with bind: two-pass — compile bind with `@self = Infer`, use tail_ty as self_ty
///   - Ephemeral with initial_value: initial_value's return type
///   - No initial_value: self_ty = raw_ty (fallback, @self not exposed without initial_value)
fn resolve_node_locals(
    interner: &Interner,
    spec: &NodeSpec,
    base_registry: &ContextTypeRegistry,
    raw_ty: Option<Ty>,
) -> Result<NodeLocalTypes, Vec<OrchError>> {
    let map_conflict = |e: RegistryConflictError| {
        vec![OrchError::new(OrchErrorKind::RegistryConflict {
            key: e.key,
            tier_a: e.tier_a,
            tier_b: e.tier_b,
        })]
    };

    let self_ty = match &spec.strategy.persistency {
        Persistency::Sequence { bind } | Persistency::Patch { bind } => {
            // Two-pass with shared TySubst:
            // Pass 1: compile initial_value → Deque<T, O> (origin allocated in subst)
            // Pass 2: compile bind with @self = Sequence<T, O, Pure> (same origin from subst)

            let mut subst = acvus_mir::ty::TySubst::new();

            // Pass 1: initial_value → Deque<T, O>
            let init_hint: Option<Ty> = raw_ty.as_ref().and_then(|ty| match ty {
                Ty::List(elem) => Some(Ty::List(elem.clone())),
                Ty::Deque(elem, origin) => Some(Ty::Deque(elem.clone(), *origin)),
                Ty::Sequence(elem, origin, effect) => Some(Ty::Sequence(elem.clone(), *origin, *effect)),
                _ => None,
            });
            let init_ty = if let Some(ref init_src) = spec.strategy.initial_value {
                let init_reg = spec.build_node_context(interner, base_registry, ContextScope::InitialValue, None)
                    .map_err(map_conflict)?;
                // Poison on failure: actual error reported during Phase 4 compilation.
                match compile_script_with_hint_subst(
                    interner, interner.resolve(*init_src), &init_reg,
                    init_hint.as_ref(), &mut subst,
                ) {
                    Ok((_, ty)) => ty,
                    Err(_) => Ty::error(),
                }
            } else {
                // No initial_value: fall back to raw_ty. For Sequence this should
                // be unreachable (MissingInitialValue error reported elsewhere).
                raw_ty.clone().unwrap_or_else(Ty::error)
            };

            // Coerce Deque<T, O> → Sequence<T, O, Pure> for @self
            let self_hint = match init_ty {
                Ty::Deque(inner, origin) => Ty::Sequence(inner, origin, Effect::Pure),
                _ => init_ty.clone(),
            };

            // Pass 2: compile bind with @self = Sequence<T, O, Pure> — same subst!
            // bind result must be compatible with self_hint (Sequence<T, O, E>).
            // If bind returns Iterator (e.g. via map), unification fails → type error.
            let raw_for_locals = raw_ty.clone().unwrap_or_else(Ty::error);
            let temp_locals = NodeLocalTypes { raw_ty: raw_for_locals, self_ty: self_hint.clone() };
            let bind_reg = spec.build_node_context(interner, base_registry, ContextScope::Bind, Some(&temp_locals))
                .map_err(map_conflict)?;
            // Poison on failure: actual error reported during Phase 4 compilation.
            let resolved_self = match compile_script_with_hint_subst(
                interner, interner.resolve(*bind), &bind_reg,
                Some(&self_hint), &mut subst,
            ) {
                Ok((_, ty)) => ty,
                Err(_) => Ty::error(),
            };

            resolved_self
        }
        _ => {
            // Ephemeral: self_ty from initial_value if present, else raw_ty
            if let Some(ref init_src) = spec.strategy.initial_value {
                let init_reg = spec.build_node_context(interner, base_registry, ContextScope::InitialValue, None)
                    .map_err(map_conflict)?;
                // Poison on failure: actual error reported during Phase 4 compilation.
                match compile_script_with_hint(interner, interner.resolve(*init_src), &init_reg, None) {
                    Ok((_, ty)) => ty,
                    Err(_) => Ty::error(),
                }
            } else {
                raw_ty.clone().unwrap_or_else(Ty::error)
            }
        }
    };

    let raw_ty = raw_ty.unwrap_or_else(Ty::error);
    Ok(NodeLocalTypes { raw_ty, self_ty })
}

/// Wrap a type as `Ty::Fn` if the node is a function node.
fn wrap_fn_ty(spec: &NodeSpec, ty: Ty) -> Ty {
    if spec.is_function {
        let param_types: Vec<Ty> = spec.fn_params.iter().map(|p| p.ty.clone()).collect();
        Ty::Fn {
            params: param_types,
            ret: Box::new(ty),
            kind: FnKind::Extern,
            captures: vec![],
            effect: Effect::Pure,
        }
    } else {
        ty
    }
}

/// Compute all externally-visible context types from node specs.
///
/// Pipeline:
/// 1. Build a temporary `working_ctx` with raw types (for intermediate compilation)
/// 2. Resolve Expr node types via topo sort (progressively adding to `working_ctx`)
/// 3. Compute node_locals → determine final `@name` types (`stored_types`, immutable)
/// 4. Register everything into registry — once per key
///
/// The result can be used for typechecking binding scripts or passed
/// to `compile_nodes_with_env` for full compilation.
pub fn compute_external_context_env(
    interner: &Interner,
    specs: &[NodeSpec],
    mut registry: PartialContextTypeRegistry,
) -> Result<ExternalContextEnv, Vec<OrchError>> {
    let map_conflict = |e: acvus_mir::context_registry::RegistryConflictError| {
        vec![OrchError::new(OrchErrorKind::RegistryConflict {
            key: e.key,
            tier_a: e.tier_a,
            tier_b: e.tier_b,
        })]
    };

    // ── Phase 1: Register system types into registry ───────────────────
    //
    // Add @turn_index and concrete node raw types directly to the registry.
    // For intermediate compilation (Phase 2-3), create temporary full registries
    // via `registry.to_full()`.

    let raw_types: Vec<Option<Ty>> = specs
        .iter()
        .map(|s| s.kind.raw_output_ty(interner))
        .collect();

    registry
        .insert_system(interner.intern(KEY_TURN_INDEX), Ty::Int)
        .map_err(map_conflict)?;

    for (spec, ty) in specs.iter().zip(raw_types.iter()) {
        let Some(ty) = ty else { continue };
        registry.insert_system(spec.name, wrap_fn_ty(spec, ty.clone()))
            .map_err(map_conflict)?;
    }

    // ── Phase 2: Resolve Expr node types via topo sort ───────────────────
    //
    // Expr nodes start with raw_type = None (inferred). Topo sort compiles
    // them in dependency order, progressively adding resolved types to the
    // registry. expr_resolved[idx] holds the resolved raw type for each node.

    let infer_indices: Vec<usize> = (0..specs.len())
        .filter(|&i| raw_types[i].is_none())
        .collect();

    let mut expr_resolved: FxHashMap<usize, Ty> = FxHashMap::default();

    if !infer_indices.is_empty() {
        let node_name_to_idx: FxHashMap<Astr, usize> =
            specs.iter().enumerate().map(|(i, s)| (s.name, i)).collect();
        let infer_set: FxHashSet<usize> = infer_indices.iter().copied().collect();

        let n = specs.len();
        let mut deps: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
        let mut rdeps: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];

        // Use a temporary full registry for analysis extraction
        let temp_reg = registry.to_full();
        for &idx in &infer_indices {
            if let NodeKind::Expression(expr_spec) = &specs[idx].kind {
                let keys =
                    analysis_extract_script_keys(interner, &expr_spec.source, &temp_reg);
                for key in keys {
                    if let Some(&dep_idx) = node_name_to_idx.get(&key) {
                        if infer_set.contains(&dep_idx) && dep_idx != idx {
                            deps[idx].insert(dep_idx);
                            rdeps[dep_idx].insert(idx);
                        }
                    }
                }
            }
        }

        // Kahn's algorithm on Infer nodes
        let mut in_degree: FxHashMap<usize, usize> = infer_indices
            .iter()
            .map(|&i| (i, deps[i].len()))
            .collect();
        let mut queue: VecDeque<usize> = infer_indices
            .iter()
            .filter(|&&i| deps[i].is_empty())
            .copied()
            .collect();
        let mut topo = Vec::with_capacity(infer_indices.len());

        while let Some(u) = queue.pop_front() {
            topo.push(u);
            for &v in &rdeps[u] {
                if let Some(deg) = in_degree.get_mut(&v) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(v);
                    }
                }
            }
        }

        if topo.len() != infer_indices.len() {
            let in_cycle: Vec<String> = infer_indices
                .iter()
                .filter(|&&i| in_degree[&i] > 0)
                .map(|&i| interner.resolve(specs[i].name).to_string())
                .collect();
            return Err(vec![OrchError::new(OrchErrorKind::CycleDetected {
                nodes: in_cycle,
            })]);
        }

        for &idx in &topo {
            if let NodeKind::Expression(expr_spec) = &specs[idx].kind {
                let hint = expr_spec.output_ty.as_ref();
                let full_reg = registry.to_full();
                let temp_raw_ty = specs[idx].kind.raw_output_ty(interner);
                let temp_locals = resolve_node_locals(interner, &specs[idx], &full_reg, temp_raw_ty)?;
                let expr_reg = specs[idx].build_node_context(interner, &full_reg, ContextScope::Body, Some(&temp_locals))
                    .map_err(map_conflict)?;
                let resolved = compile_script_with_hint(
                    interner,
                    &expr_spec.source,
                    &expr_reg,
                    hint,
                )
                .map(|(_, ty)| ty)
                .unwrap_or_else(|_| Ty::error());

                registry.insert_system(specs[idx].name, wrap_fn_ty(&specs[idx], resolved.clone()))
                    .map_err(map_conflict)?;
                expr_resolved.insert(idx, resolved);
            }
        }
    }

    // ── Phase 3: Compute node_locals → final stored_types (immutable) ────
    //
    // For each node, resolve @self and @raw via resolve_node_locals,
    // then determine the final @name type.
    // Use the now-complete registry for compilation.

    let full_reg = registry.to_full();
    let mut node_locals = FxHashMap::default();
    let mut stored_types: Vec<Ty> = Vec::with_capacity(specs.len());

    for (i, spec) in specs.iter().enumerate() {
        let raw_ty: Option<Ty> = expr_resolved.get(&i).cloned()
            .or_else(|| raw_types[i].clone());
        let locals = resolve_node_locals(interner, spec, &full_reg, raw_ty)?;
        let name_ty = if spec.strategy.initial_value.is_some() {
            locals.self_ty.clone()
        } else {
            locals.raw_ty.clone()
        };
        stored_types.push(name_ty);
        node_locals.insert(spec.name, locals);
    }

    // ── Phase 3b: Validate purity — non-Ephemeral nodes must have pure stored types ──
    for (i, spec) in specs.iter().enumerate() {
        if !matches!(spec.strategy.persistency, Persistency::Ephemeral) && !stored_types[i].is_pureable() {
            let persistency_name = match &spec.strategy.persistency {
                Persistency::Ephemeral => unreachable!(),
                Persistency::Sequence { .. } => "Sequence",
                Persistency::Patch { .. } => "Patch",
            };
            return Err(vec![OrchError::new(OrchErrorKind::UnpureStoredType {
                node: spec.name,
                persistency: persistency_name.to_string(),
                ty: stored_types[i].clone(),
            })]);
        }
    }

    let storage_types = registry.system().clone();

    Ok(ExternalContextEnv {
        registry,
        storage_types,
        node_locals,
        stored_types,
    })
}

/// Compile multiple node specs, merging their output types into context automatically.
///
/// `injected_types` are externally declared context types (from project.toml).
/// Node stored types are derived from `initial_value` scripts.
pub fn compile_nodes(
    interner: &Interner,
    specs: &[NodeSpec],
    registry: PartialContextTypeRegistry,
) -> Result<CompiledNodeGraph, Vec<OrchError>> {
    let env = compute_external_context_env(interner, specs, registry)?;
    compile_nodes_with_env(interner, specs, env)
}

/// Compile nodes using a pre-computed external context environment.
///
/// Use this when you already called `compute_external_context_env` (e.g. for
/// typechecking binding scripts) and want to avoid recomputation.
pub fn compile_nodes_with_env(
    interner: &Interner,
    specs: &[NodeSpec],
    env: ExternalContextEnv,
) -> Result<CompiledNodeGraph, Vec<OrchError>> {
    let base_registry = env.registry.to_full();
    let stored_types = env.stored_types;
    let mut errors = Vec::new();

    let map_conflict = |e: RegistryConflictError| {
        OrchError::new(OrchErrorKind::RegistryConflict {
            key: e.key,
            tier_a: e.tier_a,
            tier_b: e.tier_b,
        })
    };

    // Compile initial_value scripts — all node kinds.
    let mut initial_value_scripts: Vec<Option<CompiledScript>> = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let Some(ref init_src) = spec.strategy.initial_value else {
            initial_value_scripts.push(None);
            continue;
        };
        let hint = match &stored_types[i] {
            Ty::Error(_) => None,
            ty => Some(ty),
        };
        let init_reg = match spec.build_node_context(interner, &base_registry, ContextScope::InitialValue, env.node_locals.get(&spec.name)) {
            Ok(r) => r,
            Err(e) => {
                errors.push(map_conflict(e));
                initial_value_scripts.push(None);
                continue;
            }
        };
        let (script, _init_ty) = match compile_script_with_hint(
            interner,
            interner.resolve(*init_src),
            &init_reg,
            hint,
        ) {
            Ok(v) => v,
            Err(e) => {
                errors.push(OrchError::new(OrchErrorKind::ScriptCompile {
                    context: format!(
                        "node '{}' initial_value '{}' (stored_type={})",
                        interner.resolve(spec.name),
                        interner.resolve(*init_src),
                        stored_types[i].display(interner),
                    ),
                    errors: vec![],
                }));
                errors.push(e);
                initial_value_scripts.push(None);
                continue;
            }
        };
        // Type check is already done via hint in compile_script_with_hint above.
        initial_value_scripts.push(Some(script));
    }

    // Compile execution for each node
    let mut compiled_executions: Vec<CompiledExecution> = Vec::new();
    for spec in specs.iter() {
        let exec = match &spec.strategy.execution {
            Execution::Always => CompiledExecution::Always,
            Execution::OncePerTurn => CompiledExecution::OncePerTurn,
            Execution::IfModified { key } => {
                let (script, _ty) = match compile_script(
                    interner,
                    interner.resolve(*key),
                    &base_registry,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        errors.push(e);
                        compiled_executions.push(CompiledExecution::Always);
                        continue;
                    }
                };
                CompiledExecution::IfModified { key: script }
            }
        };
        compiled_executions.push(exec);
    }

    // Compile persistency bind scripts for each node.
    // Ephemeral → None, Sequence/Patch → Some((mode, bind_script)).
    let mut compiled_persistencies: Vec<Option<(PersistMode, CompiledScript)>> = Vec::new();
    for spec in specs.iter() {
        let persistency = match &spec.strategy.persistency {
            Persistency::Ephemeral => None,
            Persistency::Sequence { bind } | Persistency::Patch { bind } => {
                // Sequence requires initial_value (bind always references @self for accumulation)
                if matches!(&spec.strategy.persistency, Persistency::Sequence { .. })
                    && spec.strategy.initial_value.is_none()
                {
                    errors.push(OrchError::new(OrchErrorKind::MissingInitialValue {
                        node: spec.name,
                        persistency: "Sequence",
                    }));
                    compiled_persistencies.push(None);
                    continue;
                }

                // bind context: @self + @raw + all context (via ContextScope::Bind)
                let bind_reg = match spec.build_node_context(interner, &base_registry, ContextScope::Bind, env.node_locals.get(&spec.name)) {
                    Ok(r) => r,
                    Err(e) => {
                        errors.push(map_conflict(e));
                        compiled_persistencies.push(None);
                        continue;
                    }
                };
                let (script, _ty) = match compile_script(
                    interner,
                    interner.resolve(*bind),
                    &bind_reg,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        errors.push(e);
                        compiled_persistencies.push(None);
                        continue;
                    }
                };
                let mode = match &spec.strategy.persistency {
                    Persistency::Sequence { .. } => PersistMode::Sequence,
                    Persistency::Patch { .. } => PersistMode::Patch,
                    _ => unreachable!(),
                };
                Some((mode, script))
            }
        };
        compiled_persistencies.push(persistency);
    }

    // Compile assert for each node (Bind scope: @self + @raw)
    let mut compiled_asserts: Vec<Option<CompiledScript>> = Vec::new();
    for spec in specs.iter() {
        let compiled_assert = if let Some(ref assert_src) = spec.strategy.assert {
            let assert_reg = match spec.build_node_context(interner, &base_registry, ContextScope::Bind, env.node_locals.get(&spec.name)) {
                Ok(r) => r,
                Err(e) => {
                    errors.push(map_conflict(e));
                    compiled_asserts.push(None);
                    continue;
                }
            };
            let (script, _ty) = match compile_script_with_hint(
                interner,
                interner.resolve(*assert_src),
                &assert_reg,
                Some(&Ty::Bool),
            ) {
                Ok(v) => v,
                Err(e) => {
                    errors.push(e);
                    compiled_asserts.push(None);
                    continue;
                }
            };
            Some(script)
        } else {
            None
        };
        compiled_asserts.push(compiled_assert);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Tool param types must be injected from the caller (LLM node spec) side because
    // the target node alone cannot know what types its params will have — the
    // param types are declared in ToolBinding, not in the target node itself.
    let mut tool_param_types: FxHashMap<Astr, Vec<(Astr, Ty)>> = FxHashMap::default();
    for spec in specs {
        let tools = spec.kind.tools();
        if tools.is_empty() {
            continue;
        }
        for tool in tools {
            let params: Vec<(Astr, Ty)> = tool
                .params
                .iter()
                .filter_map(|(k, v)| Some((interner.intern(k), parse_type_name(&v.ty)?)))
                .collect();
            tool_param_types
                .entry(interner.intern(&tool.node))
                .or_default()
                .extend(params);
        }
    }

    let mut nodes: Vec<CompiledNode> = Vec::new();
    let mut name_to_primary: FxHashMap<Astr, NodeId> = FxHashMap::default();
    let mut next_id: usize = 0;

    let raw_key = interner.intern(KEY_RAW);
    let self_key = interner.intern(KEY_SELF);

    for (i, spec) in specs.iter().enumerate() {
        let mut node_reg = match spec.build_node_context(interner, &base_registry, ContextScope::Body, env.node_locals.get(&spec.name)) {
            Ok(r) => r,
            Err(e) => {
                errors.push(map_conflict(e));
                continue;
            }
        };
        if let Some(params) = tool_param_types.get(&spec.name) {
            node_reg = match node_reg.with_extra_scoped(params.iter().map(|(k, v)| (*k, v.clone()))) {
                Ok(r) => r,
                Err(e) => {
                    errors.push(map_conflict(e));
                    continue;
                }
            };
        }

        let compiled_execution = compiled_executions[i].clone();
        let compiled_persist = compiled_persistencies[i].clone();
        let compiled_assert = compiled_asserts[i].clone();

        match compiled_persist {
            None => {
                // Ephemeral: single node, no split
                let initial_value = initial_value_scripts[i].clone();
                let output_ty = stored_types[i].clone();
                let id = NodeId(next_id);
                next_id += 1;

                match compile_node(
                    interner,
                    spec,
                    &node_reg,
                    id,
                    initial_value,
                    compiled_execution,
                    compiled_assert,
                    output_ty,
                    NodeRole::Standalone,
                ) {
                    Ok(node) => {
                        name_to_primary.insert(spec.name, id);
                        nodes.push(node);
                    }
                    Err(errs) => {
                        errors.extend(errs);
                        continue;
                    }
                }
            }
            Some((mode, bind_script)) => {
                // Split into body + bind nodes.

                // Pre-compute both IDs so body can reference bind and vice versa.
                let body_id = NodeId(next_id);
                let bind_id = NodeId(next_id + 1);
                next_id += 2;

                // ── Body node ──
                // Executes the computation (LLM/Expr/etc), produces @raw.
                // Role = Body (body itself doesn't persist).
                // Context keys = only body kind keys (no bind/initial_value/assert).
                let body_raw_ty = env.node_locals.get(&spec.name)
                    .map(|l| l.raw_ty.clone())
                    .unwrap_or_else(|| stored_types[i].clone());

                match compile_node(
                    interner,
                    spec,
                    &node_reg,
                    body_id,
                    None, // no initial_value for body
                    compiled_execution.clone(),
                    None, // no assert for body
                    body_raw_ty,
                    NodeRole::Body { bind_id },
                ) {
                    Ok(body_node) => {
                        nodes.push(body_node);
                    }
                    Err(errs) => {
                        errors.extend(errs);
                        continue;
                    }
                }

                // ── Bind node ──
                // Executes the bind script, produces the accumulated value.
                // Kind = Expr (using the bind CompiledScript).
                // Role = Persistent (with mode + bind script — resolver uses this).
                // Context keys = bind script's keys minus "raw" and "self".

                let bind_kind = CompiledNodeKind::Expression(CompiledExpression {
                    script: bind_script.clone(),
                });

                let mut bind_context_keys: FxHashSet<Astr> = bind_script.context_keys.clone();
                bind_context_keys.remove(&raw_key);
                bind_context_keys.remove(&self_key);

                // Include initial_value and assert context keys in the bind node
                if let Some(ref iv) = initial_value_scripts[i] {
                    bind_context_keys.extend(iv.context_keys.iter().copied());
                }
                if let Some(ref assert_script) = compiled_assert {
                    bind_context_keys.extend(assert_script.context_keys.iter().copied());
                }

                let role = NodeRole::Persistent {
                    body_id,
                    mode: mode.clone(),
                    bind: bind_script,
                };

                let bind_node = CompiledNode {
                    id: bind_id,
                    name: spec.name,
                    kind: bind_kind,
                    all_context_keys: bind_context_keys,
                    strategy: CompiledStrategy {
                        execution: compiled_execution,
                        initial_value: initial_value_scripts[i].clone(),
                        retry: spec.strategy.retry,
                        assert: compiled_assert,
                    },
                    is_function: spec.is_function,
                    fn_params: spec.fn_params.clone(),
                    output_ty: stored_types[i].clone(),
                    role,
                };

                // Validate: persistent nodes must have storable output types.
                if !bind_node.output_ty.is_storable() {
                    errors.push(OrchError::new(OrchErrorKind::NonStorableOutput {
                        node: interner.resolve(spec.name).to_string(),
                        ty: bind_node.output_ty.clone(),
                    }));
                }

                name_to_primary.insert(spec.name, bind_id);
                nodes.push(bind_node);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Tool targets are not captured in all_context_keys (they are invoked
    // dynamically by the model, not via @ref in templates), so we validate
    // their existence separately.
    let node_names: FxHashSet<Astr> = name_to_primary.keys().copied().collect();
    for node in &nodes {
        {
            let tools = node.kind.tools();
            for tool in tools {
                let tool_node = interner.intern(&tool.node);
                if !node_names.contains(&tool_node) {
                    errors.push(OrchError::new(OrchErrorKind::ToolTargetNotFound {
                        tool: tool.name.clone(),
                        target: tool.node.clone(),
                    }));
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(CompiledNodeGraph {
            nodes,
            name_to_primary,
        })
    } else {
        Err(errors)
    }
}

/// Extract context keys from a script source using analysis mode.
///
/// Analysis mode assigns fresh type variables for unknown `@context` refs,
/// so context keys can be discovered even before all types are known.
fn analysis_extract_script_keys(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
) -> FxHashSet<Astr> {
    let Ok(script) = acvus_ast::parse_script(interner, source) else {
        return FxHashSet::default();
    };
    let (module, _, _, _) = acvus_mir::compile_script_analysis_with_tail_partial(
        interner,
        &script,
        registry,
        None,
    );
    extract_context_keys(&module)
}

/// Extract all context keys referenced by `ContextLoad` instructions in a module.
fn extract_context_keys(module: &MirModule) -> FxHashSet<Astr> {
    let mut keys = FxHashSet::default();

    for inst in &module.main.insts {
        if let InstKind::ContextLoad { name, .. } = &inst.kind {
            keys.insert(*name);
        }
    }

    for closure in module.closures.values() {
        for inst in &closure.body.insts {
            if let InstKind::ContextLoad { name, .. } = &inst.kind {
                keys.insert(*name);
            }
        }
    }

    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_mir::ty::Ty;
    use acvus_mir::context_registry::PartialContextTypeRegistry;
    use acvus_utils::Interner;
    use rustc_hash::FxHashMap;
    use crate::dsl::*;
    use crate::spec::{NodeKind, OpenAICompatibleSpec, MaxTokens};
    use crate::MessageSpec as MsgSpec;

    #[test]
    fn deque_bind_self_extend_raw() {
        let interner = Interner::new();
        let registry = PartialContextTypeRegistry::new(
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
        ).unwrap();

        let spec = NodeSpec {
            name: interner.intern("chat"),
            kind: NodeKind::OpenAICompatible(OpenAICompatibleSpec {
                endpoint: String::new(),
                api_key: String::new(),
                model: String::new(),
                messages: vec![MsgSpec::Block {
                    role: interner.intern("user"),
                    source: "hello".into(),
                }],
                tools: vec![],
                temperature: None,
                top_p: None,
                cache_key: None,
                max_tokens: MaxTokens::default(),
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Sequence {
                    bind: interner.intern("@self | chain(@raw | iter)"),
                },
                initial_value: Some(interner.intern("[]")),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        };

        let env = compute_external_context_env(&interner, &[spec], registry)
            .expect("compute_external_context_env should succeed");

        let chat_name = interner.intern("chat");
        let locals = env.node_locals.get(&chat_name).expect("should have locals for chat");
        eprintln!("self_ty = {}", locals.self_ty.display(&interner));
        eprintln!("raw_ty = {}", locals.raw_ty.display(&interner));

        // self_ty should be Sequence (Deque → Sequence coercion via 2-pass)
        assert!(
            matches!(&locals.self_ty, Ty::Sequence(_, _, _)),
            "self_ty should be Sequence, got: {}",
            locals.self_ty.display(&interner),
        );

        // Simulate pomollu-engine bind typecheck: re-check bind with hint = self_ty
        let full_reg = env.registry.to_full();
        let chat_spec = NodeSpec {
            name: chat_name,
            kind: NodeKind::OpenAICompatible(OpenAICompatibleSpec {
                endpoint: String::new(),
                api_key: String::new(),
                model: String::new(),
                messages: vec![],
                tools: vec![],
                temperature: None,
                top_p: None,
                cache_key: None,
                max_tokens: MaxTokens::default(),
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Sequence {
                    bind: interner.intern("@self | chain(@raw | iter)"),
                },
                initial_value: Some(interner.intern("[]")),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        };
        let bind_reg = chat_spec.build_node_context(&interner, &full_reg, ContextScope::Bind, Some(locals)).unwrap();
        eprintln!("bind_ctx @self = {:?}", bind_reg.scoped().get(&interner.intern(KEY_SELF)).map(|t| format!("{}", t.display(&interner))));
        eprintln!("bind_ctx @raw = {:?}", bind_reg.scoped().get(&interner.intern(KEY_RAW)).map(|t| format!("{}", t.display(&interner))));

        let hint = Some(&locals.self_ty);
        let result = compile_script_with_hint(
            &interner,
            "@self | chain(@raw | iter)",
            &bind_reg,
            hint.as_ref().map(|t| *t),
        );
        match &result {
            Ok((_, ty)) => eprintln!("bind typecheck OK, tail = {}", ty.display(&interner)),
            Err(e) => eprintln!("bind typecheck FAILED: {}", e.display(&interner)),
        }
        assert!(result.is_ok(), "bind re-typecheck should succeed");

        // Also check initial_value
        let init_reg = chat_spec.build_node_context(&interner, &full_reg, ContextScope::InitialValue, Some(locals)).unwrap();
        let init_result = compile_script_with_hint(
            &interner,
            "[]",
            &init_reg,
            hint.as_ref().map(|t| *t),
        );
        match &init_result {
            Ok((_, ty)) => eprintln!("init typecheck OK, tail = {}", ty.display(&interner)),
            Err(e) => eprintln!("init typecheck FAILED: {}", e.display(&interner)),
        }
    }

    // ── Helper ──────────────────────────────────────────────────────

    fn empty_registry() -> PartialContextTypeRegistry {
        PartialContextTypeRegistry::new(
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
        ).unwrap()
    }

    fn expr_node(interner: &Interner, name: &str, source: &str) -> NodeSpec {
        NodeSpec {
            name: interner.intern(name),
            kind: NodeKind::Expression(crate::spec::ExpressionSpec {
                source: source.into(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        }
    }

    fn llm_node(interner: &Interner, name: &str) -> NodeSpec {
        NodeSpec {
            name: interner.intern(name),
            kind: NodeKind::OpenAICompatible(OpenAICompatibleSpec {
                endpoint: String::new(),
                api_key: String::new(),
                model: String::new(),
                messages: vec![],
                tools: vec![],
                temperature: None,
                top_p: None,
                cache_key: None,
                max_tokens: MaxTokens::default(),
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        }
    }

    // ── Soundness tests ─────────────────────────────────────────────

    #[test]
    fn cycle_detected_between_expr_nodes() {
        let i = Interner::new();
        // A depends on @B, B depends on @A → cycle
        let specs = vec![
            expr_node(&i, "A", "@B"),
            expr_node(&i, "B", "@A"),
        ];
        let result = compile_nodes(&i, &specs, empty_registry());
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(&e.kind, OrchErrorKind::CycleDetected { .. })),
            "expected CycleDetected, got: {errors:?}");
    }

    #[test]
    fn missing_initial_value_for_sequence() {
        let i = Interner::new();
        let mut node = llm_node(&i, "chat");
        node.strategy.persistency = Persistency::Sequence {
            bind: i.intern("@self | chain(@raw | iter)"),
        };
        // No initial_value but Sequence persistency → should fail
        let result = compile_nodes(&i, &[node], empty_registry());
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(&e.kind, OrchErrorKind::MissingInitialValue { .. })),
            "expected MissingInitialValue, got: {errors:?}");
    }

    // ── Completeness tests ──────────────────────────────────────────

    #[test]
    fn ephemeral_node_compiles_without_storable_constraint() {
        let i = Interner::new();
        // Ephemeral Expr node — no storability requirement
        let specs = vec![expr_node(&i, "x", "1 + 2")];
        let result = compile_nodes(&i, &specs, empty_registry());
        assert!(result.is_ok(), "Ephemeral Expr should compile: {result:?}");
    }

    #[test]
    fn expr_type_inference_chain() {
        let i = Interner::new();
        // A = 42 (Int), B = @A + 1 (depends on A, should infer Int)
        let specs = vec![
            expr_node(&i, "A", "42"),
            expr_node(&i, "B", "@A + 1"),
        ];
        let result = compile_nodes(&i, &specs, empty_registry());
        assert!(result.is_ok(), "Expr chain should compile: {result:?}");
        let graph = result.unwrap();
        // B should have Int output type
        let b_id = graph.name_to_primary[&i.intern("B")];
        assert_eq!(graph.nodes[b_id.index()].output_ty, Ty::Int,
            "B should have Int type");
    }

    #[test]
    fn linear_dependency_compiles() {
        let i = Interner::new();
        // C references @B which references @A — linear chain
        let specs = vec![
            expr_node(&i, "A", "10"),
            expr_node(&i, "B", "@A * 2"),
            expr_node(&i, "C", "@B + 1"),
        ];
        let result = compile_nodes(&i, &specs, empty_registry());
        assert!(result.is_ok(), "Linear dependency chain should compile: {result:?}");
    }

    #[test]
    fn sequence_with_initial_value_compiles() {
        let i = Interner::new();
        let mut node = llm_node(&i, "chat");
        node.strategy.persistency = Persistency::Sequence {
            bind: i.intern("@self | chain(@raw | iter)"),
        };
        node.strategy.initial_value = Some(i.intern("[]"));
        node.kind = NodeKind::OpenAICompatible(OpenAICompatibleSpec {
            endpoint: String::new(),
            api_key: String::new(),
            model: String::new(),
            messages: vec![MsgSpec::Block {
                role: i.intern("user"),
                source: "hello".into(),
            }],
            tools: vec![],
            temperature: None,
            top_p: None,
            cache_key: None,
            max_tokens: MaxTokens::default(),
        });
        let result = compile_nodes(&i, &[node], empty_registry());
        assert!(result.is_ok(), "Sequence with initial_value should compile: {result:?}");
    }
}
