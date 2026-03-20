//! Graph resolution engine: DAG topo sort + SCC unification.

use rustc_hash::{FxHashMap, FxHashSet};
use acvus_utils::{Astr, Interner};

use crate::context_registry::ContextTypeRegistry;
use crate::error::MirError;
use crate::ty::{Ty, TySubst};
use crate::typeck::{TypeResolution, Checked, Unchecked, check_completeness};

use super::types::*;

// ── Output types ─────────────────────────────────────────────────────

/// Result of graph resolution (Phase 0 → Phase 1).
#[derive(Debug)]
pub struct ResolvedGraph {
    /// Per-unit type resolution (only for CompilationUnits, not ExternDecls).
    pub resolutions: FxHashMap<UnitId, TypeResolution<Checked>>,
    /// Context types (scope locals, externals): ContextId → Ty.
    pub resolved_types: FxHashMap<ContextId, Ty>,
    /// Per-unit output types: UnitId → Ty.
    pub unit_outputs: FxHashMap<UnitId, Ty>,
    /// Collected errors (per-unit). Does not include fatal errors.
    pub errors: Vec<GraphError>,
}

/// Result of graph compilation (Phase 0 → Phase 2).
/// Contains both type resolutions and compiled MIR modules.
#[derive(Debug)]
pub struct CompiledGraph {
    /// Per-unit compiled MIR (only for CompilationUnits, not ExternDecls).
    pub compiled: FxHashMap<UnitId, CompiledUnit>,
    /// Per-unit output types: UnitId → Ty.
    pub unit_outputs: FxHashMap<UnitId, Ty>,
    /// Context types (scope locals, externals): ContextId → Ty.
    pub resolved_types: FxHashMap<ContextId, Ty>,
    /// Collected errors.
    pub errors: Vec<GraphError>,
}

/// A compiled unit: MIR module + type resolution + metadata.
#[derive(Debug)]
pub struct CompiledUnit {
    pub module: crate::ir::MirModule,
    pub hints: crate::hints::HintTable,
    pub tail_ty: Ty,
    pub context_keys: FxHashSet<Astr>,
}

/// A graph-level error with source attribution.
#[derive(Debug)]
pub struct GraphError {
    pub unit: UnitId,
    pub errors: Vec<MirError>,
}

// ── Internal: dependency analysis ────────────────────────────────────

/// Which entity (Unit or Extern) a UnitId refers to.
enum UnitEntry<'a> {
    Unit(&'a CompilationUnit),
    Extern(&'a ExternDecl),
}

impl CompilationGraph {
    /// Resolve all types and compile to MIR. Phase 0 → Phase 2.
    pub fn compile(&self, interner: &Interner) -> CompiledGraph {
        let mut resolved = self.resolve(interner);
        let mut compiled = FxHashMap::default();
        let mut errors = resolved.errors;

        for unit in &self.units {
            let Some(resolution) = resolved.resolutions.remove(&unit.id) else {
                // Unit had errors during resolution — skip lowering.
                continue;
            };
            // We need to re-parse to lower. TypeResolution<Checked> has the type info,
            // but Lowerer needs the AST.
            let lower_result = match unit.kind {
                SourceKind::Script => {
                    let Ok(script) = acvus_ast::parse_script(interner, interner.resolve(unit.source)) else {
                        continue; // Parse error already reported during resolve.
                    };
                    crate::lower_checked_script(interner, &script, resolution)
                }
                SourceKind::Template => {
                    let Ok(template) = acvus_ast::parse(interner, interner.resolve(unit.source)) else {
                        continue;
                    };
                    crate::lower_checked_template(interner, &template, resolution)
                }
            };

            match lower_result {
                Ok((module, hints)) => {
                    let context_keys = module.extract_context_keys();
                    compiled.insert(unit.id, CompiledUnit {
                        module,
                        hints,
                        tail_ty: resolved.unit_outputs.get(&unit.id).cloned().unwrap_or_else(Ty::error),
                        context_keys,
                    });
                }
                Err(errs) => {
                    errors.push(GraphError { unit: unit.id, errors: errs });
                }
            }
        }

        CompiledGraph {
            compiled,
            unit_outputs: resolved.unit_outputs,
            resolved_types: resolved.resolved_types,
            errors,
        }
    }

    /// Resolve all types in the graph. Phase 0 → Phase 1.
    pub fn resolve(&self, interner: &Interner) -> ResolvedGraph {
        let mut subst = TySubst::new();
        let mut resolved_types: FxHashMap<ContextId, Ty> = FxHashMap::default();
        let mut unit_outputs: FxHashMap<UnitId, Ty> = FxHashMap::default(); // unit output types by UnitId
        let mut resolutions: FxHashMap<UnitId, TypeResolution<Checked>> = FxHashMap::default();
        let mut errors: Vec<GraphError> = Vec::new();

        // 0. Build lookup tables.
        let unit_map: FxHashMap<UnitId, UnitEntry> = self.build_unit_map();
        let context_to_binding: FxHashMap<ContextId, &ContextBinding> = self.build_context_binding_map();
        let unit_to_scope: FxHashMap<UnitId, ScopeId> = self.build_unit_scope_map();

        // Register external types.
        for (&ctx_id, ext_ty) in &self.externals {
            match ext_ty {
                ExternalType::Known(ty) => { resolved_types.insert(ctx_id, ty.clone()); }
                ExternalType::Infer => { resolved_types.insert(ctx_id, subst.fresh_var()); }
            }
        }

        // 1. Build dependency graph (all UnitIds — both Unit and Extern).
        let all_ids: Vec<UnitId> = self.all_unit_ids();
        let id_to_idx: FxHashMap<UnitId, usize> = all_ids.iter().enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let n = all_ids.len();

        let mut deps: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];

        // Edges from scope bindings.
        for scope in &self.scopes {
            // Collect units in this scope that reference ScopeLocal bindings.
            // These must form mutual edges (= will become an SCC via Tarjan).
            let scope_local_ctx_ids: FxHashSet<ContextId> = scope.bindings.iter()
                .filter(|b| matches!(b.source, ContextSource::ScopeLocal))
                .map(|b| b.id)
                .collect();

            let mut scope_local_unit_idxs: Vec<usize> = Vec::new();

            for unit_id in &scope.units {
                let Some(&unit_idx) = id_to_idx.get(unit_id) else { continue };
                if let Some(UnitEntry::Unit(unit)) = unit_map.get(unit_id) {
                    let mut references_scope_local = false;
                    for ctx_id in unit.name_to_id.values() {
                        if scope_local_ctx_ids.contains(ctx_id) {
                            references_scope_local = true;
                        }
                        if let Some(binding) = context_to_binding.get(ctx_id) {
                            if let ContextSource::Derived(dep_id, _) = &binding.source {
                                if let Some(&dep_idx) = id_to_idx.get(dep_id) {
                                    if dep_idx != unit_idx {
                                        deps[unit_idx].insert(dep_idx);
                                    }
                                }
                            }
                        }
                    }
                    if references_scope_local {
                        scope_local_unit_idxs.push(unit_idx);
                    }
                }
            }

            // Units referencing ScopeLocal: add mutual edges so Tarjan groups them as SCC.
            for i in 0..scope_local_unit_idxs.len() {
                for j in 0..scope_local_unit_idxs.len() {
                    if i != j {
                        deps[scope_local_unit_idxs[i]].insert(scope_local_unit_idxs[j]);
                    }
                }
            }

            // Also: units in the same scope as ScopeLocal bindings whose output
            // feeds into the ScopeLocal (via hint unification) should be in the SCC.
            // The "init" pattern: init doesn't reference @self but its output
            // unifies with the ScopeLocal variable. We handle this by including
            // ALL units in the scope that are listed alongside ScopeLocal bindings.
            // If they don't actually participate, Tarjan will make them trivial SCCs.
            if !scope_local_ctx_ids.is_empty() {
                let all_scope_idxs: Vec<usize> = scope.units.iter()
                    .filter_map(|id| id_to_idx.get(id).copied())
                    .collect();
                for i in 0..all_scope_idxs.len() {
                    for j in 0..all_scope_idxs.len() {
                        if i != j {
                            deps[all_scope_idxs[i]].insert(all_scope_idxs[j]);
                        }
                    }
                }
            }
        }

        // Edges for units NOT in any scope — they may still reference
        // bindings via name_to_id (e.g., global externals or other unit outputs).
        for unit in &self.units {
            let Some(&unit_idx) = id_to_idx.get(&unit.id) else { continue };
            if unit_to_scope.contains_key(&unit.id) { continue; } // already handled above
            for ctx_id in unit.name_to_id.values() {
                if let Some(binding) = context_to_binding.get(ctx_id) {
                    if let ContextSource::Derived(dep_id, _) = &binding.source {
                        if let Some(&dep_idx) = id_to_idx.get(dep_id) {
                            if dep_idx != unit_idx {
                                deps[unit_idx].insert(dep_idx);
                            }
                        }
                    }
                }
            }
        }

        // Edges from ExternDecl inputs.
        for ext in &self.externs {
            let Some(&ext_idx) = id_to_idx.get(&ext.id) else { continue };
            for (input_id, _) in &ext.inputs {
                if let Some(&input_idx) = id_to_idx.get(input_id) {
                    deps[ext_idx].insert(input_idx);
                }
            }
        }

        // 2. SCC detection (Tarjan).
        let sccs = tarjan_scc(n, &deps);

        // 3. Process SCCs in topo order.
        // Tarjan returns SCCs in reverse topological order of the condensation DAG:
        // dependencies appear LATER in the list. So iterate forward = process deps first.
        for scc in &sccs {
            let scc_ids: Vec<UnitId> = scc.iter().map(|&idx| all_ids[idx]).collect();

            // Allocate ScopeLocal variables for this SCC.
            let scope_locals = self.allocate_scope_locals(&scc_ids, &unit_to_scope, &mut subst, &mut resolved_types);

            // Unified SCC processing — same logic for trivial and non-trivial.
            //
            // All unknowns get fresh vars upfront. Unification during
            // typecheck connects them. No special cases for ScopeLocal
            // vs Derived, trivial vs non-trivial.
            {
                // 1. Pre-register fresh vars for ALL unit outputs in this SCC.
                let mut scc_output_vars: FxHashMap<UnitId, Ty> = FxHashMap::default();
                for &unit_id in &scc_ids {
                    if !unit_outputs.contains_key(&unit_id) {
                        let var = subst.fresh_var();
                        unit_outputs.insert(unit_id, var.clone());
                        scc_output_vars.insert(unit_id, var);
                    }
                }

                // 2. Typecheck all units. Each unit's output_binding declares
                //    which ScopeLocal α it should unify with (via hint).
                let mut scc_unchecked: Vec<(UnitId, TypeResolution<Unchecked>)> = Vec::new();

                for &unit_id in &scc_ids {
                    match unit_map.get(&unit_id) {
                        Some(UnitEntry::Unit(unit)) => {
                            let hint = unit.output_binding
                                .and_then(|ctx_id| scope_locals.get(&ctx_id))
                                .cloned();
                            let registry = Self::build_registry_for_unit(
                                unit, interner, &resolved_types, &unit_outputs, &context_to_binding, &subst,
                            );
                            match self.typecheck_unit(interner, unit, &registry, hint.as_ref(), &mut subst) {
                                Ok(unchecked) => {
                                    // Unify tail_ty with the pre-registered fresh var
                                    // so the output var gets bound to the actual type.
                                    if let Some(var) = scc_output_vars.get(&unit_id) {
                                        let _ = subst.unify(&unchecked.tail_ty, var, crate::ty::Polarity::Covariant);
                                    }
                                    scc_unchecked.push((unit_id, unchecked));
                                }
                                Err(errs) => {
                                    errors.push(GraphError { unit: unit_id, errors: errs });
                                }
                            }
                        }
                        Some(UnitEntry::Extern(ext)) => {
                            Self::process_extern(ext, &unit_outputs, &mut subst, &mut errors);
                            if let Some(var) = scc_output_vars.get(&ext.id) {
                                let _ = subst.unify(&ext.output_ty, var, crate::ty::Polarity::Covariant);
                            }
                        }
                        None => {}
                    }
                }

                // 3. check_completeness for all unchecked resolutions.
                for (unit_id, unchecked) in scc_unchecked {
                    match check_completeness(unchecked, &subst) {
                        Ok(checked) => {
                            resolutions.insert(unit_id, checked);
                        }
                        Err(errs) => {
                            errors.push(GraphError { unit: unit_id, errors: errs });
                        }
                    }
                }

                // 4. Re-resolve everything in SCC: unit_outputs, scope_locals.
                for (unit_id, var) in &scc_output_vars {
                    let resolved = subst.resolve(var);
                    unit_outputs.insert(*unit_id, resolved);
                }
                for (ctx_id, original_ty) in &scope_locals {
                    let resolved = subst.resolve(original_ty);
                    resolved_types.insert(*ctx_id, resolved);
                }
            }
        }

        ResolvedGraph { resolutions, resolved_types, unit_outputs, errors }
    }

    // ── Helpers ──────────────────────────────────────────────────────

    fn all_unit_ids(&self) -> Vec<UnitId> {
        let mut ids: Vec<UnitId> = self.units.iter().map(|u| u.id).collect();
        ids.extend(self.externs.iter().map(|e| e.id));
        ids
    }

    fn build_unit_map(&self) -> FxHashMap<UnitId, UnitEntry<'_>> {
        let mut map = FxHashMap::default();
        for unit in &self.units {
            map.insert(unit.id, UnitEntry::Unit(unit));
        }
        for ext in &self.externs {
            map.insert(ext.id, UnitEntry::Extern(ext));
        }
        map
    }

    fn build_context_binding_map(&self) -> FxHashMap<ContextId, &ContextBinding> {
        let mut map = FxHashMap::default();
        for scope in &self.scopes {
            for binding in &scope.bindings {
                map.insert(binding.id, binding);
            }
        }
        map
    }

    fn build_unit_scope_map(&self) -> FxHashMap<UnitId, ScopeId> {
        let mut map = FxHashMap::default();
        for scope in &self.scopes {
            for &unit_id in &scope.units {
                map.insert(unit_id, scope.id);
            }
        }
        map
    }

    /// Allocate fresh type variables for ScopeLocal bindings.
    /// Apply constraints if present.
    fn allocate_scope_locals(
        &self,
        scc_ids: &[UnitId],
        unit_to_scope: &FxHashMap<UnitId, ScopeId>,
        subst: &mut TySubst,
        resolved_types: &mut FxHashMap<ContextId, Ty>,
    ) -> FxHashMap<ContextId, Ty> {
        let mut locals = FxHashMap::default();
        // Find all scopes containing SCC units.
        let mut scope_ids: FxHashSet<ScopeId> = FxHashSet::default();
        for id in scc_ids {
            if let Some(&scope_id) = unit_to_scope.get(id) {
                scope_ids.insert(scope_id);
            }
        }
        for scope in &self.scopes {
            if !scope_ids.contains(&scope.id) { continue; }
            for binding in &scope.bindings {
                if !matches!(binding.source, ContextSource::ScopeLocal) { continue; }
                if resolved_types.contains_key(&binding.id) { continue; }

                let var = if let Some(ref constraint) = binding.constraint {
                    // Apply structural constraint.
                    // e.g., Sequence<β, O, Pure> — allocate fresh vars for β and O.
                    let constrained = self.instantiate_constraint(constraint, subst);
                    constrained
                } else {
                    subst.fresh_var()
                };
                resolved_types.insert(binding.id, var.clone());
                locals.insert(binding.id, var);
            }
        }
        locals
    }

    /// Instantiate a constraint type: replace all Ty::Var and Origin::Var
    /// with fresh variables from the graph engine's subst.
    /// This is necessary because the constraint was built with a different subst
    /// (e.g., during lowering) and its variables are not in this subst's space.
    /// Instantiate a constraint type: replace all Ty::Var, Origin::Var, Effect::Var
    /// with fresh variables from the graph engine's subst.
    /// This is necessary because the constraint was built with a different subst
    /// (e.g., during lowering) and its variables are not in this subst's space.
    ///
    /// Every Ty variant that can contain nested types or variables must be handled
    /// explicitly. Concrete leaf types (Int, String, Bool, etc.) pass through unchanged.
    fn instantiate_constraint(&self, constraint: &Ty, subst: &mut TySubst) -> Ty {
        use crate::ty::{Effect, FnKind};
        match constraint {
            Ty::Var(_) => subst.fresh_var(),
            Ty::Infer(_) => subst.fresh_var(),
            Ty::Error(_) => Ty::error(),

            // Leaf types — no nested variables possible.
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit | Ty::Range
            | Ty::Byte | Ty::Opaque(_) => constraint.clone(),

            // Collection types — inner + origin/effect variables.
            Ty::List(inner) => Ty::List(Box::new(self.instantiate_constraint(inner, subst))),
            Ty::Deque(inner, _origin) => {
                Ty::Deque(Box::new(self.instantiate_constraint(inner, subst)), subst.fresh_origin())
            }
            Ty::Sequence(inner, _origin, effect) => {
                let new_effect = match effect {
                    Effect::Var(_) => subst.fresh_effect_var(),
                    other => *other,
                };
                Ty::Sequence(
                    Box::new(self.instantiate_constraint(inner, subst)),
                    subst.fresh_origin(),
                    new_effect,
                )
            }
            Ty::Iterator(inner, effect) => {
                let new_effect = match effect {
                    Effect::Var(_) => subst.fresh_effect_var(),
                    other => *other,
                };
                Ty::Iterator(Box::new(self.instantiate_constraint(inner, subst)), new_effect)
            }
            Ty::Option(inner) => Ty::Option(Box::new(self.instantiate_constraint(inner, subst))),
            Ty::Tuple(elems) => {
                Ty::Tuple(elems.iter().map(|e| self.instantiate_constraint(e, subst)).collect())
            }

            // Composite types — recurse into all nested types.
            Ty::Object(fields) => {
                Ty::Object(fields.iter().map(|(k, v)| (*k, self.instantiate_constraint(v, subst))).collect())
            }
            Ty::Fn { params, ret, kind, captures, effect } => {
                let new_effect = match effect {
                    Effect::Var(_) => subst.fresh_effect_var(),
                    other => *other,
                };
                Ty::Fn {
                    params: params.iter().map(|p| self.instantiate_constraint(p, subst)).collect(),
                    ret: Box::new(self.instantiate_constraint(ret, subst)),
                    kind: *kind,
                    captures: captures.iter().map(|c| self.instantiate_constraint(c, subst)).collect(),
                    effect: new_effect,
                }
            }
            Ty::Enum { name, variants } => {
                Ty::Enum {
                    name: *name,
                    variants: variants.iter().map(|(tag, payload)| {
                        (*tag, payload.as_ref().map(|ty| Box::new(self.instantiate_constraint(ty, subst))))
                    }).collect(),
                }
            }
        }
    }

    fn build_registry_for_unit(
        unit: &CompilationUnit,
        interner: &Interner,
        resolved_types: &FxHashMap<ContextId, Ty>,
        unit_outputs: &FxHashMap<UnitId, Ty>,
        context_to_binding: &FxHashMap<ContextId, &ContextBinding>,
        subst: &TySubst,
    ) -> ContextTypeRegistry {
        let mut ctx: FxHashMap<acvus_utils::Astr, Ty> = FxHashMap::default();
        for (&name, &ctx_id) in &unit.name_to_id {
            let ty = Self::resolve_context_type(ctx_id, resolved_types, unit_outputs, context_to_binding, subst);
            ctx.insert(name, ty);
        }
        ContextTypeRegistry::all_system(ctx)
    }

    fn resolve_context_type(
        ctx_id: ContextId,
        resolved_types: &FxHashMap<ContextId, Ty>,
        unit_outputs: &FxHashMap<UnitId, Ty>,
        context_to_binding: &FxHashMap<ContextId, &ContextBinding>,
        subst: &TySubst,
    ) -> Ty {
        // First check resolved_types (scope locals, externals).
        if let Some(ty) = resolved_types.get(&ctx_id) {
            return subst.resolve(ty);
        }
        // Check binding source.
        if let Some(binding) = context_to_binding.get(&ctx_id) {
            match &binding.source {
                ContextSource::Derived(dep_id, transform) => {
                    if let Some(dep_ty) = unit_outputs.get(dep_id) {
                        let resolved = subst.resolve(dep_ty);
                        match transform {
                            TypeTransform::Identity => resolved,
                            TypeTransform::ElemOf => {
                                resolved.elem_of().cloned().unwrap_or_else(Ty::error)
                            }
                        }
                    } else {
                        Ty::error()
                    }
                }
                ContextSource::ScopeLocal => {
                    // Should already be in resolved_types (allocated by allocate_scope_locals).
                    Ty::error()
                }
                ContextSource::External => {
                    resolved_types.get(&ctx_id).cloned().unwrap_or_else(Ty::error)
                }
            }
        } else {
            Ty::error()
        }
    }


    fn typecheck_unit(
        &self,
        interner: &Interner,
        unit: &CompilationUnit,
        registry: &ContextTypeRegistry,
        hint: Option<&Ty>,
        subst: &mut TySubst,
    ) -> Result<TypeResolution<Unchecked>, Vec<MirError>> {
        match unit.kind {
            SourceKind::Script => {
                let script = acvus_ast::parse_script(interner, interner.resolve(unit.source))
                    .map_err(|e| vec![MirError {
                        kind: crate::error::MirErrorKind::ParseError(format!("parse: {e}")),
                        span: acvus_ast::Span::ZERO,
                    }])?;
                crate::typecheck_script(interner, &script, registry, hint, subst)
            }
            SourceKind::Template => {
                let template = acvus_ast::parse(interner, interner.resolve(unit.source))
                    .map_err(|e| vec![MirError {
                        kind: crate::error::MirErrorKind::ParseError(format!("parse: {e}")),
                        span: acvus_ast::Span::ZERO,
                    }])?;
                crate::typecheck_template(interner, &template, registry, subst)
            }
        }
    }

    fn register_unit_output(
        unit_id: UnitId,
        tail_ty: &Ty,
        unit_outputs: &mut FxHashMap<UnitId, Ty>,
    ) {
        unit_outputs.insert(unit_id, tail_ty.clone());
    }

    fn register_extern_output(
        ext: &ExternDecl,
        unit_outputs: &mut FxHashMap<UnitId, Ty>,
    ) {
        unit_outputs.insert(ext.id, ext.output_ty.clone());
    }

    fn process_extern(
        ext: &ExternDecl,
        unit_outputs: &FxHashMap<UnitId, Ty>,
        subst: &mut TySubst,
        errors: &mut Vec<GraphError>,
    ) {
        let mut errs = Vec::new();
        for (input_id, expected_ty) in &ext.inputs {
            if let Some(actual_ty) = unit_outputs.get(input_id) {
                // Use covariant unification: actual ≤ expected.
                // e.g., Deque<T,O> ≤ Sequence<T,O,E> is valid.
                if subst.unify(actual_ty, expected_ty, crate::ty::Polarity::Covariant).is_err()
                    && !matches!(actual_ty, Ty::Error(_))
                {
                    errs.push(MirError {
                        kind: crate::error::MirErrorKind::UnificationFailure {
                            expected: expected_ty.clone(),
                            got: actual_ty.clone(),
                        },
                        span: acvus_ast::Span::ZERO,
                    });
                }
            }
        }
        if !errs.is_empty() {
            errors.push(GraphError { unit: ext.id, errors: errs });
        }
    }
}

// ── Tarjan's SCC algorithm ───────────────────────────────────────────

fn tarjan_scc(n: usize, deps: &[FxHashSet<usize>]) -> Vec<Vec<usize>> {
    struct State {
        index: usize,
        indices: Vec<Option<usize>>,
        lowlinks: Vec<usize>,
        on_stack: Vec<bool>,
        stack: Vec<usize>,
        sccs: Vec<Vec<usize>>,
    }

    let mut state = State {
        index: 0,
        indices: vec![None; n],
        lowlinks: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        sccs: Vec::new(),
    };

    fn strongconnect(v: usize, deps: &[FxHashSet<usize>], state: &mut State) {
        state.indices[v] = Some(state.index);
        state.lowlinks[v] = state.index;
        state.index += 1;
        state.stack.push(v);
        state.on_stack[v] = true;

        for &w in &deps[v] {
            if state.indices[w].is_none() {
                strongconnect(w, deps, state);
                state.lowlinks[v] = state.lowlinks[v].min(state.lowlinks[w]);
            } else if state.on_stack[w] {
                state.lowlinks[v] = state.lowlinks[v].min(state.indices[w].unwrap());
            }
        }

        if state.lowlinks[v] == state.indices[v].unwrap() {
            let mut scc = Vec::new();
            loop {
                let w = state.stack.pop().unwrap();
                state.on_stack[w] = false;
                scc.push(w);
                if w == v { break; }
            }
            state.sccs.push(scc);
        }
    }

    for i in 0..n {
        if state.indices[i].is_none() {
            strongconnect(i, deps, &mut state);
        }
    }

    state.sccs
}
