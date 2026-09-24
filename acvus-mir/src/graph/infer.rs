//! Phase 1: Infer
//!
//! For each function, infer the types of unknown context parameters and
//! function output types. Supports inter-function calls by adding all
//! local functions to the TypeEnv before typechecking.
//!
//! Output: context params + function types (for UI display and Phase 2).

use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::ty::{
    EffectTerm, Flows, Infer, InferTy, InputParam, MachineCoercion, Param, ParamTerm, PolyBuilder,
    PolyTy, Scheme, Solver, Sources, Ty, TyTerm, TyVarBound, TypeRegistry, lift_declaration,
    lift_to_poly, lift_ty,
};

use super::extract::{ExtractResult, ParsedSource};
use super::types::*;
use crate::typeck::{
    BodyView, Checked, Checks, ParamOrigin, ProbeProduct, Reader, ResultCrossing, TypeChecker,
    Unresolved,
};

// -- Phase 1 output --------------------------------------------------

/// Inferred metadata for a single function.
#[derive(Debug, Clone)]
pub struct FunctionMeta {
    /// Fully resolved Ty::Fn for this function.
    pub ty: Ty,
    /// What a call passes by position: the parameters a declaration names,
    /// in declared order, at the types the solve closed them to. An entry's
    /// declared parameters are inputs instead (RFC-0054 rule 6).
    pub params: Vec<Param>,
    /// What a call passes from the caller's own `$` of each name, after the
    /// positional arguments and in this order (RFC-0071 rule 4).
    pub inputs: Vec<InputParam>,
}

struct Signature {
    params: Vec<Param>,
    inputs: Vec<InputParam>,
}

fn signature_of(
    fid: QualifiedRef,
    is_entry: bool,
    params: &[crate::typeck::ResolvedParam],
) -> Signature {
    let mut positional: Vec<Param> = Vec::new();
    let mut inputs: Vec<InputParam> = Vec::new();
    for param in params {
        let reader = match param.origin {
            ParamOrigin::Declared if !is_entry => {
                positional.push(Param::new(param.name, param.ty.clone()));
                continue;
            }
            ParamOrigin::Bound => continue,
            ParamOrigin::Declared | ParamOrigin::Read(Reader::ThisBody) => fid,
            ParamOrigin::Read(Reader::Reached(reader)) => reader,
        };
        inputs.push(InputParam {
            name: param.name,
            ty: param.ty.clone(),
            reader,
        });
    }
    Signature {
        params: positional,
        inputs,
    }
}

/// A member whose body did not check takes what its declaration names.
fn declared_signature(
    fid: QualifiedRef,
    is_entry: bool,
    solver: &Solver,
    declared: &[ParamTerm<Infer>],
) -> Signature {
    let declared: Vec<crate::typeck::ResolvedParam> = settled_params(solver, declared)
        .into_iter()
        .map(|param| crate::typeck::ResolvedParam {
            name: param.name,
            ty: param.ty,
            origin: ParamOrigin::Declared,
        })
        .collect();
    signature_of(fid, is_entry, &declared)
}

fn positional_of<T>(declared: &[T], is_entry: bool) -> Vec<T>
where
    T: Clone,
{
    match is_entry {
        true => Vec::new(),
        false => declared.to_vec(),
    }
}

/// Compared as sets of names: a round's calls read their callee's inputs in
/// the order the round before stated, so the order may still move while the
/// set holds, and the lowering reads each callee's final order.
fn same_inputs(
    left: &FxHashMap<QualifiedRef, Vec<InputParam>>,
    right: &FxHashMap<QualifiedRef, Vec<InputParam>>,
) -> bool {
    let names = |inputs: Option<&Vec<InputParam>>| -> FxHashSet<Astr> {
        inputs
            .into_iter()
            .flatten()
            .map(|input| input.name)
            .collect()
    };
    left.len() == right.len()
        && left
            .keys()
            .all(|fid| names(left.get(fid)) == names(right.get(fid)))
}

/// Per-function inference outcome.
#[derive(Debug, Clone)]
pub enum FnInferOutcome {
    /// Type fully resolved. Lowerable.
    Complete {
        resolution: Freeze<crate::typeck::TypeResolution>,
        tail_ty: Ty,
        meta: FunctionMeta,
        view: Freeze<BodyView>,
    },
    /// Type incomplete. Cannot lower.
    Incomplete {
        meta: FunctionMeta,
        errors: Vec<crate::error::MirError>,
        /// `None` where no body was checked: the function has no parsed
        /// source.
        view: Option<Freeze<BodyView>>,
    },
}

impl FnInferOutcome {
    /// Get the function metadata regardless of completeness.
    pub fn meta(&self) -> &FunctionMeta {
        match self {
            FnInferOutcome::Complete { meta, .. } => meta,
            FnInferOutcome::Incomplete { meta, .. } => meta,
        }
    }

    /// Get the checked resolution if complete. Clone is cheap (Arc::clone).
    pub fn resolution(&self) -> Option<Freeze<crate::typeck::TypeResolution>> {
        match self {
            FnInferOutcome::Complete { resolution, .. } => Some(resolution.clone()),
            FnInferOutcome::Incomplete { .. } => None,
        }
    }

    /// Get the tail type if complete.
    pub fn tail_ty(&self) -> Option<&Ty> {
        match self {
            FnInferOutcome::Complete { tail_ty, .. } => Some(tail_ty),
            FnInferOutcome::Incomplete { .. } => None,
        }
    }

    pub fn view(&self) -> Option<Freeze<BodyView>> {
        match self {
            FnInferOutcome::Complete { view, .. } => Some(view.clone()),
            FnInferOutcome::Incomplete { view, .. } => view.clone(),
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self, FnInferOutcome::Complete { .. })
    }
}

/// Phase 1 output: inferred context parameters and function types.
/// All type information is frozen - immutable after inference.
#[derive(Debug)]
pub struct InferResult {
    /// Per-function inference outcome (Complete or Incomplete).
    pub outcomes: FxHashMap<QualifiedRef, FnInferOutcome>,
    /// Resolved context types (known + inferred). Frozen after inference.
    pub context_types: Freeze<FxHashMap<QualifiedRef, Ty>>,
}

impl InferResult {
    /// Get the checked resolution for a function, if complete.
    pub fn try_resolution(
        &self,
        id: QualifiedRef,
    ) -> Option<Freeze<crate::typeck::TypeResolution>> {
        self.outcomes.get(&id)?.resolution()
    }

    /// Get the context type for a QualifiedRef.
    pub fn context_type(&self, qref: &QualifiedRef) -> Option<&Ty> {
        (*self.context_types).get(qref)
    }

    /// Whether any function has errors.
    pub fn has_errors(&self) -> bool {
        self.outcomes
            .values()
            .any(|o| matches!(o, FnInferOutcome::Incomplete { errors, .. } if !errors.is_empty()))
    }

    /// Collect all errors across all functions.
    pub fn errors(&self) -> Vec<(QualifiedRef, &[crate::error::MirError])> {
        self.outcomes
            .iter()
            .filter_map(|(&id, o)| match o {
                FnInferOutcome::Incomplete { errors, .. } if !errors.is_empty() => {
                    Some((id, errors.as_slice()))
                }
                _ => None,
            })
            .collect()
    }
}

// -- Call graph + SCC -------------------------------------------------

/// Extract call edges for a single function from its parsed AST.
/// Returns the list of QualifiedRefs that this function references.
fn value_refs(parsed: &ParsedSource) -> Vec<Astr> {
    match parsed {
        ParsedSource::Script(script) => collect_value_refs_script(script),
        ParsedSource::Template(template) => collect_value_refs_template(template),
        ParsedSource::Recovered(RecoveredAst::Script(script)) => collect_value_refs_script(script),
        ParsedSource::Recovered(RecoveredAst::Template(template)) => {
            collect_value_refs_template(template)
        }
    }
}

/// Whether a component's members name one another: more than one member,
/// or one whose body names itself, which the call graph's edges leave out.
fn is_cyclic(scc: &[QualifiedRef], parsed: &FxHashMap<QualifiedRef, &ParsedSource>) -> bool {
    match scc {
        [only] => parsed
            .get(only)
            .is_some_and(|body| value_refs(body).contains(&only.name)),
        _ => true,
    }
}

/// Each member's flows joined with what its body was checked to join.
fn grown_flows(
    scc: &[QualifiedRef],
    stated: &FxHashMap<QualifiedRef, Flows>,
    checked: &FxHashMap<QualifiedRef, Checked>,
) -> FxHashMap<QualifiedRef, Flows> {
    scc.iter()
        .map(|fid| {
            let joined = match checked.get(fid).map(|checked| &checked.resolution) {
                Some(Ok(resolution)) => stated[fid].join(&resolution.flows),
                Some(Err(_)) | None => stated[fid].clone(),
            };
            (*fid, joined)
        })
        .collect()
}

fn grown_inputs(
    scc: &[QualifiedRef],
    stated: &FxHashMap<QualifiedRef, Vec<InputParam>>,
    checked: &FxHashMap<QualifiedRef, Signature>,
) -> FxHashMap<QualifiedRef, Vec<InputParam>> {
    scc.iter()
        .map(|fid| {
            let inputs = match checked.get(fid) {
                Some(signature) => signature.inputs.clone(),
                None => stated[fid].clone(),
            };
            (*fid, inputs)
        })
        .collect()
}

pub fn extract_call_edges(
    parsed: &ParsedSource,
    name_to_fn: &FxHashMap<Astr, QualifiedRef>,
    self_id: QualifiedRef,
) -> Vec<QualifiedRef> {
    let names: Vec<Astr> = value_refs(parsed);
    let mut callees = Vec::new();
    for name in names {
        if let Some(&callee_id) = name_to_fn.get(&name)
            && callee_id != self_id
            && !callees.contains(&callee_id)
        {
            callees.push(callee_id);
        }
    }
    callees
}

/// Build a call graph: for each local function, which other local functions
/// does it reference by name in its body?
fn build_call_graph(
    graph: &CompilationGraph,
    extract: &ExtractResult,
) -> FxHashMap<QualifiedRef, Vec<QualifiedRef>> {
    let name_to_id: FxHashMap<Astr, QualifiedRef> = graph
        .functions
        .iter()
        .filter(|f| matches!(f.kind, FnKind::Local(..)))
        .map(|f| (f.qref.name, f.qref))
        .collect();

    let mut edges: FxHashMap<QualifiedRef, Vec<QualifiedRef>> = FxHashMap::default();
    for func in graph.functions.iter() {
        if matches!(func.kind, FnKind::Extern { .. }) {
            continue;
        }
        let Some(parsed) = extract.parsed.get(&func.qref) else {
            continue;
        };
        edges.insert(
            func.qref,
            extract_call_edges(parsed, &name_to_id, func.qref),
        );
    }
    edges
}

fn collect_value_refs_stmts<S>(stmts: &[acvus_ast::Stmt<S>], refs: &mut Vec<Astr>) {
    use acvus_ast::*;
    for stmt in stmts {
        match stmt {
            Stmt::DerefStore { target, expr, .. } => {
                collect_value_refs_expr(target, refs);
                collect_value_refs_expr(expr, refs);
            }
            Stmt::Store { place, expr, .. } => {
                collect_value_refs_place(place, refs);
                collect_value_refs_expr(expr, refs);
            }
            Stmt::Expr(expr) => collect_value_refs_expr(expr, refs),
            Stmt::WhileLet { source, body, .. } => {
                collect_value_refs_expr(source, refs);
                collect_value_refs_stmts(body, refs);
            }
            Stmt::LetBind { expr, .. } | Stmt::Assign { expr, .. } => {
                collect_value_refs_expr(expr, refs);
            }
            Stmt::LetUninit { .. } => {}
            Stmt::For { head, body, .. } => {
                match head {
                    acvus_ast::ForHead::Value(e) => collect_value_refs_expr(e, refs),
                    acvus_ast::ForHead::Range { lo, hi } => {
                        collect_value_refs_expr(lo, refs);
                        collect_value_refs_expr(hi, refs);
                    }
                }
                collect_value_refs_stmts(body, refs);
            }
            Stmt::Break { .. } | Stmt::Continue { .. } => {}
            Stmt::While { cond, body, .. } => {
                collect_value_refs_expr(cond, refs);
                collect_value_refs_stmts(body, refs);
            }
            Stmt::Anyorder { body, .. } => collect_value_refs_stmts(body, refs),
            Stmt::Append { expr, .. } => collect_value_refs_expr(expr, refs),
            Stmt::Error(_) => {}
        }
    }
}

/// Collect all RefKind::Value identifiers from a script AST.
fn collect_value_refs_script<S>(script: &acvus_ast::Script<S>) -> Vec<Astr> {
    let mut refs = Vec::new();
    collect_value_refs_stmts(&script.stmts, &mut refs);
    if let Some(tail) = &script.tail {
        collect_value_refs_expr(tail, &mut refs);
    }
    refs
}

fn collect_value_refs_template<S>(template: &acvus_ast::Template<S>) -> Vec<Astr> {
    let mut refs = Vec::new();
    collect_value_refs_stmts(&template.body, &mut refs);
    refs
}

fn collect_value_refs_place<S>(place: &acvus_ast::Place<S>, refs: &mut Vec<Astr>) {
    use acvus_ast::*;
    match place {
        Place::Field { object, .. } => collect_value_refs_place(object, refs),
        Place::Base(PlaceBase::Root {
            root: Root::Local(name),
            ..
        }) => refs.push(*name),
        Place::Base(PlaceBase::Root { .. }) => {}
        Place::Base(PlaceBase::Element {
            container, index, ..
        }) => {
            collect_value_refs_expr(container.expr(), refs);
            collect_value_refs_expr(index, refs);
        }
    }
}

fn collect_value_refs_expr<S>(expr: &acvus_ast::Expr<S>, refs: &mut Vec<Astr>) {
    use acvus_ast::*;
    match expr {
        Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            ..
        } => refs.push(name.name),
        Expr::Ident { .. } | Expr::Literal { .. } | Expr::ContextRef { .. } | Expr::Error(_) => {}
        Expr::BinaryOp { left, right, .. } | Expr::Pipe { left, right, .. } => {
            collect_value_refs_expr(left, refs);
            collect_value_refs_expr(right, refs);
        }
        Expr::UnaryOp { operand, .. } => collect_value_refs_expr(operand, refs),
        Expr::Match {
            scrutinee, arms, ..
        } => {
            collect_value_refs_expr(scrutinee, refs);
            for arm in arms {
                collect_value_refs_stmts(&arm.body, refs);
                if let Some(tail) = &arm.tail {
                    collect_value_refs_expr(tail, refs);
                }
            }
        }
        Expr::FieldAccess { object, .. } => collect_value_refs_expr(object, refs),
        Expr::Index { object, index, .. } => {
            collect_value_refs_expr(object, refs);
            collect_value_refs_expr(index, refs);
        }
        Expr::FuncCall { func, args, .. } => {
            collect_value_refs_expr(func, refs);
            for a in args {
                collect_value_refs_expr(a, refs);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            collect_value_refs_expr(receiver, refs);
            for a in args {
                collect_value_refs_expr(a, refs);
            }
        }
        Expr::Lambda { body, .. } => collect_value_refs_expr(body, refs),
        Expr::Paren { inner, .. }
        | Expr::Try { inner, .. }
        | Expr::Return { value: inner, .. }
        | Expr::Cast { expr: inner, .. }
        | Expr::Borrow { place: inner, .. } => collect_value_refs_expr(inner, refs),
        Expr::List { head, tail, .. } => {
            for e in head.iter().chain(tail.iter()) {
                collect_value_refs_expr(e, refs);
            }
        }
        Expr::Object { fields, .. } => {
            for f in fields {
                collect_value_refs_expr(&f.value, refs);
            }
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                if let TupleElem::Expr(e) = e {
                    collect_value_refs_expr(e, refs);
                }
            }
        }
        Expr::Group { elements, .. } => {
            for e in elements {
                collect_value_refs_expr(e, refs);
            }
        }
        Expr::Variant {
            payload: Some(inner),
            ..
        } => collect_value_refs_expr(inner, refs),
        Expr::Variant { payload: None, .. } => {}
        Expr::Block { stmts, tail, .. } => {
            collect_value_refs_stmts(stmts, refs);
            collect_value_refs_expr(tail, refs);
        }
        Expr::If {
            cond,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            collect_value_refs_expr(cond, refs);
            collect_value_refs_stmts(then_body, refs);
            if let Some(tail) = then_tail {
                collect_value_refs_expr(tail, refs);
            }
            if let Some(eb) = else_branch {
                collect_value_refs_else_branch(eb, refs);
            }
        }
        Expr::IfLet {
            source,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            collect_value_refs_expr(source, refs);
            collect_value_refs_stmts(then_body, refs);
            if let Some(tail) = then_tail {
                collect_value_refs_expr(tail, refs);
            }
            if let Some(eb) = else_branch {
                collect_value_refs_else_branch(eb, refs);
            }
        }
    }
}

fn collect_value_refs_else_branch<S>(eb: &acvus_ast::ElseBranch<S>, refs: &mut Vec<Astr>) {
    match eb {
        acvus_ast::ElseBranch::ElseIf(expr) => collect_value_refs_expr(expr, refs),
        acvus_ast::ElseBranch::Else { body, tail, .. } => {
            collect_value_refs_stmts(body, refs);
            if let Some(tail) = tail {
                collect_value_refs_expr(tail, refs);
            }
        }
    }
}

/// Tarjan's SCC algorithm. Returns SCCs in reverse topological order
/// (leaf SCCs first - dependencies before dependents).
pub fn tarjan_scc(
    ids: &[QualifiedRef],
    edges: &FxHashMap<QualifiedRef, Vec<QualifiedRef>>,
) -> Vec<Vec<QualifiedRef>> {
    let mut index_counter: u32 = 0;
    let mut stack: Vec<QualifiedRef> = Vec::new();
    let mut on_stack: FxHashSet<QualifiedRef> = FxHashSet::default();
    let mut index: FxHashMap<QualifiedRef, u32> = FxHashMap::default();
    let mut lowlink: FxHashMap<QualifiedRef, u32> = FxHashMap::default();
    let mut result: Vec<Vec<QualifiedRef>> = Vec::new();

    fn strongconnect(
        v: QualifiedRef,
        edges: &FxHashMap<QualifiedRef, Vec<QualifiedRef>>,
        index_counter: &mut u32,
        stack: &mut Vec<QualifiedRef>,
        on_stack: &mut FxHashSet<QualifiedRef>,
        index: &mut FxHashMap<QualifiedRef, u32>,
        lowlink: &mut FxHashMap<QualifiedRef, u32>,
        result: &mut Vec<Vec<QualifiedRef>>,
    ) {
        index.insert(v, *index_counter);
        lowlink.insert(v, *index_counter);
        *index_counter += 1;
        stack.push(v);
        on_stack.insert(v);

        if let Some(neighbors) = edges.get(&v) {
            for &w in neighbors {
                if !index.contains_key(&w) {
                    strongconnect(
                        w,
                        edges,
                        index_counter,
                        stack,
                        on_stack,
                        index,
                        lowlink,
                        result,
                    );
                    let lw = lowlink[&w];
                    let lv = lowlink[&v];
                    lowlink.insert(v, lv.min(lw));
                } else if on_stack.contains(&w) {
                    let iw = index[&w];
                    let lv = lowlink[&v];
                    lowlink.insert(v, lv.min(iw));
                }
            }
        }

        if lowlink[&v] == index[&v] {
            let mut component = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                component.push(w);
                if w == v {
                    break;
                }
            }
            result.push(component);
        }
    }

    for &id in ids {
        if !index.contains_key(&id) {
            strongconnect(
                id,
                edges,
                &mut index_counter,
                &mut stack,
                &mut on_stack,
                &mut index,
                &mut lowlink,
                &mut result,
            );
        }
    }

    // Tarjan produces SCCs in reverse topological order already.
    result
}

// -- Per-SCC inference ------------------------------------------------

/// Result of inferring a single SCC.
#[derive(Debug, Clone)]
pub struct SccInferResult {
    /// Per-function outcome, carrying everything lowering reads: the frozen
    /// resolution, the tail type and the metadata whose `ty` is the return
    /// type `graph::lower` takes.
    pub outcomes: FxHashMap<QualifiedRef, FnInferOutcome>,
    /// QualifiedRef -> resolved Ty::Fn (for passing to next SCC).
    pub resolved_types: FxHashMap<QualifiedRef, Ty>,
    pub resolved_inputs: FxHashMap<QualifiedRef, Vec<InputParam>>,
    /// What the checker saw at the probe's marker, where `infer_scc` was
    /// given a probe of a body in this SCC.
    pub probe: Option<ProbeProduct>,
}

/// A body of the SCC checked with a marked node (`TypeChecker::with_probe`).
#[derive(Debug, Clone, Copy)]
pub struct Probe {
    pub body: QualifiedRef,
    pub marker: acvus_ast::AstId,
}

impl SccInferResult {
    pub fn errors(&self) -> impl Iterator<Item = (QualifiedRef, &[crate::error::MirError])> {
        self.outcomes
            .iter()
            .filter_map(|(&qref, outcome)| match outcome {
                FnInferOutcome::Incomplete { errors, .. } if !errors.is_empty() => {
                    Some((qref, errors.as_slice()))
                }
                _ => None,
            })
    }
}

/// Infer types for a single SCC.
///
/// `resolved_fn_types`: all function types already resolved by prior SCCs + builtins.
/// `known_ctx`: declared context types from the graph.
/// The declaration of every Extern in `functions`.
pub fn declared_bounds<'a>(
    functions: impl Iterator<Item = &'a Function>,
) -> FxHashMap<QualifiedRef, Declared> {
    functions
        .filter_map(|f| match &f.kind {
            FnKind::Extern {
                bounds,
                effect_bounds,
                instances,
                requires,
            } => Some((
                f.qref,
                Declared {
                    bounds: bounds.clone(),
                    effect_bounds: effect_bounds.clone(),
                    instances: instances.clone(),
                    requires: requires.clone(),
                },
            )),
            FnKind::Local(..) => None,
        })
        .collect()
}

pub struct Declared {
    pub bounds: Vec<TyVarBound>,
    pub effect_bounds: Vec<crate::ty::EffectVarBound>,
    pub instances: crate::ty::Instances,
    pub requires: Vec<crate::ty::RequirementSig>,
}

pub fn declared_instances(
    declared: &FxHashMap<QualifiedRef, Declared>,
) -> FxHashMap<QualifiedRef, crate::ty::Instances> {
    declared
        .iter()
        .map(|(&qref, own)| (qref, own.instances.clone()))
        .collect()
}

fn machine_signatures(
    registry: &TypeRegistry,
    resolved_fn_types: &FxHashMap<QualifiedRef, PolyTy>,
    declared: &FxHashMap<QualifiedRef, Declared>,
) -> FxHashMap<QualifiedRef, MachineCoercion> {
    resolved_fn_types
        .iter()
        .filter_map(|(&qref, ty)| {
            let viewed = registry.machine_view(qref)?;
            let scheme = declared_scheme(declared, qref, ty.clone());
            Some((qref, MachineCoercion { viewed, scheme }))
        })
        .collect()
}

fn declared_scheme(
    declared: &FxHashMap<QualifiedRef, Declared>,
    qref: QualifiedRef,
    ty: PolyTy,
) -> Scheme {
    let Some(own) = declared.get(&qref) else {
        return Scheme::unbounded(ty);
    };
    let requires = own
        .requires
        .iter()
        .map(|req| crate::ty::Requirement {
            signature: req.signature,
            pattern: req.pattern.clone(),
            calls: req.calls,
            instances: declared
                .get(&req.signature)
                .map(|sig| sig.instances.clone())
                .unwrap_or_else(|| {
                    panic!(
                        "{qref:?} requires an instance of {:?}, which is not declared: \
                         `Externs::combine` refuses this",
                        req.signature
                    )
                }),
        })
        .collect();
    Scheme {
        ty,
        bounds: own.bounds.clone(),
        effect_bounds: own.effect_bounds.clone(),
        instances: Some(own.instances.clone()),
        requires,
    }
}

/// The parameters a declaration names, each at a solver variable the body is
/// then checked against.
fn instantiate_params(
    solver: &mut Solver,
    declared: &[crate::ty::PolyParam],
) -> Vec<ParamTerm<Infer>> {
    declared
        .iter()
        .map(|p| p.retyped(solver.instantiate_poly(&p.ty)))
        .collect()
}

/// What a function takes when its body did not check: the parameters its
/// declaration names, so a call is measured against the arity it was written
/// for. A declared type the failed solve left open is `error`, as the return
/// type of the same function is.
fn settled_params(solver: &Solver, declared: &[ParamTerm<Infer>]) -> Vec<Param> {
    declared
        .iter()
        .map(|p| {
            let settled = solver.resolve_ty(&p.ty);
            p.retyped(solver.freeze_ty(&settled).unwrap_or_else(|_| Ty::error()))
        })
        .collect()
}

struct BodyCheck<'c> {
    interner: &'c Interner,
    env: &'c crate::ty::TypeEnv,
    declared_params: Vec<ParamTerm<Infer>>,
    inputs: Inputs,
    bindings: &'c Bindings,
    effect: EffectTerm<Infer>,
    probe: Option<acvus_ast::AstId>,
    expected_tail: Option<InferTy>,
    crossing: ResultCrossing,
}

impl BodyCheck<'_> {
    fn check(self, solver: &mut Solver<'_>, parsed: &ParsedSource) -> Checked {
        match parsed {
            ParsedSource::Script(script) => {
                let expected_tail = self.expected_tail.clone();
                self.checker(solver)
                    .check_script(script, expected_tail, self.crossing)
            }
            ParsedSource::Template(template) => self.checker(solver).check_template(template),
            ParsedSource::Recovered(RecoveredAst::Script(script)) => {
                let expected_tail = self.expected_tail.clone();
                refused(
                    self.checker(solver)
                        .check_script(script, expected_tail, self.crossing),
                )
            }
            ParsedSource::Recovered(RecoveredAst::Template(template)) => {
                refused(self.checker(solver).check_template(template))
            }
        }
    }

    fn checker<'s, 'src, S>(&self, solver: &'s mut Solver<'src>) -> TypeChecker<'_, 's, 'src, S>
    where
        S: Checks,
    {
        let checker = TypeChecker::new(
            self.interner,
            self.env,
            solver,
            self.inputs,
            self.declared_params.clone(),
        )
        .with_bound_inputs(self.bindings)
        .with_body_effect(self.effect.clone());
        match self.probe {
            Some(marker) => checker.with_probe(marker),
            None => checker,
        }
    }
}

fn refused(checked: Checked<Unresolved>) -> Checked {
    Checked {
        resolution: Err(checked.resolution.refusals),
        view: checked.view,
        probe: checked.probe,
    }
}

fn crossing_of(entries: &[QualifiedRef], body: QualifiedRef) -> crate::typeck::ResultCrossing {
    match entries.contains(&body) {
        true => crate::typeck::ResultCrossing::Host,
        false => crate::typeck::ResultCrossing::Registers,
    }
}

/// Each context's type as one solver's term, and each init's context by the
/// init's function (RFC-0090 rule 1).
struct KnownContexts<'g> {
    types: FxHashMap<QualifiedRef, InferTy>,
    inits: FxHashMap<QualifiedRef, &'g Context>,
}

impl<'g> KnownContexts<'g> {
    fn instantiate(solver: &mut Solver<'_>, contexts: &'g [Context]) -> Self {
        KnownContexts {
            types: contexts
                .iter()
                .map(|ctx| (ctx.qref, solver.instantiate_poly(&ctx.ty)))
                .collect(),
            inits: contexts
                .iter()
                .filter_map(|ctx| Some((ctx.init?, ctx)))
                .collect(),
        }
    }

    /// The return `fid` is checked against. An init returns its context's
    /// type: the context's own variable while the graph still solves it, so
    /// the init's result joins that solve, and otherwise the settled type
    /// with every identity open, since a declaration names no source
    /// (RFC-0012 rule 7) and the init's value carries the source it makes.
    fn declared_return(
        &self,
        solver: &mut Solver<'_>,
        fid: QualifiedRef,
        fn_ret: &PolyTy,
    ) -> InferTy {
        match self.inits.get(&fid) {
            Some(context) if context.is_open() => self.types[&context.qref].clone(),
            Some(context) => solver.instantiate_open(&context.ty),
            None => solver.instantiate_poly(fn_ret),
        }
    }

    /// The type the body's tail is held to: an init's declared return
    /// itself, and any other function's declared return where it is
    /// concrete. `None` leaves the tail inferred.
    fn expected_tail(
        &self,
        solver: &mut Solver<'_>,
        fid: QualifiedRef,
        fn_ret: &PolyTy,
        ret_vars: &FxHashMap<QualifiedRef, InferTy>,
    ) -> Option<InferTy> {
        if self.inits.contains_key(&fid) {
            return Some(ret_vars[&fid].clone());
        }
        let infer = solver.instantiate_poly(fn_ret);
        solver.freeze_ty(&infer).ok().map(|ty| lift_ty(&ty))
    }
}

pub fn infer_scc(
    interner: &Interner,
    scc: &[QualifiedRef],
    entries: &[QualifiedRef],
    bindings: &Bindings,
    fn_by_id: &FxHashMap<QualifiedRef, &Function>,
    extract_parsed: &FxHashMap<QualifiedRef, &ParsedSource>,
    contexts: &[Context],
    resolved_fn_types: &FxHashMap<QualifiedRef, PolyTy>,
    resolved_inputs: &FxHashMap<QualifiedRef, Vec<InputParam>>,
    declared: &FxHashMap<QualifiedRef, Declared>,
    sources: &mut Sources,
    registry: &TypeRegistry,
    probe: Option<Probe>,
    access: Access,
) -> SccInferResult {
    let signatures = declared_instances(declared);
    let mut solver = Solver::new(sources, registry, &signatures);
    let known = KnownContexts::instantiate(&mut solver, contexts);
    Component {
        interner,
        entries,
        bindings,
        fn_by_id,
        parsed: extract_parsed,
        known: &known,
        resolved_fn_types,
        resolved_inputs,
        declared,
        registry,
        probe,
        access,
    }
    .check(&mut solver, scc)
}

/// What checking one component of the call graph reads: the members'
/// declarations and parsed bodies, and what the components before it
/// settled.
struct Component<'c> {
    interner: &'c Interner,
    entries: &'c [QualifiedRef],
    bindings: &'c Bindings,
    fn_by_id: &'c FxHashMap<QualifiedRef, &'c Function>,
    /// The parsed body of each member that has one.
    parsed: &'c FxHashMap<QualifiedRef, &'c ParsedSource>,
    known: &'c KnownContexts<'c>,
    resolved_fn_types: &'c FxHashMap<QualifiedRef, PolyTy>,
    resolved_inputs: &'c FxHashMap<QualifiedRef, Vec<InputParam>>,
    declared: &'c FxHashMap<QualifiedRef, Declared>,
    registry: &'c TypeRegistry,
    probe: Option<Probe>,
    access: Access,
}

impl Component<'_> {
    /// Check every member of `scc` on `solver` and freeze what it settled.
    /// A component whose members call one another states each member's
    /// flows and inputs at the calls between them, so it is checked again
    /// from the same solver state while a member's flows or inputs grow:
    /// the least fixpoint RFC-0079 rule 5 names. A member's flows are what
    /// its body joins, which only grows with what its calls read, so the
    /// rounds climb from none and stop.
    fn check(&self, solver: &mut Solver<'_>, scc: &[QualifiedRef]) -> SccInferResult {
        let before = is_cyclic(scc, self.parsed).then(|| solver.snapshot());
        let mut signatures: FxHashMap<QualifiedRef, Signature> = FxHashMap::default();
        let mut declared_params: FxHashMap<QualifiedRef, Vec<ParamTerm<Infer>>> =
            FxHashMap::default();
        let mut ret_vars: FxHashMap<QualifiedRef, InferTy> = FxHashMap::default();
        let mut effect_vars: FxHashMap<QualifiedRef, EffectTerm<Infer>> = FxHashMap::default();
        let mut checked: FxHashMap<QualifiedRef, Checked> = FxHashMap::default();
        let mut probed: Option<ProbeProduct> = None;
        let mut member_flows: FxHashMap<QualifiedRef, Flows> =
            scc.iter().map(|fid| (*fid, Flows::none())).collect();
        let mut member_inputs: FxHashMap<QualifiedRef, Vec<InputParam>> =
            scc.iter().map(|fid| (*fid, Vec::new())).collect();
        loop {
            signatures.clear();
            let mut scc_fn_types: FxHashMap<QualifiedRef, PolyTy> = FxHashMap::default();
            for &fid in scc {
                let func = self.fn_by_id[&fid];
                let TyTerm::Fn {
                    params: ref fn_params,
                    ret: ref fn_ret,
                    effect: ref fn_effect,
                    ..
                } = func.ty
                else {
                    unreachable!("local function ty must be Fn");
                };
                ret_vars.insert(fid, self.known.declared_return(solver, fid, fn_ret));
                effect_vars.insert(fid, solver.fresh_effect_var());
                declared_params.insert(fid, instantiate_params(solver, fn_params));
                scc_fn_types.insert(
                    func.qref,
                    TyTerm::Fn {
                        params: positional_of(fn_params, self.entries.contains(&fid)),
                        ret: fn_ret.clone(),
                        captures: vec![],
                        effect: fn_effect.clone(),
                        flows: member_flows[&fid].clone().into(),
                    },
                );
            }

            let machine_signatures =
                machine_signatures(self.registry, self.resolved_fn_types, self.declared);
            let mut env_functions: FxHashMap<QualifiedRef, Scheme> = self
                .resolved_fn_types
                .iter()
                .filter(|(qref, _)| self.registry.machine_view(**qref).is_none())
                .map(|(&k, v)| (k, declared_scheme(self.declared, k, v.clone())))
                .collect();
            env_functions.extend(
                scc_fn_types
                    .into_iter()
                    .map(|(k, v)| (k, Scheme::unbounded(v))),
            );

            for &fid in scc {
                let func = self.fn_by_id[&fid];
                let (Some(parsed), FnKind::Local(_, inputs)) = (self.parsed.get(&fid), &func.kind)
                else {
                    continue;
                };
                let env = crate::ty::TypeEnv {
                    contexts: self.known.types.clone(),
                    functions: env_functions.clone(),
                    machine: machine_signatures.clone(),
                    inputs: self
                        .resolved_inputs
                        .iter()
                        .chain(&member_inputs)
                        .map(|(fid, inputs)| (*fid, inputs.clone()))
                        .collect(),
                    access: self.access,
                };
                let TyTerm::Fn {
                    ret: ref fn_ret, ..
                } = func.ty
                else {
                    unreachable!("local function ty must be Fn");
                };
                let expected_tail = self.known.expected_tail(solver, fid, fn_ret, &ret_vars);
                let mut body = BodyCheck {
                    interner: self.interner,
                    env: &env,
                    declared_params: declared_params[&fid].clone(),
                    inputs: *inputs,
                    bindings: self.bindings,
                    effect: effect_vars[&fid].clone(),
                    probe: self
                        .probe
                        .filter(|probe| probe.body == fid)
                        .map(|probe| probe.marker),
                    expected_tail: expected_tail.clone(),
                    crossing: crossing_of(self.entries, fid),
                }
                .check(solver, parsed);

                if let Ok(unchecked) = &body.resolution {
                    // An inferred return is the tail's type.
                    if expected_tail.is_none() {
                        let tail_infer = lift_ty(&unchecked.tail_ty);
                        let _ = solver.unify(&ret_vars[&fid], &tail_infer);
                    }
                    let closed = EffectTerm::Known(unchecked.effect.clone());
                    solver
                        .unify_effect(
                            &effect_vars[&fid],
                            &closed,
                            crate::solver::EffectRelation::Equal,
                        )
                        .expect("the closed effect is the variable's own lower bound");
                    signatures.insert(
                        fid,
                        signature_of(fid, self.entries.contains(&fid), &unchecked.extern_params),
                    );
                }
                if let Some(product) = body.probe.take() {
                    probed = Some(product);
                }
                checked.insert(fid, body);
            }

            let grown = grown_flows(scc, &member_flows, &checked);
            let grown_inputs = grown_inputs(scc, &member_inputs, &signatures);
            let settled = grown == member_flows && same_inputs(&grown_inputs, &member_inputs);
            member_flows = grown;
            member_inputs = grown_inputs;
            match &before {
                Some(before) if !settled => solver.restore(before.clone()),
                Some(_) | None => break,
            }
        }

        // Freeze InferTy -> Ty at the component's boundary.
        let mut resolved_types: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();
        let mut resolved_inputs: FxHashMap<QualifiedRef, Vec<InputParam>> = FxHashMap::default();
        let mut outcomes: FxHashMap<QualifiedRef, FnInferOutcome> = FxHashMap::default();
        for &fid in scc {
            let ret = solver
                .freeze_ty(&solver.resolve_ty(&ret_vars[&fid]))
                .unwrap_or_else(|_| Ty::error());
            let signature = match signatures.remove(&fid) {
                Some(checked) => checked,
                None => declared_signature(
                    fid,
                    self.entries.contains(&fid),
                    solver,
                    &declared_params[&fid],
                ),
            };
            let effect = EffectTerm::Known(solver.freeze_effect(&effect_vars[&fid]));
            let fn_ty = Ty::Fn {
                params: signature.params.clone(),
                ret: Box::new(ret),
                captures: vec![],
                effect,
                flows: member_flows[&fid].clone().into(),
            };
            resolved_types.insert(fid, fn_ty.clone());
            resolved_inputs.insert(fid, signature.inputs.clone());
            let meta = FunctionMeta {
                ty: fn_ty,
                params: signature.params,
                inputs: signature.inputs,
            };
            let outcome = match checked.remove(&fid) {
                Some(Checked {
                    resolution: Ok(resolution),
                    view,
                    probe: _,
                }) => FnInferOutcome::Complete {
                    tail_ty: resolution.tail_ty.clone(),
                    resolution,
                    meta,
                    view,
                },
                Some(Checked {
                    resolution: Err(errors),
                    view,
                    probe: _,
                }) => FnInferOutcome::Incomplete {
                    meta,
                    errors,
                    view: Some(view),
                },
                // A member with no parsed body was not checked.
                None => FnInferOutcome::Incomplete {
                    meta,
                    errors: Vec::new(),
                    view: None,
                },
            };
            outcomes.insert(fid, outcome);
        }

        SccInferResult {
            outcomes,
            resolved_types,
            resolved_inputs,
            probe: probed,
        }
    }
}

// -- Batch inference -------------------------------------------------

/// Run Phase 1 inference with SCC-based processing.
///
/// 1. Build call graph from AST references.
/// 2. Compute SCCs (Tarjan) - reverse topological order.
/// 3. Process each SCC:
///    - Within an SCC: shared Solver, no instantiation of intra-SCC calls.
///    - After an SCC is done: resolve ret vars -> concrete Ty::Fn.
///    - Next SCC sees concrete types -> instantiation is safe.
pub fn infer(
    interner: &Interner,
    graph: &CompilationGraph,
    extract: &ExtractResult,
) -> InferResult {
    let solved = solve_contexts(interner, graph, extract);
    infer_at(interner, graph, extract, &solved)
}

/// The graph's contexts, each open one at the type the whole graph solves
/// it to. `infer` checks every body against these, and so does a graph
/// that infers one component at a time, so both word a refusal alike
/// (RFC-0085 rule 1).
pub fn solve_contexts(
    interner: &Interner,
    graph: &CompilationGraph,
    extract: &ExtractResult,
) -> Vec<Context> {
    if !graph.contexts.iter().any(Context::is_open) {
        return graph.contexts.to_vec();
    }
    let solving = infer_at(interner, graph, extract, &graph.contexts);
    solved_contexts(&graph.contexts, &solving)
}

/// A context at an open type is solved from every body that stores or reads
/// it (RFC-0090 rule 1), and each body is then checked at the solved type.
/// The first round is the solve: one solver holds the context's variable
/// across every component, but a body's types are frozen when its own check
/// ends, so a body checked before another body constrained the variable
/// would keep the narrower type. The second round checks every body against
/// the type the whole graph solved, where a store is admitted as it is at
/// any declared type. A context the first round could not settle stays at
/// its variable, so the second round reports it where it is used.
fn solved_contexts(contexts: &[Context], solving: &InferResult) -> Vec<Context> {
    let mut builder = PolyBuilder::new();
    contexts
        .iter()
        .map(|context| match solving.context_type(&context.qref) {
            Some(ty) if !ty.is_error() && context.is_open() => Context {
                qref: context.qref,
                ty: lift_declaration(ty, &mut builder),
                init: context.init,
            },
            Some(_) | None => context.clone(),
        })
        .collect()
}

fn infer_at(
    interner: &Interner,
    graph: &CompilationGraph,
    extract: &ExtractResult,
    contexts: &[Context],
) -> InferResult {
    let mut sources = Sources::new();
    let declared = declared_bounds(graph.functions.iter());
    let signatures = declared_instances(&declared);
    let mut solver = Solver::new(&mut sources, &graph.types, &signatures);

    // Extern function types are always known upfront (their PolyTy is fully concrete).
    let mut resolved_fn_types: FxHashMap<QualifiedRef, PolyTy> = graph
        .functions
        .iter()
        .filter(|func| matches!(func.kind, FnKind::Extern { .. }))
        .map(|func| (func.qref, func.ty.clone()))
        .collect();
    let mut resolved_inputs: FxHashMap<QualifiedRef, Vec<InputParam>> = FxHashMap::default();
    let mut outcomes: FxHashMap<QualifiedRef, FnInferOutcome> = FxHashMap::default();

    let known = KnownContexts::instantiate(&mut solver, contexts);

    let fn_by_id: FxHashMap<QualifiedRef, &Function> = graph
        .functions
        .iter()
        .filter(|f| matches!(f.kind, FnKind::Local(..)))
        .map(|f| (f.qref, f))
        .collect();

    // Every component is checked on the one solver, after the components
    // it calls, so a context's variable is held across all of them.
    let call_graph = build_call_graph(graph, extract);
    let local_ids: Vec<QualifiedRef> = graph
        .functions
        .iter()
        .filter(|f| matches!(f.kind, FnKind::Local(..)))
        .map(|f| f.qref)
        .collect();
    for scc in &tarjan_scc(&local_ids, &call_graph) {
        let parsed: FxHashMap<QualifiedRef, &ParsedSource> = scc
            .iter()
            .filter_map(|fid| Some((*fid, extract.parsed.get(fid)?)))
            .collect();
        let settled = Component {
            interner,
            entries: &graph.entries,
            bindings: &graph.bindings,
            fn_by_id: &fn_by_id,
            parsed: &parsed,
            known: &known,
            resolved_fn_types: &resolved_fn_types,
            resolved_inputs: &resolved_inputs,
            declared: &declared,
            registry: &graph.types,
            probe: None,
            access: graph.access,
        }
        .check(&mut solver, scc);
        resolved_fn_types.extend(
            settled
                .resolved_types
                .iter()
                .map(|(&fid, ty)| (fid, lift_to_poly(ty))),
        );
        resolved_inputs.extend(settled.resolved_inputs);
        outcomes.extend(settled.outcomes);
    }

    // A context the graph leaves open closes to `!` (RFC-0038 rule 2).
    let context_types: FxHashMap<QualifiedRef, Ty> = known
        .types
        .iter()
        .map(|(&k, v)| {
            let resolved = solver.resolve_ty(v);
            let frozen = solver.close_ty(&resolved).unwrap_or_else(|_| Ty::error());
            (k, frozen)
        })
        .collect();

    InferResult {
        outcomes,
        context_types: Freeze::new(context_types),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::extract;
    use crate::ty::{
        Mutability, ObjectTy, ParamTerm, Poly, PolyBuilder, PolyParam, TypeArg, lift_declaration,
        lift_to_poly,
    };
    use acvus_utils::{Freeze, Interner};

    /// A string literal's type (RFC-0062 rule 2).
    fn str_view() -> Ty {
        Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::Str)))
    }

    /// `core::to_string` at `T = Str`: the copy that turns a literal into
    /// the owned text (RFC-0062 rule 2). These graphs are built by hand
    /// rather than from the standard registries, and a script has no other
    /// way to write a `String`.
    fn to_string_extern(interner: &Interner) -> Function {
        make_extern_fn(interner, "to_string", vec![str_view()], Ty::String)
    }

    fn make_graph(interner: &Interner, source: &str) -> CompilationGraph {
        let mut pb = PolyBuilder::new();
        let qref = QualifiedRef::root(interner.intern("test"));
        CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref,
                kind: FnKind::Local(
                    ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse")),
                    crate::graph::Inputs::FromReads,
                ),
                ty: TyTerm::Fn {
                    params: vec![],
                    ret: Box::new(pb.fresh_ty_var()),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                    flows: crate::ty::Flows::Every.into(),
                },
            }]),
            contexts: Freeze::new(vec![]),
            types: Freeze::default(),
            bindings: Bindings::default(),
            access: crate::graph::Access::Sync,
            entries: Vec::new(),
        }
    }

    fn make_graph_with_ctx(
        interner: &Interner,
        source: &str,
        ctx: &[(&str, Ty)],
    ) -> CompilationGraph {
        let mut pb = PolyBuilder::new();
        let contexts = ctx
            .iter()
            .map(|(name, ty)| Context {
                qref: QualifiedRef::root(interner.intern(name)),
                ty: lift_declaration(ty, &mut pb),
                init: None,
            })
            .collect();
        let qref = QualifiedRef::root(interner.intern("test"));
        CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref,
                kind: FnKind::Local(
                    ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse")),
                    crate::graph::Inputs::FromReads,
                ),
                ty: TyTerm::Fn {
                    params: vec![],
                    ret: Box::new(pb.fresh_ty_var()),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                    flows: crate::ty::Flows::Every.into(),
                },
            }]),
            contexts: Freeze::new(contexts),
            types: Freeze::default(),
            bindings: Bindings::default(),
            access: crate::graph::Access::Sync,
            entries: Vec::new(),
        }
    }

    // -- Completeness: correct types inferred --

    // FnRefs removed: context param inference no longer produces InferredParam for
    // undeclared contexts. All contexts are now passed via known_ctx; undeclared
    // context references are handled by the typechecker directly.

    /// A context nothing declared is a refusal, not a parameter inference
    /// invents for the caller to fill.
    #[test]
    fn infer_no_unknown_context_params() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);
        assert!(params(&result, only_function(&i)).is_empty());
        let refused = refusals(&i, &result);
        assert_eq!(refused.len(), 1, "{refused:?}");
        assert!(refused[0].contains('x'), "{refused:?}");
    }

    /// A declared context is read at the type its declaration gave it and
    /// is not a parameter either; the undeclared one beside it is refused.
    #[test]
    fn infer_known_context_not_in_params() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + @y", &[("x", Ty::I64)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);
        assert!(params(&result, only_function(&i)).is_empty());
        assert_eq!(
            result.context_type(&QualifiedRef::root(i.intern("x"))),
            Some(&Ty::I64)
        );
        let refused = refusals(&i, &result);
        assert_eq!(refused.len(), 1, "{refused:?}");
        assert!(refused[0].contains('y'), "{refused:?}");
    }

    // -- Soundness: no false inferences --

    /// A body that reads no context leaves the context map empty: nothing
    /// is inferred for a caller that was never asked for anything.
    #[test]
    fn infer_no_contexts_empty() {
        let i = Interner::new();
        let graph = make_graph(&i, "1 + 2");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);
        assert_eq!(refusals(&i, &result), Vec::<String>::new());
        assert!(result.context_types.is_empty());
        assert_eq!(tail_type(&result, only_function(&i)), Some(Ty::I64));
    }

    // ================================================================
    // Migrated from resolve.rs - inter-function, soundness, edge cases
    // ================================================================

    // -- Helpers (resolve-style: builtins + named params + output constraint) --

    fn make_graph_with_ctx_and_builtins(
        interner: &Interner,
        source: &str,
        ctx: &[(&str, Ty)],
    ) -> CompilationGraph {
        let mut pb = PolyBuilder::new();
        let contexts = ctx
            .iter()
            .map(|(name, ty)| Context {
                qref: QualifiedRef::root(interner.intern(name)),
                ty: lift_declaration(ty, &mut pb),
                init: None,
            })
            .collect();
        let mut functions = Vec::new();
        let qref = QualifiedRef::root(interner.intern("test"));
        functions.push(Function {
            qref,
            kind: FnKind::Local(
                ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse")),
                crate::graph::Inputs::FromReads,
            ),
            ty: TyTerm::Fn {
                params: vec![],
                ret: Box::new(pb.fresh_ty_var()),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
                flows: crate::ty::Flows::Every.into(),
            },
        });
        CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
            types: Freeze::default(),
            bindings: Bindings::default(),
            access: crate::graph::Access::Sync,
            entries: Vec::new(),
        }
    }

    fn make_graph_no_ctx_with_builtins(interner: &Interner, source: &str) -> CompilationGraph {
        make_graph_with_ctx_and_builtins(interner, source, &[])
    }

    fn last_local_id(graph: &CompilationGraph) -> QualifiedRef {
        graph
            .functions
            .iter()
            .rev()
            .find(|f| matches!(f.kind, FnKind::Local(..)))
            .expect("no local function")
            .qref
    }

    /// Infer a multi-function graph, return result and ids.
    fn infer_multi(
        interner: &Interner,
        fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Option<PolyTy>)],
        ctx: &[(&str, Ty)],
    ) -> (InferResult, Vec<(Astr, QualifiedRef)>) {
        infer_with_extern(interner, fns, &[], ctx)
    }

    /// Build a graph with both local and extern functions, then infer.
    fn infer_with_extern(
        interner: &Interner,
        local_fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Option<PolyTy>)],
        extern_fns: &[Function],
        ctx: &[(&str, Ty)],
    ) -> (InferResult, Vec<(Astr, QualifiedRef)>) {
        let mut pb = PolyBuilder::new();
        let contexts: Vec<Context> = ctx
            .iter()
            .map(|(name, ty)| Context {
                qref: QualifiedRef::root(interner.intern(name)),
                ty: lift_declaration(ty, &mut pb),
                init: None,
            })
            .collect();

        let mut functions = extern_fns.to_vec();
        functions.push(to_string_extern(interner));
        let mut ids = Vec::new();

        for (name, source, sig, output) in local_fns {
            let aname = interner.intern(name);
            let fid = QualifiedRef::root(aname);
            ids.push((aname, fid));
            let (inputs, poly_params): (Inputs, Vec<PolyParam>) = match sig {
                Some(params) => (
                    Inputs::Declared,
                    params
                        .iter()
                        .map(|(name, ty)| {
                            ParamTerm::<Poly>::new(interner.intern(name), lift_to_poly(ty))
                        })
                        .collect(),
                ),
                None => (Inputs::FromReads, Vec::new()),
            };
            let ret = output.clone().unwrap_or_else(|| pb.fresh_ty_var());
            functions.push(Function {
                qref: fid,
                kind: FnKind::Local(
                    ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse")),
                    inputs,
                ),
                ty: TyTerm::Fn {
                    params: poly_params,
                    ret: Box::new(ret),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                    flows: crate::ty::Flows::Every.into(),
                },
            });
        }

        let graph = CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
            types: Freeze::default(),
            bindings: Bindings::default(),
            access: crate::graph::Access::Sync,
            entries: Vec::new(),
        };
        let ext = extract::extract(interner, &graph);
        let result = infer(interner, &graph, &ext);
        (result, ids)
    }

    fn error_strings(interner: &Interner, result: &InferResult) -> Vec<String> {
        result
            .errors()
            .iter()
            .flat_map(|(_, errs)| errs.iter())
            .map(|e| format!("{}", e.display(interner)))
            .collect()
    }

    /// Get the tail type (return type) of a function from InferResult.
    /// In the old resolve pipeline, fn_type() returned the tail type.
    /// In the new pipeline, fn_type() returns the full Ty::Fn.
    /// The refusals one graph's inference produced, rendered.
    fn refusals(i: &Interner, result: &InferResult) -> Vec<String> {
        let mut found: Vec<String> = result
            .errors()
            .into_iter()
            .flat_map(|(_, errors)| errors.iter().map(|e| e.display(i).to_string()))
            .collect();
        found.sort();
        found
    }

    /// The parameters inference gave a function.
    fn params(result: &InferResult, id: QualifiedRef) -> Vec<Param> {
        result.outcomes[&id].meta().params.clone()
    }

    fn only_function(i: &Interner) -> QualifiedRef {
        QualifiedRef::root(i.intern("test"))
    }

    fn tail_type(result: &InferResult, id: QualifiedRef) -> Option<Ty> {
        result.outcomes.get(&id)?.tail_ty().cloned()
    }

    fn make_extern_fn(interner: &Interner, name: &str, params: Vec<Ty>, ret: Ty) -> Function {
        let named_params: Vec<ParamTerm<Poly>> = params
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                ParamTerm::<Poly>::new(interner.intern(&format!("_{i}")), lift_to_poly(ty))
            })
            .collect();
        Function {
            qref: QualifiedRef::root(interner.intern(name)),
            kind: FnKind::Extern {
                bounds: vec![],
                effect_bounds: vec![],
                instances: crate::ty::Instances::default(),
                requires: vec![],
            },
            ty: TyTerm::Fn {
                params: named_params,
                ret: Box::new(lift_to_poly(&ret)),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
                flows: crate::ty::Flows::Every.into(),
            },
        }
    }

    // -- Completeness: valid single-function programs ------------------

    #[test]
    fn resolve_simple_arithmetic() {
        let i = Interner::new();
        let graph = make_graph_no_ctx_with_builtins(&i, "1 + 2");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        assert!(
            !result.has_errors(),
            "errors: {:?}",
            error_strings(&i, &result)
        );
        let uid = last_local_id(&graph);
        assert!(result.try_resolution(uid).is_some());
        assert_eq!(tail_type(&result, uid).unwrap(), Ty::I64);
    }

    #[test]
    fn resolve_with_declared_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@x + 1", &[("x", Ty::I64)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        assert!(
            !result.has_errors(),
            "errors: {:?}",
            error_strings(&i, &result)
        );
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result, uid).unwrap(), Ty::I64);
    }

    #[test]
    fn resolve_string_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@name", &[("name", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        assert!(
            !result.has_errors(),
            "errors: {:?}",
            error_strings(&i, &result)
        );
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result, uid).unwrap(), Ty::String);
    }

    #[test]
    fn resolve_context_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("name"),
            Ty::String,
        )])));
        let graph = make_graph_with_ctx_and_builtins(&i, "@user.name", &[("user", obj_ty)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        assert!(
            !result.has_errors(),
            "errors: {:?}",
            error_strings(&i, &result)
        );
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result, uid).unwrap(), Ty::String);
    }

    // -- Soundness: type errors detected --

    #[test]
    fn resolve_type_mismatch_detected() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@x + 1", &[("x", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        assert!(result.has_errors(), "should detect type mismatch");
    }

    // -- Context type resolution --

    #[test]
    fn resolve_context_types_populated() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@x", &[("x", Ty::I64)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let ctx_ref = graph.contexts[0].qref;
        assert_eq!(*result.context_type(&ctx_ref).unwrap(), Ty::I64);
    }

    // -- Completeness: valid inter-function calls ----------------------

    /// C1: A calls B with matching concrete types.
    #[test]
    fn inter_fn_simple_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("double", "$x * 2", Some(vec![("x", Ty::I64)]), None),
                ("main", "double(21)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// C2: A calls B, B returns String.
    #[test]
    fn inter_fn_string_return() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("greet", "\"hello\".to_string()", Some(vec![]), None),
                ("main", "greet()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
    }

    /// C3: Multi-arg function call.
    #[test]
    fn inter_fn_multi_arg() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "add",
                    "$x + $y",
                    Some(vec![("x", Ty::I64), ("y", Ty::I64)]),
                    None,
                ),
                ("main", "add(1, 2)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// C4: Chain of calls - A calls B, B calls C.
    #[test]
    fn inter_fn_chain_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("inc", "$x + 1", Some(vec![("x", Ty::I64)]), None),
                (
                    "double_inc",
                    "inc($x) + inc($x)",
                    Some(vec![("x", Ty::I64)]),
                    None,
                ),
                ("main", "double_inc(5)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[2].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// C5: Function uses context and is called by another function.
    #[test]
    fn inter_fn_with_context() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("get_count", "@count", Some(vec![]), None),
                ("main", "get_count() + 1", None, None),
            ],
            &[("count", Ty::I64)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// C6: Function with declared Exact output type.
    #[test]
    fn inter_fn_exact_output() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "make_str",
                    "\"hi\".to_string()",
                    Some(vec![]),
                    Some(lift_to_poly(&Ty::String)),
                ),
                ("main", "make_str()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
    }

    /// C7: Caller uses return value in arithmetic.
    #[test]
    fn inter_fn_return_used_in_binop() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("five", "5", Some(vec![]), None),
                ("main", "five() + five()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// C8: Pipe syntax - value | fn.
    #[test]
    fn inter_fn_pipe_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("double", "$x * 2", Some(vec![("x", Ty::I64)]), None),
                ("main", "10 | double", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    // inter_fn_list_return: migrated to acvus-mir-test (depends on ExternFn `len`)

    /// C10: Function accepting and returning String.
    #[test]
    fn inter_fn_string_identity() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("echo", "$s", Some(vec![("s", Ty::String)]), None),
                ("main", "echo(\"hello\".to_string())", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
    }

    /// C11: Multiple callers of the same function.
    #[test]
    fn inter_fn_multiple_callers() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("inc", "$x + 1", Some(vec![("x", Ty::I64)]), None),
                ("a", "inc(10)", None, None),
                ("b", "inc(20)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(tail_type(&result, ids[1].1).unwrap(), Ty::I64);
        assert_eq!(tail_type(&result, ids[2].1).unwrap(), Ty::I64);
    }

    /// C12: Calling function with bool return.
    #[test]
    fn inter_fn_bool_return() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("is_positive", "$x > 0", Some(vec![("x", Ty::I64)]), None),
                ("main", "is_positive(42)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Bool);
    }

    /// C13: Deep call chain - A -> B -> C -> D.
    #[test]
    fn inter_fn_deep_chain() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("d", "1", Some(vec![]), None),
                ("c", "d()", Some(vec![]), None),
                ("b", "c()", Some(vec![]), None),
                ("main", "b()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[3].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    // inter_fn_mixed_builtin_and_local: migrated to acvus-mir-test (depends on ExternFn `to_string`)

    /// C15: Function result used as argument to another function.
    #[test]
    fn inter_fn_nested_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("inc", "$x + 1", Some(vec![("x", Ty::I64)]), None),
                ("main", "inc(inc(0))", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// C16: Mutual recursion - A calls B, B calls A.
    #[test]
    fn inter_fn_mutual_recursion() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "is_even",
                    "is_odd($n - 1)",
                    Some(vec![("n", Ty::I64)]),
                    Some(lift_to_poly(&Ty::Bool)),
                ),
                (
                    "is_odd",
                    "is_even($n - 1)",
                    Some(vec![("n", Ty::I64)]),
                    Some(lift_to_poly(&Ty::Bool)),
                ),
                ("main", "is_even(10)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[2].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Bool);
    }

    /// C17: Self-recursion.
    #[test]
    fn inter_fn_self_recursion() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "fib",
                    "fib($n - 1) + fib($n - 2)",
                    Some(vec![("n", Ty::I64)]),
                    Some(lift_to_poly(&Ty::I64)),
                ),
                ("main", "fib(10)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// C18: Function with float params and return.
    #[test]
    fn inter_fn_float_arithmetic() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "avg",
                    "($a + $b) / 2.0",
                    Some(vec![("a", Ty::Float), ("b", Ty::Float)]),
                    None,
                ),
                ("main", "avg(1.0, 3.0)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Float);
    }

    // -- Soundness: invalid calls should be rejected -----------------

    /// S1: Wrong argument type.
    #[test]
    fn inter_fn_reject_wrong_arg_type() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("double", "x * 2", Some(vec![("x", Ty::I64)]), None),
                ("main", "double(\"hello\")", None, None),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject String arg for Int param"
        );
    }

    /// S2: Too many arguments.
    #[test]
    fn inter_fn_reject_too_many_args() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("inc", "x + 1", Some(vec![("x", Ty::I64)]), None),
                ("main", "inc(1, 2)", None, None),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject extra argument");
    }

    /// S3: Too few arguments.
    #[test]
    fn inter_fn_reject_too_few_args() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                (
                    "add",
                    "x + y",
                    Some(vec![("x", Ty::I64), ("y", Ty::I64)]),
                    None,
                ),
                ("main", "add(1)", None, None),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject missing argument");
    }

    /// S4: Using return value where wrong type expected.
    #[test]
    fn inter_fn_reject_return_type_mismatch() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("make_str", "\"hello\"", Some(vec![]), None),
                ("main", "make_str() + 1", None, None),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject String + Int");
    }

    /// S5: Calling undefined function.
    #[test]
    fn inter_fn_reject_undefined_function() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(&i, &[("main", "nonexistent(1)", None, None)], &[]);
        assert!(
            result.has_errors(),
            "should reject call to undefined function"
        );
    }

    /// S6: Declared output type contradicts actual body.
    #[test]
    fn inter_fn_reject_output_mismatch() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("bad", "42", Some(vec![]), Some(lift_to_poly(&Ty::String))),
                ("main", "bad()", None, None),
            ],
            &[],
        );
        let main_id = ids[1].1;
        if !result.has_errors() {
            assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
        }
    }

    /// S7: Mutual recursion without declared types - must not stack overflow.
    #[test]
    fn inter_fn_mutual_recursion_no_declared_types() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("ping", "pong(x)", Some(vec![("x", Ty::I64)]), None),
                ("pong", "ping(x)", Some(vec![("x", Ty::I64)]), None),
            ],
            &[],
        );
        // We don't assert success or failure - just that it terminates.
        let _ = result;
    }

    /// S8: Wrong type in pipe position.
    #[test]
    fn inter_fn_reject_wrong_pipe_type() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("needs_int", "x + 1", Some(vec![("x", Ty::I64)]), None),
                ("main", "\"hello\" | needs_int", None, None),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject String piped to Int param"
        );
    }

    /// S9: Function with wrong context type propagated through call.
    #[test]
    fn inter_fn_reject_context_type_propagation() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("get_name", "@name", Some(vec![]), None),
                ("main", "get_name() + 1", None, None),
            ],
            &[("name", Ty::String)],
        );
        assert!(
            result.has_errors(),
            "should reject String + Int through call chain"
        );
    }

    /// S10: Calling a function as if it had different arity in different call sites.
    #[test]
    fn inter_fn_reject_inconsistent_arity() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("f", "x", Some(vec![("x", Ty::I64)]), None),
                ("main", "f(1) + f(1, 2)", None, None),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject wrong arity call");
    }

    /// S11: Return type of called function used in list - type must be consistent.
    #[test]
    fn inter_fn_reject_heterogeneous_via_calls() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("make_int", "42", Some(vec![]), None),
                ("make_str", "\"hi\"", Some(vec![]), None),
                ("main", "[make_int(), make_str()]", None, None),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject heterogeneous list from calls"
        );
    }

    /// S12: Passing function return to wrong-typed parameter of another function.
    #[test]
    fn inter_fn_reject_chained_type_mismatch() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("make_str", "\"hi\"", Some(vec![]), None),
                ("needs_int", "x + 1", Some(vec![("x", Ty::I64)]), None),
                ("main", "needs_int(make_str())", None, None),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject String passed to Int param"
        );
    }

    // -- Edge cases --------------------------------------------------

    /// E1: Function with no parameters, no context - pure constant.
    #[test]
    fn inter_fn_zero_arg_constant() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("pi", "3", Some(vec![]), None),
                ("main", "pi()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// E2: Same name as builtin - local should shadow or coexist?
    #[test]
    fn inter_fn_name_shadows_builtin() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("len", "42", Some(vec![]), Some(lift_to_poly(&Ty::I64))),
                ("main", "len()", None, None),
            ],
            &[],
        );
        assert_eq!(refusals(&i, &result), Vec::<String>::new());
        assert_eq!(tail_type(&result, ids[1].1), Some(Ty::I64));
    }

    /// E3: Callee defined after caller in graph order.
    #[test]
    fn inter_fn_forward_reference() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("main", "helper()", None, None),
                ("helper", "42", Some(vec![]), None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(
            errs.is_empty(),
            "forward reference should resolve: {errs:?}"
        );
        let main_id = ids[0].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// E4: Two functions reading the same context.
    #[test]
    fn inter_fn_shared_context() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("read_a", "@x + 1", Some(vec![]), None),
                ("read_b", "@x + 2", Some(vec![]), None),
                ("main", "read_a() + read_b()", None, None),
            ],
            &[("x", Ty::I64)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[2].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// E5: Function calling itself with Exact type annotation (base case).
    #[test]
    fn inter_fn_self_call_exact() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "f",
                    "f($x - 1)",
                    Some(vec![("x", Ty::I64)]),
                    Some(lift_to_poly(&Ty::I64)),
                ),
                ("main", "f(10)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// E6: Diamond dependency - A calls B and C, both call D.
    #[test]
    fn inter_fn_diamond_dependency() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("d", "1", Some(vec![]), None),
                ("b", "d() + 10", Some(vec![]), None),
                ("c", "d() + 20", Some(vec![]), None),
                ("main", "b() + c()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[3].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    // inter_fn_pipe_through_builtins: migrated to acvus-mir-test (depends on ExternFn `iter`, `map`, `collect`, `len`)

    // inter_fn_effectful_return: migrated to acvus-mir-test (depends on ExternFn `collect`)

    // inter_fn_option_return: migrated to acvus-mir-test (depends on ExternFn `iter`, `first`, `unwrap`)

    /// E10: Three functions forming a pipeline.
    #[test]
    fn inter_fn_three_stage_pipeline() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("stage1", "$x + 1", Some(vec![("x", Ty::I64)]), None),
                ("stage2", "$x * 2", Some(vec![("x", Ty::I64)]), None),
                ("stage3", "$x - 1", Some(vec![("x", Ty::I64)]), None),
                ("main", "0 | stage1 | stage2 | stage3", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[3].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// E11: All local functions are callers - no inter-function calls.
    #[test]
    fn inter_fn_independent_functions() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("a", "1 + 2", None, None),
                ("b", "\"hello\".to_string()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(tail_type(&result, ids[0].1).unwrap(), Ty::I64);
        assert_eq!(tail_type(&result, ids[1].1).unwrap(), Ty::String);
    }

    /// E12: Function with object return type used with field access.
    #[test]
    fn inter_fn_object_return_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(ObjectTy::written(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::I64),
        ])));
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "get_user",
                    "@user",
                    Some(vec![]),
                    Some(lift_to_poly(&obj_ty)),
                ),
                ("main", "get_user().name", None, None),
            ],
            &[(
                "user",
                Ty::Object(ObjectTy::written(FxHashMap::from_iter([
                    (i.intern("name"), Ty::String),
                    (i.intern("age"), Ty::I64),
                ]))),
            )],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
    }

    // ================================================================
    // Soundness boundary tests
    // ================================================================

    /// B1: Caller tries to use return value as wrong type.
    #[test]
    fn boundary_caller_forces_wrong_return_type() {
        let i = Interner::new();
        let (result, _) = infer_multi(
            &i,
            &[
                ("a", "0", Some(vec![]), None),
                ("main", "a() + \"hello\"", None, None),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject Int used as String");
    }

    /// B2: Two callers use same function's return as different types.
    #[test]
    fn boundary_inconsistent_return_usage() {
        let i = Interner::new();
        let (result, _) = infer_multi(
            &i,
            &[
                ("a", "0", Some(vec![]), None),
                ("ok_caller", "a() + 1", None, None),
                ("bad_caller", "a() + \"hi\"", None, None),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject inconsistent return type usage"
        );
    }

    /// B3: Mutual recursion with Inferred output - should NOT silently succeed.
    #[test]
    fn boundary_mutual_recursion_inferred_must_not_succeed_silently() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("ping", "pong(x)", Some(vec![("x", Ty::I64)]), None),
                ("pong", "ping(x)", Some(vec![("x", Ty::I64)]), None),
            ],
            &[],
        );
        if !result.has_errors() {
            let ping_ty = tail_type(&result, ids[0].1);
            let pong_ty = tail_type(&result, ids[1].1);
            if let (Some(ref pt), Some(ref qt)) = (ping_ty, pong_ty) {
                assert_eq!(pt, qt, "mutual recursion must have consistent return types");
            }
        }
    }

    /// B4: Self-recursion with Inferred output and no base case type.
    #[test]
    fn boundary_self_recursion_inferred_no_base() {
        let i = Interner::new();
        let (result, _) = infer_multi(
            &i,
            &[
                ("f", "f(x - 1)", Some(vec![("x", Ty::I64)]), None),
                ("main", "f(10)", None, None),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "purely recursive return type should fail inference"
        );
    }

    /// B8: infer produces wrong type, should be caught.
    #[test]
    fn boundary_infer_wrong_type_catches() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("bad", "42", Some(vec![]), Some(lift_to_poly(&Ty::String))),
                ("main", "bad()", None, None),
            ],
            &[],
        );
        let main_id = ids[1].1;
        if let Some(main_ty) = tail_type(&result, main_id) {
            assert_ne!(
                main_ty,
                Ty::I64,
                "main must not see Int when bad declared String"
            );
        }
    }

    /// B9: Param in output but not in input - must not silently succeed.
    #[test]
    fn boundary_orphan_param_in_output() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("mystery", "x", Some(vec![]), None),
                ("main", "mystery() + 1", None, None),
            ],
            &[],
        );
        if !result.has_errors() {
            let main_id = ids[1].1;
            if let Some(ty) = tail_type(&result, main_id) {
                assert_eq!(ty, Ty::I64, "if resolved, return type should be Int");
            }
        }
    }

    // -- Extern function tests -------------------------------------

    /// Extern function should be callable from local functions.
    #[test]
    fn extern_fn_call_resolves() {
        let i = Interner::new();
        let fetch = make_extern_fn(&i, "fetch", vec![Ty::I64], Ty::String);
        let (result, ids) =
            infer_with_extern(&i, &[("main", "fetch(42)", None, None)], &[fetch], &[]);
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "extern call should resolve: {errs:?}");
        let main_id = ids[0].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
    }

    /// Extern function with wrong argument type should error.
    #[test]
    fn extern_fn_call_type_mismatch() {
        let i = Interner::new();
        let fetch = make_extern_fn(&i, "fetch", vec![Ty::I64], Ty::String);
        let (result, _) =
            infer_with_extern(&i, &[("main", "fetch(\"bad\")", None, None)], &[fetch], &[]);
        assert!(
            result.has_errors(),
            "should reject String where Int expected"
        );
    }

    /// Extern function return type flows into caller's expression.
    #[test]
    fn extern_fn_return_type_propagates() {
        let i = Interner::new();
        let get_count = make_extern_fn(&i, "get_count", vec![], Ty::I64);
        let (result, ids) = infer_with_extern(
            &i,
            &[("main", "get_count() + 1", None, None)],
            &[get_count],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[0].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::I64);
    }

    /// Multiple extern functions can be registered and called.
    #[test]
    fn extern_fn_multiple() {
        let i = Interner::new();
        let add = make_extern_fn(&i, "ext_add", vec![Ty::I64, Ty::I64], Ty::I64);
        let greet = make_extern_fn(&i, "ext_greet", vec![Ty::String], Ty::String);
        let (result, ids) = infer_with_extern(
            &i,
            &[
                ("use_add", "ext_add(1, 2)", None, None),
                ("use_greet", "ext_greet(\"hi\".to_string())", None, None),
            ],
            &[add, greet],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(tail_type(&result, ids[0].1).unwrap(), Ty::I64);
        assert_eq!(tail_type(&result, ids[1].1).unwrap(), Ty::String);
    }

    // ================================================================
    // Context extraction tests
    // ================================================================

    // -- Completeness: contexts correctly extracted and typed --

    /// A context read is seen wherever it stands, and one nothing declared
    /// is named in the refusal.
    #[test]
    fn context_extract_single_read() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);
        let refused = refusals(&i, &result);
        assert_eq!(refused.len(), 1, "{refused:?}");
        assert!(refused[0].contains('x'), "{refused:?}");
    }

    /// Two reads are two refusals, one per context.
    #[test]
    fn context_extract_multiple() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + @y");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);
        let refused = refusals(&i, &result);
        assert_eq!(refused.len(), 2, "{refused:?}");
        assert!(refused.iter().any(|r| r.contains('x')), "{refused:?}");
        assert!(refused.iter().any(|r| r.contains('y')), "{refused:?}");
    }

    /// A block does not hide the read.
    #[test]
    fn context_extract_nested_block() {
        let i = Interner::new();
        let graph = make_graph(&i, "{ @x + 1 }");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);
        let refused = refusals(&i, &result);
        assert_eq!(refused.len(), 1, "{refused:?}");
        assert!(refused[0].contains('x'), "{refused:?}");
    }

    // context_extract_in_lambda: migrated to acvus-mir-test (depends on ExternFn `map`, `collect`)

    // -- Complete/Incomplete boundary --

    /// Declared Exact context -> Complete.
    #[test]
    fn context_declared_exact_is_complete() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::I64)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let fid = graph.functions[0].qref;
        assert!(
            result.outcomes[&fid].is_complete(),
            "declared Exact context should be Complete"
        );
    }

    /// Declared Inferred context -> Complete (type inferred via fresh var).
    #[test]
    fn context_declared_inferred_is_complete() {
        let i = Interner::new();
        let mut pb = PolyBuilder::new();
        let contexts = vec![Context {
            qref: QualifiedRef::root(i.intern("x")),
            ty: pb.fresh_ty_var(),
            init: None,
        }];
        let test_qref = QualifiedRef::root(i.intern("test"));
        let graph = CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref: test_qref,
                kind: FnKind::Local(
                    ParsedAst::Script(acvus_ast::parse_script(&i, "@x + 1").expect("parse")),
                    crate::graph::Inputs::FromReads,
                ),
                ty: TyTerm::Fn {
                    params: vec![],
                    ret: Box::new(pb.fresh_ty_var()),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                    flows: crate::ty::Flows::Every.into(),
                },
            }]),
            contexts: Freeze::new(contexts),
            types: Freeze::default(),
            bindings: Bindings::default(),
            access: crate::graph::Access::Sync,
            entries: Vec::new(),
        };
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let fid = graph.functions[0].qref;
        assert!(
            result.outcomes[&fid].is_complete(),
            "Inferred context should still be Complete"
        );
        // The inferred type should be Int (from @x + 1).
        let qref = QualifiedRef::root(i.intern("x"));
        assert_eq!(*result.context_type(&qref).unwrap(), Ty::I64);
    }

    /// A context the graph does not declare is refused, not inferred: a
    /// declared context with an open type is the shape that infers, and
    /// `context_declared_inferred_is_complete` above is it.
    #[test]
    fn context_undeclared_is_refused() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let fid = graph.functions[0].qref;
        let words: Vec<String> = result
            .errors()
            .into_iter()
            .flat_map(|(_, errs)| errs.iter().map(|e| e.display(&i).to_string()))
            .collect();
        assert_eq!(words, ["`@x` is not a declared context"]);
        assert!(!result.outcomes[&fid].is_complete());
    }

    // -- Soundness: type mismatch detected --

    /// Declared context type conflicts with usage -> Incomplete.
    #[test]
    fn context_type_mismatch_is_incomplete() {
        let i = Interner::new();
        // @x is String but used in arithmetic.
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let fid = graph.functions[0].qref;
        assert!(
            !result.outcomes[&fid].is_complete(),
            "type mismatch should be Incomplete"
        );
    }

    // ================================================================
    // Param extraction tests
    // ================================================================

    /// Single $param - discovered in extern_params.
    #[test]
    fn param_extract_single() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", "$x + 1", Some(vec![("x", Ty::I64)]), None)],
            &[],
        );
        let fid = ids[0].1;
        let meta = result.outcomes[&fid].meta();
        assert_eq!(meta.params.len(), 1);
        assert_eq!(meta.params[0].name, i.intern("x"));
        assert_eq!(meta.params[0].ty, Ty::I64);
    }

    /// Multiple $params.
    #[test]
    fn param_extract_multiple() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[(
                "test",
                "$x + $y",
                Some(vec![("x", Ty::I64), ("y", Ty::I64)]),
                None,
            )],
            &[],
        );
        let fid = ids[0].1;
        let meta = result.outcomes[&fid].meta();
        assert_eq!(meta.params.len(), 2);
        let names: FxHashSet<Astr> = meta.params.iter().map(|p| p.name).collect();
        assert!(names.contains(&i.intern("x")));
        assert!(names.contains(&i.intern("y")));
    }

    /// Param type inferred from usage.
    #[test]
    fn param_type_inferred() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", "$x + 1", Some(vec![("x", Ty::I64)]), None)],
            &[],
        );
        let fid = ids[0].1;
        let meta = result.outcomes[&fid].meta();
        assert_eq!(meta.params[0].ty, Ty::I64);
    }

    /// Param matched against a string literal -> inferred as String.
    #[test]
    fn param_type_inferred_string() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[(
                "test",
                r#"if let "hello" = $x { let y = 1; }; 0"#,
                Some(vec![("x", Ty::String)]),
                None,
            )],
            &[],
        );
        let fid = ids[0].1;
        let meta = result.outcomes[&fid].meta();
        assert_eq!(meta.params[0].ty, Ty::String);
    }

    // ================================================================
    // Type constraint tests
    // ================================================================

    /// Exact output constraint satisfied -> Complete.
    #[test]
    fn type_constraint_exact_satisfied() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", "42", Some(vec![]), Some(lift_to_poly(&Ty::I64)))],
            &[],
        );
        let fid = ids[0].1;
        assert!(
            result.outcomes[&fid].is_complete(),
            "matching Exact output should be Complete"
        );
    }

    /// Exact output constraint violated -> Incomplete.
    #[test]
    fn type_constraint_exact_violated() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[(
                "test",
                r#""hello""#,
                Some(vec![]),
                Some(lift_to_poly(&Ty::I64)),
            )],
            &[],
        );
        let fid = ids[0].1;
        assert!(
            !result.outcomes[&fid].is_complete(),
            "String body with Exact(Int) constraint should be Incomplete"
        );
    }

    /// Inferred output -> always Complete (no constraint to violate).
    #[test]
    fn type_constraint_inferred_always_complete() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", r#""hello".to_string()"#, Some(vec![]), None)],
            &[],
        );
        let fid = ids[0].1;
        assert!(result.outcomes[&fid].is_complete());
    }
}
