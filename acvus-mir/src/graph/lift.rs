//! The lift of RFC-0100 rule 4. This is the one place a script's `fn`
//! becomes a function node, and every graph builder reaches it through
//! `graph::extract`.
//!
//! A `fn` is typed as a lambda, one instance per call site (rule 3). The
//! lift makes one instance of a `fn`'s call-graph component for each call
//! from outside that component. Each instance is typed together with the
//! body whose call it serves, the way a lambda is typed within the body
//! that writes it. A call between members of one component reaches the
//! members of the same instance.

use acvus_ast::report::Label;
use acvus_ast::{
    AstId, Binder, Clean, ElseBranch, ErrorNode, Expr, FnDecl, ForHead, Pattern, Place, PlaceBase,
    RefKind, Script, Slot, Span, Stmt, TupleElem, TuplePatternElem,
};
use acvus_utils::{Astr, FnScope, Interner, QualifiedRef};
use rustc_hash::{FxHashMap, FxHashSet};

use super::types::*;
use crate::error::{MirError, MirErrorKind};
use crate::ty::{Effect, Flows, ParamTerm, PolyBuilder, PolyTy, TyTerm};

pub struct Lift {
    pub functions: Vec<Function>,
    pub facts: FxHashMap<QualifiedRef, LiftFacts>,
}

pub fn lift<F>(interner: &Interner, script: &Function, host_names_reached: F) -> Option<Lift>
where
    F: Fn(Astr) -> Vec<String>,
{
    let FnKind::Local(ast, _) = &script.kind else {
        return None;
    };
    match ast {
        ParsedAst::Script(body) => {
            Some(Lifter::of(interner, script.qref, body).lift(host_names_reached))
        }
        ParsedAst::Recovered(RecoveredAst::Script(body)) => {
            Some(Lifter::of(interner, script.qref, body).lift(host_names_reached))
        }
        ParsedAst::Template(_)
        | ParsedAst::Recovered(RecoveredAst::Template(_))
        | ParsedAst::Fn(_) => None,
    }
}

trait IntoFnBody: Slot {
    fn fn_body(decl: &FnDecl<Self>) -> FnBody;
}

impl IntoFnBody for Clean {
    fn fn_body(decl: &FnDecl<Self>) -> FnBody {
        FnBody::Parsed(decl.clone())
    }
}

impl IntoFnBody for ErrorNode {
    fn fn_body(decl: &FnDecl<Self>) -> FnBody {
        FnBody::Recovered(decl.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DeclAt(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ComponentAt(usize);

struct BareCall {
    callee: AstId,
    name: Astr,
}

#[derive(Clone, Copy)]
struct Site {
    callee: AstId,
    target: DeclAt,
}

struct Bound {
    name: Astr,
    span: Span,
    id: AstId,
}

struct Components {
    component_of: Vec<ComponentAt>,
    members: Vec<Vec<DeclAt>>,
}

struct Lifter<'a, S> {
    interner: &'a Interner,
    script: QualifiedRef,
    body: &'a Script<S>,
    decls_in_source_order: Vec<&'a FnDecl<S>>,
    first_of_name: FxHashMap<Astr, DeclAt>,
    sites_of: Vec<Vec<Site>>,
    components: Components,
    next_instance: u32,
    functions: Vec<Function>,
    facts: FxHashMap<QualifiedRef, LiftFacts>,
}

impl<'a, S> Lifter<'a, S>
where
    S: IntoFnBody,
{
    fn of(interner: &'a Interner, script: QualifiedRef, body: &'a Script<S>) -> Self {
        let mut found = AllDeclarations { decls: Vec::new() };
        walk_script(body, &mut found);
        let decls_in_source_order = found.decls;
        let mut first_of_name: FxHashMap<Astr, DeclAt> = FxHashMap::default();
        for (at, decl) in decls_in_source_order.iter().enumerate() {
            first_of_name.entry(decl.name.name).or_insert(DeclAt(at));
        }
        let sites_of: Vec<Vec<Site>> = decls_in_source_order
            .iter()
            .map(|decl| {
                let mut found = OwnBodyCalls { calls: Vec::new() };
                walk_declared_body(decl, &mut found);
                sites_naming_fns(&found.calls, &first_of_name)
            })
            .collect();
        let components = components(&sites_of);
        Self {
            interner,
            script,
            body,
            decls_in_source_order,
            first_of_name,
            sites_of,
            components,
            next_instance: 0,
            functions: Vec::new(),
            facts: FxHashMap::default(),
        }
    }

    fn decl(&self, at: DeclAt) -> &'a FnDecl<S> {
        self.decls_in_source_order[at.0]
    }

    fn component_of(&self, at: DeclAt) -> ComponentAt {
        self.components.component_of[at.0]
    }

    fn lift<F>(mut self, host_names_reached: F) -> Lift
    where
        F: Fn(Astr) -> Vec<String>,
    {
        let refusals = self.declaration_refusals(host_names_reached);
        let mut found = OwnBodyCalls { calls: Vec::new() };
        walk_script(self.body, &mut found);
        let own_sites = sites_naming_fns(&found.calls, &self.first_of_name);

        let mut reached: FxHashSet<ComponentAt> = FxHashSet::default();
        let mut calls: FxHashMap<AstId, QualifiedRef> = FxHashMap::default();
        for site in own_sites {
            let component = self.component_of(site.target);
            let instance = self.instantiate(component, Some(self.script));
            calls.insert(site.callee, instance[&site.target]);
            reached.insert(component);
        }
        for (at, sites) in self.sites_of.iter().enumerate() {
            let caller = self.components.component_of[at];
            for site in sites {
                let callee = self.component_of(site.target);
                if callee != caller {
                    reached.insert(callee);
                }
            }
        }
        // A component nothing outside it calls is still checked, once, as a
        // lambda nothing calls is checked where it is written.
        for component in (0..self.components.members.len()).map(ComponentAt) {
            if !reached.contains(&component) {
                self.instantiate(component, None);
            }
        }
        self.facts.insert(
            self.script,
            LiftFacts {
                calls,
                typed_with: None,
                refusals,
            },
        );
        Lift {
            functions: self.functions,
            facts: self.facts,
        }
    }

    fn declaration_refusals<F>(&self, host_names_reached: F) -> Vec<MirError>
    where
        F: Fn(Astr) -> Vec<String>,
    {
        let mut refusals = Vec::new();
        for (at, decl) in self.decls_in_source_order.iter().enumerate() {
            let name = decl.name.name;
            let spelled = self.interner.resolve(name).to_string();
            let first = self.first_of_name[&name];
            if first != DeclAt(at) {
                refusals.push(MirError {
                    kind: MirErrorKind::FnDeclaredTwice(spelled.clone()),
                    span: decl.name.span,
                    labels: vec![Label::at(
                        self.decl(first).name.span,
                        format!("`{spelled}` is first declared here"),
                    )],
                });
                continue;
            }
            let shadows = host_names_reached(name);
            if !shadows.is_empty() {
                refusals.push(MirError {
                    kind: MirErrorKind::FnShadows {
                        name: spelled,
                        shadowed: shadows,
                    },
                    span: decl.name.span,
                    labels: Vec::new(),
                });
            }
        }
        refusals
    }

    fn instantiate(
        &mut self,
        component: ComponentAt,
        typed_with: Option<QualifiedRef>,
    ) -> FxHashMap<DeclAt, QualifiedRef> {
        let members = self.components.members[component.0].clone();
        let group: FxHashMap<DeclAt, QualifiedRef> = members
            .iter()
            .map(|&member| (member, self.instance_ref(member)))
            .collect();
        for &member in &members {
            let qref = group[&member];
            let mut calls: FxHashMap<AstId, QualifiedRef> = FxHashMap::default();
            for site in self.sites_of[member.0].clone() {
                let reached = match group.get(&site.target) {
                    Some(&within) => within,
                    None => {
                        let callee = self.component_of(site.target);
                        self.instantiate(callee, Some(qref))[&site.target]
                    }
                };
                calls.insert(site.callee, reached);
            }
            let decl = self.decl(member);
            self.functions.push(Function {
                qref,
                kind: FnKind::Local(
                    ParsedAst::Fn(LiftedFn {
                        decl: S::fn_body(decl),
                        outside: self.locals_outside(member),
                    }),
                    Inputs::Declared,
                ),
                ty: open_fn_ty(&decl.params),
            });
            self.facts.insert(
                qref,
                LiftFacts {
                    calls,
                    typed_with,
                    refusals: Vec::new(),
                },
            );
        }
        group
    }

    fn instance_ref(&mut self, member: DeclAt) -> QualifiedRef {
        let instance = self.next_instance;
        self.next_instance += 1;
        QualifiedRef {
            namespace: self.script.namespace,
            name: self.decl(member).name.name,
            scope: Some(FnScope {
                script: self.script.name,
                instance,
            }),
        }
    }

    fn locals_outside(&self, member: DeclAt) -> FxHashMap<Astr, Span> {
        let mut inside = AllBinders { found: Vec::new() };
        walk_declaration(self.decl(member), &mut inside);
        let inside: FxHashSet<AstId> = inside.found.iter().map(|bound| bound.id).collect();
        let mut all = AllBinders { found: Vec::new() };
        walk_script(self.body, &mut all);
        let mut outside: FxHashMap<Astr, Span> = FxHashMap::default();
        for bound in all.found {
            if inside.contains(&bound.id) {
                continue;
            }
            outside
                .entry(bound.name)
                .and_modify(|first: &mut Span| {
                    if bound.span.start < first.start {
                        *first = bound.span;
                    }
                })
                .or_insert(bound.span);
        }
        outside
    }
}

fn open_fn_ty(params: &[Binder]) -> PolyTy {
    let mut builder = PolyBuilder::new();
    TyTerm::Fn {
        params: params
            .iter()
            .map(|param| ParamTerm::new(param.name, builder.fresh_ty_var()))
            .collect(),
        ret: Box::new(builder.fresh_ty_var()),
        captures: vec![],
        effect: Effect::OPAQUE.into(),
        flows: Flows::Every.into(),
    }
}

fn sites_naming_fns(calls: &[BareCall], first_of_name: &FxHashMap<Astr, DeclAt>) -> Vec<Site> {
    calls
        .iter()
        .filter_map(|call| {
            Some(Site {
                callee: call.callee,
                target: *first_of_name.get(&call.name)?,
            })
        })
        .collect()
}

fn components(sites_of: &[Vec<Site>]) -> Components {
    struct Tarjan<'c> {
        sites_of: &'c [Vec<Site>],
        counter: usize,
        index: Vec<Option<usize>>,
        low: Vec<usize>,
        stack: Vec<usize>,
        on_stack: Vec<bool>,
        found: Components,
    }
    fn connect(state: &mut Tarjan<'_>, v: usize) {
        state.index[v] = Some(state.counter);
        state.low[v] = state.counter;
        state.counter += 1;
        state.stack.push(v);
        state.on_stack[v] = true;
        for site in state.sites_of[v].iter() {
            let w = site.target.0;
            match state.index[w] {
                None => {
                    connect(state, w);
                    state.low[v] = state.low[v].min(state.low[w]);
                }
                Some(index) if state.on_stack[w] => state.low[v] = state.low[v].min(index),
                Some(_) => {}
            }
        }
        if Some(state.low[v]) == state.index[v] {
            let component = ComponentAt(state.found.members.len());
            let mut members = Vec::new();
            while let Some(w) = state.stack.pop() {
                state.on_stack[w] = false;
                state.found.component_of[w] = component;
                members.push(DeclAt(w));
                if w == v {
                    break;
                }
            }
            members.sort_unstable_by_key(|member| member.0);
            state.found.members.push(members);
        }
    }
    let count = sites_of.len();
    let mut state = Tarjan {
        sites_of,
        counter: 0,
        index: vec![None; count],
        low: vec![0; count],
        stack: Vec::new(),
        on_stack: vec![false; count],
        found: Components {
            component_of: vec![ComponentAt(0); count],
            members: Vec::new(),
        },
    };
    for v in 0..count {
        if state.index[v].is_none() {
            connect(&mut state, v);
        }
    }
    state.found
}

trait Visit<'a, S> {
    fn enter_declaration(&mut self, decl: &'a FnDecl<S>) -> bool;
    fn binder(&mut self, _bound: Bound) {}
    fn call(&mut self, _call: BareCall) {}
}

struct AllDeclarations<'a, S> {
    decls: Vec<&'a FnDecl<S>>,
}

impl<'a, S> Visit<'a, S> for AllDeclarations<'a, S> {
    fn enter_declaration(&mut self, decl: &'a FnDecl<S>) -> bool {
        self.decls.push(decl);
        true
    }
}

struct OwnBodyCalls {
    calls: Vec<BareCall>,
}

impl<'a, S> Visit<'a, S> for OwnBodyCalls {
    fn enter_declaration(&mut self, _: &'a FnDecl<S>) -> bool {
        false
    }

    fn call(&mut self, call: BareCall) {
        self.calls.push(call);
    }
}

struct AllBinders {
    found: Vec<Bound>,
}

impl<'a, S> Visit<'a, S> for AllBinders {
    fn enter_declaration(&mut self, _: &'a FnDecl<S>) -> bool {
        true
    }

    fn binder(&mut self, bound: Bound) {
        self.found.push(bound);
    }
}

fn walk_script<'a, S, V>(script: &'a Script<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    walk_branch(&script.stmts, script.tail.as_deref(), visit);
}

fn walk_declared_body<'a, S, V>(decl: &'a FnDecl<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    walk_branch(&decl.body, decl.tail.as_deref(), visit);
}

fn walk_declaration<'a, S, V>(decl: &'a FnDecl<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    for param in &decl.params {
        walk_binder::<S, V>(param, visit);
    }
    walk_declared_body(decl, visit);
}

fn walk_binder<'a, S, V>(binder: &Binder, visit: &mut V)
where
    V: Visit<'a, S>,
{
    visit.binder(Bound {
        name: binder.name,
        span: binder.span,
        id: binder.id,
    });
}

fn walk_stmts<'a, S, V>(stmts: &'a [Stmt<S>], visit: &mut V)
where
    V: Visit<'a, S>,
{
    acvus_utils::grow(|| walk_stmts_level::<S, V>(stmts, visit))
}

fn walk_stmts_level<'a, S, V>(stmts: &'a [Stmt<S>], visit: &mut V)
where
    V: Visit<'a, S>,
{
    for stmt in stmts {
        walk_stmt(stmt, visit);
    }
}

fn walk_stmt<'a, S, V>(stmt: &'a Stmt<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    acvus_utils::grow(|| walk_stmt_level::<S, V>(stmt, visit))
}

fn walk_stmt_level<'a, S, V>(stmt: &'a Stmt<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    match stmt {
        Stmt::Store { place, expr, .. } => {
            walk_place(place, visit);
            walk_expr(expr, visit);
        }
        Stmt::DerefStore { target, expr, .. } => {
            walk_expr(target, visit);
            walk_expr(expr, visit);
        }
        Stmt::Expr(expr) | Stmt::Assign { expr, .. } | Stmt::Append { expr, .. } => {
            walk_expr(expr, visit)
        }
        Stmt::LetBind { binder, expr, .. } => {
            walk_expr(expr, visit);
            walk_binder::<S, V>(binder, visit);
        }
        Stmt::LetUninit { binder, .. } => walk_binder::<S, V>(binder, visit),
        Stmt::While { cond, body, .. } => {
            walk_expr(cond, visit);
            walk_stmts(body, visit);
        }
        Stmt::For {
            binder, head, body, ..
        } => {
            match head {
                ForHead::Value(value) => walk_expr(value, visit),
                ForHead::Range { lo, hi } => {
                    walk_expr(lo, visit);
                    walk_expr(hi, visit);
                }
            }
            walk_binder::<S, V>(binder, visit);
            walk_stmts(body, visit);
        }
        Stmt::WhileLet {
            pattern,
            source,
            body,
            ..
        } => {
            walk_expr(source, visit);
            walk_pattern(pattern, visit);
            walk_stmts(body, visit);
        }
        Stmt::Anyorder { body, .. } => walk_stmts(body, visit),
        Stmt::FnDecl(decl) => {
            if visit.enter_declaration(decl) {
                walk_declaration(decl, visit);
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Error(_) => {}
    }
}

fn walk_place<'a, S, V>(place: &'a Place<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    acvus_utils::grow(|| walk_place_level::<S, V>(place, visit))
}

fn walk_place_level<'a, S, V>(place: &'a Place<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    match place {
        Place::Field { object, .. } => walk_place(object, visit),
        Place::Base(PlaceBase::Root { .. }) => {}
        Place::Base(PlaceBase::Element {
            container, index, ..
        }) => {
            walk_expr(container.expr(), visit);
            walk_expr(index, visit);
        }
    }
}

fn walk_pattern<'a, S, V>(pattern: &'a Pattern<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    acvus_utils::grow(|| walk_pattern_level::<S, V>(pattern, visit))
}

fn walk_pattern_level<'a, S, V>(pattern: &'a Pattern<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    match pattern {
        Pattern::Binding {
            id,
            name,
            ref_kind: RefKind::Value,
            span,
        } => visit.binder(Bound {
            name: *name,
            span: *span,
            id: *id,
        }),
        Pattern::Binding { .. }
        | Pattern::ContextBind { .. }
        | Pattern::Literal { .. }
        | Pattern::Wildcard { .. }
        | Pattern::Error(_) => {}
        Pattern::List { head, tail, .. } => {
            for part in head.iter().chain(tail) {
                walk_pattern(part, visit);
            }
        }
        Pattern::Object { fields, .. } => {
            for field in fields {
                walk_pattern(&field.pattern, visit);
            }
        }
        Pattern::Tuple { elements, .. } => {
            for element in elements {
                if let TuplePatternElem::Pattern(part) = element {
                    walk_pattern(part, visit);
                }
            }
        }
        Pattern::Variant { payload, .. } => {
            if let Some(payload) = payload {
                walk_pattern(payload, visit);
            }
        }
    }
}

fn bare_callee<S>(func: &Expr<S>) -> Option<BareCall> {
    match func {
        Expr::Ident {
            id,
            name:
                QualifiedRef {
                    namespace: None,
                    name,
                    scope: None,
                },
            ref_kind: RefKind::Value,
            ..
        } => Some(BareCall {
            callee: *id,
            name: *name,
        }),
        _ => None,
    }
}

fn walk_call<'a, S, V>(func: &'a Expr<S>, args: &'a [Expr<S>], visit: &mut V)
where
    V: Visit<'a, S>,
{
    if let Some(call) = bare_callee(func) {
        visit.call(call);
    }
    walk_expr(func, visit);
    for arg in args {
        walk_expr(arg, visit);
    }
}

fn walk_expr<'a, S, V>(expr: &'a Expr<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    acvus_utils::grow(|| walk_expr_level::<S, V>(expr, visit))
}

fn walk_expr_level<'a, S, V>(expr: &'a Expr<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    match expr {
        Expr::Ident { .. }
        | Expr::Literal { .. }
        | Expr::ContextRef { .. }
        | Expr::Variant { payload: None, .. }
        | Expr::Error(_) => {}
        Expr::BinaryOp { left, right, .. } => {
            walk_expr(left, visit);
            walk_expr(right, visit);
        }
        Expr::Pipe { left, right, .. } => {
            walk_expr(left, visit);
            match right.as_ref() {
                Expr::FuncCall { func, args, .. } => walk_call(func, args, visit),
                stage => walk_call(stage, &[], visit),
            }
        }
        Expr::FuncCall { func, args, .. } => walk_call(func, args, visit),
        Expr::UnaryOp { operand: inner, .. }
        | Expr::FieldAccess { object: inner, .. }
        | Expr::Paren { inner, .. }
        | Expr::Borrow { place: inner, .. }
        | Expr::Cast { expr: inner, .. }
        | Expr::Try { inner, .. }
        | Expr::Return { value: inner, .. }
        | Expr::Variant {
            payload: Some(inner),
            ..
        } => walk_expr(inner, visit),
        Expr::Index { object, index, .. } => {
            walk_expr(object, visit);
            walk_expr(index, visit);
        }
        Expr::MethodCall { receiver, args, .. } => {
            walk_expr(receiver, visit);
            for arg in args {
                walk_expr(arg, visit);
            }
        }
        Expr::Lambda { params, body, .. } => {
            for param in params {
                walk_binder::<S, V>(param, visit);
            }
            walk_expr(body, visit);
        }
        Expr::List { head, tail, .. } => {
            for element in head.iter().chain(tail) {
                walk_expr(element, visit);
            }
        }
        Expr::Group { elements, .. } => {
            for element in elements {
                walk_expr(element, visit);
            }
        }
        Expr::Object { fields, .. } => {
            for field in fields {
                walk_expr(&field.value, visit);
            }
        }
        Expr::Tuple { elements, .. } => {
            for element in elements {
                if let TupleElem::Expr(element) = element {
                    walk_expr(element, visit);
                }
            }
        }
        Expr::Block { stmts, tail, .. } => {
            walk_stmts(stmts, visit);
            walk_expr(tail, visit);
        }
        Expr::If {
            cond,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            walk_expr(cond, visit);
            walk_branch(then_body, then_tail.as_deref(), visit);
            if let Some(branch) = else_branch {
                walk_else(branch, visit);
            }
        }
        Expr::IfLet {
            pattern,
            source,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            walk_expr(source, visit);
            walk_pattern(pattern, visit);
            walk_branch(then_body, then_tail.as_deref(), visit);
            if let Some(branch) = else_branch {
                walk_else(branch, visit);
            }
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            walk_expr(scrutinee, visit);
            for arm in arms {
                walk_pattern(&arm.pattern, visit);
                walk_branch(&arm.body, arm.tail.as_deref(), visit);
            }
        }
    }
}

fn walk_branch<'a, S, V>(body: &'a [Stmt<S>], tail: Option<&'a Expr<S>>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    walk_stmts(body, visit);
    if let Some(tail) = tail {
        walk_expr(tail, visit);
    }
}

fn walk_else<'a, S, V>(branch: &'a ElseBranch<S>, visit: &mut V)
where
    V: Visit<'a, S>,
{
    match branch {
        ElseBranch::ElseIf(expr) => walk_expr(expr, visit),
        ElseBranch::Else { body, tail, .. } => walk_branch(body, tail.as_deref(), visit),
    }
}
