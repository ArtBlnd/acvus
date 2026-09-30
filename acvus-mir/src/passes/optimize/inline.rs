//! Inlining (`Editor<Inline>::inline`, S2) over the module's call graph
//! (`analysis::CallGraph`), whose strongly connected components are visited
//! callees first.
//!
//! In each body, every call site the body holds when its visit begins is
//! offered to `inline` once. The editor refuses a call whose callee is not
//! a closure made in the body (`NoBody`), which every direct call is, since
//! a script declares no function whose body the module holds (D61); and a
//! call of the body's own closure (`Recursive`), so no body is inlined into
//! itself. A call that an inlined body brings in is not offered: its callee
//! was visited first, so what could be inlined into it already was, and
//! what is left is a call into the same component or one the editor refuses.
//!
//! Form: F1, SSA, in and out.
//!
//! Cost: O(bodies + instructions) for the call graph and its components;
//! each call site the module holds before the pass is offered at most once,
//! and an inlining costs the core's O(callee size). The pass is bounded by
//! the module's call sites.
//!
//! Meaning: a call means its callee's body, run with its parameters bound
//! to the arguments and its captures to the closure's captured values; the
//! callee's token chain continues the caller's, so the order of effects is
//! the call's (effects §3), and each return of the callee goes to what
//! followed the call with its value. Every copied value keeps its type.

use rustc_hash::FxHashSet;

use crate::ir::cfg::edit::{CannotInline, Editor, Inline as InlineScope, NoFacts};
use crate::ir::cfg::{BodyId, Changes, Module, Op};
use crate::ir::ValueId;
use crate::passes::analysis::{AnalysisManager, CallGraph};
use crate::passes::optimize::ModuleTransform;

pub struct Inline;

impl ModuleTransform for Inline {
    const NAME: &'static str = "inline";

    fn run(module: &mut Module, _am: &mut AnalysisManager<'_>) -> Vec<(BodyId, Changes)> {
        let graph = CallGraph::of(module);
        let mut changed = Vec::new();
        for component in graph.components_callees_first() {
            for body in component {
                changed.push((body, inline_into(module, body)));
            }
        }
        changed
    }
}

/// Offers every call site `body` holds now to `inline`, once each. A site
/// is named by the call's result, a value that stays the call's until the
/// call is inlined, whatever moves around it; an inlined body's calls have
/// results of their own, so they are not offered.
fn inline_into(module: &mut Module, body: BodyId) -> Changes {
    let mut sites: FxHashSet<ValueId> = module
        .body(body)
        .blocks()
        .flat_map(|(_, block)| block.insts())
        .filter(|inst| matches!(inst.op(), Op::FunctionCall { .. }))
        .map(|inst| inst.result().expect("a call yields a value"))
        .collect();
    let mut no_facts = NoFacts;
    let mut editor = Editor::<InlineScope>::new(module, body, &mut no_facts);
    let mut cursor = editor.cursor();
    while let Some(inst) = cursor.next() {
        let editor = cursor.editor();
        let record = editor.body().inst(inst.id());
        let Op::FunctionCall { .. } = record.op() else {
            continue;
        };
        if !sites.remove(&record.result().expect("a call yields a value")) {
            continue;
        }
        match editor.inline(inst) {
            Ok(()) | Err(CannotInline::NoBody | CannotInline::Recursive) => {}
        }
    }
    editor.finish()
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::Callee;
    use crate::ir::cfg::{Aspect, Inst, OrderUse, Terminator};
    use crate::passes::optimize::tests::{Fixture, add, assert_whole, direct_call, ret};

    fn calls(f: &Fixture, body: BodyId) -> usize {
        let insts = f.module.body(body).blocks().flat_map(|(_, block)| block.insts());
        insts.filter(|inst| matches!(inst.op(), Op::FunctionCall { .. })).count()
    }

    fn changes_of(changed: &[(BodyId, Changes)], body: BodyId) -> &Changes {
        let found = changed.iter().find(|(of, _)| *of == body);
        let Some((_, changes)) = found else { panic!("{body:?} was visited") };
        changes
    }

    /// The callee `|x| x + x` and the caller `f = the closure; r = f(a);
    /// return r`.
    #[test]
    fn a_closure_called_once_is_inlined() {
        let mut f = Fixture::new();
        let (int, closure_ty, na, nx) = (f.int, f.pure_unary, f.name("a"), f.name("x"));
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(nx, int, acvus_ast::Span::ZERO);
        let doubled = builder.push(add(x, x), int, Span::ZERO);
        builder.terminate(ret(doubled), acvus_ast::Span::ZERO);
        let callee = builder.seal().expect("the callee has no open state");
        let mut builder = f.module.build_body(false, int);
        let a = builder.param(na, int, acvus_ast::Span::ZERO);
        let closure = builder.push(Op::MakeClosure { bodies: vec![callee], captures: Vec::new() }, closure_ty, Span::ZERO);
        let call = Op::FunctionCall { callee: Callee::Indirect(closure), callee_ty: closure_ty, args: vec![a] };
        let r = builder.push(call, int, Span::ZERO);
        builder.terminate(ret(r), acvus_ast::Span::ZERO);
        let caller = builder.seal().expect("the caller has no open state");
        let callee_before = f.snapshot(callee);

        let changed = f.run_module::<Inline>();

        assert_eq!(calls(&f, caller), 0, "the call is gone");
        let body = f.module.body(caller);
        let copied = body
            .blocks()
            .flat_map(|(_, block)| block.insts())
            .find(|inst| matches!(inst.op(), Op::BinOp { left, right, .. } if (*left, *right) == (a, a)));
        assert!(copied.is_some(), "the callee's body stands in the caller with the argument for its parameter");
        assert!(changes_of(&changed, caller).touched(Aspect::CfgShape));
        assert_eq!(f.snapshot(callee), callee_before, "the callee is unchanged");
        assert_whole(&f.module, caller);
    }

    /// A body `x -> f(x)` where `f` is a closure of this very body.
    #[test]
    fn a_recursive_closure_is_not_inlined() {
        let mut f = Fixture::new();
        let (int, closure_ty, nx) = (f.int, f.pure_unary, f.name("x"));
        let own = f.module.next_body_id();
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(nx, int, acvus_ast::Span::ZERO);
        let closure = builder.push(Op::MakeClosure { bodies: vec![own], captures: Vec::new() }, closure_ty, Span::ZERO);
        let call = Op::FunctionCall { callee: Callee::Indirect(closure), callee_ty: closure_ty, args: vec![x] };
        let r = builder.push(call, int, Span::ZERO);
        builder.terminate(ret(r), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        assert_eq!(id, own);
        let before = f.snapshot(id);

        f.run_module::<Inline>();

        assert_eq!(f.snapshot(id), before);
    }

    /// The callee `|x| { g(); x + x }` and the caller `f = the closure;
    /// r = f(a); h(); return r`, both ordered.
    #[test]
    fn the_callers_token_chain_runs_through_the_inlined_body() {
        let mut f = Fixture::new();
        let (int, closure_ty, opaque, na, nx, ng, nh) =
            (f.int, f.opaque_unary, f.opaque, f.name("a"), f.name("x"), f.name("g"), f.name("h"));
        let mut builder = f.module.build_body(true, int);
        let x = builder.param(nx, int, acvus_ast::Span::ZERO);
        let g = builder.push_effect(direct_call(ng, opaque, Vec::new()), int, Span::ZERO);
        let doubled = builder.push(add(x, x), int, Span::ZERO);
        builder.terminate(ret(doubled), acvus_ast::Span::ZERO);
        let callee = builder.seal().expect("the callee has no open state");
        let mut builder = f.module.build_body(true, int);
        let a = builder.param(na, int, acvus_ast::Span::ZERO);
        let closure = builder.push(Op::MakeClosure { bodies: vec![callee], captures: Vec::new() }, closure_ty, Span::ZERO);
        let call = Op::FunctionCall { callee: Callee::Indirect(closure), callee_ty: closure_ty, args: vec![a] };
        let r = builder.push_effect(call, int, Span::ZERO);
        let h = builder.push_effect(direct_call(nh, opaque, Vec::new()), int, Span::ZERO);
        builder.terminate(ret(r), acvus_ast::Span::ZERO);
        let caller = builder.seal().expect("the caller has no open state");
        let o0 = f.module.body(caller).order_param().expect("the caller is ordered");
        let g_in_callee = f.module.body(callee).blocks().flat_map(|(_, block)| block.insts()).find(|inst| inst.result() == Some(g));
        assert!(g_in_callee.is_some());

        f.run_module::<Inline>();

        let body = f.module.body(caller);
        let insts: Vec<&Inst> = body.blocks().flat_map(|(_, block)| block.insts()).collect();
        let edge = |inst: &Inst| match inst.order() {
            Some(OrderUse::Edge(edge)) => edge,
            other => panic!("{:?} is on the chain, not {other:?}", inst.op()),
        };
        let named = |inst: &&Inst, name| matches!(inst.op(), Op::FunctionCall { callee: Callee::Direct(called), .. } if called.name == name);
        let copied_g = insts.iter().find(|inst| named(inst, ng)).expect("the callee's call is copied into the caller");
        let h_call = insts.iter().find(|inst| inst.result() == Some(h)).expect("the caller's later call stays");
        assert_eq!(edge(copied_g).before, o0, "the inlined body consumes the caller's token");
        let (continuation, _) = body
            .blocks()
            .find(|(_, block)| block.insts().any(|inst| inst.result() == Some(h)))
            .expect("the later call stands in the continuation");
        let [_, token] = body.block(continuation).params() else {
            panic!("the continuation takes the call's result and its token")
        };
        assert_eq!(edge(h_call).before, *token, "the caller's chain goes on from the continuation's token");
        let passes_on = body.blocks().any(|(_, block)| {
            matches!(block.terminator(), Terminator::Jump { to, args } if *to == continuation && args.get(1) == Some(&edge(copied_g).after))
        });
        assert!(passes_on, "the inlined body's return hands its last token to the continuation");
        assert_eq!(calls(&f, caller), 2, "g's copy and h are the caller's only calls");
        assert_whole(&f.module, caller);
    }
}
