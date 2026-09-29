//! borrowck over programs built as `lower/tests.rs` builds them: parsed,
//! checked by typeck over verified declarations, lowered, and run through
//! the required steps up to `remove_dead_context_ops`.

use acvus_ast::{Script, Span};
use acvus_utils::Interner;

use super::{Conflict, conflicts, met};
use crate::check::ConflictTrace;
use crate::decl::{Core, Declarations, LoopMarkers, RawDeclarations};
use crate::ir::cfg::{Module, Op, Terminator};
use crate::lower::lower_script;
use crate::lower::required::{clone_strings, remove_dead_context_ops};
use crate::passes::analysis::{AnalysisManager, Loans};
use crate::ty::{
    Alignment, Calls, Effect, EffectTerm, Flow, FlowEnd, FlowTerm, Flows, Holds, IntTy, Mutability, TyKind, TypeArg, Types,
};
use crate::typeck::check_script;

/// What borrowck finds in `source`: each conflict, one per fact.
struct Found {
    source: String,
    conflicts: Vec<Conflict>,
}

impl Found {
    fn of(source: &str) -> Found {
        let names = Interner::new();
        let raw = RawDeclarations {
            types: Types::new(),
            functions: Vec::new(),
            lang_impls: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            inputs: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(&names),
        };
        let script: Script = acvus_ast::parse_script(&names, source).expect("the test's source parses");
        let decls = Declarations::verify(raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let typeck = check_script(&names, &decls, &mut types, &script);
        assert!(typeck.refusals.is_empty(), "typeck refused {source:?}: {:?}", typeck.refusals);
        let mut module: Module = lower_script(&decls, &typeck, types, &script);
        let mut am = AnalysisManager::new(&decls);
        clone_strings(&mut module, &mut am);
        remove_dead_context_ops(&mut module, &mut am);
        let conflicts = module
            .body_ids()
            .flat_map(|body| {
                let met = met(&module, body, &mut am);
                conflicts(module.body(body), &am.get::<Loans>(&module, body), &met)
            })
            .collect();
        Found { source: source.to_owned(), conflicts }
    }

    fn text(&self, span: Span) -> &str {
        &self.source[span.start..span.end]
    }

    /// Each conflict's text: where it stands, and its trace.
    fn texts(&self) -> Vec<(&str, Vec<&str>, Vec<&str>, Vec<&str>)> {
        let texts = |spans: &[Span]| spans.iter().map(|span| self.text(*span)).collect::<Vec<&str>>();
        self.conflicts
            .iter()
            .map(|Conflict { at, trace }| (self.text(*at), texts(&trace.earlier), texts(&trace.later), texts(&trace.uses)))
            .collect()
    }

    fn admitted(&self) -> bool {
        self.conflicts.is_empty()
    }

    /// The span of the `nth` occurrence of `text` in the source, from 0.
    fn span_of(&self, text: &str, nth: usize) -> Span {
        let start = self.source.match_indices(text).nth(nth).unwrap_or_else(|| panic!("{text:?} occurs {nth} times before")).0;
        Span { start, end: start + text.len() }
    }
}

/// An access by name meets `&x`, which `r` holds.
#[test]
fn a_write_while_a_shared_loan_lives_is_refused() {
    let found = Found::of("let x = 1; let r = &x; x = 2; *r");
    let none: Vec<&str> = Vec::new();
    assert_eq!(found.texts(), [("x = 2;", vec!["&x"], none.clone(), none)], "{:?}", found.conflicts);
}

/// A read by name of `x` meets the loan `m`'s store made within `&mut x`.
#[test]
fn a_read_while_a_mut_loan_lives_is_refused() {
    let found = Found::of("let x = 1; let m = &mut x; let y = x; *m = 3; y");
    let [Conflict { at, trace }] = &found.conflicts[..] else {
        panic!("one conflict: {:?}", found.texts());
    };
    let binding = found.span_of("let y = x;", 0);
    assert!(binding.start <= at.start && at.end <= binding.end, "at the read: {at:?}");
    let [(_, earlier, later, uses)] = &found.texts()[..] else { unreachable!() };
    assert_eq!((earlier.as_slice(), later.as_slice(), uses.as_slice()), (&["m", "&mut x"][..], &[][..], &[][..]), "{trace:?}");
}

#[test]
fn two_mut_loans_on_disjoint_fields_of_one_binding_are_admitted() {
    let found = Found::of("let p = { a: 1, b: 2, }; let a = &mut p.a; let b = &mut p.b; *a = 3; *b = 4; p");
    assert!(found.admitted(), "{:?}", found.texts());
}

/// One fact: the second `&mut p.a` made while the first lives, held by
/// `a`'s store. `*a = 3;` reaches `p.a` through the first while the second
/// lives, a use of the same pair; storing the second into `b` reborrows it,
/// which is making it, not a use.
#[test]
fn two_mut_loans_on_one_field_are_refused() {
    let found = Found::of("let p = { a: 1, b: 2, }; let a = &mut p.a; let b = &mut p.a; *a = 3; *b = 4; p");
    let expected = Conflict {
        at: found.span_of("&mut p.a", 1),
        trace: ConflictTrace {
            earlier: vec![found.span_of("a", 1), found.span_of("&mut p.a", 0)],
            later: vec![found.span_of("&mut p.a", 1)],
            uses: vec![found.span_of("*a = 3;", 0)],
        },
    };
    assert_eq!(found.conflicts, [expected], "{:?}", found.texts());
}

#[test]
fn as_mut_on_a_tuple_with_both_parts_written_is_admitted() {
    let found = Found::of("let t = (1, 2); let m = t.as_mut(); match m { (a, b) => { *a = 3; *b = 4; 0 }, }; t");
    assert!(found.admitted(), "{:?}", found.texts());
}

/// A shared reborrow of a `&mut` is a loan of its own, shared: a second
/// one beside it is admitted, and a write through the `&mut` while one
/// lives is one refusal, at the write: the shared reborrow `a` holds, which
/// its store does not reborrow again, made through `m`'s store's, met by
/// the write through `m`'s store's.
#[test]
fn a_mut_loan_reached_through_a_shared_reborrow_reads_shared() {
    let found = Found::of("let x = 1; let m = &mut x; let a = &*m; let b = &*m; (*a, *b)");
    assert!(found.admitted(), "{:?}", found.texts());

    let found = Found::of("let x = 1; let m = &mut x; let a = &*m; *m = 2; *a");
    let expected = ("*m = 2;", vec!["&*m", "m", "&mut x"], vec!["m", "&mut x"], Vec::new());
    assert_eq!(found.texts(), [expected], "{:?}", found.conflicts);
}

#[test]
fn a_use_through_the_loans_own_reference_is_admitted() {
    let found = Found::of("let x = 1; let m = &mut x; *m = 2; let y = *m; *m = y; *m");
    assert!(found.admitted(), "{:?}", found.texts());
}

#[test]
fn a_write_after_the_loans_holder_is_dead_is_admitted() {
    let found = Found::of("let x = 1; let r = &x; let y = *r; x = 2; (x, y)");
    assert!(found.admitted(), "{:?}", found.texts());
}


/// A module whose entry makes a closure of a body that returns its
/// parameter, at `fn(&i64) -> &i64` claiming `flows`; what `check` refuses
/// of it.
fn lambda_returning_its_parameter(flows: Flows) -> Vec<crate::check::Refusal> {
    let names = Interner::new();
    let mut types = Types::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let ref_int = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
    let signature = types.intern(TyKind::Fn {
        params: Box::new([ref_int]),
        ret: ref_int,
        effect: EffectTerm::Known(Effect::pure()),
        flows: FlowTerm::Known(flows),
        calls: Calls::Many,
        holds: Holds::Nothing,
        values: (),
    });
    let mut module = Module::new(types);
    let mut lambda = module.build_body(false, ref_int);
    let p = lambda.param(names.intern("p"), ref_int, Span::ZERO);
    lambda.terminate(Terminator::Return { value: p, order: None, span: Span::ZERO }, Span::ZERO);
    let lambda = lambda.seal().expect("the lambda has no open state");
    let mut entry = module.build_body(false, signature);
    let closure = entry.push(Op::MakeClosure { bodies: vec![lambda], captures: Vec::new() }, signature, Span::ZERO);
    entry.terminate(Terminator::Return { value: closure, order: None, span: Span::ZERO }, Span::ZERO);
    let entry = entry.seal().expect("the entry has no open state");
    module.set_entry(entry);
    let raw = RawDeclarations {
        types: Types::new(),
        functions: Vec::new(),
        lang_impls: Vec::new(),
        user_types: Vec::new(),
        casts: Vec::new(),
        conversions: Vec::new(),
        contexts: Vec::new(),
        inputs: Vec::new(),
        loop_markers: LoopMarkers::default(),
        core: Core::of(&names),
    };
    let decls = Declarations::verify(raw).expect("no declarations verify");
    let mut am = AnalysisManager::new(&decls);
    crate::check::check(&module, &mut am)
}

/// A lambda's flows are what its body does: returning its parameter is a
/// flow from it into the result, which the claim states.
#[test]
fn a_lambda_body_within_its_claimed_flows_holds() {
    let flows = Flows::of([Flow { to: FlowEnd::Result, from: FlowEnd::Param(0), alignment: Alignment::Any }]);
    assert_eq!(lambda_returning_its_parameter(flows), []);
}

/// A claim that states no flow the body makes is the compiler's defect.
#[test]
#[should_panic(expected = "breaks its claimed flows")]
fn a_lambda_body_outside_its_claimed_flows_is_a_defect() {
    lambda_returning_its_parameter(Flows::none());
}
