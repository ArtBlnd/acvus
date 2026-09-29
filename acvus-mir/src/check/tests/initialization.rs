//! initck and moveck over programs built from source: parsed, checked by
//! typeck, lowered, and taken through the required steps up to
//! `remove_dead_context_ops`, as the checks see them.

use acvus_ast::{Script, Span};
use acvus_utils::{Astr, Interner, QualifiedRef};

use crate::check::{Refusal, RefusalKind, check};
use crate::decl::{Core, Declarations, DeclaredFacts, RawDeclarations, Scheme, TyVarBound, UserTypeDecl};
use crate::lower::{lower_script, required};
use crate::passes::analysis::AnalysisManager;
use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, ObjectTy, Poly, TyId, TyKind, TypeArg, Types};
use crate::typeck::check_script;

struct Fixture {
    names: Interner,
    raw: RawDeclarations,
}

impl Fixture {
    fn new() -> Self {
        let names = Interner::new();
        let core = Core::of(&names);
        Fixture {
            names,
            raw: RawDeclarations {
                types: Types::new(),
                functions: Vec::new(),
                lang_impls: Vec::new(),
                user_types: Vec::new(),
                casts: Vec::new(),
                conversions: Vec::new(),
                contexts: Vec::new(),
                inputs: Vec::new(),
                loop_markers: Default::default(),
                core,
            },
        }
    }

    fn name(&self, name: &str) -> Astr {
        self.names.intern(name)
    }

    fn ty(&mut self, kind: TyKind<Poly>) -> TyId<Poly> {
        self.raw.types.intern(kind)
    }

    fn int(&mut self) -> TyId<Poly> {
        self.ty(TyKind::Int(IntTy::I64))
    }

    fn object(&mut self, fields: &[(&str, TyId<Poly>)]) -> TyId<Poly> {
        let fields = fields.iter().map(|(name, ty)| (self.name(name), *ty)).collect();
        self.ty(TyKind::Object(ObjectTy::Structural(fields, ())))
    }

    /// A pure extern fn `name` of these parameters and result.
    fn declare(&mut self, name: &str, params: &[TyId<Poly>], ret: TyId<Poly>) -> QualifiedRef {
        let ty = self.ty(TyKind::Fn {
            params: params.into(),
            ret,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        let id = QualifiedRef::root(self.name(name));
        let param_names = (0..params.len()).map(|at| self.name(&format!("p{at}"))).collect();
        let scheme = Scheme { ty, param_names, bounds: Box::new([]), impls: None, requires: Box::new([]), facts: DeclaredFacts::default() };
        self.raw.functions.push((id, scheme));
        id
    }

    fn input(&mut self, name: &str, ty: TyId<Poly>) {
        let name = self.name(name);
        self.raw.inputs.push((name, ty));
    }

    /// An extern `Vec<i64>`, which does not copy (ownership §1).
    fn vec_of_ints(&mut self) -> TyId<Poly> {
        let id = QualifiedRef::root(self.name("Vec"));
        self.raw.user_types.push(UserTypeDecl {
            id,
            type_params: Box::new([TyVarBound::Any]),
            specializes: Box::new([false]),
            effect_params: 0,
            identity_params: 0,
            lifetimes: 0,
        });
        let int = self.int();
        self.ty(TyKind::UserDefined {
            id,
            type_args: Box::new([TypeArg::Uniform(int)]),
            effect_args: Box::new([]),
            identity_args: Box::new([]),
        })
    }

    /// `make() -> Vec<i64>` and `consume(Vec<i64>) -> i64`; the type
    /// `Vec<i64>`.
    fn vecs(&mut self) -> TyId<Poly> {
        let (vec, int) = (self.vec_of_ints(), self.int());
        self.declare("make", &[], vec);
        self.declare("consume", &[vec], int);
        vec
    }

    /// The refusals of `check` over `source`, which typeck admits.
    fn refusals(self, source: &str) -> Vec<Refusal> {
        let script: Script = acvus_ast::parse_script(&self.names, source).expect("the test's source parses");
        let decls = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut types = Types::new();
        let typeck = check_script(&decls, &mut types, &script);
        assert!(typeck.refusals.is_empty(), "typeck refused: {:?}", typeck.refusals);
        let mut module = lower_script(&decls, &typeck, types, &script);
        let mut am = AnalysisManager::new(&decls);
        required::clone_strings(&mut module, &mut am);
        required::remove_dead_context_ops(&mut module, &mut am);
        check(&module, &mut am)
    }
}

/// The span of the `nth` occurrence of `text` in `source`, from 0.
fn span_of(source: &str, text: &str, nth: usize) -> Span {
    let start = source.match_indices(text).nth(nth).unwrap_or_else(|| panic!("{text:?} occurs {nth} times before in {source:?}")).0;
    Span { start, end: start + text.len() }
}

fn covers(outer: Span, inner: Span) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

/// ownership §4's example: on the one path that reaches the whole use,
/// every field of the binding's type is written before it.
#[test]
fn a_binding_built_field_by_field_is_used_whole_where_every_field_is_written() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let int = fx.int();
    let x = fx.object(&[("x", int)]);
    fx.declare("f", &[x], int);
    let refusals = fx.refusals("let a; if $c { a.x = 0; f(a) } else { 1 }");
    assert_eq!(refusals, []);
}

/// ownership §4: a field is read only where every path to the read has
/// written it.
#[test]
fn a_field_one_path_did_not_write_is_not_read() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let source = "let a; if $c { a.x = 0; 1 } else { 2 }; a.x";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::NotWritten, span }] = refusals[..] else {
        panic!("one NotWritten: {refusals:?}");
    };
    assert!(covers(span, span_of(source, "a.x", 1)), "at the read: {span:?}");
}

/// ownership §4: the value is used whole only where every path to the use
/// has written every field of its type, which grew by a later store.
#[test]
fn a_whole_use_before_a_field_of_its_type_is_written_is_refused() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let source = "let a; if $c { a.x = 1; 1 } else { 2 }; a.y = true; a";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::NotWritten, span }] = refusals[..] else {
        panic!("one NotWritten: {refusals:?}");
    };
    assert!(covers(span, span_of(source, "a", 3)), "at the whole use: {span:?}");
}

/// ownership §4: a binding declared without a value is not read before a
/// store writes it.
#[test]
fn a_binding_read_before_any_store_is_refused() {
    let fx = Fixture::new();
    let source = "let a; let b = a; a = 1; b";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::NotWritten, span }] = refusals[..] else {
        panic!("one NotWritten: {refusals:?}");
    };
    assert!(covers(span, span_of(source, "a", 1)), "at the read: {span:?}");
}

/// ownership §2: a use moves a value that does not copy, and a later use
/// is refused, naming the move.
#[test]
fn a_value_that_does_not_copy_is_not_used_after_its_move() {
    let mut fx = Fixture::new();
    fx.vecs();
    let source = "let v = make(); consume(v); consume(v)";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::UsedAfterMove { moved }, span }] = refusals[..] else {
        panic!("one UsedAfterMove: {refusals:?}");
    };
    assert!(covers(span, span_of(source, "v", 2)), "at the second use: {span:?}");
    assert!(covers(moved, span_of(source, "v", 1)), "naming the first use: {moved:?}");
}

/// ownership §6: a value a part of which moved out is not used whole
/// again.
#[test]
fn a_value_whose_part_moved_out_is_not_used_whole() {
    let mut fx = Fixture::new();
    fx.vecs();
    let source = "let p = { a: make(), b: 1, }; consume(p.a); p";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::UsedAfterMove { moved }, span }] = refusals[..] else {
        panic!("one UsedAfterMove: {refusals:?}");
    };
    assert!(covers(span, span_of(source, "p", 2)), "at the whole use: {span:?}");
    assert!(covers(moved, span_of(source, "p.a", 0)), "naming the part's move: {moved:?}");
}

/// ownership §1-2: a value that copies clones by its copy, so every use
/// reads it.
#[test]
fn a_value_that_copies_is_used_again() {
    let fx = Fixture::new();
    let refusals = fx.refusals("let x = 1; let y = x; (x, y, x)");
    assert_eq!(refusals, []);
}

/// ownership §3: a `String`'s use that is not its last is a clone, so only
/// its last use moves it.
#[test]
fn a_string_used_twice_is_cloned_before_its_last_use() {
    let mut fx = Fixture::new();
    let string = fx.ty(TyKind::String);
    fx.declare("name", &[], string);
    let refusals = fx.refusals("let s = name(); (s, s)");
    assert_eq!(refusals, []);
}
