//! The graph over units built from source: each unit's interface after its
//! callees', the instantiations the entries reach, a cycle refused.

use acvus_utils::{Astr, Interner, QualifiedRef};

use super::{Graph, Refusal, Source, Stop, Unit};
use crate::decl::{Core, Declarations, EntryDecl, ImplSig, Impls, LoopMarkers, RawDeclarations};
use crate::passes::pipeline::Level;
use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Poly, TyId, TyKind, TypeArg, Types};

struct Fixture {
    names: Interner,
    raw: RawDeclarations,
    units: Vec<(QualifiedRef, Source)>,
}

impl Fixture {
    /// `core::add` at `u8` and `i64`, each `(&w, &w) -> w`.
    fn new() -> Self {
        let names = Interner::new();
        let core = Core::of(&names);
        let mut fx = Fixture {
            raw: RawDeclarations {
                lang_impls: Vec::new(),
                types: Types::new(),
                functions: Vec::new(),
                user_types: Vec::new(),
                casts: Vec::new(),
                conversions: Vec::new(),
                contexts: Vec::new(),
                entries: Vec::new(),
                loop_markers: LoopMarkers::default(),
                core,
            },
            names,
            units: Vec::new(),
        };
        let concrete = [IntTy::U8, IntTy::I64]
            .into_iter()
            .map(|width| {
                let w = fx.raw.types.intern(TyKind::Int(width));
                let shared = fx.raw.types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(w)));
                let ty = fx.pure(&[shared, shared], w);
                ImplSig { ty, param_names: Box::new([fx.names.intern("p0"), fx.names.intern("p1")]), requires: Box::new([]), facts: Default::default() }
            })
            .collect();
        let add = fx.raw.core.add;
        fx.raw.lang_impls.push((add, Impls { concrete, generic: None }));
        fx
    }

    fn pure(&mut self, params: &[TyId<Poly>], ret: TyId<Poly>) -> TyId<Poly> {
        self.raw.types.intern(TyKind::Fn {
            params: params.into(),
            ret,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        })
    }

    fn name(&self, path: &[&str], name: &str) -> QualifiedRef {
        let path: Vec<Astr> = path.iter().map(|segment| self.names.intern(segment)).collect();
        QualifiedRef::qualified(self.names.namespace(&path), self.names.intern(name))
    }

    fn script(&mut self, path: &[&str], name: &str, source: &str) -> QualifiedRef {
        let unit = self.name(path, name);
        let script = acvus_ast::parse_script(&self.names, source).expect("the unit's source parses");
        self.units.push((unit, Source::Script(script)));
        unit
    }

    /// An entry at `function`, its inputs each at `IntTy`, its result
    /// declared where given.
    fn entry(&mut self, function: QualifiedRef, inputs: &[(&str, IntTy)], result: Option<IntTy>) {
        let inputs = inputs.iter().map(|(name, width)| (self.names.intern(name), self.raw.types.intern(TyKind::Int(*width)))).collect();
        let result = result.map(|width| self.raw.types.intern(TyKind::Int(width)));
        self.raw.entries.push(EntryDecl { function, inputs, result });
    }

    fn graph(self) -> Graph {
        let declarations = Declarations::verify(self.raw).expect("the test's declarations verify");
        let mut graph = Graph::new(self.names, declarations);
        for (name, source) in self.units {
            graph.set_unit(Unit { name, source });
        }
        graph
    }
}

/// programs, host: an entry is its unit instantiated at the declared
/// types, one body of the unit's module.
#[test]
fn an_entry_is_its_unit_at_the_declared_types() {
    let mut fx = Fixture::new();
    let main = fx.script(&["app"], "main", "$n + 1");
    fx.entry(main, &[("n", IntTy::I64)], Some(IntTy::I64));
    let mut graph = fx.graph();
    let compiled = graph.compile(Stop::Optimized(Level::Full)).unwrap_or_else(|refused| panic!("refused: {refused:?}"));
    assert_eq!(compiled.entries.len(), 1);
    let (unit, entry) = compiled.entries[0];
    assert_eq!((unit, entry.module), (main, main));
    let (_, module) = compiled.modules.iter().find(|(name, _)| *name == main).expect("the entry's module");
    assert_eq!(module.instantiations.len(), 1);
    assert_eq!(module.module.types().kind(module.module.body(entry.body).ret()), &TyKind::Int(IntTy::I64));
}

/// programs §1-§2: a unit called at two types from the entry is two
/// instantiations, two bodies of its one module.
#[test]
fn a_unit_called_at_two_types_is_two_bodies_of_its_module() {
    let mut fx = Fixture::new();
    let inc = fx.script(&["lib"], "inc", "$x + 1");
    let main = fx.script(&["app"], "main", "(lib::inc(x: $n), lib::inc(x: 1u8))");
    fx.entry(main, &[("n", IntTy::I64)], None);
    let mut graph = fx.graph();
    let compiled = graph.compile(Stop::Optimized(Level::Full)).unwrap_or_else(|refused| panic!("refused: {refused:?}"));
    let (_, module) = compiled.modules.iter().find(|(name, _)| *name == inc).expect("the callee's module");
    let widths: Vec<&TyKind<crate::ty::Concrete>> =
        module.instantiations.iter().map(|(types, _)| module.module.types().kind(types[0])).collect();
    assert_eq!(widths, vec![&TyKind::Int(IntTy::I64), &TyKind::Int(IntTy::U8)]);
}

/// functions §4: a cycle of calls through no `fn` is refused, at each unit
/// on it.
#[test]
fn a_cycle_of_unit_calls_is_refused() {
    let mut fx = Fixture::new();
    let a = fx.script(&["m"], "a", "m::b(x: $x)");
    let b = fx.script(&["m"], "b", "m::a(x: $x)");
    fx.entry(a, &[("x", IntTy::I64)], None);
    let mut graph = fx.graph();
    let Err(refused) = graph.compile(Stop::Optimized(Level::Full)) else {
        panic!("a cycle is refused");
    };
    let mut on: Vec<QualifiedRef> = refused
        .iter()
        .filter(|refused| matches!(refused.refusals[..], [Refusal::Cycle { .. }]))
        .map(|refused| refused.unit)
        .collect();
    on.sort();
    let mut expected = vec![a, b];
    expected.sort();
    assert_eq!(on, expected);
}

/// host: an entry whose declared inputs are not the unit's parameters, or
/// whose declared result is not the unit's at them, does not fit.
#[test]
fn an_entry_that_does_not_fit_its_unit_is_refused() {
    for (inputs, result) in [(vec![("m", IntTy::I64)], None), (vec![("n", IntTy::I64)], Some(IntTy::U8))] {
        let mut fx = Fixture::new();
        let main = fx.script(&["app"], "main", "$n + 1");
        fx.entry(main, &inputs, result);
        let mut graph = fx.graph();
        let Err(refused) = graph.compile(Stop::Optimized(Level::Full)) else {
            panic!("the entry does not fit");
        };
        assert!(matches!(refused[..], [super::Refused { unit, ref refusals }] if unit == main && matches!(refusals[..], [Refusal::EntryDoesNotFit { .. }])), "{refused:?}");
    }
}
