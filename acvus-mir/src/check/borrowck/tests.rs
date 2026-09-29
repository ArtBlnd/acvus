//! borrowck over programs built as `lower/tests.rs` builds them: parsed,
//! checked by typeck over verified declarations, lowered, and run through
//! the required steps up to `remove_dead_context_ops`.

use acvus_ast::Script;
use acvus_utils::Interner;

use super::{Met, met};
use crate::decl::{Core, Declarations, LoopMarkers, RawDeclarations};
use crate::ir::PathSeg;
use crate::ir::cfg::Module;
use crate::lower::lower_script;
use crate::lower::required::{clone_strings, remove_dead_context_ops};
use crate::passes::analysis::{AnalysisManager, Mode, Storage};
use crate::ty::Types;
use crate::typeck::check_script;

/// What borrowck finds in `source`: each live loan an access meets, with
/// the text of the access.
struct Found {
    source: String,
    names: Interner,
    met: Vec<Met>,
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
        let typeck = check_script(&decls, &mut types, &script);
        assert!(typeck.refusals.is_empty(), "typeck refused {source:?}: {:?}", typeck.refusals);
        let mut module: Module = lower_script(&decls, &typeck, types, &script);
        let mut am = AnalysisManager::new(&decls);
        clone_strings(&mut module, &mut am);
        remove_dead_context_ops(&mut module, &mut am);
        let met = module.body_ids().flat_map(|body| met(&module, body, &mut am)).collect();
        Found { source: source.to_owned(), names, met }
    }

    /// The text of each access that meets a loan, with the loan's mode,
    /// storage kind and path.
    fn accesses(&self) -> Vec<(&str, Mode, Vec<PathSeg>)> {
        self.met
            .iter()
            .map(|Met { access, loan }| (&self.source[access.start..access.end], loan.mode, loan.path.clone()))
            .collect()
    }

    fn name(&self, name: &str) -> acvus_utils::Astr {
        self.names.intern(name)
    }

    fn admitted(&self) -> bool {
        self.met.is_empty()
    }

    fn on_slots_only(&self) -> bool {
        self.met.iter().all(|Met { loan, .. }| matches!(loan.storage, Storage::Slot(_)))
    }
}

#[test]
fn a_write_while_a_shared_loan_lives_is_refused() {
    let found = Found::of("let x = 1; let r = &x; x = 2; *r");
    assert!(found.on_slots_only(), "{:?}", found.met);
    assert_eq!(found.accesses(), [("x = 2;", Mode::Shared, Vec::new())], "{:?}", found.met);
}

#[test]
fn a_read_while_a_mut_loan_lives_is_refused() {
    let found = Found::of("let x = 1; let m = &mut x; let y = x; *m = 3; y");
    assert!(found.on_slots_only(), "{:?}", found.met);
    let accesses = found.accesses();
    assert!(!accesses.is_empty(), "the read of `x` meets `m`'s loan");
    assert!(accesses.iter().all(|(_, mode, path)| *mode == Mode::Exclusive && path.is_empty()), "{accesses:?}");
}

#[test]
fn two_mut_loans_on_disjoint_fields_of_one_binding_are_admitted() {
    let found = Found::of("let p = { a: 1, b: 2, }; let a = &mut p.a; let b = &mut p.b; *a = 3; *b = 4; p");
    assert!(found.admitted(), "{:?}", found.accesses());
}

#[test]
fn two_mut_loans_on_one_field_are_refused() {
    let found = Found::of("let p = { a: 1, b: 2, }; let a = &mut p.a; let b = &mut p.a; *a = 3; *b = 4; p");
    assert_eq!(found.accesses(), [("&mut p.a", Mode::Exclusive, vec![PathSeg::Field(found.name("a"))])], "{:?}", found.met);
}

#[test]
fn as_mut_on_a_tuple_with_both_parts_written_is_admitted() {
    let found = Found::of("let t = (1, 2); let m = t.as_mut(); match m { (a, b) => { *a = 3; *b = 4; 0 }, }; t");
    assert!(found.admitted(), "{:?}", found.accesses());
}

/// A shared reborrow of a `&mut` holds its loan shared: a second one beside
/// it is admitted, and a write through the `&mut` while one lives is not.
#[test]
fn a_mut_loan_reached_through_a_shared_reborrow_reads_shared() {
    let found = Found::of("let x = 1; let m = &mut x; let a = &*m; let b = &*m; (*a, *b)");
    assert!(found.admitted(), "{:?}", found.accesses());

    let found = Found::of("let x = 1; let m = &mut x; let a = &*m; *m = 2; *a");
    assert!(found.on_slots_only(), "{:?}", found.met);
    assert_eq!(found.accesses(), [("*m = 2;", Mode::Shared, Vec::new())], "{:?}", found.met);
}

#[test]
fn a_use_through_the_loans_own_reference_is_admitted() {
    let found = Found::of("let x = 1; let m = &mut x; *m = 2; let y = *m; *m = y; *m");
    assert!(found.admitted(), "{:?}", found.accesses());
}

#[test]
fn a_write_after_the_loans_holder_is_dead_is_admitted() {
    let found = Found::of("let x = 1; let r = &x; let y = *r; x = 2; (x, y)");
    assert!(found.admitted(), "{:?}", found.accesses());
}

