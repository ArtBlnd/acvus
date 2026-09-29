//! Takings of a `Take`, by name and through a reference, each test citing
//! the ledger rule it checks.

use acvus_ast::{Literal, Span};
use acvus_utils::Interner;

use super::*;
use crate::ir::cfg::Module;
use crate::ir::PathSeg;
use crate::ty::{IntTy, ObjectTy, TypeArg};

struct Fixture {
    module: Module,
    int: Ty,
    string: Ty,
    /// `&mut String`.
    string_mut: Ty,
}

fn fixture() -> Fixture {
    let mut types = Types::<Concrete>::new();
    let int = types.intern(TyKind::Int(IntTy::I64));
    let string = types.intern(TyKind::String);
    let string_mut = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(string)));
    Fixture { module: Module::new(types), int, string, string_mut }
}

fn take(slot: ValueId, path: Vec<PathSeg>) -> Op {
    Op::Take { target: RefTarget::Var(slot), path, taken_out: false }
}

fn assign(slot: ValueId, value: ValueId) -> Op {
    Op::Assign { target: RefTarget::Var(slot), path: Vec::new(), value, restores: false }
}

/// The takings of every `Take` in `body`'s entry block, in order.
fn take_takings(module: &Module, body: crate::ir::cfg::BodyId) -> Vec<Vec<Operand>> {
    let body = module.body(body);
    body.block(body.entry())
        .insts()
        .filter(|inst| matches!(inst.op(), Op::Take { .. }))
        .map(|inst| inst_takings(body, module.types(), inst).to_vec())
        .collect()
}

/// `s = "a"; r = &mut s; r; r`: each take of the slot `r` reborrows the
/// `&mut` it holds, so `r` keeps it and both are read (ownership §5).
#[test]
fn a_mut_ref_slot_taken_twice_is_read_both_times() {
    let mut f = fixture();
    let mut builder = f.module.build_body(false, f.int);
    let s = builder.slot(f.string, Span::ZERO);
    let text = builder.push(Op::Const { value: Literal::String("a".to_string()) }, f.string, Span::ZERO);
    builder.push_unit(assign(s, text), Span::ZERO);
    let r = builder.slot(f.string_mut, Span::ZERO);
    let lent = builder.push(
        Op::Ref { target: RefTarget::Var(s), path: Vec::new(), mutability: Mutability::Mut },
        f.string_mut,
        Span::ZERO,
    );
    builder.push_unit(assign(r, lent), Span::ZERO);
    builder.push(take(r, Vec::new()), f.string_mut, Span::ZERO);
    builder.push(take(r, Vec::new()), f.string_mut, Span::ZERO);
    let zero = builder.push(Op::Const { value: Literal::Int(0) }, f.int, Span::ZERO);
    builder.terminate(Terminator::Return { value: zero, order: None, span: Span::ZERO }, Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let read = Operand { value: r, taken: Taken::Read, names_place: true, lent: None };
    let takings = take_takings(&f.module, id);
    assert_eq!(takings, [[read], [read]]);
}

/// `o = { m: &mut s }; o.m; o`: the field `m` is a `&mut`, so taking it by
/// path reborrows it and `o` keeps it (ownership §5); `o` itself does not
/// copy (ownership §1), so taking it whole moves it (ownership §2).
#[test]
fn an_object_holding_a_mut_ref_is_moved_whole_and_its_mut_ref_field_is_read() {
    let mut f = fixture();
    let interner = Interner::new();
    let m = interner.intern("m");
    let mut builder = f.module.build_body(false, f.int);
    let holder = builder.intern(TyKind::Object(ObjectTy::Structural(Box::new([(m, f.string_mut)]), ())));
    let s = builder.slot(f.string, Span::ZERO);
    let text = builder.push(Op::Const { value: Literal::String("a".to_string()) }, f.string, Span::ZERO);
    builder.push_unit(assign(s, text), Span::ZERO);
    let lent = builder.push(
        Op::Ref { target: RefTarget::Var(s), path: Vec::new(), mutability: Mutability::Mut },
        f.string_mut,
        Span::ZERO,
    );
    let object = builder.push(Op::MakeObject { fields: vec![(m, lent)] }, holder, Span::ZERO);
    let o = builder.slot(holder, Span::ZERO);
    builder.push_unit(assign(o, object), Span::ZERO);
    builder.push(take(o, vec![PathSeg::Field(m)]), f.string_mut, Span::ZERO);
    builder.push(take(o, Vec::new()), holder, Span::ZERO);
    let zero = builder.push(Op::Const { value: Literal::Int(0) }, f.int, Span::ZERO);
    builder.terminate(Terminator::Return { value: zero, order: None, span: Span::ZERO }, Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let field = Operand { value: o, taken: Taken::Read, names_place: true, lent: None };
    let whole = Operand { value: o, taken: Taken::Moved, names_place: true, lent: None };
    let takings = take_takings(&f.module, id);
    assert_eq!(takings, [[field], [whole]]);
}

/// `s = "a"; r = &mut s; *r`: what the take reaches through `r` does not
/// copy and is not a `&mut`, so it is moved out of the place `r` lends
/// (ownership §2, §4); `r` itself is read.
#[test]
fn a_take_through_a_reference_moves_what_it_lends_and_reads_the_reference() {
    let mut f = fixture();
    let mut builder = f.module.build_body(false, f.int);
    let s = builder.slot(f.string, Span::ZERO);
    let text = builder.push(Op::Const { value: Literal::String("a".to_string()) }, f.string, Span::ZERO);
    builder.push_unit(assign(s, text), Span::ZERO);
    let r = builder.push(
        Op::Ref { target: RefTarget::Var(s), path: Vec::new(), mutability: Mutability::Mut },
        f.string_mut,
        Span::ZERO,
    );
    builder.push(Op::Take { target: RefTarget::Through(r), path: Vec::new(), taken_out: false }, f.string, Span::ZERO);
    let zero = builder.push(Op::Const { value: Literal::Int(0) }, f.int, Span::ZERO);
    builder.terminate(Terminator::Return { value: zero, order: None, span: Span::ZERO }, Span::ZERO);
    let id = builder.seal().expect("the body has no open state");

    let through = Operand { value: r, taken: Taken::Read, names_place: false, lent: Some(Taken::Moved) };
    let takings = take_takings(&f.module, id);
    assert_eq!(takings, [[through]]);
}
