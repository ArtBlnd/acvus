//! Global value numbering over the dominator tree, with scoped hash tables:
//! walking the tree in preorder, an instruction whose key is already in the
//! table of a dominating block takes that block's value, and otherwise
//! enters its key until its subtree is left.
//!
//! A key is one of:
//! - a numbered instruction's kind, immediate fields, result type and
//!   operands, the operands sorted where the operation commutes on their
//!   type. The operands are already numbered, since every replacement
//!   rewrites the uses it takes over.
//! - a `Merge`'s inputs taken as a set, flattened through the merges among
//!   them (effects §3: associative, commutative, idempotent). A merge whose
//!   inputs, or whose flattened inputs, are one token is that token.
//!
//! An instruction is numbered when it stands on no order chain, names no
//! storage, touches no context, and, for a call, calls a callee whose
//! reissue is pure and that writes no context. Its operands and its result
//! must also copy (ownership §1). A value that does not copy is moved by
//! its one use and ends at one place (ownership §2, §6), so two such values
//! are never one, and a `&mut` operand may be written through between the
//! two reads. A shared reference's storage is not written while a value
//! holding its loan lives (ownership §5), and the dominated instruction
//! reads the same reference, so reads through a shared reference number.
//!
//! A float `+` and `*` are keyed in operand order: IEEE leaves open which
//! NaN operand's payload the result carries, so the two orders are not
//! known to give the same bits.
//!
//! The replaced instruction stays, unread, for `dce`.
//!
//! Form: F1, SSA, in and out.
//!
//! Cost: O(instructions) expected table operations; each replacement costs
//! the uses it rewrites; a merge's flattened set costs its inputs'.
//!
//! Meaning: a dominating equal computation already ran on every path to
//! the replaced one, with the same operands and no state between them that
//! it reads, and a trapping one already trapped where this one would. A
//! merge orders only what follows it, and equal sets order it equally, so
//! every token stays defined before its uses (effects §3). A replacement
//! has the replaced value's type, which is part of the key.

use std::collections::BTreeSet;

use acvus_ast::Literal;
use acvus_utils::{Astr, QualifiedRef};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::ir::cfg::edit::{DoesNotDominate, Editor, InstRef, Value};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Inst, Module, Op};
use crate::ir::{BinOp, Callee, Chosen, IndexBound, IndexMode, UnaryOp, ValueId, fn_effect};
use crate::passes::analysis::{AnalysisManager, Dominators};
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;
use crate::ty::{CastTy, Concrete, Mutability, Reissue, Ty, TyKind, Types, copies};

pub struct Gvn;

impl BodyTransform for Gvn {
    const NAME: &'static str = "gvn";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let dominators = am.get::<Dominators>(module, body);
        let children = dominator_children(module.body(body), &dominators);
        let entry = module.body(body).entry();
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Value>::new(module, body, &mut facts);
        let mut numbering = Numbering::default();
        let mut visits = vec![Visit::Enter(entry)];
        while let Some(visit) = visits.pop() {
            match visit {
                Visit::Enter(block) => {
                    visits.push(Visit::Leave(numbering.entered.len()));
                    visits.extend(children[block.index()].iter().rev().map(|child| Visit::Enter(*child)));
                    for inst in editor.insts_of(block) {
                        numbering.number(&mut editor, inst);
                    }
                }
                Visit::Leave(kept) => numbering.leave(kept),
            }
        }
        editor.finish()
    }
}

/// Each block's children in the dominator tree, by `BlockId::index`, in
/// block order. A block the entry does not reach has no parent and is not
/// walked. Cost: O(blocks).
fn dominator_children(body: &Body, dominators: &Dominators) -> Vec<Vec<BlockId>> {
    let mut children = vec![Vec::new(); body.blocks().count()];
    for (block, _) in body.blocks() {
        if let Some(parent) = dominators.idom(block) {
            children[parent.index()].push(block);
        }
    }
    children
}

enum Visit {
    Enter(BlockId),
    /// Leaves a subtree, dropping every key entered since `entered` held
    /// this many.
    Leave(usize),
}

#[derive(Default)]
struct Numbering {
    table: FxHashMap<Key, ValueId>,
    entered: Vec<Key>,
    /// Every merge walked so far, by its result, with its inputs flattened
    /// through the merges among them.
    merges: FxHashMap<ValueId, BTreeSet<ValueId>>,
}

impl Numbering {
    fn number(&mut self, editor: &mut Editor<'_, Value>, inst: InstRef) {
        let inst = editor.inst_of(inst);
        let Some(result) = inst.result() else {
            return;
        };
        let found = match inst.op() {
            Op::Merge { orders } => self.merge(result, orders),
            _ => match key_of(editor.body(), editor.types(), inst) {
                Some(key) => Found::Key(key),
                None => return,
            },
        };
        let key = match found {
            Found::Token(token) => {
                replace(editor, result, token);
                return;
            }
            Found::Key(key) => key,
        };
        match self.table.get(&key) {
            Some(earlier) => replace(editor, result, *earlier),
            None => {
                self.table.insert(key.clone(), result);
                self.entered.push(key);
            }
        }
    }

    /// A merge's key, or the one token it is.
    fn merge(&mut self, result: ValueId, orders: &[ValueId]) -> Found {
        let inputs: BTreeSet<ValueId> = orders.iter().copied().collect();
        let flattened: BTreeSet<ValueId> = inputs
            .iter()
            .flat_map(|input| match self.merges.get(input) {
                Some(nested) => nested.iter().copied().collect::<Vec<_>>(),
                None => vec![*input],
            })
            .collect();
        self.merges.insert(result, flattened.clone());
        match (one_of(&inputs), one_of(&flattened)) {
            (Some(token), _) | (None, Some(token)) => Found::Token(token),
            (None, None) => Found::Key(Key::Merge(flattened.into_iter().collect())),
        }
    }

    fn leave(&mut self, kept: usize) {
        for key in self.entered.drain(kept..) {
            self.table.remove(&key).expect("an entered key stays in the table until its subtree is left");
        }
    }
}

fn one_of(set: &BTreeSet<ValueId>) -> Option<ValueId> {
    match set.len() {
        1 => set.first().copied(),
        _ => None,
    }
}

/// Uses of `of` take `with`. A refusal leaves `of` as it is; `with` comes
/// from a dominating block or earlier in this one, so none is expected.
fn replace(editor: &mut Editor<'_, Value>, of: ValueId, with: ValueId) {
    match editor.replace_uses(of, with) {
        Ok(()) | Err(DoesNotDominate { .. }) => {}
    }
}

enum Found {
    Key(Key),
    Token(ValueId),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Inst { shape: Shape, ty: Ty, operands: SmallVec<[ValueId; 4]> },
    /// A merge's flattened inputs, in order.
    Merge(Vec<ValueId>),
}

/// An instruction's kind with its immediate fields.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Shape {
    Const(LiteralKey),
    ConstStr(String),
    StringConcat,
    StringEq,
    StringClone,
    AsSlice { mutability: Mutability, id: QualifiedRef, r#impl: Option<usize> },
    ArrayView(Mutability),
    SliceLen,
    /// An `Index`: whether it yields the element itself, and whether it
    /// checks its bound.
    Index { copies_element: bool, checked: bool },
    FieldGet { field: Astr, rest: Vec<Astr> },
    FieldSet { field: Astr, rest: Vec<Astr> },
    Binary(BinOp),
    Unary(UnaryOp),
    Cast(CastTy),
    LoadFunction(QualifiedRef),
    Call { callee: CalleeKey, callee_ty: Ty },
    ArrayBegin(usize),
    ArrayPush,
    MakeObject(Vec<Astr>),
    MakeTuple,
    TupleIndex(usize),
    TestLiteral(LiteralKey),
    TestObjectKey(Astr),
    ArrayIndex(usize),
    ObjectGet(Astr),
    MakeClosure(Vec<BodyId>),
    MakeVariant { tag: Astr, has_payload: bool },
    TestVariant(Astr),
    UnwrapVariant,
    Undef,
}

/// A callee without its operand: an indirect callee is an operand of the
/// call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CalleeKey {
    Direct(QualifiedRef),
    Extern { id: QualifiedRef, r#impl: Option<usize>, required: Vec<Chosen> },
    Indirect,
}

/// A literal's value, a float by its bits, so that equal keys are equal
/// values; the width a suffix names is the literal's type, which the key
/// holds beside it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum LiteralKey {
    String(String),
    Int(i128),
    Float(u64),
    Char(char),
    Bool(bool),
    List(Vec<LiteralKey>),
    Unit,
}

impl LiteralKey {
    fn of(literal: &Literal) -> LiteralKey {
        match literal {
            Literal::String(text) => LiteralKey::String(text.clone()),
            Literal::Int(value) => LiteralKey::Int(*value),
            Literal::IntOf(suffixed) => LiteralKey::Int(suffixed.value),
            Literal::Float(value) => LiteralKey::Float(value.to_bits()),
            Literal::Char(value) => LiteralKey::Char(*value),
            Literal::Bytes(bytes) => LiteralKey::List(bytes.iter().map(|byte| LiteralKey::Int(i128::from(*byte))).collect()),
            Literal::Bool(value) => LiteralKey::Bool(*value),
            Literal::List(items) => LiteralKey::List(items.iter().map(LiteralKey::of).collect()),
            Literal::Unit => LiteralKey::Unit,
        }
    }
}

/// `inst`'s key where it is numbered.
fn key_of(body: &Body, types: &Types<Concrete>, inst: &Inst) -> Option<Key> {
    let result = inst.result()?;
    if inst.order().is_some() {
        return None;
    }
    let shape = shape_of(types, inst.op())?;
    let mut operands = inst.operands();
    let copying = |value: &ValueId| copies(types, body.ty_of(*value));
    if !copying(&result) || !operands.iter().all(copying) {
        return None;
    }
    if let (Shape::Binary(op), [left, _]) = (&shape, &operands[..])
        && commutes(*op, types.kind(body.ty_of(*left)))
    {
        operands.sort();
    }
    Some(Key::Inst { shape, ty: body.ty_of(result), operands })
}

/// The shape of an operation numbering may key; none for one that names
/// storage, touches a context, stands on the order chain, yields nothing,
/// or calls a callee that is not pure.
fn shape_of(types: &Types<Concrete>, op: &Op) -> Option<Shape> {
    Some(match op {
        Op::Const { value } => Shape::Const(LiteralKey::of(value)),
        Op::ConstStr { text } => Shape::ConstStr(text.clone()),
        Op::StringConcat { .. } => Shape::StringConcat,
        Op::StringEq { .. } => Shape::StringEq,
        Op::StringClone { .. } => Shape::StringClone,
        Op::AsSlice { mutability, r#impl, .. } => {
            Shape::AsSlice { mutability: *mutability, id: r#impl.id, r#impl: r#impl.r#impl }
        }
        Op::ArrayView { mutability, .. } => Shape::ArrayView(*mutability),
        Op::SliceLen { .. } => Shape::SliceLen,
        Op::Index { mode, bound, .. } => Shape::Index {
            copies_element: match mode {
                IndexMode::Copy => true,
                IndexMode::Ref => false,
            },
            checked: match bound {
                IndexBound::Checked => true,
                IndexBound::Proven => false,
            },
        },
        Op::FieldGet { field, rest, .. } => Shape::FieldGet { field: *field, rest: rest.clone() },
        Op::FieldSet { field, rest, .. } => Shape::FieldSet { field: *field, rest: rest.clone() },
        Op::BinOp { op, .. } => Shape::Binary(*op),
        Op::UnaryOp { op, .. } => Shape::Unary(*op),
        Op::Cast { to, .. } => Shape::Cast(*to),
        Op::LoadFunction { id } => Shape::LoadFunction(*id),
        Op::FunctionCall { callee, callee_ty, .. } => {
            let Some(effect) = fn_effect(types, *callee_ty) else {
                panic!("a call's callee type is a function type")
            };
            if effect.reissue() != Reissue::Pure || !effect.writes().is_empty() {
                return None;
            }
            let callee = match callee {
                Callee::Direct(id) => CalleeKey::Direct(*id),
                Callee::Extern { id, r#impl, required } => {
                    CalleeKey::Extern { id: *id, r#impl: *r#impl, required: required.clone() }
                }
                Callee::Indirect(_) => CalleeKey::Indirect,
            };
            Shape::Call { callee, callee_ty: *callee_ty }
        }
        Op::ArrayBegin { capacity } => Shape::ArrayBegin(*capacity),
        Op::ArrayPush { .. } => Shape::ArrayPush,
        Op::MakeObject { fields } => Shape::MakeObject(fields.iter().map(|(name, _)| *name).collect()),
        Op::MakeTuple { .. } => Shape::MakeTuple,
        // `ArrayMap` calls a function value whose purity the op does not
        // carry, and `Gather` makes a reference: numbering keys neither.
        Op::Intrinsic { .. } => return None,
        Op::TupleIndex { index, .. } => Shape::TupleIndex(*index),
        Op::TestLiteral { value, .. } => Shape::TestLiteral(LiteralKey::of(value)),
        Op::TestObjectKey { key, .. } => Shape::TestObjectKey(*key),
        Op::ArrayIndex { index, .. } => Shape::ArrayIndex(*index),
        Op::ObjectGet { key, .. } => Shape::ObjectGet(*key),
        Op::MakeClosure { bodies, .. } => Shape::MakeClosure(bodies.clone()),
        // A specialized closure is its closure at fewer signatures; left to
        // the closure's own number.
        Op::Specialize { .. } => return None,
        Op::MakeVariant { tag, payload } => Shape::MakeVariant { tag: *tag, has_payload: payload.is_some() },
        Op::TestVariant { tag, .. } => Shape::TestVariant(*tag),
        Op::UnwrapVariant { .. } => Shape::UnwrapVariant,
        Op::Undef => Shape::Undef,
        Op::Ref { .. }
        | Op::Take { .. }
        | Op::Assign { .. }
        | Op::IndexSet { .. }
        | Op::StringAppend { .. }
        | Op::Fetch { .. }
        | Op::Commit { .. }
        | Op::Spawn { .. }
        | Op::TakeEffect { .. }
        | Op::Merge { .. }
        | Op::Check { .. }
        | Op::CheckSteps { .. }
        | Op::Drop { .. } => return None,
    })
}

/// Whether `op` gives the same result for its operands in either order, on
/// operands of `operand`'s type.
fn commutes(op: BinOp, operand: &TyKind<Concrete>) -> bool {
    match operand {
        TyKind::Int(_) => matches!(
            op,
            BinOp::Add(_)
                | BinOp::Mul(_)
                | BinOp::Eq
                | BinOp::Neq
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::Xor
                | BinOp::Min
                | BinOp::Max
        ),
        TyKind::Bool => matches!(op, BinOp::Eq | BinOp::Neq | BinOp::And | BinOp::Or | BinOp::Xor),
        TyKind::Float | TyKind::Char => matches!(op, BinOp::Eq | BinOp::Neq),
        TyKind::String
        | TyKind::Unit
        | TyKind::Never
        | TyKind::Order
        | TyKind::Handle(_)
        | TyKind::Closures(_)
        | TyKind::Array(..)
        | TyKind::Slice(_)
        | TyKind::Str
        | TyKind::Tuple(.., _)
        | TyKind::Option(.., _)
        | TyKind::Result(..)
        | TyKind::Object(_)
        | TyKind::Enum { .. }
        | TyKind::Fn { .. }
        | TyKind::UserDefined { .. }
        | TyKind::Ref(..) => false,
        TyKind::Var(var) => match *var {},
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;
    use acvus_utils::LocalIdOps;

    use super::*;
    use crate::ir::Overflow;
    use crate::ir::cfg::Terminator;
    use crate::passes::optimize::tests::{Fixture, assert_whole, ret, wrapping_add};

    fn sub(left: ValueId, right: ValueId) -> Op {
        Op::BinOp { op: BinOp::Sub(Overflow::Wrap), left, right }
    }

    /// ```text
    /// entry(x, y, c): s = x + y; diamond c { then } else { else } join
    /// then:           t = x + y; jump join(t)
    /// else:           jump join(s)
    /// join(p):        return p
    /// ```
    #[test]
    fn a_pure_op_repeated_below_its_dominator_is_numbered_once() {
        let mut f = Fixture::new();
        let (int, bool) = (f.int, f.bool);
        let (nx, ny, nc) = (f.name("x"), f.name("y"), f.name("c"));
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(nx, int, acvus_ast::Span::ZERO);
        let y = builder.param(ny, int, acvus_ast::Span::ZERO);
        let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
        let s = builder.push(wrapping_add(x, y), int, Span::ZERO);
        let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
        let join = builder.new_block(&[int]);
        builder.terminate(Terminator::Diamond {
            cond: c,
            then_to: then_block,
            then_args: Vec::new(),
            else_to: else_block,
            else_args: Vec::new(),
            join,
        }, acvus_ast::Span::ZERO);
        builder.switch_to(then_block);
        let t = builder.push(wrapping_add(x, y), int, Span::ZERO);
        builder.terminate(Terminator::Jump { to: join, args: vec![t] }, acvus_ast::Span::ZERO);
        builder.switch_to(else_block);
        builder.terminate(Terminator::Jump { to: join, args: vec![s] }, acvus_ast::Span::ZERO);
        builder.switch_to(join);
        let p = builder.block_params(join)[0];
        builder.terminate(ret(p), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let changes = f.run::<Gvn>(id);

        let body = f.module.body(id);
        assert_eq!(body.use_count(t), 0, "t's uses took s: {}", f.snapshot(id));
        assert!(matches!(body.block(then_block).terminator(), Terminator::Jump { args, .. } if args[..] == [s]));
        assert!(changes.touched(crate::ir::cfg::Aspect::Instructions));
        assert_whole(&f.module, id);
    }

    /// `a = x + y; b = y + x; c = x - y; d = y - x`, all returned in a
    /// tuple: `b` is `a`; `d` is not `c`.
    #[test]
    fn commutative_operands_match_in_either_order() {
        let mut f = Fixture::new();
        let int = f.int;
        let tuple = f.quad;
        let (nx, ny) = (f.name("x"), f.name("y"));
        let mut builder = f.module.build_body(false, tuple);
        let x = builder.param(nx, int, acvus_ast::Span::ZERO);
        let y = builder.param(ny, int, acvus_ast::Span::ZERO);
        let a = builder.push(wrapping_add(x, y), int, Span::ZERO);
        let b = builder.push(wrapping_add(y, x), int, Span::ZERO);
        let c = builder.push(sub(x, y), int, Span::ZERO);
        let d = builder.push(sub(y, x), int, Span::ZERO);
        let all = builder.push(Op::MakeTuple { elements: vec![a, b, c, d] }, tuple, Span::ZERO);
        builder.terminate(ret(all), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();

        f.run::<Gvn>(id);

        let body = f.module.body(id);
        let made = f.ops(id, entry)[4];
        assert!(matches!(made, Op::MakeTuple { elements } if elements[..] == [a, a, c, d]), "{made:?}");
        assert_eq!((body.use_count(b), body.use_count(d)), (0, 1));
        assert_whole(&f.module, id);
    }

    /// `a = g(); b = g(); p = h(); q = h()`, `g` opaque and `h` pure: the
    /// opaque calls stay two, the pure ones become one.
    #[test]
    fn an_effectful_call_is_never_numbered_and_a_pure_one_is() {
        let mut f = Fixture::new();
        let int = f.int;
        let tuple = f.quad;
        let (opaque_call, pure_call) = (f.call(f.opaque), f.call(f.pure));
        let mut builder = f.module.build_body(true, tuple);
        let a = builder.push_effect(opaque_call.clone(), int, Span::ZERO);
        let b = builder.push_effect(opaque_call, int, Span::ZERO);
        let p = builder.push(pure_call.clone(), int, Span::ZERO);
        let q = builder.push(pure_call, int, Span::ZERO);
        let all = builder.push(Op::MakeTuple { elements: vec![a, b, p, q] }, tuple, Span::ZERO);
        builder.terminate(ret(all), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let entry = f.module.body(id).entry();

        f.run::<Gvn>(id);

        let made = f.ops(id, entry)[4];
        assert!(matches!(made, Op::MakeTuple { elements } if elements[..] == [a, b, p, p]), "{made:?}");
        assert_whole(&f.module, id);
    }

    /// An `anyorder` scope of two opaque calls: the Builder merges what each
    /// yields into what the scope gathered so far, so the second merge nests
    /// the first. Their flattened sets differ, and both stay.
    #[test]
    fn nested_merges_of_different_sets_stay() {
        let mut f = Fixture::new();
        let int = f.int;
        let tuple = f.pair;
        let opaque_call = f.call(f.opaque);
        let mut builder = f.module.build_body(true, tuple);
        builder.open_anyorder();
        let a = builder.push_effect(opaque_call.clone(), int, Span::ZERO);
        let b = builder.push_effect(opaque_call, int, Span::ZERO);
        builder.close_anyorder();
        let all = builder.push(Op::MakeTuple { elements: vec![a, b] }, tuple, Span::ZERO);
        builder.terminate(ret(all), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let before = f.snapshot(id);
        let merges = before.matches("Merge").count();
        assert_eq!(merges, 2, "{before}");

        f.run::<Gvn>(id);

        assert_eq!(f.snapshot(id), before);
        assert_whole(&f.module, id);
    }

    /// The token an effectful call consumes.
    fn consumed(f: &Fixture, body: BodyId, call: ValueId) -> ValueId {
        let inst = f
            .module
            .body(body)
            .blocks()
            .flat_map(|(_, block)| block.insts())
            .find(|inst| inst.result() == Some(call));
        let Some(order) = inst.and_then(|inst| inst.order()) else {
            panic!("{call:?} is an effectful call's result")
        };
        order.consumed().expect("an effectful call consumes a token")
    }

    /// `a = g(); t = token; m = merge(t, t); b = g()` with `b` consuming
    /// `m`: `m` is `t`.
    #[test]
    fn a_merge_of_one_token_becomes_that_token() {
        let mut f = Fixture::new();
        let (int, tuple, opaque_call) = (f.int, f.pair, f.call(f.opaque));
        let mut builder = f.module.build_body(true, tuple);
        let a = builder.push_effect(opaque_call.clone(), int, Span::ZERO);
        let t = builder.current_token();
        let m = builder.push_merge(vec![t, t], Span::ZERO);
        let b = builder.push_effect(opaque_call, int, Span::ZERO);
        let all = builder.push(Op::MakeTuple { elements: vec![a, b] }, tuple, Span::ZERO);
        builder.terminate(ret(all), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        assert_eq!(consumed(&f, id, b), m);

        f.run::<Gvn>(id);

        assert_eq!(consumed(&f, id, b), t, "{}", f.snapshot(id));
        assert_eq!(f.module.body(id).use_count(m), 0);
        assert_whole(&f.module, id);
    }

    /// ```text
    /// a = g(); t1 = token; b = g(); t2 = token
    /// m1 = merge(t1, t2); m2 = merge(m1, t1); m3 = merge(m2, t2, t2)
    /// c = g()   consuming m3
    /// ```
    /// `m2` nests `m1` over the same set, and `m3` repeats it: both are
    /// `m1`.
    #[test]
    fn nested_and_duplicate_merges_collapse() {
        let mut f = Fixture::new();
        let (int, triple, opaque_call) = (f.int, f.triple, f.call(f.opaque));
        let mut builder = f.module.build_body(true, triple);
        let a = builder.push_effect(opaque_call.clone(), int, Span::ZERO);
        let t1 = builder.current_token();
        let b = builder.push_effect(opaque_call.clone(), int, Span::ZERO);
        let t2 = builder.current_token();
        let m1 = builder.push_merge(vec![t1, t2], Span::ZERO);
        let m2 = builder.push_merge(vec![m1, t1], Span::ZERO);
        let m3 = builder.push_merge(vec![m2, t2, t2], Span::ZERO);
        let c = builder.push_effect(opaque_call, int, Span::ZERO);
        let all = builder.push(Op::MakeTuple { elements: vec![a, b, c] }, triple, Span::ZERO);
        builder.terminate(ret(all), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        assert_eq!(consumed(&f, id, c), m3);

        f.run::<Gvn>(id);

        let body = f.module.body(id);
        assert_eq!(consumed(&f, id, c), m1, "{}", f.snapshot(id));
        assert_eq!((body.use_count(m2), body.use_count(m3)), (0, 0));
        assert_whole(&f.module, id);
    }

    fn value(index: usize) -> ValueId {
        ValueId::try_from_raw(index).expect("a small index is a value id")
    }

    /// The merge key on values directly: a merge nesting another over the
    /// same tokens, and one repeating a token, key as the set they order
    /// after, and a merge of one token, directly or once flattened, is that
    /// token.
    #[test]
    fn merges_key_by_their_flattened_set_and_a_merge_of_one_token_is_it() {
        let (a, b, c) = (value(1), value(2), value(3));
        let (inner, outer, repeated, single, nested_single) = (value(10), value(11), value(12), value(13), value(14));
        let mut numbering = Numbering::default();
        let key = |found: Found| match found {
            Found::Key(key) => key,
            Found::Token(token) => panic!("a merge of more than one token keys, not {token:?}"),
        };
        let token = |found: Found| match found {
            Found::Token(token) => token,
            Found::Key(key) => panic!("a merge of one token is that token, not {key:?}"),
        };

        let inner_key = key(numbering.merge(inner, &[a, b]));
        let outer_key = key(numbering.merge(outer, &[inner, c]));
        let repeated_key = key(numbering.merge(repeated, &[c, a, b, a]));
        assert_eq!(inner_key, Key::Merge(vec![a, b]));
        assert_eq!(outer_key, Key::Merge(vec![a, b, c]), "the nested merge flattens");
        assert_eq!(repeated_key, outer_key, "a repeated token counts once, in any order");
        assert_eq!(token(numbering.merge(single, &[a, a])), a);
        assert_eq!(token(numbering.merge(nested_single, &[inner, inner])), inner);
    }
}
