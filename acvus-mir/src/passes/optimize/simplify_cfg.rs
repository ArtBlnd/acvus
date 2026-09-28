//! CFG simplification, scope `Cfg`. Any form in, the same form out.
//!
//! Algorithm: rounds of three edits until a round makes none. First, a block
//! that holds no instruction and only jumps on, passing its parameters
//! through, is bypassed: every edge into it goes to its target, with the
//! same arguments. Then a block joins its sole predecessor, whose sole
//! successor it is (`merge_into_predecessor`). Last, `prune` removes the
//! blocks no edge reaches. Legality is the editor's: a bypass is refused for
//! the entry and for a block whose parameters are read beyond its jump, and
//! a merge for a `Diamond`'s or a `Switch`'s join and a loop's stage or exit
//! (D45, D57). A bypass of such a label moves the structure with the edges
//! (core rule 6), so the `Diamond`, `Switch` or loop still holds it.
//!
//! Cost: O(blocks) edits per round, each at the editor's cost. A round that
//! changes the body removes at least one block, so there are at most as many
//! rounds as the body has blocks.
//!
//! Meaning: only edges and empty blocks change. A bypassed block does nothing
//! but pass its parameters on, so an edge into it and an edge into its target
//! with the same arguments run the same instructions; a merge runs the same
//! instructions in the same order; a pruned block runs on no run. Types,
//! extern-defined identity included, are untouched (compilation §3).

use crate::ir::cfg::edit::{CannotMerge, CannotRetarget, Cfg, Editor, NoFacts};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Module, Terminator};
use crate::passes::analysis::AnalysisManager;
use crate::passes::optimize::BodyTransform;

pub struct SimplifyCfg;

impl BodyTransform for SimplifyCfg {
    const NAME: &'static str = "simplify_cfg";

    fn run(module: &mut Module, body: BodyId, _am: &mut AnalysisManager<'_>) -> Changes {
        let mut no_facts = NoFacts;
        let mut editor = Editor::<Cfg>::new(module, body, &mut no_facts);
        loop {
            let before = editor.body().blocks().count();
            let bypassed = bypass_empty_jumps(&mut editor);
            let merged = merge_straight_lines(&mut editor);
            editor.prune();
            let after = editor.body().blocks().count();
            if !bypassed && !merged && after == before {
                break;
            }
            assert!(
                after < before,
                "a round of simplify_cfg that bypassed or merged a block removes one, so the rounds are bounded by the blocks; {before} blocks became {after}"
            );
        }
        editor.finish()
    }
}

/// A block that holds no instruction and only jumps on, and where to.
struct Bypass {
    block: BlockId,
    to: BlockId,
}

/// Tries a bypass of every block `empty_jump` finds, each against the body
/// as the bypasses before it left it. A retarget keeps every block's id, so
/// one walk over the ids visits each block once. Returns whether it
/// bypassed any.
fn bypass_empty_jumps(editor: &mut Editor<'_, Cfg>) -> bool {
    let mut bypassed = false;
    for block in block_ids(editor.body()) {
        let Some(Bypass { block, to }) = empty_jump(editor.body(), block) else {
            continue;
        };
        match editor.retarget(block, to) {
            Ok(()) => bypassed = true,
            Err(CannotRetarget::Entry | CannotRetarget::ParamsUsedBeyond(_) | CannotRetarget::ParamsDisagree) => {}
        }
    }
    bypassed
}

/// `block` and the block it jumps to, where it holds no instruction and
/// jumps elsewhere with its own parameters, in order, as the arguments.
fn empty_jump(body: &Body, block: BlockId) -> Option<Bypass> {
    let record = body.block(block);
    let Terminator::Jump { to, args } = record.terminator() else {
        return None;
    };
    let empty_jump = record.insts().is_empty() && args.as_slice() == record.params() && *to != block;
    empty_jump.then_some(Bypass { block, to: *to })
}

/// Merges every block that can join its predecessor. The walk runs from the
/// last block down: a merge removes the merged block and renumbers only the
/// blocks after it, which the walk has passed. Returns whether it merged
/// any.
fn merge_straight_lines(editor: &mut Editor<'_, Cfg>) -> bool {
    let mut merged = false;
    for block in block_ids(editor.body()).into_iter().rev() {
        match editor.merge_into_predecessor(block) {
            Ok(()) => merged = true,
            Err(CannotMerge::NotStraight | CannotMerge::Structural) => {}
        }
    }
    merged
}

fn block_ids(body: &Body) -> Vec<BlockId> {
    body.blocks().map(|(block, _)| block).collect()
}

#[cfg(test)]
mod tests {
    use acvus_ast::{Literal, Span};
    use acvus_utils::Interner;

    use super::*;
    use crate::decl::{Core, Declarations, LoopMarkers, RawDeclarations};
    use crate::ir::cfg::{Aspect, Inst, Op};
    use crate::ir::{BinOp, Overflow, ValueId};
    use crate::ty::{Concrete, IntTy, Ty, TyKind, Types};

    struct Fixture {
        module: Module,
        interner: Interner,
        int: Ty,
        bool: Ty,
    }

    fn fixture() -> Fixture {
        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let bool = types.intern(TyKind::Bool);
        Fixture { module: Module::new(types), interner: Interner::new(), int, bool }
    }

    fn declarations(interner: &Interner) -> Declarations {
        Declarations::verify(RawDeclarations {
            types: Types::new(),
            functions: Vec::new(),
            lang_impls: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            inputs: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(interner),
        })
        .expect("empty declarations verify")
    }

    fn constant(n: i128) -> Op {
        Op::Const { value: Literal::Int(n) }
    }

    fn jump(to: BlockId, args: Vec<ValueId>) -> Terminator {
        Terminator::Jump { to, args }
    }

    fn ret(value: ValueId) -> Terminator {
        Terminator::Return { value, order: None, span: Span::ZERO }
    }

    fn run(f: &mut Fixture, body: BodyId) -> Changes {
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        SimplifyCfg::run(&mut f.module, body, &mut am)
    }

    fn ops(body: &Body) -> Vec<&Op> {
        body.blocks().flat_map(|(_, block)| block.insts()).map(Inst::op).collect()
    }

    /// ```text
    /// entry:  x = 1; jump a(x)
    /// a(p):   jump b(p)
    /// b(q):   jump c(q)
    /// c(r):   y = r + r; return y
    /// ```
    /// The two empty jumps are bypassed and `c` joins the entry: one block.
    #[test]
    fn a_chain_of_empty_jumps_collapses() {
        let mut f = fixture();
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let x = builder.push(constant(1), int, Span::ZERO);
        let [a, b, c] = [(); 3].map(|()| builder.new_block(&[int]));
        builder.terminate(jump(a, vec![x]), acvus_ast::Span::ZERO);
        builder.switch_to(a);
        let p = builder.block_params(a)[0];
        builder.terminate(jump(b, vec![p]), acvus_ast::Span::ZERO);
        builder.switch_to(b);
        let q = builder.block_params(b)[0];
        builder.terminate(jump(c, vec![q]), acvus_ast::Span::ZERO);
        builder.switch_to(c);
        let r = builder.block_params(c)[0];
        let y = builder.push(Op::BinOp { op: BinOp::Add(Overflow::Trap), left: r, right: r }, int, Span::ZERO);
        builder.terminate(ret(y), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let changes = run(&mut f, id);

        let body = f.module.body(id);
        assert_eq!(body.blocks().count(), 1);
        let entry = body.block(body.entry());
        assert!(entry.params().is_empty());
        assert!(matches!(entry.terminator(), Terminator::Return { value, .. } if *value == y));
        let [Op::Const { .. }, Op::BinOp { left, right, .. }] = ops(body)[..] else {
            panic!("the constant and the addition stay, in order")
        };
        assert_eq!((*left, *right), (x, x), "the addition reads what the entry passed down the chain");
        assert!(changes.touched(Aspect::CfgShape));
    }

    /// ```text
    /// entry(c): diamond c { then } else { else } join
    /// then:     t = 1; jump join(t)
    /// else:     jump join2
    /// join2:    e = 2; jump join(e)
    /// join(p):  jump tail(p)
    /// tail(q):  return q
    /// ```
    /// `else` is bypassed, so `join2` is the else arm. The join, empty and
    /// passing `p` on, is bypassed too, and the `Diamond`'s join moves with
    /// its edges to `tail`: the structure is kept, never merged away.
    #[test]
    fn a_diamond_join_survives() {
        let mut f = fixture();
        let (name, int, bool) = (f.interner.intern("c"), f.int, f.bool);
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(name, bool);
        let entry = builder.current_block();
        let [then_block, else_block, join2] = [(); 3].map(|()| builder.new_block(&[]));
        let [join, tail] = [(); 2].map(|()| builder.new_block(&[int]));
        builder.terminate(Terminator::Diamond {
            cond: c,
            then_to: then_block,
            then_args: Vec::new(),
            else_to: else_block,
            else_args: Vec::new(),
            join,
        }, acvus_ast::Span::ZERO);
        builder.switch_to(then_block);
        let t = builder.push(constant(1), int, Span::ZERO);
        builder.terminate(jump(join, vec![t]), acvus_ast::Span::ZERO);
        builder.switch_to(else_block);
        builder.terminate(jump(join2, Vec::new()), acvus_ast::Span::ZERO);
        builder.switch_to(join2);
        let e = builder.push(constant(2), int, Span::ZERO);
        builder.terminate(jump(join, vec![e]), acvus_ast::Span::ZERO);
        builder.switch_to(join);
        let p = builder.block_params(join)[0];
        builder.terminate(jump(tail, vec![p]), acvus_ast::Span::ZERO);
        builder.switch_to(tail);
        let q = builder.block_params(tail)[0];
        builder.terminate(ret(q), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        run(&mut f, id);

        let body = f.module.body(id);
        assert_eq!(body.blocks().count(), 4, "entry, then, join2 as the else arm, and tail as the join");
        let Terminator::Diamond { then_to, else_to, join, .. } = body.block(entry).terminator() else {
            panic!("the entry still ends in its Diamond")
        };
        let join_block = body.block(*join);
        assert_eq!(join_block.params(), &[q], "tail is the join now, with its own parameter");
        assert!(matches!(join_block.terminator(), Terminator::Return { value, .. } if *value == q));
        for arm in [*then_to, *else_to] {
            let Terminator::Jump { to, .. } = body.block(arm).terminator() else {
                panic!("each arm jumps to the join")
            };
            assert_eq!(to, join);
        }
        assert_eq!(body.block(*else_to).insts().len(), 1, "join2, with its constant, is the else arm");
    }

    /// ```text
    /// entry: x = 1; return x
    /// dead:  y = 2; jump dead2
    /// dead2: return y
    /// ```
    #[test]
    fn unreachable_blocks_go() {
        let mut f = fixture();
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let x = builder.push(constant(1), int, Span::ZERO);
        let [dead, dead2] = [(); 2].map(|()| builder.new_block(&[]));
        builder.terminate(ret(x), acvus_ast::Span::ZERO);
        builder.switch_to(dead);
        let y = builder.push(constant(2), int, Span::ZERO);
        builder.terminate(jump(dead2, Vec::new()), acvus_ast::Span::ZERO);
        builder.switch_to(dead2);
        builder.terminate(ret(y), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        run(&mut f, id);

        let body = f.module.body(id);
        assert_eq!(body.blocks().count(), 1);
        assert_eq!(ops(body).len(), 1);
    }

    /// A body with nothing to simplify comes back unchanged.
    #[test]
    fn a_single_block_is_left_alone() {
        let mut f = fixture();
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let x = builder.push(constant(1), int, Span::ZERO);
        builder.terminate(ret(x), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let changes = run(&mut f, id);

        assert_eq!(f.module.body(id).blocks().count(), 1);
        assert!(!changes.touched(Aspect::CfgShape));
    }
}
