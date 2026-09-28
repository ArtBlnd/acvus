//! The editor's facts over the analysis manager (passes-analyses.md): one
//! adapter serves every scope. `refresh` invalidates the manager for what the
//! last edit changed and recomputes what the adapter serves; every query
//! reads only facts kept since the last refresh, never the module.
//!
//! A `Position::End(block)` is the point after every instruction of the
//! block and before its terminator.

use std::collections::BTreeSet;
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};

use smallvec::SmallVec;

use crate::ir::cfg::edit::{DischargeFacts, DominanceFacts, Facts, InstRef, LoanFacts, OrderFacts, Position, Unobserved};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Definition, Module, Op, Site, Terminator};
use crate::ir::{PathSeg, RefTarget, ValueId, fn_effect};
use crate::passes::analysis::{AnalysisManager, Dominators, Escapes, LoopTree, Loans, Point, PostDominators, Storage};
use crate::ty::{Concrete, Ty, Types};

pub struct EditFacts<'a, 'd> {
    am: &'a mut AnalysisManager<'d>,
    dominators: Rc<Dominators>,
    post_dominators: Rc<PostDominators>,
    loop_tree: Rc<LoopTree>,
    loans: Rc<Loans>,
    escapes: Rc<Escapes>,
    defined: FxHashMap<ValueId, Defined>,
    /// The storages each instruction's operands and result name, and those
    /// it reads and writes, by block and index.
    named: FxHashMap<Site, BTreeSet<Storage>>,
    /// The instructions, by block and index, that are calls, or a call's
    /// start or taking, whose callee's declared effect commutes.
    commuting: FxHashSet<(BlockId, usize)>,
    /// The fact that makes each instruction unobservable, by block and
    /// index, where one is proven.
    unobserved: FxHashMap<(BlockId, usize), Unobserved>,
}

/// Where a value becomes defined.
#[derive(Debug, Clone, Copy)]
enum Defined {
    /// A parameter, a capture, the order parameter or a slot: before the
    /// entry runs.
    BeforeEntry,
    At(Point),
}

impl<'a, 'd> EditFacts<'a, 'd> {
    /// Facts for editing `body`, computed now. Cost: the analyses it serves,
    /// and O(values + instructions) for its own tables.
    pub fn new(am: &'a mut AnalysisManager<'d>, module: &Module, body: BodyId) -> Self {
        let of = module.body(body);
        let loans = am.get::<Loans>(module, body);
        EditFacts {
            dominators: am.get::<Dominators>(module, body),
            post_dominators: am.get::<PostDominators>(module, body),
            loop_tree: am.get::<LoopTree>(module, body),
            escapes: am.get::<Escapes>(module, body),
            defined: definitions(of),
            named: named_storages(of, &loans),
            commuting: commuting_calls(of, module.types()),
            unobserved: unobserved_insts(of, &loans),
            loans,
            am,
        }
    }

    fn defined(&self, value: ValueId) -> Defined {
        *self
            .defined
            .get(&value)
            .unwrap_or_else(|| panic!("{value:?} is not a value of the body the facts were refreshed for"))
    }
}

impl Facts for EditFacts<'_, '_> {
    /// Cost: the analyses `changes` made stale, and O(values +
    /// instructions) for the adapter's own tables.
    fn refresh(&mut self, module: &Module, body: BodyId, changes: &Changes) {
        self.am.invalidate(body, changes);
        self.dominators = self.am.get::<Dominators>(module, body);
        self.post_dominators = self.am.get::<PostDominators>(module, body);
        self.loop_tree = self.am.get::<LoopTree>(module, body);
        self.loans = self.am.get::<Loans>(module, body);
        self.escapes = self.am.get::<Escapes>(module, body);
        let of = module.body(body);
        self.defined = definitions(of);
        self.named = named_storages(of, &self.loans);
        self.commuting = commuting_calls(of, module.types());
        self.unobserved = unobserved_insts(of, &self.loans);
    }
}

impl DominanceFacts for EditFacts<'_, '_> {
    /// Cost: O(1).
    fn dominates(&self, value: ValueId, at: Position) -> bool {
        match self.defined(value) {
            Defined::BeforeEntry => true,
            Defined::At(defined) => self.dominators.point_dominates(defined, point(at)),
        }
    }

    /// Cost: O(1).
    fn position_dominates(&self, at: Position, site: Site) -> bool {
        let read = match site {
            Site::Inst(block, index) => Point::Before(block, index),
            Site::Term(block) => Point::End(block),
        };
        self.dominators.point_dominates(point(at), read)
    }

    /// Cost: O(blocks + edges), for the loops between the two.
    fn control_equivalent(&self, a: Position, b: Position) -> bool {
        let (a, b) = (point(a), point(b));
        let (dominators, post_dominators) = (&self.dominators, &self.post_dominators);
        let in_order = |earlier: Point, later: Point| {
            dominators.point_dominates(earlier, later)
                && post_dominators.point_post_dominates(later, earlier)
                && !self.loop_tree.loop_between(earlier.block(), later.block())
        };
        in_order(a, b) || in_order(b, a)
    }
}

impl LoanFacts for EditFacts<'_, '_> {
    /// Cost: O(storages the two touch · log).
    fn conflicts(&self, inst: InstRef, write: InstRef) -> bool {
        let written = self.loans.writes(site(write));
        let named = self
            .named
            .get(&site(inst))
            .unwrap_or_else(|| panic!("{inst:?} is an instruction of the body the facts were refreshed for"));
        written.iter().any(|storage| named.contains(storage))
    }

    /// Cost: O(1).
    fn escapes(&self, slot: ValueId) -> Option<Site> {
        self.escapes.escape(slot)
    }
}

impl OrderFacts for EditFacts<'_, '_> {
    /// Both are calls, or a call's start or taking (effects §6), whose
    /// callee's declared effect commutes (effects §1), read from the table
    /// the last refresh made from the callee types. Cost: O(1).
    fn may_reorder(&self, a: InstRef, b: InstRef) -> bool {
        let commutes = |inst: InstRef| self.commuting.contains(&(inst.block(), inst.index()));
        commutes(a) && commutes(b)
    }
}

impl DischargeFacts for EditFacts<'_, '_> {
    /// Read from the table the last refresh made: a store or a load never
    /// read, and a commit of a context never written. No analysis proves a
    /// check never fires yet, so none is `NeverFires`. Cost: O(1).
    fn discharged(&self, inst: InstRef) -> Option<Unobserved> {
        self.unobserved.get(&(inst.block(), inst.index())).copied()
    }
}

fn site(inst: InstRef) -> Site {
    Site::Inst(inst.block(), inst.index())
}

/// The storages each instruction of `body` names: through the loans of its
/// operands and result, a slot it names directly, and what it reads and
/// writes. Cost: O(operands · positions · loans).
fn named_storages(body: &Body, loans: &Loans) -> FxHashMap<Site, BTreeSet<Storage>> {
    let mut named = FxHashMap::default();
    for (block, record) in body.blocks() {
        for (index, inst) in record.insts().iter().enumerate() {
            let at = Site::Inst(block, index);
            let mut storages: BTreeSet<Storage> = loans.reads(at).iter().chain(loans.writes(at)).copied().collect();
            for value in inst.operands().into_iter().chain(inst.result()) {
                match body.definition(value) {
                    Definition::Slot(_) => {
                        storages.insert(Storage::Slot(value));
                    }
                    Definition::Param(_)
                    | Definition::Capture(_)
                    | Definition::OrderParam
                    | Definition::BlockParam(..)
                    | Definition::Inst(..) => {
                        storages.extend(loans.all_of(value).into_iter().map(|loan| loan.storage));
                    }
                }
            }
            named.insert(at, storages);
        }
    }
    named
}

fn point(at: Position) -> Point {
    match at {
        Position::Before(inst) => Point::Before(inst.block(), inst.index()),
        Position::End(block) => Point::End(block),
    }
}

/// Where each value of `body` becomes defined: a block parameter at its
/// block's start, an instruction's result and yielded token just after it.
/// Cost: O(values).
fn definitions(body: &Body) -> FxHashMap<ValueId, Defined> {
    body.values()
        .map(|value| {
            let defined = match body.definition(value) {
                Definition::Param(_) | Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => {
                    Defined::BeforeEntry
                }
                Definition::BlockParam(block, _) => Defined::At(Point::Before(block, 0)),
                Definition::Inst(block, index) => Defined::At(Point::Before(block, index + 1)),
            };
            (value, defined)
        })
        .collect()
}

/// The instructions of `body` a fact proves unobservable (D55, G1):
/// - a store whose storage no read reaches before a write replaces it or
///   the body ends, per `loans`;
/// - a load whose result has no use;
/// - a commit of the slot its value is taken whole from, where no write to
///   that slot stands on a path from the body's entry to it.
///
/// Cost: O(stores · storages each writes · the body's sites and edges), and
/// O(commits · the body's sites and edges).
fn unobserved_insts(body: &Body, loans: &Loans) -> FxHashMap<(BlockId, usize), Unobserved> {
    let preds = body.predecessors();
    let mut unobserved = FxHashMap::default();
    for (block, record) in body.blocks() {
        for (index, inst) in record.insts().iter().enumerate() {
            let proven = match inst.op() {
                Op::Assign { .. } => store_never_read(body, loans, block, index).then_some(Unobserved::NeverRead),
                Op::Fetch { .. } => {
                    let result = inst.result().expect("a load yields a value");
                    (body.use_count(result) == 0).then_some(Unobserved::NeverRead)
                }
                Op::Commit { value, .. } => {
                    commit_never_written(body, loans, &preds, block, index, *value).then_some(Unobserved::NeverWritten)
                }
                _ => None,
            };
            if let Some(fact) = proven {
                unobserved.insert((block, index), fact);
            }
        }
    }
    unobserved
}

/// The storage and path an `Assign` names directly, by a slot or a
/// by-value parameter's own storage; `None` for one through a reference.
fn direct_place(op: &Op) -> Option<(Storage, &[PathSeg])> {
    let Op::Assign { target, path, .. } = op else {
        return None;
    };
    match target {
        RefTarget::Var(slot) => Some((Storage::Slot(*slot), path)),
        RefTarget::Param(param) => Some((Storage::Param(*param), path)),
        RefTarget::Through(_) => None,
    }
}

/// Whether no read of any storage the store at `index` of `block` writes is
/// reached from just after it, before a write that replaces what it stored
/// or the body's end. A write replaces it where it assigns a place that
/// holds the stored one: a prefix of its path in the same storage, or the
/// whole storage where the store went through a reference. The body's end
/// ends its own slots and parameters' storage, and not an outside storage,
/// which the caller reads after the return; a run that dies owes nothing
/// (effects §5).
fn store_never_read(body: &Body, loans: &Loans, block: BlockId, index: usize) -> bool {
    let store = &body.block(block).insts()[index];
    let stored_path = direct_place(store.op()).map(|(_, path)| path);
    let replaces = |site_op: &Op, storage: Storage| {
        let Some((written, path)) = direct_place(site_op) else {
            return false;
        };
        let covers = match stored_path {
            Some(stored) => stored.starts_with(path),
            None => path.is_empty(),
        };
        written == storage && covers
    };
    loans.writes(Site::Inst(block, index)).iter().all(|storage| {
        let mut visited: FxHashSet<BlockId> = FxHashSet::default();
        let mut pending: Vec<(BlockId, usize)> = vec![(block, index + 1)];
        while let Some((at, from)) = pending.pop() {
            let record = body.block(at);
            let mut replaced = false;
            for (offset, inst) in record.insts().iter().enumerate().skip(from) {
                if loans.reads(Site::Inst(at, offset)).contains(storage) {
                    return false;
                }
                if replaces(inst.op(), *storage) {
                    replaced = true;
                    break;
                }
            }
            if replaced {
                continue;
            }
            if loans.reads(Site::Term(at)).contains(storage) {
                return false;
            }
            match record.terminator() {
                Terminator::Return { .. } if matches!(storage, Storage::Outside { .. }) => return false,
                terminator => {
                    for edge in terminator.edges() {
                        if visited.insert(edge.to) {
                            pending.push((edge.to, 0));
                        }
                    }
                }
            }
        }
        true
    })
}

/// Whether the commit at `index` of `block`, whose value is a whole `Take`
/// of a slot, has no write to that slot on any path from the body's entry
/// to it. A commit whose value is taken from no slot has no slot to read
/// this from, and is not proven.
fn commit_never_written(
    body: &Body,
    loans: &Loans,
    preds: &FxHashMap<BlockId, SmallVec<[BlockId; 2]>>,
    block: BlockId,
    index: usize,
    value: ValueId,
) -> bool {
    let Definition::Inst(taken_at, taken_index) = body.definition(value) else {
        return false;
    };
    let slot = match body.block(taken_at).insts()[taken_index].op() {
        Op::Take { target: RefTarget::Var(slot), path, .. } if path.is_empty() => *slot,
        _ => return false,
    };
    let storage = Storage::Slot(slot);
    let writes = |site: Site| loans.writes(site).contains(&storage);
    if (0..index).any(|earlier| writes(Site::Inst(block, earlier))) {
        return false;
    }
    let mut visited: FxHashSet<BlockId> = FxHashSet::default();
    let mut pending: Vec<BlockId> = preds[&block].to_vec();
    while let Some(at) = pending.pop() {
        if !visited.insert(at) {
            continue;
        }
        let sites = (0..body.block(at).insts().len()).map(|offset| Site::Inst(at, offset));
        if sites.chain([Site::Term(at)]).any(writes) {
            return false;
        }
        pending.extend(preds[&at].iter().copied());
    }
    true
}

/// The calls, starts and takings of `body` whose callee's declared effect
/// commutes (effects §1, §6). A taking's callee is its start's. Cost:
/// O(instructions).
fn commuting_calls(body: &Body, types: &Types<Concrete>) -> FxHashSet<(BlockId, usize)> {
    let commutes = |callee_ty: Ty| {
        let Some(effect) = fn_effect(types, callee_ty) else {
            panic!("a call's callee type is a function type")
        };
        effect.commutes()
    };
    let mut commuting = FxHashSet::default();
    for (block, record) in body.blocks() {
        for (index, inst) in record.insts().iter().enumerate() {
            let callee_ty = match inst.op() {
                Op::FunctionCall { callee_ty, .. } | Op::Spawn { callee_ty, .. } => *callee_ty,
                Op::TakeEffect { src } => started_callee(body, *src),
                _ => continue,
            };
            if commutes(callee_ty) {
                commuting.insert((block, index));
            }
        }
    }
    commuting
}

/// The callee type of the start whose handle is `handle`.
///
/// # Panics
/// `handle` is not a `Spawn`'s result: a taking takes a start's handle
/// (effects §6).
fn started_callee(body: &Body, handle: ValueId) -> Ty {
    let Definition::Inst(block, index) = body.definition(handle) else {
        panic!("a taking's handle {handle:?} is a start's result (effects §6)")
    };
    match body.block(block).insts()[index].op() {
        Op::Spawn { callee_ty, .. } => *callee_ty,
        other => panic!("a taking's handle is a start's result (effects §6), not {other:?}'s"),
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::{Literal, Span};
    use acvus_utils::{Interner, QualifiedRef};

    use super::*;
    use crate::decl::{Core, Declarations, LoopMarkers, RawDeclarations};
    use crate::ir::cfg::edit::{CannotDischarge, CannotMove, Discharge, Editor, Motion, NoFacts, Remove, Value};
    use crate::ir::cfg::{Inst, Terminator};
    use crate::ir::{BinOp, Callee, Overflow, RefTarget};
    use crate::ty::{
        Calls, Contexts, Cost, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Reissue, SplitCost,
        TyKind, TypeArg,
    };

    struct Fixture {
        module: Module,
        interner: Interner,
        int: Ty,
        bool: Ty,
        /// Opaque, and declared to commute with other calls.
        commuting: Ty,
        /// Opaque, commuting, and splittable.
        commuting_splittable: Ty,
        /// Opaque and not commuting.
        opaque: Ty,
        /// `&i64`.
        ref_int: Ty,
    }

    fn fixture() -> Fixture {
        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let bool = types.intern(TyKind::Bool);
        let mut call_ty = |effect: Effect| {
            types.intern(TyKind::Fn {
                params: Box::new([]),
                ret: int,
                effect: EffectTerm::Known(effect),
                flows: FlowTerm::Known(Flows::none()),
                calls: Calls::Many,
                holds: Holds::Nothing,
            })
        };
        let opaque_effect = |commutes: bool, split: SplitCost| {
            Effect::new(Reissue::Opaque, commutes, Contexts::new(), Contexts::new(), split)
        };
        let commuting = call_ty(opaque_effect(true, SplitCost::Unavailable));
        let commuting_splittable = call_ty(opaque_effect(true, SplitCost::WithCost(Cost::stated(1))));
        let opaque = call_ty(Effect::opaque());
        let ref_int = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
        Fixture {
            module: Module::new(types),
            interner: Interner::new(),
            int,
            bool,
            commuting,
            commuting_splittable,
            opaque,
            ref_int,
        }
    }

    fn declarations(interner: &Interner) -> Declarations {
        Declarations::verify(RawDeclarations {
            types: Types::new(),
            functions: Vec::new(),
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

    fn add(left: ValueId, right: ValueId) -> Op {
        Op::BinOp { op: BinOp::Add(Overflow::Trap), left, right }
    }

    fn call(f: &Fixture, callee_ty: Ty) -> Op {
        Op::FunctionCall { callee: Callee::Direct(QualifiedRef::root(f.interner.intern("f"))), callee_ty, args: Vec::new() }
    }

    /// Every instruction of `body`, in block order, as the editor names them.
    fn inst_refs(module: &mut Module, body: BodyId) -> Vec<InstRef> {
        let mut no_facts = NoFacts;
        let mut editor = Editor::<Remove>::new(module, body, &mut no_facts);
        let mut cursor = editor.cursor();
        std::iter::from_fn(|| cursor.next()).collect()
    }

    struct DiamondBody {
        id: BodyId,
        entry: BlockId,
        then_block: BlockId,
        else_block: BlockId,
        join: BlockId,
        c: ValueId,
        x: ValueId,
        t: ValueId,
        p: ValueId,
    }

    /// ```text
    /// entry(c): x = 0; diamond c { then } else { else } join
    /// then:     t = x + x; jump join(t)
    /// else:     jump join(x)
    /// join(p):  return p
    /// ```
    fn diamond_body(f: &mut Fixture) -> DiamondBody {
        let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool);
        let entry = builder.current_block();
        let x = builder.push(constant(0), int, Span::ZERO);
        let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
        let join = builder.new_block(&[int]);
        let p = builder.block_params(join)[0];
        builder.terminate(Terminator::Diamond {
            cond: c,
            then_to: then_block,
            then_args: Vec::new(),
            else_to: else_block,
            else_args: Vec::new(),
            join,
        });
        builder.switch_to(then_block);
        let t = builder.push(add(x, x), int, Span::ZERO);
        builder.terminate(Terminator::Jump { to: join, args: vec![t] });
        builder.switch_to(else_block);
        builder.terminate(Terminator::Jump { to: join, args: vec![x] });
        builder.switch_to(join);
        builder.terminate(Terminator::Return { value: p, order: None, span: Span::ZERO });
        let id = builder.seal().expect("the body has no open state");
        DiamondBody { id, entry, then_block, else_block, join, c, x, t, p }
    }

    #[test]
    fn dominates_reads_where_each_value_is_defined() {
        let mut f = fixture();
        let d = diamond_body(&mut f);
        let [x_at, t_at] = inst_refs(&mut f.module, d.id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let facts = EditFacts::new(&mut am, &f.module, d.id);

        assert!(facts.dominates(d.c, Position::Before(x_at)), "a parameter is defined before the entry runs");
        assert!(!facts.dominates(d.x, Position::Before(x_at)), "a result is not defined before its instruction");
        assert!(facts.dominates(d.x, Position::End(d.entry)));
        assert!(facts.dominates(d.x, Position::Before(t_at)));
        assert!(facts.dominates(d.x, Position::End(d.join)));
        assert!(facts.dominates(d.t, Position::End(d.then_block)));
        assert!(!facts.dominates(d.t, Position::End(d.join)), "the else arm reaches the join without t");
        assert!(!facts.dominates(d.t, Position::End(d.else_block)));
        assert!(facts.dominates(d.p, Position::End(d.join)), "a block parameter is defined at its block's start");
        assert!(!facts.dominates(d.p, Position::End(d.then_block)));
    }

    #[test]
    fn position_dominates_orders_a_block_and_includes_the_site_just_after() {
        let mut f = fixture();
        let d = diamond_body(&mut f);
        let [x_at, t_at] = inst_refs(&mut f.module, d.id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let facts = EditFacts::new(&mut am, &f.module, d.id);

        let t_site = Site::Inst(t_at.block(), t_at.index());
        assert!(facts.position_dominates(Position::Before(t_at), t_site), "standing just before the site");
        assert!(!facts.position_dominates(Position::End(d.then_block), t_site));
        assert!(facts.position_dominates(Position::End(d.then_block), Site::Term(d.then_block)));
        assert!(facts.position_dominates(Position::Before(x_at), Site::Term(d.join)));
        assert!(facts.position_dominates(Position::End(d.entry), t_site));
        assert!(!facts.position_dominates(Position::End(d.then_block), Site::Term(d.join)));
    }

    #[test]
    fn control_equivalence_needs_dominance_one_way_and_post_dominance_the_other() {
        let mut f = fixture();
        let d = diamond_body(&mut f);
        let [x_at, t_at] = inst_refs(&mut f.module, d.id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let facts = EditFacts::new(&mut am, &f.module, d.id);

        assert!(facts.control_equivalent(Position::Before(x_at), Position::End(d.join)));
        assert!(facts.control_equivalent(Position::End(d.join), Position::Before(x_at)), "symmetric");
        assert!(facts.control_equivalent(Position::Before(t_at), Position::End(d.then_block)));
        assert!(!facts.control_equivalent(Position::Before(x_at), Position::Before(t_at)), "t runs on one arm");
        assert!(!facts.control_equivalent(Position::End(d.then_block), Position::End(d.else_block)));
    }

    /// `a = commuting(); b = commuting(); c = opaque(); d = a + b; return d`,
    /// in an ordered body.
    #[test]
    fn may_reorder_holds_between_two_calls_that_commute() {
        let mut f = fixture();
        let (int, commuting, opaque) = (f.int, f.commuting, f.opaque);
        let (commuting_call, opaque_call) = (call(&f, commuting), call(&f, opaque));
        let mut builder = f.module.build_body(true, int);
        let a = builder.push_effect(commuting_call.clone(), int, Span::ZERO);
        let b = builder.push_effect(commuting_call, int, Span::ZERO);
        builder.push_effect(opaque_call, int, Span::ZERO);
        let d = builder.push(add(a, b), int, Span::ZERO);
        builder.terminate(Terminator::Return { value: d, order: None, span: Span::ZERO });
        let id = builder.seal().expect("the body has no open state");
        let [a_at, b_at, c_at, d_at] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has four instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let facts = EditFacts::new(&mut am, &f.module, id);

        assert!(facts.may_reorder(a_at, b_at));
        assert!(facts.may_reorder(b_at, a_at));
        assert!(!facts.may_reorder(a_at, c_at), "the opaque call does not commute");
        assert!(!facts.may_reorder(c_at, b_at));
        assert!(!facts.may_reorder(a_at, d_at), "an addition is not a call");
    }

    /// An instruction inserted through `Editor<Value>` is known to the next
    /// check the editor makes: replacing `t` by it passes only if the refresh
    /// after the insertion saw where it stands.
    #[test]
    fn refresh_after_an_insertion_through_the_value_editor_reflects_it() {
        let mut f = fixture();
        let int = f.int;
        let d = diamond_body(&mut f);
        let [_, t_at] = inst_refs(&mut f.module, d.id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, &f.module, d.id);

        let one = {
            let mut editor = Editor::<Value>::new(&mut f.module, d.id, &mut facts);
            let one = editor.insert(Position::Before(t_at), constant(1), int, Span::ZERO);
            editor.replace_uses(d.t, one).expect("the inserted value dominates every use of t");
            editor.finish();
            one
        };
        assert!(facts.dominates(one, Position::End(d.then_block)));
        assert!(!facts.dominates(one, Position::End(d.join)));
        let [_, one_at, t_now] = inst_refs(&mut f.module, d.id)[..] else {
            panic!("the body has three instructions")
        };
        assert_eq!((one_at.block(), one_at.index()), (d.then_block, 0));
        assert!(!facts.dominates(d.t, Position::Before(t_now)), "t moved down by one");
        assert!(facts.dominates(one, Position::Before(t_now)));
    }

    /// A split through `Editor<Effect>`: the start and the taking of a
    /// commuting call are known to commute after the refresh, at the places
    /// the split gave them.
    #[test]
    fn refresh_after_a_split_through_the_effect_editor_reflects_it() {
        let mut f = fixture();
        let (int, splittable, commuting) = (f.int, f.commuting_splittable, f.commuting);
        let (split_call, other_call) = (call(&f, splittable), call(&f, commuting));
        let mut builder = f.module.build_body(true, int);
        let a = builder.push_effect(split_call, int, Span::ZERO);
        builder.push_effect(other_call, int, Span::ZERO);
        builder.terminate(Terminator::Return { value: a, order: None, span: Span::ZERO });
        let id = builder.seal().expect("the body has no open state");
        let [a_at, _] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, &f.module, id);

        {
            let mut editor = Editor::<crate::ir::cfg::edit::Effect>::new(&mut f.module, id, &mut facts);
            editor.split_call(a_at).expect("the callee lets the call be split");
            editor.finish();
        }
        let refs = inst_refs(&mut f.module, id);
        let insts: Vec<&Inst> = f.module.body(id).blocks().flat_map(|(_, block)| block.insts()).collect();
        let start = refs.iter().zip(&insts).find(|(_, inst)| matches!(inst.op(), Op::Spawn { .. }));
        let take = refs.iter().zip(&insts).find(|(_, inst)| matches!(inst.op(), Op::TakeEffect { .. }));
        let other = refs.iter().zip(&insts).find(|(_, inst)| matches!(inst.op(), Op::FunctionCall { .. }));
        let (Some((start, _)), Some((take, _)), Some((other, _))) = (start, take, other) else {
            panic!("the split left a start, a taking and the other call")
        };
        assert!(facts.may_reorder(*start, *other));
        assert!(facts.may_reorder(*take, *other));
        assert!(facts.may_reorder(*start, *take));
    }

    fn assign(slot: ValueId, value: ValueId) -> Op {
        Op::Assign { target: RefTarget::Var(slot), path: Vec::new(), value, restores: false }
    }

    fn take(slot: ValueId) -> Op {
        Op::Take { target: RefTarget::Var(slot), path: Vec::new(), taken_out: false }
    }

    /// ```text
    /// entry(c): x = 0; jump header
    /// header:   while c { body } else exit
    /// body:     jump header
    /// exit:     y = x + x; return y
    /// ```
    /// The entry dominates the exit and the exit post-dominates the entry,
    /// but the loop between them may never end.
    #[test]
    fn control_equivalence_is_refused_across_a_loop() {
        let mut f = fixture();
        let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool);
        let entry = builder.current_block();
        let x = builder.push(constant(0), int, Span::ZERO);
        let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
        builder.terminate(Terminator::Jump { to: header, args: Vec::new() });
        builder.switch_to(header);
        builder.terminate(Terminator::While {
            cond: c,
            stages: crate::ir::cfg::Stages::lowered(loop_body),
            exit,
            exit_args: Vec::new(),
        });
        builder.switch_to(loop_body);
        builder.terminate(Terminator::Jump { to: header, args: Vec::new() });
        builder.switch_to(exit);
        let y = builder.push(add(x, x), int, Span::ZERO);
        builder.terminate(Terminator::Return { value: y, order: None, span: Span::ZERO });
        let id = builder.seal().expect("the body has no open state");
        let [x_at, y_at] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let facts = EditFacts::new(&mut am, &f.module, id);

        assert!(facts.dominates(x, Position::Before(y_at)));
        assert!(!facts.control_equivalent(Position::End(entry), Position::Before(y_at)), "a loop lies between");
        assert!(!facts.control_equivalent(Position::Before(y_at), Position::Before(x_at)));
        assert!(!facts.control_equivalent(Position::End(header), Position::End(loop_body)));
        assert!(facts.control_equivalent(Position::Before(x_at), Position::End(entry)), "one block, no loop");
        assert!(facts.control_equivalent(Position::Before(y_at), Position::End(exit)));
    }

    struct StorageBody {
        id: BodyId,
        x: ValueId,
        y: ValueId,
        /// The load of `y`.
        u: ValueId,
    }

    /// ```text
    /// x = 1; y = 2; r = &x; t = *r; u = y; return t
    /// ```
    /// with `x`, `y` slots and the stores and loads whole.
    fn storage_body(f: &mut Fixture) -> StorageBody {
        let int = f.int;
        let ref_int = f.ref_int;
        let mut builder = f.module.build_body(false, int);
        let x = builder.slot(int, Span::ZERO);
        let y = builder.slot(int, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(x, one), Span::ZERO);
        builder.push_unit(assign(y, one), Span::ZERO);
        let r = builder.push(Op::Ref { target: RefTarget::Var(x), path: Vec::new(), mutability: Mutability::Shared }, ref_int, Span::ZERO);
        let t = builder.push(Op::Take { target: RefTarget::Through(r), path: Vec::new(), taken_out: false }, int, Span::ZERO);
        let u = builder.push(take(y), int, Span::ZERO);
        builder.terminate(Terminator::Return { value: t, order: None, span: Span::ZERO });
        let id = builder.seal().expect("the body has no open state");
        StorageBody { id, x, y, u }
    }

    #[test]
    fn conflicts_intersects_what_a_write_writes_with_what_an_instruction_names() {
        let mut f = fixture();
        let s = storage_body(&mut f);
        let [_, assign_x, assign_y, make_ref, read_through, take_y] = inst_refs(&mut f.module, s.id)[..] else {
            panic!("the body has six instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let facts = EditFacts::new(&mut am, &f.module, s.id);

        assert!(facts.conflicts(read_through, assign_x), "the read through r reads x");
        assert!(facts.conflicts(make_ref, assign_x), "r names x");
        assert!(!facts.conflicts(read_through, assign_y));
        assert!(facts.conflicts(take_y, assign_y));
        assert!(!facts.conflicts(take_y, assign_x));
        assert!(!facts.conflicts(assign_x, read_through), "a read writes nothing");
    }

    #[test]
    fn escapes_names_the_first_use_a_promotion_cannot_rewrite() {
        let mut f = fixture();
        let s = storage_body(&mut f);
        let [.., make_ref, _, _] = inst_refs(&mut f.module, s.id)[..] else {
            panic!("the body has six instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let facts = EditFacts::new(&mut am, &f.module, s.id);

        assert_eq!(facts.escapes(s.x), Some(Site::Inst(make_ref.block(), make_ref.index())));
        assert_eq!(facts.escapes(s.y), None);
    }

    /// A move through `Editor<Motion>` above the store to `y` is refused,
    /// since the load of `y` names what it writes; a move above the reads of
    /// `x` is made, and the refresh after it knows the load's new place.
    #[test]
    fn refresh_after_a_move_through_the_motion_editor_reflects_it() {
        let mut f = fixture();
        let s = storage_body(&mut f);
        let [_, _, assign_y, make_ref, _, take_y] = inst_refs(&mut f.module, s.id)[..] else {
            panic!("the body has six instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, &f.module, s.id);

        {
            let mut editor = Editor::<Motion>::new(&mut f.module, s.id, &mut facts);
            assert_eq!(
                editor.move_to(take_y, Position::Before(assign_y)),
                Err(CannotMove::CrossesConflictingWrite(assign_y))
            );
            editor.move_to(take_y, Position::Before(make_ref)).expect("the load crosses only reads of x");
            editor.finish();
        }
        let [_, _, assign_y, moved, make_ref, _] = inst_refs(&mut f.module, s.id)[..] else {
            panic!("the body has six instructions")
        };
        assert!(matches!(f.module.body(s.id).block(moved.block()).insts()[moved.index()].op(), Op::Take { .. }));
        assert!(facts.dominates(s.u, Position::Before(make_ref)), "the load now stands before the Ref");
        assert!(!facts.dominates(s.u, Position::Before(moved)));
        assert!(facts.conflicts(moved, assign_y), "the load's storage is known at its new place");
    }

    fn commit(context: QualifiedRef, value: ValueId) -> Op {
        Op::Commit { context, value, wrote: true }
    }

    /// ```text
    /// one = 1; s = one; s = one; t = s; check(one + one); return t
    /// ```
    /// with `s` a slot: the first store is replaced before any read, and the
    /// second is read.
    #[test]
    fn a_store_replaced_before_any_read_is_never_read() {
        let mut f = fixture();
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let s = builder.slot(int, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        let t = builder.push(take(s), int, Span::ZERO);
        builder.push_unit(Op::Check { op: crate::ir::Checked::Add, left: one, right: one }, Span::ZERO);
        builder.terminate(Terminator::Return { value: t, order: None, span: Span::ZERO });
        let id = builder.seal().expect("the body has no open state");
        let [_, first, second, _, check] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has five instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let facts = EditFacts::new(&mut am, &f.module, id);

        assert_eq!(facts.discharged(first), Some(Unobserved::NeverRead));
        assert_eq!(facts.discharged(second), None, "the load of s reads it");
        assert_eq!(facts.discharged(check), None, "no analysis proves a check never fires yet");
    }

    /// `s = 1; if c { t = s } else {}; return 0`: the store is read on one
    /// arm, and a store no path reads before the end is never read.
    #[test]
    fn a_store_read_on_one_path_is_read_and_one_left_at_the_end_is_not() {
        let mut f = fixture();
        let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool);
        let (s, unread) = (builder.slot(int, Span::ZERO), builder.slot(int, Span::ZERO));
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        builder.push_unit(assign(unread, one), Span::ZERO);
        let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
        builder.terminate(Terminator::Diamond {
            cond: c,
            then_to: then_block,
            then_args: Vec::new(),
            else_to: else_block,
            else_args: Vec::new(),
            join,
        });
        builder.switch_to(then_block);
        builder.push(take(s), int, Span::ZERO);
        builder.terminate(Terminator::Jump { to: join, args: Vec::new() });
        builder.switch_to(else_block);
        builder.terminate(Terminator::Jump { to: join, args: Vec::new() });
        builder.switch_to(join);
        builder.terminate(Terminator::Return { value: one, order: None, span: Span::ZERO });
        let id = builder.seal().expect("the body has no open state");
        let [_, read_store, unread_store, _] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has four instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let facts = EditFacts::new(&mut am, &f.module, id);

        assert_eq!(facts.discharged(read_store), None);
        assert_eq!(facts.discharged(unread_store), Some(Unobserved::NeverRead));
    }

    /// ```text
    /// v = fetch page; w = fetch page; t = c; commit page = t;
    /// u = s; commit page = u; commit page = w; return w
    /// ```
    /// with `c` and `s` slots, `s` written first.
    #[test]
    fn a_load_nothing_reads_and_a_commit_of_a_slot_never_written_are_unobserved() {
        let mut f = fixture();
        let int = f.int;
        let page = QualifiedRef::root(f.interner.intern("page"));
        let mut builder = f.module.build_body(false, int);
        let (c, s) = (builder.slot(int, Span::ZERO), builder.slot(int, Span::ZERO));
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        builder.push(Op::Fetch { context: page }, int, Span::ZERO);
        let w = builder.push(Op::Fetch { context: page }, int, Span::ZERO);
        let t = builder.push(take(c), int, Span::ZERO);
        builder.push_unit(commit(page, t), Span::ZERO);
        let u = builder.push(take(s), int, Span::ZERO);
        builder.push_unit(commit(page, u), Span::ZERO);
        builder.push_unit(commit(page, w), Span::ZERO);
        builder.terminate(Terminator::Return { value: w, order: None, span: Span::ZERO });
        let id = builder.seal().expect("the body has no open state");
        let [_, _, unread, read, _, never_written, _, written, from_no_slot] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has nine instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let facts = EditFacts::new(&mut am, &f.module, id);

        assert_eq!(facts.discharged(unread), Some(Unobserved::NeverRead));
        assert_eq!(facts.discharged(read), None);
        assert_eq!(facts.discharged(never_written), Some(Unobserved::NeverWritten));
        assert_eq!(facts.discharged(written), None, "s is written before its commit");
        assert_eq!(facts.discharged(from_no_slot), None, "its value is taken from no slot");
    }

    /// Through `Editor<Discharge>` over `EditFacts`: a dead store, an unread
    /// load and an unwritten commit go, each on the fact the refresh after
    /// the one before still proves, and the facts refuse what is observed.
    #[test]
    fn discharge_through_the_editor_removes_what_the_facts_prove_unobserved() {
        let mut f = fixture();
        let int = f.int;
        let page = QualifiedRef::root(f.interner.intern("page"));
        let mut builder = f.module.build_body(false, int);
        let (c, s) = (builder.slot(int, Span::ZERO), builder.slot(int, Span::ZERO));
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        builder.push(Op::Fetch { context: page }, int, Span::ZERO);
        let t = builder.push(take(c), int, Span::ZERO);
        builder.push_unit(commit(page, t), Span::ZERO);
        let u = builder.push(take(s), int, Span::ZERO);
        builder.terminate(Terminator::Return { value: u, order: None, span: Span::ZERO });
        let id = builder.seal().expect("the body has no open state");
        let [_, dead, live, load, _, unwritten, _] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has seven instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, &f.module, id);

        {
            let mut editor = Editor::<Discharge>::new(&mut f.module, id, &mut facts);
            assert_eq!(editor.discharge(live), Err(CannotDischarge::NotProven));
            assert_eq!(editor.discharge(unwritten), Ok(()));
            assert_eq!(editor.discharge(load), Ok(()));
            assert_eq!(editor.discharge(dead), Ok(()));
            editor.finish();
        }
        let ops: Vec<&Op> = f.module.body(id).blocks().flat_map(|(_, block)| block.insts()).map(Inst::op).collect();
        assert_eq!(ops.len(), 4);
        assert!(!ops.iter().any(|op| matches!(op, Op::Fetch { .. } | Op::Commit { .. })));
        assert_eq!(ops.iter().filter(|op| matches!(op, Op::Assign { .. })).count(), 1);
    }
}
