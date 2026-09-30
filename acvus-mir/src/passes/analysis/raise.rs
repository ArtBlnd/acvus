//! What a run of an instruction, a loop or a body may do besides giving its
//! value: trap, or not finish. Whether a run traps and whether it ends are
//! part of its meaning (compilation §1, effects §4), so a pass that removes,
//! moves or runs ahead an instruction asks here first.
//!
//! What traps is the language's (operators): integer division or remainder
//! by zero, an index outside its container, and `unwrap` of `None` or
//! `Err`. Nothing else traps; integer overflow is undefined, not a trap
//! (undefined behaviour §2). An operation that may trap by its kind does not
//! where the value analyses prove its operand clear of the case: a divisor
//! whose interval excludes zero, an index the bounds proof places inside its
//! slice, an `unwrap` whose variant the match facts know.
//!
//! What may not finish: a `While` the scalar evolution names no trip count
//! for, and a call whose ending is not known to be one. A `For` always
//! finishes. A call's ending composes (extern fn §4): an extern's declared
//! `Ends` states its own code, and the work the call runs ends and traps as
//! that work does, a closure argument by the bodies its function type's
//! values name, and work a value it is given holds by nothing known, so as
//! may-trap and may-not-finish. A call of a body of the module ends as that
//! body does, decided callees first over the call graph; a callee in the
//! caller's own strongly connected component may trap and may not finish.
//!
//! The analysis is a query the manager keeps per body; it writes nothing
//! into the IR. It asks the manager for each callee's endings outside its own
//! component, so the manager never meets a body it is computing.
//!
//! Cost: O(instructions + loops) per body over the facts it reads (the
//! value analyses, the match facts, the scalar evolution and the callees'
//! endings), plus the call graph's O(bodies + instructions) once per module.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::decl::{Declarations, Ends};
use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Definition, Inst, InstId, Module, Op, Terminator};
use crate::ir::{Callee, Chosen, IndexBound, Intrinsic, ValueId};
use crate::passes::analysis::{
    Analysis, AnalysisManager, AffineValues, CallGraph, Dominators, Evolutions, IntervalValues, LoopId, LoopTree, MatchFacts, ScalarEvolution,
    SparseFacts, ValueBounds,
};
use crate::passes::analysis::match_facts::MatchDomain;
use crate::passes::domain::{Bound, Interval, Tags};
use crate::ty::{CastTy, Concrete, Ty, TyKind, Types};

/// How a run of a body ends, as a caller sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ending {
    /// Some run may end in a trap.
    pub may_trap: bool,
    /// Every run that does not trap ends.
    pub finishes: bool,
}

impl Ending {
    const CLEAN: Ending = Ending { may_trap: false, finishes: true };
    const UNKNOWN: Ending = Ending { may_trap: true, finishes: false };

    fn traps_unless(cleared: bool) -> Ending {
        Ending { may_trap: !cleared, finishes: true }
    }

    /// Running both: either may trap, and both must end.
    fn and(self, other: Ending) -> Ending {
        Ending { may_trap: self.may_trap || other.may_trap, finishes: self.finishes && other.finishes }
    }

    /// The call's own code, as its declaration states it (extern fn §4).
    fn declared(ends: Ends) -> Ending {
        match ends {
            Ends::Unstated => Ending::UNKNOWN,
            Ends::Returns => Ending { may_trap: true, finishes: true },
            Ends::Total => Ending::CLEAN,
        }
    }
}

/// The endings of one body's instructions and loops, and of the body.
pub struct Endings {
    of_inst: FxHashMap<InstId, Ending>,
    of_loop: FxHashMap<LoopId, bool>,
    body: Ending,
}

impl Endings {
    /// Whether a run of `inst` may trap.
    pub fn may_trap(&self, inst: InstId) -> bool {
        self.of(inst).may_trap
    }

    /// Whether every run of `inst` that does not trap ends.
    pub fn finishes(&self, inst: InstId) -> bool {
        self.of(inst).finishes
    }

    /// Whether every run of `over` that does not trap ends: a `For` always,
    /// a `While` where the scalar evolution names its trip count and every
    /// instruction of the loop finishes.
    pub fn loop_finishes(&self, over: LoopId) -> bool {
        *self.of_loop.get(&over).unwrap_or_else(|| panic!("{over:?} is no loop of the body these endings are of"))
    }

    /// The body's own ending, which its callers read.
    pub fn body(&self) -> Ending {
        self.body
    }

    fn of(&self, inst: InstId) -> Ending {
        *self.of_inst.get(&inst).unwrap_or_else(|| panic!("{inst:?} is no instruction of the body these endings are of"))
    }
}

/// The analysis the manager keeps.
pub struct Raise;

impl Analysis for Raise {
    type Facts = Endings;
    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Endings {
        let dominators = am.get::<Dominators>(module, body);
        let loops = am.get::<LoopTree>(module, body);
        let evolutions = am.get::<ScalarEvolution>(module, body);
        let intervals = am.get::<IntervalValues>(module, body);
        let affine = am.get::<AffineValues>(module, body);
        let matches = am.get::<MatchFacts>(module, body);
        let decls = am.decls();
        let raiser = Raiser {
            module,
            body: module.body(body),
            component: own_component(module, body),
            decls,
            bounds: ValueBounds { intervals: &intervals, affine: &affine, decls, body: module.body(body), types: module.types() },
            matches: &matches,
        };

        let mut works = FxHashMap::default();
        let mut of_inst = FxHashMap::default();
        for (_, block) in raiser.body.blocks() {
            for inst in block.insts() {
                of_inst.insert(inst.id(), raiser.ending(am, inst, &mut works));
            }
        }

        let reached: Vec<BlockId> = raiser.body.blocks().map(|(block, _)| block).filter(|block| dominators.reachable(*block)).collect();
        let ending_at = |block: BlockId| {
            raiser.body.block(block).insts().fold(Ending::CLEAN, |ending, inst| ending.and(of_inst[&inst.id()]))
        };
        let cycles = CyclesOutsideLoops::of(raiser.body, &dominators, &loops);
        let of_loop = loop_endings(raiser.body, &loops, &evolutions, &cycles, &reached, ending_at);
        let straight = reached.iter().fold(Ending::CLEAN, |ending, block| ending.and(ending_at(*block)));
        let body = Ending {
            may_trap: straight.may_trap,
            finishes: straight.finishes && of_loop.values().all(|finishes| *finishes) && !cycles.any,
        };
        Endings { of_inst, of_loop, body }
    }
}

/// The bodies in `body`'s own strongly connected component of the call
/// graph, `body` among them. Cost: the call graph's, O(bodies +
/// instructions).
fn own_component(module: &Module, body: BodyId) -> FxHashSet<BodyId> {
    let component = CallGraph::of(module).components_callees_first().into_iter().find(|component| component.contains(&body));
    component.expect("every body of the module is in one component of its call graph").into_iter().collect()
}

/// What an instruction's ending reads: the body, its value facts, the
/// declarations, and the component whose callees are taken as unknown.
struct Raiser<'a> {
    module: &'a Module,
    body: &'a Body,
    component: FxHashSet<BodyId>,
    decls: &'a Declarations,
    bounds: ValueBounds<'a>,
    matches: &'a SparseFacts<MatchDomain>,
}

impl Raiser<'_> {
    fn types(&self) -> &Types<Concrete> {
        self.module.types()
    }

    /// Whether `inst` may trap, cleared by the value analyses where its
    /// kind may, and whether it ends, by the work it runs. `works` keeps,
    /// per type, whether a value of it holds work.
    fn ending(&self, am: &mut AnalysisManager<'_>, inst: &Inst, works: &mut FxHashMap<Ty, bool>) -> Ending {
        match inst.op() {
            Op::BinOp { op, right, .. } if op.can_trap_on_integers() => {
                Ending::traps_unless(!self.is_integer(*right) || excludes_zero(&self.bounds.value_interval(*right)))
            }
            Op::Index { slice, index, bound: IndexBound::Checked, .. } | Op::IndexSet { slice, index, bound: IndexBound::Checked, .. } => {
                Ending::traps_unless(self.bounds.index_in_bounds(*slice, *index))
            }
            Op::UnwrapVariant { src } => Ending::traps_unless(self.holds_a_payload(*src)),
            Op::FunctionCall { callee, args, .. } => self.call(am, callee, args, works),
            Op::Spawn { .. } => Ending::CLEAN,
            Op::TakeEffect { src } => {
                let Definition::Inst(start) = self.body.definition(*src) else {
                    panic!("a taking's handle is a start's result (effects §6)")
                };
                let Op::Spawn { callee, args, .. } = self.body.inst(start).op() else {
                    panic!("a taking's handle is a start's result (effects §6)")
                };
                self.call(am, callee, args, works)
            }
            Op::Intrinsic { intrinsic: Intrinsic::ArrayMap, args } => {
                let function = *args.last().expect("an array map's last argument is its function");
                self.work_of(am, function)
            }
            op => Ending { may_trap: op.may_trap(), finishes: true },
        }
    }

    fn is_integer(&self, value: ValueId) -> bool {
        matches!(CastTy::of_kind(self.types().kind(self.body.ty_of(value))), Some(CastTy::Int(_)))
    }

    /// Whether every variant the match facts leave `src` holding is one an
    /// unwrap takes a payload of: `Some`, `Ok`, or an enum's variant with a
    /// payload. No run reaches an unwrap of a value no run defines.
    fn holds_a_payload(&self, src: ValueId) -> bool {
        let tags = match &self.matches.of(src).tags {
            Tags::None => return true,
            Tags::Any => return false,
            Tags::Of(tags) => tags,
        };
        let primitive = self.decls.core().tags;
        match self.types().kind(self.body.ty_of(src)) {
            TyKind::Option(..) => tags.iter().all(|tag| *tag == primitive.some),
            TyKind::Result(..) => tags.iter().all(|tag| *tag == primitive.ok),
            TyKind::Enum { variants, .. } => tags.iter().all(|tag| variants.iter().any(|(name, payload)| name == tag && payload.is_some())),
            _ => false,
        }
    }

    /// A call's ending (extern fn §4): an extern's declared ending of its
    /// own code, and of the impls the checker chose for its requirements,
    /// with the work each argument holds; a closure's, its bodies'; any
    /// other callee's is unknown.
    fn call(&self, am: &mut AnalysisManager<'_>, callee: &Callee, args: &[ValueId], works: &mut FxHashMap<Ty, bool>) -> Ending {
        match callee {
            Callee::Extern { id, r#impl, required, .. } => {
                let own = Ending::declared(self.decls.call_facts(*id, *r#impl).ends);
                let with_required = required.iter().fold(own, |ending, chosen| ending.and(self.chosen(chosen)));
                args.iter().fold(with_required, |ending, arg| ending.and(self.work_given(am, *arg, works)))
            }
            Callee::Indirect(function) => match self.closure_bodies(*function) {
                Some(bodies) => self.bodies_ending(am, bodies),
                None => Ending::UNKNOWN,
            },
            Callee::Direct(_) => Ending::UNKNOWN,
        }
    }

    /// The declared ending of a chosen impl and of those chosen for its own
    /// requirements.
    fn chosen(&self, chosen: &Chosen) -> Ending {
        let own = Ending::declared(self.decls.call_facts(chosen.generic_function, Some(chosen.r#impl)).ends);
        chosen.required.iter().fold(own, |ending, required| ending.and(self.chosen(required)))
    }

    /// The work an argument holds, which the call may run: none for a value
    /// whose type holds no work.
    fn work_given(&self, am: &mut AnalysisManager<'_>, arg: ValueId, works: &mut FxHashMap<Ty, bool>) -> Ending {
        match holds_work(self.types(), self.body.ty_of(arg), works) {
            true => self.work_of(am, arg),
            false => Ending::CLEAN,
        }
    }

    /// Running `value`: a closure made here runs one of its bodies; any
    /// other value's work is unknown.
    fn work_of(&self, am: &mut AnalysisManager<'_>, value: ValueId) -> Ending {
        match self.closure_bodies(value) {
            Some(bodies) => self.bodies_ending(am, bodies),
            None => Ending::UNKNOWN,
        }
    }

    /// The bodies of the closure `value` is, where this body makes it,
    /// through any selection of its signatures.
    fn closure_bodies(&self, mut value: ValueId) -> Option<&[BodyId]> {
        loop {
            let Definition::Inst(made) = self.body.definition(value) else {
                return None;
            };
            match self.body.inst(made).op() {
                Op::MakeClosure { bodies, .. } => return Some(bodies),
                Op::Specialize { closure, .. } => value = *closure,
                _ => return None,
            }
        }
    }

    /// A run of any one of `bodies`: each ends as the manager's endings of
    /// it say, one in this body's own component as unknown.
    fn bodies_ending(&self, am: &mut AnalysisManager<'_>, bodies: &[BodyId]) -> Ending {
        bodies.iter().fold(Ending::CLEAN, |ending, callee| {
            let callee_ending = match self.component.contains(callee) {
                true => Ending::UNKNOWN,
                false => am.get::<Raise>(self.module, *callee).body(),
            };
            ending.and(callee_ending)
        })
    }
}

fn excludes_zero(interval: &Interval) -> bool {
    match interval {
        Interval::Empty => true,
        Interval::Range { lo, hi } => *lo > Bound::Finite(0) || *hi < Bound::Finite(0),
    }
}

/// Whether a value of `ty` holds work a call it is given may run: a
/// function value, a started call, an extern-defined type that carries an
/// effect (extern fn §8), or a type with such a part. Cost: O(1) per type
/// of the table once `works` holds it, so O(the types `ty` is made of).
fn holds_work(types: &Types<Concrete>, ty: Ty, works: &mut FxHashMap<Ty, bool>) -> bool {
    if let Some(known) = works.get(&ty) {
        return *known;
    }
    let holds = acvus_utils::grow(|| match types.kind(ty) {
        TyKind::Fn { .. } | TyKind::Closures(_) | TyKind::Handle(_) | TyKind::UserDefined { effect: Some(_), .. } => true,
        TyKind::Int(_)
        | TyKind::Float
        | TyKind::Char
        | TyKind::String
        | TyKind::Bool
        | TyKind::Unit
        | TyKind::Never
        | TyKind::Order
        | TyKind::Str => false,
        TyKind::Array(part, ..) | TyKind::Slice(part) | TyKind::Option(part, _) | TyKind::Ref(_, part) => {
            holds_work(types, *part, works)
        }
        TyKind::Result(ok, err, _) => holds_work(types, *ok, works) || holds_work(types, *err, works),
        TyKind::Tuple(parts, _) => parts.iter().any(|part| holds_work(types, *part, works)),
        TyKind::Object { fields, .. } => fields.iter().any(|(_, part)| holds_work(types, *part, works)),
        TyKind::Enum { variants, .. } => variants.iter().filter_map(|(_, payload)| *payload).any(|part| holds_work(types, part, works)),
        TyKind::UserDefined { type_args, effect: None, .. } => type_args.iter().any(|part| holds_work(types, *part, works)),
        TyKind::Var(var) => match *var {},
    });
    works.insert(ty, holds);
    holds
}

/// The cycles of the body that no natural loop is: a retreating edge of a
/// depth-first walk from the entry whose target does not dominate its
/// source. Such a cycle has no trip count, so no run of it is known to
/// end; a loop that holds one does not finish either.
struct CyclesOutsideLoops {
    any: bool,
    /// The loops that hold such a cycle whole.
    holding: FxHashSet<LoopId>,
}

impl CyclesOutsideLoops {
    /// Cost: O(blocks + edges), and O(the loop depth) per such edge.
    fn of(body: &Body, dominators: &Dominators, loops: &LoopTree) -> CyclesOutsideLoops {
        let mut found = CyclesOutsideLoops { any: false, holding: FxHashSet::default() };
        let count = body.blocks().count();
        let mut visited = vec![false; count];
        let mut on_path = vec![false; count];
        let mut path: Vec<(BlockId, usize)> = vec![(body.entry(), 0)];
        visited[body.entry().index()] = true;
        on_path[body.entry().index()] = true;
        while let Some((block, next)) = path.last_mut() {
            let from = *block;
            let edges = body.block(from).terminator().edges();
            let Some(edge) = edges.get(*next) else {
                on_path[from.index()] = false;
                path.pop();
                continue;
            };
            *next += 1;
            let to = edge.to;
            if on_path[to.index()] {
                if !dominators.dominates(to, from) {
                    found.record(loops, from, to);
                }
            } else if !visited[to.index()] {
                visited[to.index()] = true;
                on_path[to.index()] = true;
                path.push((to, 0));
            }
        }
        found
    }

    fn record(&mut self, loops: &LoopTree, from: BlockId, to: BlockId) {
        self.any = true;
        let around = std::iter::successors(loops.innermost(from), |of| loops.parent(*of));
        self.holding.extend(around.filter(|of| loops.contains(*of, to)));
    }
}

/// Whether each loop finishes: its head yields a bounded count, every
/// instruction of its own blocks finishes, it holds no cycle outside the
/// loops, and every loop inside it finishes. Cost: O(loops + the
/// instructions of the reached blocks).
fn loop_endings<E>(
    body: &Body,
    loops: &LoopTree,
    evolutions: &Evolutions,
    cycles: &CyclesOutsideLoops,
    reached: &[BlockId],
    ending_at: E,
) -> FxHashMap<LoopId, bool>
where
    E: Fn(BlockId) -> Ending,
{
    let mut own: FxHashMap<LoopId, bool> =
        loops.loops().map(|of| (of, head_bounds(body, loops, evolutions, of) && !cycles.holding.contains(&of))).collect();
    for block in reached {
        if let Some(of) = loops.innermost(*block)
            && !ending_at(*block).finishes
        {
            own.insert(of, false);
        }
    }
    let mut children: FxHashMap<LoopId, Vec<LoopId>> = FxHashMap::default();
    let mut roots = Vec::new();
    for of in loops.loops() {
        match loops.parent(of) {
            Some(parent) => children.entry(parent).or_default().push(of),
            None => roots.push(of),
        }
    }
    let mut finishes = FxHashMap::default();
    for root in roots {
        let mut walk: Vec<(LoopId, bool)> = vec![(root, false)];
        while let Some((of, children_done)) = walk.pop() {
            let inner = children.get(&of).map(Vec::as_slice).unwrap_or_default();
            if children_done {
                let all_inner = inner.iter().all(|child| finishes[child]);
                finishes.insert(of, own[&of] && all_inner);
                continue;
            }
            walk.push((of, true));
            walk.extend(inner.iter().map(|child| (*child, false)));
        }
    }
    finishes
}

/// Whether the terminator heading `of` bounds its iterations: a `For` whose
/// body is inside the loop and whose exit is not, and a `While` whose trip
/// count the scalar evolution names. Any other head runs a count nothing
/// bounds.
fn head_bounds(body: &Body, loops: &LoopTree, evolutions: &Evolutions, of: LoopId) -> bool {
    match body.block(loops.header(of)).terminator() {
        Terminator::For { stages, exit, .. } => loops.contains(of, stages.body()) && !loops.contains(of, *exit),
        Terminator::While { .. } => evolutions.trip_count(of).is_some(),
        Terminator::Jump { .. }
        | Terminator::JumpIf { .. }
        | Terminator::Diamond { .. }
        | Terminator::Switch { .. }
        | Terminator::Return { .. }
        | Terminator::Diverge
        | Terminator::NoArmMatched { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use acvus_ast::Span;
    use acvus_utils::QualifiedRef;

    use super::*;
    use crate::decl::{Core, DeclaredFacts, LoopMarkers, PairMember, RawDeclarations, Scheme, declared_fn_flows};
    use crate::ir::cfg::{Builder, Stages};
    use crate::ir::{BinOp, ExitTrip, ForSource, IndexMode};
    use crate::passes::optimize::tests::{Fixture, add, int_const, ret};
    use crate::ty::{Calls, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Poly, TyId};

    /// `apply(f: fn(i64) -> i64) -> i64`, whose own code is declared
    /// `Total`: the one extern these tests call.
    struct Apply {
        decls: Declarations,
        id: QualifiedRef,
        /// The call's callee type, `fn(fn(i64) -> i64) -> i64`, pure.
        callee_ty: Ty,
    }

    fn apply(f: &mut Fixture) -> Apply {
        let mut types = Types::<Poly>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let poly_fn = |types: &mut Types<Poly>, params: &[TyId<Poly>]| {
            types.intern(TyKind::Fn {
                params: params.into(),
                ret: int,
                effect: EffectTerm::Known(Effect::pure()),
                flows: FlowTerm::Known(declared_fn_flows()),
                calls: Calls::Many,
                holds: Holds::Nothing,
                values: (),
            })
        };
        let callback = poly_fn(&mut types, &[int]);
        let ty = poly_fn(&mut types, &[callback]);
        let id = QualifiedRef::root(f.name("apply"));
        let scheme = Scheme {
            ty,
            param_names: Box::new([f.name("f")]),
            bounds: Box::new([]),
            impls: None,
            requires: Box::new([]),
            facts: DeclaredFacts { ends: Ends::Total, ..DeclaredFacts::default() },
            runs: Box::new([]),
            specialized: Box::new([]),
        };
        let decls = Declarations::verify(RawDeclarations {
            types,
            functions: vec![(id, scheme)],
            lang_impls: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            entries: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(&f.interner),
        })
        .expect("the declarations verify");
        let callback = f.pure_unary;
        let callee_ty = f.module.types_mut().intern(TyKind::Fn {
            params: Box::new([callback]),
            ret: f.int,
            effect: EffectTerm::Known(Effect::pure()),
            flows: FlowTerm::Known(Flows::none()),
            calls: Calls::Many,
            holds: Holds::Nothing,
            values: (),
        });
        Apply { decls, id, callee_ty }
    }

    fn endings(f: &Fixture, decls: &Declarations, body: BodyId) -> (Rc<Endings>, Rc<LoopTree>) {
        let mut am = AnalysisManager::new(decls);
        (am.get::<Raise>(&f.module, body), am.get::<LoopTree>(&f.module, body))
    }

    fn empty_endings(f: &Fixture, body: BodyId) -> (Rc<Endings>, Rc<LoopTree>) {
        let decls = crate::passes::optimize::tests::declarations(&f.interner);
        endings(f, &decls, body)
    }

    /// The instruction whose result is `value`.
    fn made(f: &Fixture, body: BodyId, value: ValueId) -> InstId {
        let Definition::Inst(inst) = f.module.body(body).definition(value) else {
            panic!("{value:?} is an instruction's result")
        };
        inst
    }

    fn binary(op: BinOp, left: ValueId, right: ValueId) -> Op {
        Op::BinOp { op, left, right }
    }

    fn jump(to: BlockId, args: Vec<ValueId>) -> Terminator {
        Terminator::Jump { to, args }
    }

    fn while_loop(cond: ValueId, body: BlockId, exit: BlockId) -> Terminator {
        Terminator::While { cond, stages: Stages::lowered(body), exit, exit_args: Vec::new() }
    }

    fn closure(bodies: Vec<BodyId>) -> Op {
        Op::MakeClosure { bodies, captures: Vec::new() }
    }

    fn call_of(function: ValueId, callee_ty: Ty, args: Vec<ValueId>) -> Op {
        Op::FunctionCall { callee: Callee::Indirect(function), callee_ty, args }
    }

    /// `|x| x + x`.
    fn doubling(f: &mut Fixture) -> BodyId {
        let (int, nx) = (f.int, f.name("x"));
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(nx, int, Span::ZERO);
        let doubled = builder.push(add(x, x), int, Span::ZERO);
        builder.terminate(ret(doubled), Span::ZERO);
        builder.seal().expect("the body has no open state")
    }

    /// `|x| { while x > 0 {} x }`: the test never changes, so the scalar
    /// evolution names no trip count.
    ///
    /// ```text
    /// entry(x):  zero = 0; jump header
    /// header:    c = x > zero; while c { body } else exit
    /// body:      jump header
    /// exit:      return x
    /// ```
    fn spinning(f: &mut Fixture) -> (BodyId, BlockId) {
        let (int, bool, nx) = (f.int, f.bool, f.name("x"));
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(nx, int, Span::ZERO);
        let zero = builder.push(int_const(0), int, Span::ZERO);
        let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
        builder.terminate(jump(header, Vec::new()), Span::ZERO);
        builder.switch_to(header);
        let c = builder.push(binary(BinOp::Gt, x, zero), bool, Span::ZERO);
        builder.terminate(while_loop(c, loop_body, exit), Span::ZERO);
        builder.switch_to(loop_body);
        builder.terminate(jump(header, Vec::new()), Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(x), Span::ZERO);
        (builder.seal().expect("the body has no open state"), header)
    }

    /// `i = 0; while i < 2j { i += 2 }`, `j = max(x, 0)`: the scalar
    /// evolution names the count `j`.
    ///
    /// ```text
    /// entry(x):   zero = 0; j = max(x, zero); bound = j + j; jump header(zero)
    /// header(i):  c = i < bound; while c { body } else exit
    /// body:       two = 2; i2 = i + two; jump header(i2)
    /// exit:       return i
    /// ```
    fn counted(f: &mut Fixture) -> (BodyId, BlockId) {
        let (int, bool, nx) = (f.int, f.bool, f.name("x"));
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(nx, int, Span::ZERO);
        let zero = builder.push(int_const(0), int, Span::ZERO);
        let j = builder.push(binary(BinOp::Max, x, zero), int, Span::ZERO);
        let bound = builder.push(add(j, j), int, Span::ZERO);
        let header = builder.new_block(&[int]);
        let i = builder.block_params(header)[0];
        let [loop_body, exit] = [(); 2].map(|()| builder.new_block(&[]));
        builder.terminate(jump(header, vec![zero]), Span::ZERO);
        builder.switch_to(header);
        let c = builder.push(binary(BinOp::Lt, i, bound), bool, Span::ZERO);
        builder.terminate(while_loop(c, loop_body, exit), Span::ZERO);
        builder.switch_to(loop_body);
        let two = builder.push(int_const(2), int, Span::ZERO);
        let i2 = builder.push(add(i, two), int, Span::ZERO);
        builder.terminate(jump(header, vec![i2]), Span::ZERO);
        builder.switch_to(exit);
        builder.terminate(ret(i), Span::ZERO);
        (builder.seal().expect("the body has no open state"), header)
    }

    /// `entry(a, x): q = a / x; two = 2; r = a / two; return r`.
    #[test]
    fn a_division_by_a_possibly_zero_value_may_trap_and_by_a_proven_non_zero_one_does_not() {
        let mut f = Fixture::new();
        let (int, na, nx) = (f.int, f.name("a"), f.name("x"));
        let mut builder = f.module.build_body(false, int);
        let a = builder.param(na, int, Span::ZERO);
        let x = builder.param(nx, int, Span::ZERO);
        let q = builder.push(binary(BinOp::Div, a, x), int, Span::ZERO);
        let two = builder.push(int_const(2), int, Span::ZERO);
        let r = builder.push(binary(BinOp::Mod, a, two), int, Span::ZERO);
        builder.terminate(ret(r), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let (endings, _) = empty_endings(&f, id);

        assert!(endings.may_trap(made(&f, id, q)), "x may be zero");
        assert!(!endings.may_trap(made(&f, id, r)), "two is not zero");
        assert!(endings.finishes(made(&f, id, q)));
        assert_eq!(endings.body(), Ending { may_trap: true, finishes: true });
    }

    /// ```text
    /// entry(s, x):  n = len(s); at = 0; jump header
    /// header:       for i in at..n { body } else exit
    /// body(i):      e = s[i]; g = s[x]; jump header
    /// exit:         z = 0; return z
    /// ```
    #[test]
    fn a_proven_in_bounds_index_does_not_trap() {
        let mut f = Fixture::new();
        let int = f.int;
        let (index, slice_ref) = {
            let types = f.module.types_mut();
            let index = types.intern(TyKind::Int(IntTy::U64));
            let slice = types.intern(TyKind::Slice(int));
            (index, types.intern(TyKind::Ref(Mutability::Shared, slice)))
        };
        let (ns, nx) = (f.name("s"), f.name("x"));
        let mut builder = f.module.build_body(false, int);
        let s = builder.param(ns, slice_ref, Span::ZERO);
        let x = builder.param(nx, index, Span::ZERO);
        let n = builder.push(Op::SliceLen { src: s }, index, Span::ZERO);
        let at = builder.push(int_const(0), index, Span::ZERO);
        let header = builder.new_block(&[]);
        let loop_body = builder.new_block(&[index]);
        let exit = builder.new_block(&[]);
        builder.terminate(jump(header, Vec::new()), Span::ZERO);
        builder.switch_to(header);
        let source = ForSource::Range { at, hi: n };
        builder.terminate(
            Terminator::For { source, stages: Stages::lowered(loop_body), exit, exit_trip: ExitTrip::Absent, exit_args: Vec::new() },
            Span::ZERO,
        );
        builder.switch_to(loop_body);
        let i = builder.block_params(loop_body)[0];
        let element = |index| Op::Index { slice: s, index, mode: IndexMode::Copy, bound: IndexBound::Checked };
        let e = builder.push(element(i), int, Span::ZERO);
        let g = builder.push(element(x), int, Span::ZERO);
        builder.terminate(jump(header, Vec::new()), Span::ZERO);
        builder.switch_to(exit);
        let z = builder.push(int_const(0), int, Span::ZERO);
        builder.terminate(ret(z), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let (endings, loops) = empty_endings(&f, id);

        assert!(!endings.may_trap(made(&f, id, e)), "i < len(s) for i in 0..len(s)");
        assert!(endings.may_trap(made(&f, id, g)), "x may be past the end");
        let over = loops.innermost(header).expect("the header heads a loop");
        assert!(endings.loop_finishes(over), "a For finishes");
    }

    #[test]
    fn a_while_with_a_trip_count_finishes_and_one_without_does_not() {
        let mut f = Fixture::new();
        let (bounded, bounded_header) = counted(&mut f);
        let (unbounded, unbounded_header) = spinning(&mut f);

        let (endings, loops) = empty_endings(&f, bounded);
        let over = loops.innermost(bounded_header).expect("the header heads a loop");
        assert!(endings.loop_finishes(over));
        assert_eq!(endings.body(), Ending { may_trap: false, finishes: true });

        let (endings, loops) = empty_endings(&f, unbounded);
        let over = loops.innermost(unbounded_header).expect("the header heads a loop");
        assert!(!endings.loop_finishes(over));
        assert_eq!(endings.body(), Ending { may_trap: false, finishes: false });
    }

    /// `entry(a): f = |x| <callee>; r = apply(f); return r`.
    fn applying(f: &mut Fixture, apply: &Apply, callee: BodyId) -> (BodyId, ValueId) {
        let (int, closure_ty, na) = (f.int, f.pure_unary, f.name("a"));
        let mut builder = f.module.build_body(false, int);
        builder.param(na, int, Span::ZERO);
        let closure = builder.push(closure(vec![callee]), closure_ty, Span::ZERO);
        let call = Op::FunctionCall {
            callee: Callee::Extern { id: apply.id, r#impl: None, required: Vec::new(), member: PairMember::Uniform },
            callee_ty: apply.callee_ty,
            args: vec![closure],
        };
        let r = builder.push(call, int, Span::ZERO);
        builder.terminate(ret(r), Span::ZERO);
        (builder.seal().expect("the body has no open state"), r)
    }

    #[test]
    fn an_extern_stated_total_with_a_total_closure_argument_is_total() {
        let mut f = Fixture::new();
        let apply = apply(&mut f);
        let callee = doubling(&mut f);
        let (caller, r) = applying(&mut f, &apply, callee);

        let (endings, _) = endings(&f, &apply.decls, caller);

        let call = made(&f, caller, r);
        assert!(!endings.may_trap(call));
        assert!(endings.finishes(call));
        assert_eq!(endings.body(), Ending { may_trap: false, finishes: true });
    }

    #[test]
    fn an_extern_stated_total_with_a_non_finishing_closure_argument_does_not_finish() {
        let mut f = Fixture::new();
        let apply = apply(&mut f);
        let (callee, _) = spinning(&mut f);
        let (caller, r) = applying(&mut f, &apply, callee);

        let (endings, _) = endings(&f, &apply.decls, caller);

        let call = made(&f, caller, r);
        assert!(!endings.may_trap(call), "neither apply's own code nor the closure traps");
        assert!(!endings.finishes(call), "the closure's loop has no trip count");
        assert!(!endings.body().finishes);
    }

    /// `own(x): f = |x| own(x); r = f(x); return r`: a call into the
    /// body's own component, which nothing in the body can trap or loop.
    #[test]
    fn recursion_is_conservative() {
        let mut f = Fixture::new();
        let (int, closure_ty, nx) = (f.int, f.pure_unary, f.name("x"));
        let own = f.module.next_body_id();
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(nx, int, Span::ZERO);
        let function = builder.push(closure(vec![own]), closure_ty, Span::ZERO);
        let r = builder.push(call_of(function, closure_ty, vec![x]), int, Span::ZERO);
        builder.terminate(ret(r), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        assert_eq!(id, own);

        let (endings, _) = empty_endings(&f, id);

        let call = made(&f, id, r);
        assert!(endings.may_trap(call));
        assert!(!endings.finishes(call));
        assert_eq!(endings.body(), Ending { may_trap: true, finishes: false });
    }

    /// The callee `|x| x / x` and the caller `f = the callee; r = f(a)`;
    /// beside it, the same caller of `|x| x + x`.
    #[test]
    fn a_callees_trap_reaches_its_caller() {
        let mut f = Fixture::new();
        let (int, closure_ty, na, nx) = (f.int, f.pure_unary, f.name("a"), f.name("x"));
        let mut builder = f.module.build_body(false, int);
        let x = builder.param(nx, int, Span::ZERO);
        let q = builder.push(binary(BinOp::Div, x, x), int, Span::ZERO);
        builder.terminate(ret(q), Span::ZERO);
        let trapping = builder.seal().expect("the body has no open state");
        let clean = doubling(&mut f);
        let caller_of = |f: &mut Fixture, callee: BodyId| {
            let mut builder: Builder<'_> = f.module.build_body(false, int);
            let a = builder.param(na, int, Span::ZERO);
            let function = builder.push(closure(vec![callee]), closure_ty, Span::ZERO);
            let r = builder.push(call_of(function, closure_ty, vec![a]), int, Span::ZERO);
            builder.terminate(ret(r), Span::ZERO);
            (builder.seal().expect("the body has no open state"), r)
        };
        let (traps, trapping_call) = caller_of(&mut f, trapping);
        let (clean_caller, clean_call) = caller_of(&mut f, clean);

        let (endings, _) = empty_endings(&f, traps);
        assert!(endings.may_trap(made(&f, traps, trapping_call)));
        assert!(endings.finishes(made(&f, traps, trapping_call)));
        assert_eq!(endings.body(), Ending { may_trap: true, finishes: true });

        let (endings, _) = empty_endings(&f, clean_caller);
        assert!(!endings.may_trap(made(&f, clean_caller, clean_call)));
        assert_eq!(endings.body(), Ending { may_trap: false, finishes: true });
    }

    /// `entry(a, o): s = Some(a); u = unwrap s; v = unwrap o; return u`.
    #[test]
    fn an_unwrap_of_a_known_some_does_not_trap_and_of_an_unknown_option_may() {
        let mut f = Fixture::new();
        let int = f.int;
        let option = f.module.types_mut().intern(TyKind::Option(int, ()));
        let (na, no) = (f.name("a"), f.name("o"));
        let some = Core::of(&f.interner).tags.some;
        let mut builder = f.module.build_body(false, int);
        let a = builder.param(na, int, Span::ZERO);
        let o = builder.param(no, option, Span::ZERO);
        let s = builder.push(Op::MakeVariant { tag: some, payload: Some(a) }, option, Span::ZERO);
        let u = builder.push(Op::UnwrapVariant { src: s }, int, Span::ZERO);
        let v = builder.push(Op::UnwrapVariant { src: o }, int, Span::ZERO);
        builder.terminate(ret(u), Span::ZERO);
        let id = builder.seal().expect("the body has no open state");

        let (endings, _) = empty_endings(&f, id);

        assert!(!endings.may_trap(made(&f, id, u)));
        assert!(endings.may_trap(made(&f, id, v)));
    }
}
