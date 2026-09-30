//! The scalar evolution: how each integer value of a body evolves over the
//! iterations of the loops around it, and how many iterations a loop runs.
//!
//! A value is an element of the value ring (`domain::ring`) over the body's
//! atoms, or a first-degree recurrence `{start, +, step}_L` over one loop `L`:
//! its value at `L`'s k-th iteration is `start + k·step`, where `step` is
//! invariant in `L` and `start` is invariant in `L` too (it may itself be a
//! recurrence of a loop around `L`). No recurrence is of a higher degree in its
//! own loop: a value whose step changes with its loop's own iteration is
//! unknown (the user's decision: each loop first degree, coefficients in the
//! ring; no full ring of recurrences).
//!
//! Arithmetic is exact over the integers: overflow is undefined (operators
//! §intro), so a recurrence needs no wrap flag and no pass reasons about one.
//! Sign and order are not the evolution's; a reader asks the interval domain.
//!
//! The evolution is a query over the SSA, kept by the analysis manager; it
//! writes nothing into the IR (compilation §4: the backend decides what to do
//! with a loop).
//!
//! Cost: two passes over the body's values in dominance order, and no
//! fixpoint. The first builds each integer value as an element of the ring
//! over the body's atoms, each loop's header parameters and counter standing
//! as atoms for their value on the current iteration. The second reads each
//! loop's header parameters off their entry and back-edge arguments once,
//! outermost loop first, and resolves every value into its evolution. A ring
//! operation is O(MAX_TERMS²); an operation on evolutions takes one ring
//! operation per recurrence of its chain, at most d for the loop nesting depth
//! d; resolving a value multiplies at most MAX_DEGREE evolutions for each of its
//! at most MAX_TERMS terms. So O(values · d · MAX_DEGREE · MAX_TERMS³) for the
//! values, O(edges · MAX_TERMS) for block arguments, and O(d · (blocks +
//! edges)) for the loops' exits, beside the loop tree and the interval values
//! the manager keeps.

use std::rc::Rc;

use acvus_ast::Literal;
use rustc_hash::FxHashMap;

use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Definition, Inst, Module, Op, Terminator};
use crate::ir::{BinOp, ForSource, Overflow, UnaryOp, ValueId, known_len};
use crate::passes::analysis::dominators::Graph;
use crate::passes::analysis::{Analysis, AnalysisManager, IntervalValues, LoopId, LoopTree, SparseFacts};
use crate::passes::domain::{Bound, Interval, IntervalDomain, Polynomial, Symbol};
use crate::ty::{CastTy, Concrete, TyKind, Types};

/// How a value evolves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Evolution {
    /// The same element of the ring on every iteration of every loop around
    /// the value's definition that it is read in.
    Invariant(Polynomial<Symbol>),
    /// `{start, +, step}` over `over`: `start + k·step` at the loop's k-th
    /// iteration, both invariant in `over`.
    Recurrence(Rc<Recurrence>),
    /// No evolution this analysis names.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recurrence {
    pub over: LoopId,
    pub start: Evolution,
    pub step: Evolution,
}

/// How many times a loop's body runs, where the analysis names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TripCount {
    /// Exactly this many, on every run that enters the loop, as an
    /// evolution invariant in the loop (it may vary with a loop around it).
    Exact(Evolution),
    /// At most this many: a loop the body may leave early, or a pull loop
    /// whose count its maker's declared relation bounds (extern fn §3,
    /// relation; source, count).
    AtMost(Evolution),
}

impl Recurrence {
    /// The loop of this recurrence, then of each recurrence its start is,
    /// innermost first.
    fn loops(&self) -> impl Iterator<Item = LoopId> + '_ {
        std::iter::successors(Some(self), |recurrence| match &recurrence.start {
            Evolution::Recurrence(outer) => Some(&**outer),
            Evolution::Invariant(_) | Evolution::Unknown => None,
        })
        .map(|recurrence| recurrence.over)
    }
}

/// The evolution facts of one body.
pub struct Evolutions {
    of: FxHashMap<ValueId, Evolution>,
    trips: FxHashMap<LoopId, TripCount>,
    /// The innermost loop around each value's definition, for the values
    /// defined inside a loop.
    defined_in: FxHashMap<ValueId, LoopId>,
    loops: Rc<LoopTree>,
}

impl Evolutions {
    /// The evolution of `value`; `Unknown` for a value the analysis does not
    /// name.
    pub fn of(&self, value: ValueId) -> &Evolution {
        const UNKNOWN: &Evolution = &Evolution::Unknown;
        self.of.get(&value).unwrap_or(UNKNOWN)
    }

    /// Whether `value` is invariant in `over`: its evolution holds no
    /// recurrence of `over` or of a loop inside it.
    ///
    /// A value of no named evolution is invariant exactly where it is
    /// defined outside `over`: one defined inside may change on every
    /// iteration, and nothing here says it does not.
    pub fn is_invariant(&self, value: ValueId, over: LoopId) -> bool {
        let inside_over = |of: LoopId| self.loops.contains(over, self.loops.header(of));
        match self.of(value) {
            Evolution::Invariant(_) => true,
            Evolution::Recurrence(recurrence) => recurrence.loops().all(|of| !inside_over(of)),
            Evolution::Unknown => self.defined_in.get(&value).is_none_or(|of| !inside_over(*of)),
        }
    }

    /// The trip count of `over`, where the analysis names one. A `For` over a
    /// range runs `max(hi - at, 0)` times, over a slice its length, over an
    /// array its length; a `While` whose only exit compares a recurrence of it
    /// with a value invariant in it runs the count that comparison gives; a
    /// pull loop runs at most what its iterator's maker's relation states.
    pub fn trip_count(&self, over: LoopId) -> Option<&TripCount> {
        self.trips.get(&over)
    }
}

/// The analysis the manager keeps. It reads the body's values and control
/// flow, the loop tree, and the declarations' relations and sources.
pub struct ScalarEvolution;

impl Analysis for ScalarEvolution {
    type Facts = Evolutions;
    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Evolutions {
        let loops = am.get::<LoopTree>(module, body);
        let intervals = am.get::<IntervalValues>(module, body);
        let body = module.body(body);
        let graph = Graph::of(body);
        let order = reverse_postorder(body, &graph);
        let mut reached = vec![false; graph.blocks.len()];
        for block in &order {
            reached[block.index()] = true;
        }
        let mut scev = Scev {
            body,
            types: module.types(),
            loops: &loops,
            intervals: &intervals,
            graph,
            reached,
            inductions: FxHashMap::default(),
            forms: FxHashMap::default(),
            resolved: FxHashMap::default(),
        };

        scev.find_inductions();
        let mut defined_in = FxHashMap::default();
        scev.build_body_params();
        for block in order {
            scev.build_block(block, &mut defined_in);
        }

        for of in outermost_first(&loops) {
            scev.resolve_inductions(of);
        }
        let of = scev
            .forms
            .iter()
            .map(|(value, form)| (*value, scev.resolve(form)))
            .filter(|(_, evolution)| *evolution != Evolution::Unknown)
            .collect();
        let trips = loops.loops().filter_map(|over| Some((over, scev.trip_count(over)?))).collect();
        Evolutions { of, trips, defined_in, loops: Rc::clone(&loops) }
    }
}

/// The blocks the entry reaches, each after every block that dominates it.
/// Cost: O(blocks + edges).
fn reverse_postorder(body: &Body, graph: &Graph) -> Vec<BlockId> {
    let entry = body.entry().index();
    let mut visited = vec![false; graph.blocks.len()];
    let mut postorder = Vec::new();
    let mut path: Vec<(usize, usize)> = vec![(entry, 0)];
    visited[entry] = true;
    while let Some((node, next)) = path.last_mut() {
        let Some(to) = graph.successors[*node].get(*next).copied() else {
            postorder.push(graph.blocks[*node]);
            path.pop();
            continue;
        };
        *next += 1;
        if !visited[to] {
            visited[to] = true;
            path.push((to, 0));
        }
    }
    postorder.reverse();
    postorder
}

/// Every loop, each after the loops around it. Cost: O(loops · d).
fn outermost_first(loops: &LoopTree) -> Vec<LoopId> {
    let depth = |of: LoopId| std::iter::successors(Some(of), |inner| loops.parent(*inner)).count();
    let mut order: Vec<LoopId> = loops.loops().collect();
    order.sort_by_key(|of| depth(*of));
    order
}

/// One body's evolutions as they are built.
struct Scev<'a> {
    body: &'a Body,
    types: &'a Types<Concrete>,
    loops: &'a LoopTree,
    intervals: &'a SparseFacts<IntervalDomain>,
    graph: Graph,
    /// Whether the entry reaches each block, by block index.
    reached: Vec<bool>,
    /// The loop of each induction atom: an integer header parameter of the
    /// loop, or the loop's `For` counter.
    inductions: FxHashMap<ValueId, LoopId>,
    /// Each integer value's form, where one is known: the ring element it is
    /// over the body's atoms and the induction atoms of the loops around its
    /// definition, each induction atom standing for its value on the current
    /// iteration.
    forms: FxHashMap<ValueId, Polynomial<Symbol>>,
    /// The evolution of each induction atom, once its loop is resolved.
    resolved: FxHashMap<ValueId, Evolution>,
}

impl Scev<'_> {
    fn is_integer(&self, value: ValueId) -> bool {
        matches!(CastTy::of_kind(self.types.kind(self.body.ty_of(value))), Some(CastTy::Int(_)))
    }

    fn outside_every_loop(&self, value: ValueId) -> bool {
        match self.body.definition(value) {
            Definition::Inst(inst) => self.loops.innermost(self.body.block_of(inst)).is_none(),
            Definition::BlockParam(block, _) => self.loops.innermost(block).is_none(),
            Definition::Param(_) | Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => true,
        }
    }

    /// Whether `inner` is a loop inside `outer`, not `outer` itself.
    fn strictly_inside(&self, inner: LoopId, outer: LoopId) -> bool {
        inner != outer && self.loops.contains(outer, self.loops.header(inner))
    }

    fn predecessors(&self, block: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.graph.predecessors[block.index()]
            .iter()
            .filter(|pred| self.reached[**pred])
            .map(|pred| self.graph.blocks[*pred])
    }

    fn find_inductions(&mut self) {
        for of in self.loops.loops() {
            let header = self.body.block(self.loops.header(of));
            let counter = match header.terminator() {
                Terminator::For { source, stages, .. } if self.loops.contains(of, stages.body()) => {
                    Some(self.body.block(stages.body()).params()[source.counter_param()])
                }
                _ => None,
            };
            for value in header.params().iter().copied().chain(counter) {
                if self.is_integer(value) {
                    self.inductions.insert(value, of);
                }
            }
        }
    }

    /// The loops whose induction atoms `form` holds.
    fn inductions_of<'f>(&'f self, form: &'f Polynomial<Symbol>) -> impl Iterator<Item = LoopId> + 'f {
        form.terms().flat_map(|(monomial, _)| monomial.factors()).filter_map(|(atom, _)| match atom {
            Symbol::Value(value) => self.inductions.get(value).copied(),
            Symbol::Len(_) | Symbol::Remaining(_) => None,
        })
    }

    /// `value`'s form as a value of block `at` reads it. A form that holds an
    /// induction atom of a loop that does not hold `at` is read after that
    /// loop is left, where the atom no longer stands for the iteration the
    /// value was made in, so the read gives none.
    fn read(&self, value: ValueId, at: BlockId) -> Option<Polynomial<Symbol>> {
        let form = self.forms.get(&value)?;
        self.inductions_of(form).all(|of| self.loops.contains(of, at)).then(|| form.clone())
    }

    /// `value`'s form, where each induction atom it holds is of a loop
    /// around `of`: a form that is the same on every iteration of `of`.
    fn read_around(&self, value: ValueId, of: LoopId) -> Option<Polynomial<Symbol>> {
        let form = self.forms.get(&value)?;
        self.is_around(form, of).then(|| form.clone())
    }

    fn is_around(&self, form: &Polynomial<Symbol>, of: LoopId) -> bool {
        self.inductions_of(form).all(|outer| self.strictly_inside(of, outer))
    }

    /// A value the analysis does not look into: the atom that is itself
    /// outside every loop, where it is one value on every run, and no form
    /// inside a loop.
    fn opaque(&self, value: ValueId, at: BlockId) -> Option<Polynomial<Symbol>> {
        self.loops.innermost(at).is_none().then(|| Polynomial::atom(Symbol::Value(value)))
    }

    fn build_body_params(&mut self) {
        let params = self.body.params().iter().chain(self.body.captures()).map(|(_, value)| *value);
        let integers: Vec<ValueId> = params.filter(|value| self.is_integer(*value)).collect();
        for value in integers {
            self.forms.insert(value, Polynomial::atom(Symbol::Value(value)));
        }
    }

    fn build_block(&mut self, at: BlockId, defined_in: &mut FxHashMap<ValueId, LoopId>) {
        let block = self.body.block(at);
        let innermost = self.loops.innermost(at);
        for (index, param) in block.params().iter().copied().enumerate() {
            if let Some(of) = innermost {
                defined_in.insert(param, of);
            }
            if !self.is_integer(param) {
                continue;
            }
            let form = match self.inductions.contains_key(&param) {
                true => Some(Polynomial::atom(Symbol::Value(param))),
                false => self.joined_argument(at, index),
            };
            if let Some(form) = form.or_else(|| self.opaque(param, at)) {
                self.forms.insert(param, form);
            }
        }
        for inst in block.insts() {
            let Some(result) = inst.result() else {
                continue;
            };
            if let Some(of) = innermost {
                defined_in.insert(result, of);
            }
            if !self.is_integer(result) {
                continue;
            }
            if let Some(form) = self.transfer(inst, at).or_else(|| self.opaque(result, at)) {
                self.forms.insert(result, form);
            }
        }
    }

    /// The one form every edge into `block` passes to its parameter at
    /// `index`; none where the edges pass different ones, or a terminator
    /// supplies it.
    fn joined_argument(&self, block: BlockId, index: usize) -> Option<Polynomial<Symbol>> {
        let mut arguments: Vec<Option<Polynomial<Symbol>>> = Vec::new();
        for pred in self.predecessors(block) {
            for edge in self.body.block(pred).terminator().edges().iter().filter(|edge| edge.to == block) {
                let argument = index.checked_sub(edge.supplied).map(|at| edge.args[at]);
                arguments.push(argument.and_then(|argument| self.read(argument, block)));
            }
        }
        let (first, rest) = arguments.split_first()?;
        rest.iter().all(|other| other == first).then(|| first.clone())?
    }

    /// The form of an instruction's integer result: exact ring arithmetic
    /// on `+`, `-`, `*` and negation, which trap rather than wrap, and a
    /// slice's length where the slice is one value on every run.
    fn transfer(&self, inst: &Inst, at: BlockId) -> Option<Polynomial<Symbol>> {
        match inst.op() {
            Op::Const { value } => match value.desugared() {
                Literal::Int(n) => Some(Polynomial::constant(n)),
                other => panic!("an integer constant is an integer literal, not {other:?}"),
            },
            Op::BinOp { op: op @ (BinOp::Add(Overflow::Trap) | BinOp::Sub(Overflow::Trap) | BinOp::Mul(Overflow::Trap)), left, right } => {
                let (left, right) = (self.read(*left, at)?, self.read(*right, at)?);
                match op {
                    BinOp::Add(_) => left.checked_add(&right),
                    BinOp::Sub(_) => left.checked_sub(&right),
                    _ => left.checked_mul(&right),
                }
            }
            Op::UnaryOp { op: UnaryOp::Neg(Overflow::Trap), operand } => self.read(*operand, at)?.checked_neg(),
            Op::SliceLen { src } if self.outside_every_loop(*src) => Some(Polynomial::atom(Symbol::Len(*src))),
            _ => None,
        }
    }

    fn algebra(&self) -> Algebra<'_> {
        Algebra { loops: self.loops }
    }

    /// The evolution of each induction atom of `of`; every loop around `of`
    /// is resolved already.
    fn resolve_inductions(&mut self, of: LoopId) {
        let header = self.body.block(self.loops.header(of));
        for (index, param) in header.params().iter().copied().enumerate() {
            if self.inductions.get(&param) == Some(&of) {
                let evolution = self.header_recurrence(of, index, param);
                self.resolved.insert(param, evolution);
            }
        }
        let Terminator::For { source, stages, .. } = header.terminator() else {
            return;
        };
        let counter = self.body.block(stages.body()).params()[source.counter_param()];
        if self.inductions.get(&counter) != Some(&of) {
            return;
        }
        let start = match source {
            ForSource::Range { at, .. } => self.read_around(*at, of).map_or(Evolution::Unknown, |at| self.resolve(&at)),
            ForSource::Slice(_) | ForSource::SliceMut(_) | ForSource::Array(_) => Evolution::Invariant(Polynomial::zero()),
        };
        let evolution = self.algebra().recurrence(of, start, Evolution::Invariant(Polynomial::constant(1)));
        self.resolved.insert(counter, evolution);
    }

    /// `{start, +, step}` for the header parameter `param` of `of`, at
    /// `index`: every edge from outside `of` passes one `start` the same on
    /// every iteration, and every back edge passes `param + step` for one
    /// `step` the same on every iteration.
    fn header_recurrence(&self, of: LoopId, index: usize, param: ValueId) -> Evolution {
        let header = self.loops.header(of);
        let mut starts: Vec<Polynomial<Symbol>> = Vec::new();
        let mut steps: Vec<Polynomial<Symbol>> = Vec::new();
        for pred in self.predecessors(header) {
            let back = self.loops.contains(of, pred);
            for edge in self.body.block(pred).terminator().edges().iter().filter(|edge| edge.to == header) {
                let Some(argument) = index.checked_sub(edge.supplied).and_then(|at| self.forms.get(&edge.args[at])) else {
                    return Evolution::Unknown;
                };
                if !back {
                    if !self.is_around(argument, of) {
                        return Evolution::Unknown;
                    }
                    starts.push(argument.clone());
                    continue;
                }
                let Some(linear) = argument.linear_in(&Symbol::Value(param)) else {
                    return Evolution::Unknown;
                };
                if linear.slope.as_constant() != Some(1) || !self.is_around(&linear.base, of) {
                    return Evolution::Unknown;
                }
                steps.push(linear.base);
            }
        }
        let (Some(start), Some(step)) = (the_one(&starts), the_one(&steps)) else {
            return Evolution::Unknown;
        };
        self.algebra().recurrence(of, self.resolve(start), self.resolve(step))
    }

    /// The evolution `form` is, each induction atom replaced by its own.
    fn resolve(&self, form: &Polynomial<Symbol>) -> Evolution {
        if self.inductions_of(form).next().is_none() {
            return Evolution::Invariant(form.clone());
        }
        let algebra = self.algebra();
        let mut sum = Evolution::Invariant(Polynomial::zero());
        for (monomial, coefficient) in form.terms() {
            let mut term = Evolution::Invariant(Polynomial::constant(coefficient));
            for (atom, power) in monomial.factors() {
                let factor = match atom {
                    Symbol::Value(value) if self.inductions.contains_key(value) => {
                        self.resolved.get(value).cloned().unwrap_or(Evolution::Unknown)
                    }
                    Symbol::Value(_) | Symbol::Len(_) | Symbol::Remaining(_) => Evolution::Invariant(Polynomial::atom(*atom)),
                };
                for _ in 0..power {
                    term = algebra.mul(&term, &factor);
                }
            }
            sum = algebra.add(&sum, &term);
            if sum == Evolution::Unknown {
                return sum;
            }
        }
        sum
    }

    fn trip_count(&self, of: LoopId) -> Option<TripCount> {
        let header = self.loops.header(of);
        let heads_the_loop = |body: BlockId, exit: BlockId| self.loops.contains(of, body) && !self.loops.contains(of, exit);
        let (count, exit) = match self.body.block(header).terminator() {
            Terminator::For { source, stages, exit, .. } if heads_the_loop(stages.body(), *exit) => {
                (self.for_count(of, source)?, *exit)
            }
            Terminator::While { cond, stages, exit, .. } if heads_the_loop(stages.body(), *exit) => {
                (self.while_count(of, *cond)?, *exit)
            }
            _ => return None,
        };
        Some(match self.leaves_only_at_head(of, exit) {
            true => TripCount::Exact(count),
            false => TripCount::AtMost(count),
        })
    }

    /// Whether the only way out of `of` is its header's edge to `exit`: no
    /// other edge leaves it and no block of it returns.
    fn leaves_only_at_head(&self, of: LoopId, exit: BlockId) -> bool {
        let header = self.loops.header(of);
        self.loops.blocks(of).iter().all(|block| {
            let terminator = self.body.block(*block).terminator();
            !matches!(terminator, Terminator::Return { .. })
                && terminator.edges().iter().all(|edge| self.loops.contains(of, edge.to) || (*block == header && edge.to == exit))
        })
    }

    /// A range's `max(hi - at, 0)`, where the interval domain proves
    /// `hi - at` never negative or never positive; a slice's length, where
    /// the slice is one value on every run; an array's length, from its type.
    fn for_count(&self, of: LoopId, source: &ForSource) -> Option<Evolution> {
        match source {
            ForSource::Range { at, hi } => {
                let span = self.read_around(*hi, of)?.checked_sub(&self.read_around(*at, of)?)?;
                let span = self.resolve(&span);
                match (self.never_negative(&span), self.at_most(&span, 0)) {
                    (true, _) => Some(span),
                    (false, true) => Some(Evolution::Invariant(Polynomial::zero())),
                    (false, false) => None,
                }
            }
            ForSource::Slice(slice) | ForSource::SliceMut(slice) => {
                self.outside_every_loop(*slice).then(|| Evolution::Invariant(Polynomial::atom(Symbol::Len(*slice))))
            }
            ForSource::Array(array) => {
                let TyKind::Array(_, len, _) = self.types.kind(self.body.ty_of(*array)) else {
                    panic!("a For over an array reads an array, and {array:?} is none")
                };
                let len = i128::try_from(known_len(len)).expect("an array's length fits an i128");
                Some(Evolution::Invariant(Polynomial::constant(len)))
            }
        }
    }

    /// The count of a `While` whose condition compares two integers whose
    /// difference is a recurrence of `of` with a constant step: the loop
    /// runs while `start + k·step` stands in the comparison to zero.
    fn while_count(&self, of: LoopId, cond: ValueId) -> Option<Evolution> {
        let Definition::Inst(made) = self.body.definition(cond) else {
            return None;
        };
        let Op::BinOp { op, left, right } = self.body.inst(made).op() else {
            return None;
        };
        if !self.is_integer(*left) {
            return None;
        }
        let header = self.loops.header(of);
        let difference = self.read(*left, header)?.checked_sub(&self.read(*right, header)?)?;
        let Evolution::Recurrence(difference) = self.resolve(&difference) else {
            return None;
        };
        let Evolution::Invariant(step) = &difference.step else {
            return None;
        };
        let step = step.as_constant()?;
        if difference.over != of {
            return None;
        }
        let approach = Approach::of(*op, &difference.start, step)?;

        if let Some(steps) = exact_quotient(&self.algebra(), &approach.distance, approach.per_step)
            && self.never_negative(&steps)
        {
            let last = Evolution::Invariant(Polynomial::constant(i128::from(approach.bound_included)));
            return Some(self.algebra().add(&steps, &last)).filter(|count| *count != Evolution::Unknown);
        }
        let never_entered = approach.ordered && self.at_most(&approach.distance, -i128::from(approach.bound_included));
        never_entered.then(|| Evolution::Invariant(Polynomial::zero()))
    }

    fn interval(&self, evolution: &Evolution) -> Option<Interval> {
        match evolution {
            Evolution::Invariant(polynomial) => Some(self.polynomial_interval(polynomial)),
            Evolution::Recurrence(recurrence) => {
                let iterations = Interval::exactly(0).at_least_lo();
                let risen = iterations.product(&self.interval(&recurrence.step)?);
                Some(self.interval(&recurrence.start)?.sum(&risen))
            }
            Evolution::Unknown => None,
        }
    }

    fn polynomial_interval(&self, polynomial: &Polynomial<Symbol>) -> Interval {
        polynomial.terms().fold(Interval::exactly(0), |sum, (monomial, coefficient)| {
            let term = monomial.factors().fold(Interval::exactly(coefficient), |product, (atom, power)| {
                let atom = self.atom_interval(atom);
                (0..power).fold(product, |product, _| product.product(&atom))
            });
            sum.sum(&term)
        })
    }

    fn atom_interval(&self, atom: &Symbol) -> Interval {
        match atom {
            Symbol::Value(value) => self.intervals.of(*value).clone(),
            Symbol::Len(_) | Symbol::Remaining(_) => Interval::exactly(0).at_least_lo(),
        }
    }

    fn never_negative(&self, evolution: &Evolution) -> bool {
        matches!(self.interval(evolution), Some(Interval::Range { lo, .. }) if lo >= Bound::Finite(0))
    }

    fn at_most(&self, evolution: &Evolution, bound: i128) -> bool {
        matches!(self.interval(evolution), Some(Interval::Range { hi, .. }) if hi <= Bound::Finite(bound))
    }
}

/// The one element every entry of `all` is; none where `all` is empty or
/// its entries differ.
fn the_one<T>(all: &[T]) -> Option<&T>
where
    T: PartialEq,
{
    let (first, rest) = all.split_first()?;
    rest.iter().all(|other| other == first).then_some(first)
}

/// A `While`'s exit comparison of `start + k·step` with zero, read as a
/// distance covered in steps: the loop runs `distance / per_step` times,
/// once more where it runs on the bound itself, and none where an ordered
/// comparison fails on the first test.
struct Approach {
    distance: Evolution,
    per_step: i128,
    bound_included: bool,
    ordered: bool,
}

impl Approach {
    /// None where the comparison holds forever once it holds, or is `==`.
    fn of(op: BinOp, start: &Evolution, step: i128) -> Option<Approach> {
        let negated = |start: &Evolution| Algebra::negate(start);
        let (distance, per_step, bound_included, ordered) = match op {
            BinOp::Lt if step > 0 => (negated(start)?, step, false, true),
            BinOp::Lte if step > 0 => (negated(start)?, step, true, true),
            BinOp::Gt if step < 0 => (start.clone(), -step, false, true),
            BinOp::Gte if step < 0 => (start.clone(), -step, true, true),
            BinOp::Neq => (negated(start)?, step, false, false),
            _ => return None,
        };
        Some(Approach { distance, per_step, bound_included, ordered })
    }
}

/// `evolution / divisor` where every coefficient divides exactly.
fn exact_quotient(algebra: &Algebra<'_>, evolution: &Evolution, divisor: i128) -> Option<Evolution> {
    match evolution {
        Evolution::Invariant(polynomial) => polynomial_quotient(polynomial, divisor).map(Evolution::Invariant),
        Evolution::Recurrence(recurrence) => {
            let start = exact_quotient(algebra, &recurrence.start, divisor)?;
            let step = exact_quotient(algebra, &recurrence.step, divisor)?;
            Some(algebra.recurrence(recurrence.over, start, step))
        }
        Evolution::Unknown => None,
    }
}

fn polynomial_quotient(polynomial: &Polynomial<Symbol>, divisor: i128) -> Option<Polynomial<Symbol>> {
    let mut quotient = Polynomial::zero();
    for (monomial, coefficient) in polynomial.terms() {
        if coefficient.checked_rem(divisor)? != 0 {
            return None;
        }
        let mut term = Polynomial::constant(coefficient.checked_div(divisor)?);
        for (atom, power) in monomial.factors() {
            for _ in 0..power {
                term = term.checked_mul(&Polynomial::atom(*atom))?;
            }
        }
        quotient = quotient.checked_add(&term)?;
    }
    Some(quotient)
}

/// Arithmetic on evolutions: each operation recurses once per recurrence
/// of its operands' chains, each step one ring operation.
struct Algebra<'t> {
    loops: &'t LoopTree,
}

impl Algebra<'_> {
    fn add(&self, a: &Evolution, b: &Evolution) -> Evolution {
        match (a, b) {
            (Evolution::Unknown, _) | (_, Evolution::Unknown) => Evolution::Unknown,
            (Evolution::Invariant(a), Evolution::Invariant(b)) => known(a.checked_add(b)),
            (Evolution::Recurrence(a), Evolution::Recurrence(b)) if a.over == b.over => {
                self.recurrence(a.over, self.add(&a.start, &b.start), self.add(&a.step, &b.step))
            }
            _ => match self.inner_and_invariant(a, b) {
                Some((inner, invariant)) => self.recurrence(inner.over, self.add(&inner.start, invariant), inner.step.clone()),
                None => Evolution::Unknown,
            },
        }
    }

    /// A product of two recurrences of one loop is of the second degree in
    /// it, which no evolution here is.
    fn mul(&self, a: &Evolution, b: &Evolution) -> Evolution {
        match (a, b) {
            (Evolution::Unknown, _) | (_, Evolution::Unknown) => Evolution::Unknown,
            (Evolution::Invariant(a), Evolution::Invariant(b)) => known(a.checked_mul(b)),
            _ => match self.inner_and_invariant(a, b) {
                Some((inner, invariant)) => {
                    self.recurrence(inner.over, self.mul(&inner.start, invariant), self.mul(&inner.step, invariant))
                }
                None => Evolution::Unknown,
            },
        }
    }

    fn negate(evolution: &Evolution) -> Option<Evolution> {
        match evolution {
            Evolution::Invariant(polynomial) => polynomial.checked_neg().map(Evolution::Invariant),
            Evolution::Recurrence(recurrence) => Some(Evolution::Recurrence(Rc::new(Recurrence {
                over: recurrence.over,
                start: Algebra::negate(&recurrence.start)?,
                step: Algebra::negate(&recurrence.step)?,
            }))),
            Evolution::Unknown => None,
        }
    }

    /// Of two operands, a recurrence and the other, where the other is
    /// invariant in the recurrence's loop.
    fn inner_and_invariant<'e>(&self, a: &'e Evolution, b: &'e Evolution) -> Option<(&'e Recurrence, &'e Evolution)> {
        match (a, b) {
            (Evolution::Recurrence(inner), other) | (other, Evolution::Recurrence(inner)) if self.invariant_in(other, inner.over) => {
                Some((inner, other))
            }
            _ => None,
        }
    }

    fn invariant_in(&self, evolution: &Evolution, of: LoopId) -> bool {
        match evolution {
            Evolution::Invariant(_) => true,
            Evolution::Recurrence(recurrence) => {
                recurrence.over != of && self.loops.contains(recurrence.over, self.loops.header(of))
            }
            Evolution::Unknown => false,
        }
    }

    /// `{start, +, step}` over `over`, which is `start` itself where `step`
    /// is zero. A step is an element of the ring, the user's decision that
    /// each loop's recurrence has its coefficients in the ring, so a step
    /// that is a recurrence of a loop around gives no evolution.
    fn recurrence(&self, over: LoopId, start: Evolution, step: Evolution) -> Evolution {
        if start == Evolution::Unknown {
            return Evolution::Unknown;
        }
        match &step {
            Evolution::Invariant(by) if by.as_constant() == Some(0) => start,
            Evolution::Invariant(_) => Evolution::Recurrence(Rc::new(Recurrence { over, start, step })),
            Evolution::Recurrence(_) | Evolution::Unknown => Evolution::Unknown,
        }
    }
}

fn known(polynomial: Option<Polynomial<Symbol>>) -> Evolution {
    polynomial.map_or(Evolution::Unknown, Evolution::Invariant)
}
