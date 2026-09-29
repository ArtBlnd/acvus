//! The sparse engine: a value domain over SSA def-use, conditional on which
//! edges may run (Wegman and Zadeck, "Constant Propagation with Conditional
//! Branches", 1991).
//!
//! Cost: a value's fact only rises, each rise re-evaluates the sites that
//! read it, and a site costs its operands, plus for a terminator the
//! parameters of its targets over their incoming edges. So the engine is
//! O(h · Σ over values of the cost of their users' sites), where h bounds how
//! often one fact rises: the domain's height, or, for a domain of infinite
//! height, the length of its widening's chains. Every def-use cycle of a body
//! in SSA passes through a parameter of a loop header, since along a cycle
//! with no back edge every definition would precede the next in reverse
//! postorder, and that parameter widens, so h is finite whenever the
//! widening terminates.
//!
//! An order token carries order, not a value, so it is outside every sparse
//! analysis: the engine holds no fact for one, never queues one, and hands
//! no analysis a token's fact among an instruction's or a terminator's
//! operands. Effect order is read through the editor's `OrderFacts`.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::ir::cfg::{BlockId, Body, Inst, InstId, Site, Terminator};
use crate::ir::ValueId;
use crate::passes::domain::{JoinSemiLattice, ValueDomain, Widen};

/// Which successors of a terminator may run, read from the facts of its
/// operands (conditional propagation, Wegman-Zadeck): only edges that may
/// run carry facts into their targets.
pub enum Executable {
    All,
    Only(Vec<BlockId>),
}

/// A sparse analysis: which value domain it runs, and how an instruction
/// that is not a value operation maps to it (calls read declared facts only).
pub trait SparseAnalysis {
    type Domain: ValueDomain;

    fn domain(&self) -> &Self::Domain;

    /// The fact of an instruction's result from its operands' facts, in the
    /// order the instruction lists its operands, order tokens left out.
    fn transfer(
        &self,
        inst: &Inst,
        operands: &[&<Self::Domain as ValueDomain>::Value],
    ) -> <Self::Domain as ValueDomain>::Value;

    /// Which successors may run, from the facts of the terminator's
    /// operands, order tokens left out.
    fn executable(
        &self,
        terminator: &Terminator,
        operands: &[&<Self::Domain as ValueDomain>::Value],
    ) -> Executable;

    /// The fact of the parameter of `to` that `terminator` fills itself on
    /// its executable edge into `to`: the `index`th of the leading
    /// parameters that edge supplies, such as a `For`'s counter and element
    /// or its exit's trip count. `operands` are the facts of the
    /// terminator's operands, order tokens left out.
    fn supplied(
        &self,
        terminator: &Terminator,
        to: BlockId,
        index: usize,
        operands: &[&<Self::Domain as ValueDomain>::Value],
    ) -> <Self::Domain as ValueDomain>::Value;
}

pub struct SparseFacts<D>
where
    D: ValueDomain,
{
    values: FxHashMap<ValueId, D::Value>,
    tokens: FxHashSet<ValueId>,
}

impl<D> SparseFacts<D>
where
    D: ValueDomain,
{
    /// # Panics
    /// `value` is an order token, which has no value fact, or is not a
    /// value of the body the facts were computed for.
    pub fn of(&self, value: ValueId) -> &D::Value {
        assert!(
            !self.tokens.contains(&value),
            "{value:?} is an order token; a token carries order, not a value, and has no value fact"
        );
        self.values.get(&value).expect("sparse facts cover every value of their body")
    }
}

/// Every value's fact over the blocks the entry reaches through executable
/// edges. A value those blocks never define keeps `bottom`; the body's
/// parameters, captures and slots hold `top`, since they come from outside
/// the body.
pub fn solve_sparse<A>(analysis: &A, body: &Body) -> SparseFacts<A::Domain>
where
    A: SparseAnalysis,
{
    let mut solver = Solver::new(analysis, body);
    solver.reach(body.entry());
    solver.run();
    let tokens = body.values().filter(|value| body.is_token(*value)).collect();
    SparseFacts { values: solver.facts, tokens }
}

type Value<A> = <<A as SparseAnalysis>::Domain as ValueDomain>::Value;

struct Solver<'a, A>
where
    A: SparseAnalysis,
{
    analysis: &'a A,
    body: &'a Body,
    facts: FxHashMap<ValueId, Value<A>>,
    preds: FxHashMap<BlockId, SmallVec<[BlockId; 2]>>,
    loop_headers: FxHashSet<BlockId>,
    /// Executable edges, by source block and position in its terminator.
    executable: FxHashSet<(BlockId, usize)>,
    reached: FxHashSet<BlockId>,
    blocks_to_visit: Vec<BlockId>,
    risen: Vec<ValueId>,
}

impl<'a, A> Solver<'a, A>
where
    A: SparseAnalysis,
{
    fn new(analysis: &'a A, body: &'a Body) -> Self {
        let is_token = |value: &ValueId| body.is_token(*value);
        let mut facts: FxHashMap<ValueId, Value<A>> = body
            .values()
            .filter(|value| !is_token(value))
            .map(|value| (value, Value::<A>::bottom()))
            .collect();
        let outside = body
            .params()
            .iter()
            .chain(body.captures())
            .map(|(_, value)| *value)
            .chain(body.slots().iter().copied())
            .filter(|value| !is_token(value));
        for value in outside {
            facts.insert(value, analysis.domain().top());
        }
        Solver {
            analysis,
            body,
            facts,
            preds: body.predecessors(),
            loop_headers: body.loop_headers(),
            executable: FxHashSet::default(),
            reached: FxHashSet::default(),
            blocks_to_visit: Vec::new(),
            risen: Vec::new(),
        }
    }

    fn run(&mut self) {
        loop {
            if let Some(block) = self.blocks_to_visit.pop() {
                self.visit_block(block);
            } else if let Some(value) = self.risen.pop() {
                let sites: SmallVec<[Site; 4]> = self.body.users(value).collect();
                for site in sites {
                    self.revisit(site);
                }
            } else {
                return;
            }
        }
    }

    fn reach(&mut self, block: BlockId) {
        if self.reached.insert(block) {
            self.blocks_to_visit.push(block);
        }
    }

    fn visit_block(&mut self, block: BlockId) {
        self.eval_params(block);
        let body = self.body;
        for inst in body.block(block).insts() {
            self.eval_inst(inst.id());
        }
        self.eval_terminator(block);
    }

    /// A site in a block not reached yet is evaluated when the block is.
    fn revisit(&mut self, site: Site) {
        match site {
            Site::Inst(inst) if self.reached.contains(&self.body.block_of(inst)) => self.eval_inst(inst),
            Site::Term(block) if self.reached.contains(&block) => self.eval_terminator(block),
            Site::Inst(..) | Site::Term(_) => {}
        }
    }

    fn is_token(&self, value: ValueId) -> bool {
        self.body.is_token(value)
    }

    fn fact(&self, value: ValueId) -> &Value<A> {
        self.facts.get(&value).expect("the solver holds a fact for every value of its body but tokens")
    }

    /// The facts of `operands` that are not order tokens, in their order.
    fn operand_facts(&self, operands: &[ValueId]) -> SmallVec<[&Value<A>; 4]> {
        operands
            .iter()
            .filter(|value| !self.is_token(**value))
            .map(|value| self.fact(*value))
            .collect()
    }

    /// Joins `next` into `value`'s fact, or widens it by `next`, and queues
    /// `value`'s users when the fact rose.
    fn rise(&mut self, value: ValueId, next: &Value<A>, widen: bool) {
        let fact = self
            .facts
            .get_mut(&value)
            .expect("the solver holds a fact for every value of its body but tokens");
        let rose = if widen { fact.widen(next) } else { fact.join(next) };
        if rose {
            self.risen.push(value);
        }
    }

    /// Each parameter's fact is the join, over the executable edges into
    /// `block`, of its argument or, where the edge's terminator supplies it,
    /// of what the analysis says the terminator supplies; widened where
    /// `block` is a loop header.
    fn eval_params(&mut self, block: BlockId) {
        let body = self.body;
        let widen = self.loop_headers.contains(&block);
        let preds = self.preds.get(&block).expect("every block has a predecessor entry").clone();
        for (index, param) in body.block(block).params().iter().enumerate() {
            if self.is_token(*param) {
                continue;
            }
            let mut incoming = Value::<A>::bottom();
            for pred in &preds {
                let terminator = body.block(*pred).terminator();
                for (position, edge) in terminator.edges().iter().enumerate() {
                    if edge.to != block || !self.executable.contains(&(*pred, position)) {
                        continue;
                    }
                    match index.checked_sub(edge.supplied) {
                        Some(at) => {
                            incoming.join(self.fact(edge.args[at]));
                        }
                        None => {
                            let operands = terminator.operands();
                            let supplied =
                                self.analysis.supplied(terminator, block, index, &self.operand_facts(&operands));
                            incoming.join(&supplied);
                        }
                    }
                }
            }
            self.rise(*param, &incoming, widen);
        }
    }

    /// A result that is an order token, such as a `Merge`'s, is not
    /// evaluated, and the token an instruction yields has no fact.
    fn eval_inst(&mut self, id: InstId) {
        let body = self.body;
        let inst = body.inst(id);
        let Some(result) = inst.result().filter(|result| !self.is_token(*result)) else {
            return;
        };
        let operands = inst.operands();
        let next = self.analysis.transfer(inst, &self.operand_facts(&operands));
        self.rise(result, &next, false);
    }

    /// Marks the edges the analysis allows as executable and re-evaluates
    /// the parameters of every target they reach, whose arguments may have
    /// risen with this terminator's operands.
    fn eval_terminator(&mut self, block: BlockId) {
        let body = self.body;
        let terminator = body.block(block).terminator();
        let operands = terminator.operands();
        let allowed = self.analysis.executable(terminator, &self.operand_facts(&operands));
        for (position, edge) in terminator.edges().iter().enumerate() {
            let runs = match &allowed {
                Executable::All => true,
                Executable::Only(targets) => targets.contains(&edge.to),
            };
            if !runs {
                continue;
            }
            self.executable.insert((block, position));
            if self.reached.contains(&edge.to) {
                self.eval_params(edge.to);
            } else {
                self.reach(edge.to);
            }
        }
    }
}
