//! The module's call graph. Its nodes are the module's bodies. A body has an
//! edge to every body it makes a closure of, and to the body of every
//! closure it calls whose callee is a `MakeClosure` result it holds; the
//! second kind is always among the first, since such a callee is made in
//! the calling body. The strongly connected components are visited callees
//! first, by Tarjan's algorithm, which completes a component only after
//! every component it reaches.
//!
//! The manager keeps it once per module, as a module analysis: it reads
//! the bodies' instructions, so an edit of any body's instructions drops
//! it, and it ranges over the module's bodies, so an added body drops it.
//!
//! Cost: O(bodies + instructions) for the graph and its components, since
//! each edge is one instruction's.

use rustc_hash::FxHashMap;

use crate::ir::cfg::{Aspect, BodyId, Definition, Module, Op};
use crate::ir::{Callee, ValueId};
use crate::passes::analysis::{AnalysisManager, ModuleAnalysis};

/// The module's bodies, by their place among `Module::body_ids`, each with
/// the bodies it makes closures of or calls through a closure it makes, and
/// the graph's strongly connected components, callees first.
pub(crate) struct CallGraph {
    components: Vec<Vec<BodyId>>,
    /// Each body's component, by its index in `components`.
    component_of: FxHashMap<BodyId, usize>,
}

impl ModuleAnalysis for CallGraph {
    type Facts = CallGraph;

    const READS: &'static [Aspect] = &[Aspect::Instructions];

    fn compute(_am: &mut AnalysisManager<'_>, module: &Module) -> CallGraph {
        CallGraph::of(module)
    }
}

impl CallGraph {
    /// Cost: O(bodies + instructions).
    fn of(module: &Module) -> CallGraph {
        let bodies: Vec<BodyId> = module.body_ids().collect();
        let place: FxHashMap<BodyId, usize> = bodies.iter().enumerate().map(|(at, body)| (*body, at)).collect();
        let callees: Vec<Vec<usize>> = bodies
            .iter()
            .map(|caller| {
                let body = module.body(*caller);
                let mut callees: Vec<usize> = Vec::new();
                for inst in body.blocks().flat_map(|(_, block)| block.insts()) {
                    let made: Vec<BodyId> = match inst.op() {
                        Op::MakeClosure { bodies, .. } => bodies.clone(),
                        Op::FunctionCall { callee: Callee::Indirect(function), .. } => closure_body(module, *caller, *function).into_iter().collect(),
                        _ => Vec::new(),
                    };
                    callees.extend(made.iter().map(|callee| place[callee]));
                }
                callees.sort_unstable();
                callees.dedup();
                callees
            })
            .collect();
        let components = components_callees_first(&bodies, &callees);
        let component_of = components
            .iter()
            .enumerate()
            .flat_map(|(at, component)| component.iter().map(move |body| (*body, at)))
            .collect();
        CallGraph { components, component_of }
    }

    /// The strongly connected components, each after every component its
    /// bodies reach. Cost: O(1).
    pub(crate) fn components_callees_first(&self) -> &[Vec<BodyId>] {
        &self.components
    }

    /// Whether `a` and `b` lie in one strongly connected component. Cost:
    /// O(1).
    ///
    /// # Panics
    /// Either is a body the graph was not made over: the manager drops the
    /// graph when a seal adds a body, so this is a module whose body set
    /// changed without `AnalysisManager::invalidate_body_set`, a compiler
    /// defect.
    pub(crate) fn same_component(&self, a: BodyId, b: BodyId) -> bool {
        let component = |body: BodyId| {
            *self.component_of.get(&body).unwrap_or_else(|| {
                panic!(
                    "{body:?} is not among the bodies the call graph was made over; a kept module analysis ranges over every body, so the body set changed without AnalysisManager::invalidate_body_set"
                )
            })
        };
        component(a) == component(b)
    }
}

/// The strongly connected components of `callees` over `bodies`, each after
/// every component its bodies reach (Tarjan 1972, with an explicit stack).
/// Cost: O(bodies + edges).
fn components_callees_first(bodies: &[BodyId], callees: &[Vec<usize>]) -> Vec<Vec<BodyId>> {
    let mut tarjan = Tarjan::new(callees.len());
    for root in 0..callees.len() {
        if tarjan.visited[root].is_none() {
            tarjan.walk_from(root, callees);
        }
    }
    tarjan.components.into_iter().map(|component| component.into_iter().map(|at| bodies[at]).collect()).collect()
}

/// Tarjan's walk: each node's visit order and the lowest visit order it
/// reaches on the stack, and the components completed so far.
struct Tarjan {
    visited: Vec<Option<usize>>,
    low: Vec<usize>,
    on_stack: Vec<bool>,
    stack: Vec<usize>,
    next: usize,
    components: Vec<Vec<usize>>,
}

/// A node on the walk, and the index of its next edge to follow.
struct Frame {
    node: usize,
    next_edge: usize,
}

impl Tarjan {
    fn new(count: usize) -> Tarjan {
        Tarjan {
            visited: vec![None; count],
            low: vec![0; count],
            on_stack: vec![false; count],
            stack: Vec::new(),
            next: 0,
            components: Vec::new(),
        }
    }

    fn enter(&mut self, node: usize) -> Frame {
        self.visited[node] = Some(self.next);
        self.low[node] = self.next;
        self.next += 1;
        self.stack.push(node);
        self.on_stack[node] = true;
        Frame { node, next_edge: 0 }
    }

    fn walk_from(&mut self, root: usize, edges: &[Vec<usize>]) {
        let mut walk = vec![self.enter(root)];
        while let Some(frame) = walk.last_mut() {
            let node = frame.node;
            if let Some(callee) = edges[node].get(frame.next_edge).copied() {
                frame.next_edge += 1;
                match self.visited[callee] {
                    None => {
                        let entered = self.enter(callee);
                        walk.push(entered);
                    }
                    Some(order) if self.on_stack[callee] => self.low[node] = self.low[node].min(order),
                    Some(_) => {}
                }
                continue;
            }
            walk.pop();
            if let Some(caller) = walk.last() {
                self.low[caller.node] = self.low[caller.node].min(self.low[node]);
            }
            if self.visited[node] == Some(self.low[node]) {
                self.complete(node);
            }
        }
    }

    /// Pops the component whose root is `root`.
    fn complete(&mut self, root: usize) {
        let mut component = Vec::new();
        loop {
            let member = self.stack.pop().expect("a component's root is on the stack below its members");
            self.on_stack[member] = false;
            component.push(member);
            if member == root {
                break;
            }
        }
        self.components.push(component);
    }
}

/// The body of the closure `function` is, where `caller` makes it.
fn closure_body(module: &Module, caller: BodyId, function: ValueId) -> Option<BodyId> {
    let body = module.body(caller);
    let Definition::Inst(made) = body.definition(function) else {
        return None;
    };
    match body.inst(made).op() {
        Op::MakeClosure { bodies, .. } => match bodies[..] {
            [one] => Some(one),
            _ => None,
        },
        _ => None,
    }
}
