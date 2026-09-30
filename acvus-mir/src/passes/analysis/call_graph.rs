//! The module's call graph. Its nodes are the module's bodies. A body has an
//! edge to every body it makes a closure of, and to the body of every
//! closure it calls whose callee is a `MakeClosure` result it holds; the
//! second kind is always among the first, since such a callee is made in
//! the calling body. The strongly connected components are visited callees
//! first, by Tarjan's algorithm, which completes a component only after
//! every component it reaches.
//!
//! Cost: O(bodies + instructions) for the graph, O(bodies + edges) for its
//! components.

use rustc_hash::FxHashMap;

use crate::ir::cfg::{BodyId, Definition, Module, Op};
use crate::ir::{Callee, ValueId};

/// The module's bodies, by their place among `Module::body_ids`, each with
/// the bodies it makes closures of or calls through a closure it makes.
pub(crate) struct CallGraph {
    bodies: Vec<BodyId>,
    callees: Vec<Vec<usize>>,
}

impl CallGraph {
    /// Cost: O(bodies + instructions).
    pub(crate) fn of(module: &Module) -> CallGraph {
        let bodies: Vec<BodyId> = module.body_ids().collect();
        let place: FxHashMap<BodyId, usize> = bodies.iter().enumerate().map(|(at, body)| (*body, at)).collect();
        let callees = bodies
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
        CallGraph { bodies, callees }
    }

    /// The strongly connected components, each after every component its
    /// bodies reach (Tarjan 1972, with an explicit stack). Cost: O(bodies +
    /// edges).
    pub(crate) fn components_callees_first(&self) -> Vec<Vec<BodyId>> {
        let mut tarjan = Tarjan::new(self.callees.len());
        for root in 0..self.callees.len() {
            if tarjan.visited[root].is_none() {
                tarjan.walk_from(root, &self.callees);
            }
        }
        tarjan.components.into_iter().map(|component| component.into_iter().map(|at| self.bodies[at]).collect()).collect()
    }
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
