//! Recording (l.224–238): the constraints a body's text gives, recorded in the
//! equivalence classes as they are given, whatever is solved (l.230) — a
//! meeting as a lower bound and a boundary as `T ⇝ U` (l.226), an invariant
//! place as an equation, and a use, a lend, a projection and a binding as a
//! node — and the classes of a type a declaration writes (l.127, l.218).

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::class::{ClassId, ClassName, KindSource};
use crate::instance::Instance;
use crate::node::{NodeKind, NodeState};
use crate::program::{Function, Item, Origin, Place, Program, TextClass};
use crate::relation::{Recorded, Recorder, Relation};
use crate::sig::{SigTy, SigVar};
use crate::solve::State;
use crate::ty::{Name, PartKey, Term, Ty};

impl State {
    /// Records the relation, which the equivalence classes take at once
    /// (l.230).
    pub(crate) fn record(&mut self, relation: Relation, recorder: Recorder) {
        self.hold(&relation);
        self.log.push(Recorded { relation, recorder });
    }

    /// The types the text gives its classes: a literal's LitVar, a type the
    /// text or the host writes, and a lambda's function value (l.230).
    pub(crate) fn record_origins(&mut self, program: &Program, classes: &[&TextClass]) {
        for class in classes {
            let id = self.text(&class.name);
            match &class.origin {
                Origin::Var => {}
                Origin::Lit(lit) => self.record(Relation::Lit(id, lit.clone()), Recorder::Origin),
                Origin::Written(term) => {
                    let term = term.clone().map(|_, part| self.text(&part));
                    let relation = Relation::Kind(id, term, KindSource::Written);
                    self.record(relation, Recorder::Origin);
                }
                Origin::Host(term) => {
                    self.hosts.insert(id);
                    let term = term.clone().map(|_, part| self.text(&part));
                    let relation = Relation::Kind(id, term, KindSource::Written);
                    self.record(relation, Recorder::Origin);
                }
                Origin::Lambda(lambda) => {
                    let function = program.function(lambda).expect("the lambda is a function");
                    let arity = function.params.len();
                    let term = Term::FunctionValue { arity };
                    let relation = Relation::Kind(id, term, KindSource::Written);
                    self.record(relation, Recorder::Origin);
                    self.record(Relation::Values(id, lambda.clone()), Recorder::Origin);
                }
            }
        }
    }

    /// The equations the text's invariant places give (l.236).
    pub(crate) fn record_equations(&mut self, items: &[(usize, &Item)]) {
        for &(i, item) in items {
            if let Item::Equation(a, b) = item {
                let relation = Relation::Equation(self.text(a), self.text(b));
                self.record(relation, Recorder::Text(i));
            }
        }
    }

    /// The rest of the text: each value passing into a type, as a meeting or
    /// a boundary (l.226, l.234–235), each call of a function (l.199–202), and
    /// each other use, lend, projection and binding, as a node (l.238).
    pub(crate) fn record_items(
        &mut self,
        program: &Program,
        items: &[(usize, &Item)],
        solved: &BTreeMap<Name, Rc<State>>,
    ) {
        for &(i, item) in items {
            match item {
                Item::Pass { value, into, place } => {
                    let (value, into) = (self.text(value), self.text(into));
                    let meeting = match place {
                        Place::ThroughRef => false,
                        Place::ByName => {
                            self.is_unknown(into) || self.enclosing_unknowns.contains(&into)
                        }
                        Place::Slot => self.is_unknown(into),
                    };
                    let relation = if meeting {
                        Relation::LowerBound {
                            lower: value,
                            upper: into,
                        }
                    } else {
                        Relation::Boundary {
                            from: value,
                            to: into,
                        }
                    };
                    self.record(relation, Recorder::Text(i));
                }
                Item::CallValue {
                    name,
                    callee,
                    args,
                    result,
                } => {
                    let kind = NodeKind::CallValue {
                        callee: self.text(callee),
                        args: args.iter().map(|arg| self.text(arg)).collect(),
                        result: self.text(result),
                    };
                    self.add_node(name, kind);
                }
                Item::Use {
                    callee,
                    args,
                    result,
                    ..
                } if program.function(callee).is_some() => {
                    let function = program.function(callee).expect("the function is declared");
                    let args: Vec<ClassId> = args.iter().map(|arg| self.text(arg)).collect();
                    let result = self.text(result);
                    self.call(i, function, &args, result, solved);
                }
                Item::Use {
                    name,
                    callee,
                    args,
                    result,
                } => {
                    let kind = NodeKind::Use {
                        callee: callee.clone(),
                        args: args.iter().map(|arg| self.text(arg)).collect(),
                        result: self.text(result),
                    };
                    self.add_node(name, kind);
                }
                Item::Lend {
                    name,
                    mutability,
                    place,
                    result,
                } => {
                    let kind = NodeKind::Lend {
                        mutability: *mutability,
                        place: self.text(place),
                        result: self.text(result),
                    };
                    self.add_node(name, kind);
                }
                Item::Project {
                    name,
                    projection,
                    operand,
                    result,
                } => {
                    let kind = NodeKind::Project {
                        projection: projection.clone(),
                        operand: self.text(operand),
                        result: self.text(result),
                    };
                    self.add_node(name, kind);
                }
                Item::Bind {
                    name,
                    path,
                    position,
                    binding,
                } => {
                    let kind = NodeKind::Bind {
                        path: path.clone(),
                        position: self.text(position),
                        binding: self.text(binding),
                    };
                    self.add_node(name, kind);
                }
                Item::Equation(..) => {}
            }
        }
    }

    pub(crate) fn add_node(&mut self, name: &Name, kind: NodeKind) {
        let node = NodeState {
            kind,
            decision: None,
            shared: BTreeSet::new(),
            references: BTreeSet::new(),
        };
        if self.nodes.insert(name.clone(), node).is_some() {
            panic!("the program names the node {name} once");
        }
    }

    /// A use that calls a function (l.199–202). Within the function's group it
    /// takes that body's own types: each argument passes into its parameter's
    /// type, universal and so a boundary (Example 10), and the body's result
    /// type enters the use's result. From another group it makes an instance
    /// of the group's C (l.222).
    fn call(
        &mut self,
        item: usize,
        function: &Function,
        args: &[ClassId],
        result: ClassId,
        solved: &BTreeMap<Name, Rc<State>>,
    ) {
        let recorder = Recorder::Text(item);
        if function.group == self.group || self.enclosing.contains(&function.group) {
            let params: Vec<ClassId> = function.params.iter().map(|p| self.text(p)).collect();
            let returned = self.text(&function.result);
            self.record_call(args, &params, returned, result, &recorder);
            return;
        }
        let from = Rc::clone(
            solved
                .get(&function.group)
                .unwrap_or_else(|| panic!("{} is solved first (l.220)", function.group)),
        );
        let at = Name::new(&format!("{}@{item}", function.name));
        let mut copy = Instance::new(&at, recorder.clone(), &from);
        let params: Vec<ClassId> = function
            .params
            .iter()
            .map(|param| copy.class(self, from.text(param)))
            .collect();
        let returned = copy.class(self, from.text(&function.result));
        copy.record_rest(self);
        self.record_call(args, &params, returned, result, &recorder);
    }

    pub(crate) fn record_call(
        &mut self,
        args: &[ClassId],
        params: &[ClassId],
        returned: ClassId,
        result: ClassId,
        recorder: &Recorder,
    ) {
        for (&arg, &param) in args.iter().zip(params) {
            let relation = if self.is_unknown(param) {
                Relation::LowerBound {
                    lower: arg,
                    upper: param,
                }
            } else {
                Relation::Boundary {
                    from: arg,
                    to: param,
                }
            };
            self.record(relation, recorder.clone());
        }
        self.record(
            Relation::LowerBound {
                lower: returned,
                upper: result,
            },
            recorder.clone(),
        );
    }

    /// A class made anew that holds `term` as a declaration writes it.
    pub(crate) fn written(
        &mut self,
        name: ClassName,
        term: Term<ClassId>,
        recorder: &Recorder,
    ) -> ClassId {
        let class = self.arena.intern(name);
        self.classes.grow(&self.arena);
        self.decided.insert(class);
        self.record(
            Relation::Kind(class, term, KindSource::Written),
            recorder.clone(),
        );
        class
    }

    /// The classes of a type a declaration writes: the type named `name`, its
    /// parts named after it, and its variables those of `vars`, each made
    /// where first met and named by `var_name`. A variable is an unknown, and
    /// a written term is decided (l.218).
    pub(crate) fn instantiate<F>(
        &mut self,
        ty: &SigTy,
        vars: &mut BTreeMap<SigVar, ClassId>,
        name: ClassName,
        var_name: &F,
        recorder: &Recorder,
    ) -> ClassId
    where
        F: Fn(SigVar) -> ClassName,
    {
        match ty {
            SigTy::Var(var) => {
                if let Some(&class) = vars.get(var) {
                    return class;
                }
                let class = self.arena.intern(var_name(*var));
                self.classes.grow(&self.arena);
                vars.insert(*var, class);
                class
            }
            SigTy::Term(term) => {
                let shared = Rc::new(name.clone());
                let term = (**term).clone().map(|key: PartKey, part| {
                    let part_name = ClassName::Part(shared.clone(), key);
                    self.instantiate(&part, vars, part_name, var_name, recorder)
                });
                self.written(name, term, recorder)
            }
        }
    }

    /// A class made anew that holds the closed type `ty`, written.
    pub(crate) fn closed(&mut self, name: ClassName, ty: &Ty, recorder: &Recorder) -> ClassId {
        let shared = Rc::new(name.clone());
        let term = ty
            .term()
            .clone()
            .map(|key, part| self.closed(ClassName::Part(shared.clone(), key), &part, recorder));
        self.written(name, term, recorder)
    }
}
