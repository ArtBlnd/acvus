//! Phase 5: Optimize
//!
//! Runs the full optimization pipeline on lowered MIR modules.

use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::inst_info;
use crate::analysis::raise::FunctionSummary;
use crate::cfg::{self, CfgBody};
use crate::graph::inliner;
use acvus_utils::QualifiedRef;
use crate::ir::{Callee, InstKind, Label, MirBody, MirModule, ValueId};
use crate::laws::LawTable;
use crate::optimize;
use crate::decl::Declarations;
use crate::ty::{Concrete, Ty, TyKind, Types};


#[cfg(debug_assertions)]
use crate::validate::type_check::ValidationErrorKind;
use crate::validate::{self, ValidationError};

/// Result of the optimization pipeline.
pub struct OptimizeResult {
    /// Optimized modules, keyed by function QualifiedRef.
    pub modules: FxHashMap<QualifiedRef, MirModule>,
    /// Validation errors per function (empty = valid).
    pub errors: Vec<(QualifiedRef, Vec<ValidationError>)>,
    /// Read off the code that survived the passes, which is what makes this
    /// the set RFC-0071 rule 5 calls required.
    pub inputs: FxHashMap<QualifiedRef, Vec<RequiredInput>>,
}

/// An input a module's body still reads after the passes, which a host
/// must pass for the module to run (programs §4).
#[derive(Debug, Clone, PartialEq)]
pub struct RequiredInput {
    pub name: QualifiedRef,
    pub ty: Ty,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Opt {
    /// Only what a program needs to reach the machine and mean what the
    /// source says.
    None,
    /// Every pass.
    Full,
}

pub fn optimize(
    interner: &Interner,
    laws: &LawTable,
    decls: &Declarations,
    modules: FxHashMap<QualifiedRef, MirModule>,
    opt: Opt,
) -> OptimizeResult {
    acvus_utils::grow(|| optimize_level(interner, laws, decls, modules, opt))
}

fn optimize_level(
    interner: &Interner,
    laws: &LawTable,
    decls: &Declarations,
    modules: FxHashMap<QualifiedRef, MirModule>,
    opt: Opt,
) -> OptimizeResult {
    // -- Pass 0: moves, borrows and exhaustiveness as the source wrote
    // them (RFC-0029, RFC-0051) --

    let mut all_errors = Vec::new();
    let mut recursive: FxHashSet<QualifiedRef> = FxHashSet::default();
    let sccs = call_graph_sccs(&modules);
    for members in &sccs {
        if Component::of(members, &modules).is_cyclic() {
            recursive.extend(members);
        }
        for &qref in members {
            let module = &modules[&qref];
            let mut errors = validate::move_check::check_moves(module);
            errors.extend(validate::borrow_check::check_outputs_and_borrows(module, decls));
            // A `match` is exhaustive (RFC-0051). Like the move check, it reads
            // the shape the source wrote, before any pass has moved it.
            errors.extend(validate::exhaustive::check_exhaustive(module));
            if !errors.is_empty() {
                all_errors.push((qref, errors));
            }
        }
    }

    let called: FxHashSet<QualifiedRef> = modules.values().flat_map(named_callees).collect();
    let functions = FunctionSummary::of(
        sccs.iter()
            .flatten()
            .filter(|qref| called.contains(qref))
            .map(|qref| (*qref, &modules[qref])),
        decls,
        laws,
    );

    // -- Pass 1: SSA (per-module) -> Inline (cross-module) -----

    let mut ssa_modules = modules;
    let inlined = match opt {
        Opt::None => inliner::InlineResult {
            modules: ssa_modules,
        },
        Opt::Full => {
            for module in ssa_modules.values_mut() {
                run_pass1_body(laws, decls, &functions, &module.types, &mut module.main);
                for closure in module.closures.values_mut() {
                    run_pass1_body(laws, decls, &functions, &module.types, closure);
                }
                // RFC-0099 rule 5: before the inliner, which then splices
                // each closure the fused loop alone calls (RFC-0060).
                // A declined pipeline runs its handlers, the program the
                // source wrote.
                let _declined: Vec<optimize::fusion::Declined> =
                    optimize::fusion::run(interner, laws, decls, module);
            }
            inliner::inline(&ssa_modules, &recursive)
        }
    };

    // -- Pass 2: Optimize + Validate (per-module, direct calls) ------

    let refused: FxHashSet<QualifiedRef> = all_errors.iter().map(|(qref, _)| *qref).collect();
    let mut result_modules = FxHashMap::default();
    let mut inputs = FxHashMap::default();

    let mut undropped: Vec<Undropped> = inlined
        .modules
        .into_iter()
        .map(|(qref, module)| Undropped::optimized(interner, laws, decls, &functions, qref, module, opt))
        .collect();
    settle_inputs(&mut undropped, laws, decls, &functions);

    for module in undropped {
        let qref = module.qref;
        inputs.insert(qref, required_inputs(&module.main));
        let module = module.finished(laws, decls);

        let mut errors = validate::type_check::check_types(&module, decls);
        errors.extend(validate::bounds::check_bounds(&module, laws));
        // RFC-0089 rules 1, 3 and 5, asked of the module after
        // `insert_drops`, which is the last pass: the form must hold on the
        // body the machine runs.
        errors.extend(validate::stages::check(&module, decls, laws));
        if !errors.is_empty() {
            all_errors.push((qref, errors));
        }
        if !refused.contains(&qref) {
            debug_rules_pass_0_held(qref, &module, decls);
        }

        result_modules.insert(qref, module);
    }

    OptimizeResult {
        modules: result_modules,
        errors: all_errors,
        inputs,
    }
}

/// A name every read of which a fold removed is absent here: it constrains
/// nothing, so its type closes to `!` and it is not required (RFC-0071
/// rule 5).
fn required_inputs(cfg: &CfgBody) -> Vec<RequiredInput> {
    let read = values_read(cfg);
    cfg.params
        .iter()
        .filter(|(_, slot)| read.contains(slot))
        .map(|(name, slot)| RequiredInput {
            name: QualifiedRef::root(*name),
            ty: *cfg
                .val_types
                .get(slot)
                .expect("lowering gives every parameter of a body its type"),
        })
        .collect()
}

fn named_callees(module: &MirModule) -> Vec<QualifiedRef> {
    let bodies = std::iter::once(&module.main).chain(module.closures.values());
    let mut callees: Vec<QualifiedRef> = bodies
        .flat_map(|body| &body.insts)
        .filter_map(|inst| match &inst.kind {
            InstKind::FunctionCall { callee, .. } | InstKind::Spawn { callee, .. } => {
                match callee {
                    Callee::Direct(id) => Some(*id),
                    Callee::Extern { .. } | Callee::Indirect(_) => None,
                }
            }
            _ => None,
        })
        .collect();
    callees.sort_unstable();
    callees.dedup();
    callees
}

/// The call graph's strongly connected components.
fn call_graph_sccs(modules: &FxHashMap<QualifiedRef, MirModule>) -> Vec<Vec<QualifiedRef>> {
    let mut ids: Vec<QualifiedRef> = modules.keys().copied().collect();
    ids.sort_unstable();
    let edges = ids
        .iter()
        .map(|qref| {
            let local = named_callees(&modules[qref])
                .into_iter()
                .filter(|callee| modules.contains_key(callee))
                .collect();
            (*qref, local)
        })
        .collect();
    Tarjan::sccs(&ids, &edges)
}

/// Tarjan's strongly connected components, in reverse topological order:
/// a component comes after every component it calls into.
struct Tarjan<'e> {
    edges: &'e FxHashMap<QualifiedRef, Vec<QualifiedRef>>,
    next_index: u32,
    stack: Vec<QualifiedRef>,
    on_stack: FxHashSet<QualifiedRef>,
    index: FxHashMap<QualifiedRef, u32>,
    lowlink: FxHashMap<QualifiedRef, u32>,
    components: Vec<Vec<QualifiedRef>>,
}

impl<'e> Tarjan<'e> {
    fn sccs(ids: &[QualifiedRef], edges: &'e FxHashMap<QualifiedRef, Vec<QualifiedRef>>) -> Vec<Vec<QualifiedRef>> {
        let mut tarjan = Tarjan {
            edges,
            next_index: 0,
            stack: Vec::new(),
            on_stack: FxHashSet::default(),
            index: FxHashMap::default(),
            lowlink: FxHashMap::default(),
            components: Vec::new(),
        };
        for &id in ids {
            if !tarjan.index.contains_key(&id) {
                tarjan.connect(id);
            }
        }
        tarjan.components
    }

    fn connect(&mut self, v: QualifiedRef) {
        acvus_utils::grow(|| self.connect_level(v))
    }

    fn connect_level(&mut self, v: QualifiedRef) {
        self.index.insert(v, self.next_index);
        self.lowlink.insert(v, self.next_index);
        self.next_index += 1;
        self.stack.push(v);
        self.on_stack.insert(v);

        let edges = self.edges;
        for &w in edges.get(&v).into_iter().flatten() {
            if !self.index.contains_key(&w) {
                self.connect(w);
                let low = self.lowlink[&v].min(self.lowlink[&w]);
                self.lowlink.insert(v, low);
            } else if self.on_stack.contains(&w) {
                let low = self.lowlink[&v].min(self.index[&w]);
                self.lowlink.insert(v, low);
            }
        }

        if self.lowlink[&v] == self.index[&v] {
            let mut component = Vec::new();
            while let Some(w) = self.stack.pop() {
                self.on_stack.remove(&w);
                component.push(w);
                if w == v {
                    break;
                }
            }
            self.components.push(component);
        }
    }
}

/// One strongly connected component of the call graph: whether its members
/// call one another is what the inliner is told, so a call inside its own
/// callee is left standing.
struct Component {
    /// For each member, by its index in the component, the members that
    /// call it.
    callers: Vec<Vec<usize>>,
}

impl Component {
    fn of(members: &[QualifiedRef], modules: &FxHashMap<QualifiedRef, MirModule>) -> Self {
        let mut callers = vec![Vec::new(); members.len()];
        for (at, caller) in members.iter().enumerate() {
            for callee in named_callees(&modules[caller]) {
                if let Some(called) = members.iter().position(|member| *member == callee) {
                    callers[called].push(at);
                }
            }
        }
        Self { callers }
    }

    fn is_cyclic(&self) -> bool {
        match self.callers.as_slice() {
            [only] => only.contains(&0),
            _ => true,
        }
    }
}

fn run_pass1_body(
    laws: &LawTable,
    decls: &Declarations,
    functions: &FunctionSummary,
    types: &Types<Concrete>,
    body: &mut crate::ir::MirBody,
) {
    let mut cfg = cfg::promote(std::mem::take(body));
    optimize::ssa_pass::run(&mut cfg);
    optimize::string_copy::run(&mut cfg, types, decls);
    optimize::dse::run(&mut cfg);
    optimize::dce::run(&mut cfg, types, decls, laws, functions);
    *body = cfg::demote(cfg);
}

/// Inputs settle across modules before any value is dropped, so the value
/// of an argument a call no longer passes is released by its caller's own
/// drops.
struct Undropped {
    qref: QualifiedRef,
    shell: MirModule,
    main: CfgBody,
    closures: Vec<(Label, CfgBody)>,
}

impl Undropped {
    fn optimized(
        interner: &Interner,
        laws: &LawTable,
        decls: &Declarations,
        functions: &FunctionSummary,
        qref: QualifiedRef,
        mut module: MirModule,
        opt: Opt,
    ) -> Undropped {
        let optimized = |body: MirBody, types: &mut Types<Concrete>| {
            let mut cfg = cfg::promote(body);
            match opt {
                Opt::None => run_pass2_required(interner, laws, functions, &mut cfg, types, decls),
                Opt::Full => run_pass2(interner, laws, functions, &mut cfg, types, decls),
            }
            cfg
        };
        let main = optimized(std::mem::take(&mut module.main), &mut module.types);
        let closures = std::mem::take(&mut module.closures)
            .into_iter()
            .map(|(label, closure)| (label, optimized(closure, &mut module.types)))
            .collect();
        Undropped {
            qref,
            shell: module,
            main,
            closures,
        }
    }

    /// Every body, and the table their types are ids of.
    fn bodies_mut(&mut self) -> (&mut Types<Concrete>, impl Iterator<Item = &mut CfgBody>) {
        (
            &mut self.shell.types,
            std::iter::once(&mut self.main).chain(self.closures.iter_mut().map(|(_, body)| body)),
        )
    }

    fn finished(self, laws: &LawTable, decls: &Declarations) -> MirModule {
        let Undropped {
            shell,
            main,
            closures,
            ..
        } = self;
        let mut shell = shell;
        let types = &mut shell.types;
        let mut finish = |mut cfg: CfgBody| {
            debug_validate(&cfg);
            let val_types = cfg.val_types.clone();
            optimize::drop_insertion::insert_drops(&mut cfg, types, decls, &val_types, laws);
            let mut body = cfg::demote(cfg);
            optimize::rejoin::run(&mut body);
            body
        };
        MirModule {
            main: finish(main),
            closures: closures
                .into_iter()
                .map(|(label, closure)| (label, finish(closure)))
                .collect(),
            ..shell
        }
    }
}

/// An input a module's body no longer reads after the folds is not required
/// and is no parameter (RFC-0071 rule 5). A call passes its callee's inputs,
/// so a parameter removed here is removed from every call to the module, and
/// a value a caller read only to pass it is unread in turn: the removal
/// repeats until no module loses one.
fn settle_inputs(undropped: &mut [Undropped], laws: &LawTable, decls: &Declarations, functions: &FunctionSummary) {
    loop {
        let mut removed: FxHashMap<QualifiedRef, Vec<usize>> = FxHashMap::default();
        for module in undropped.iter_mut() {
            let read = values_read(&module.main);
            let unread: Vec<usize> = (module.shell.declared_params..module.main.params.len())
                .filter(|at| !read.contains(&module.main.params[*at].1))
                .collect();
            for at in unread.iter().rev() {
                module.main.params.remove(*at);
            }
            if !unread.is_empty() {
                removed.insert(module.qref, unread);
            }
        }
        if removed.is_empty() {
            return;
        }
        for module in undropped.iter_mut() {
            let (types, bodies) = module.bodies_mut();
            for body in bodies {
                if strip_arguments(body, types, &removed) {
                    optimize::dce::run(body, types, decls, laws, functions);
                    // RFC-0089 rule 6: what `dce` removed may have freed a
                    // stage, and cutting again merges the boundaries it left.
                    optimize::stages::run(body, types, decls, laws);
                }
            }
        }
    }
}

fn values_read(cfg: &CfgBody) -> FxHashSet<ValueId> {
    let mut read: FxHashSet<ValueId> = FxHashSet::default();
    for block in &cfg.blocks {
        for inst in &block.insts {
            read.extend(inst_info::uses(&inst.kind));
            if let InstKind::Ref { target, .. } | InstKind::Take { target, .. } = &inst.kind {
                read.extend(inst_info::storage(target));
            }
        }
        read.extend(inst_info::terminator_uses(&block.terminator));
    }
    read
}

fn strip_arguments(
    cfg: &mut CfgBody,
    types: &mut Types<Concrete>,
    removed: &FxHashMap<QualifiedRef, Vec<usize>>,
) -> bool {

    let mut stripped = false;
    for inst in cfg.blocks.iter_mut().flat_map(|block| &mut block.insts) {
        let (InstKind::FunctionCall {
            callee: Callee::Direct(callee),
            callee_ty,
            args,
            ..
        }
        | InstKind::Spawn {
            callee: Callee::Direct(callee),
            callee_ty,
            args,
            ..
        }) = &mut inst.kind
        else {
            continue;
        };
        let Some(unread) = removed.get(callee) else {
            continue;
        };
        let TyKind::Fn {
            params,
            ret,
            effect,
            flows,
            calls,
            holds,
        } = types.kind(*callee_ty).clone()
        else {
            panic!("a call to {callee:?} is typed by its function type, not {callee_ty:?}")
        };
        let mut params = params.into_vec();
        for at in unread.iter().rev() {
            args.remove(*at);
            params.remove(*at);
        }
        *callee_ty = types.intern(TyKind::Fn {
            params: params.into_boxed_slice(),
            ret,
            effect,
            flows,
            calls,
            holds,
        });
        stripped = true;
    }
    stripped
}

/// Which inputs a body requires is a fact about the language and not an
/// optimization, so the two folds that decide it run at every level: a bound
/// `$` is a constant here as well, and the arms it decides against are gone
/// from both bodies alike (RFC-0071 rule 5).
fn run_pass2_required(
    interner: &Interner,
    laws: &LawTable,
    functions: &FunctionSummary,
    cfg: &mut CfgBody,
    types: &mut Types<Concrete>,
    decls: &Declarations,
) {
    optimize::ssa_pass::run(cfg);
    // A `String` copies (RFC-0018), and the copy is emitted here: without
    // it two names own one string and each drops it.
    optimize::string_copy::run(cfg, types, decls);
    optimize::reborrow::run(cfg, types);
    optimize::fold::run(cfg, types);
    optimize::branch::run(interner, cfg, types);
    optimize::dce::run(cfg, types, decls, laws, functions);
}

fn run_pass2(
    interner: &Interner,
    laws: &LawTable,
    functions: &FunctionSummary,
    cfg: &mut CfgBody,
    types: &mut Types<Concrete>,
    decls: &Declarations,
) {
    optimize::commute::run(cfg, types, decls);
    optimize::spawn_split::run(cfg, types);
    // RFC-0050: an aggregate no use lets out of the body never exists.
    // Before `ssa_pass`, whose builder places the phis its parts need;
    // before `dce`, which sweeps the constructor left with no reader.
    optimize::sroa::run(cfg, types);
    optimize::ssa_pass::run(cfg);
    optimize::string_copy::run(cfg, types, decls);
    // RFC-0055: after `ssa_pass`, which brings a constant and its reader
    // into one body; before `dse`/`dce`, which sweep the operands a fold
    // left with no reader.
    optimize::fold::run(cfg, types);
    // RFC-0071: after the fold, which is what makes a condition constant;
    // before `dse`/`dce`, which sweep what the arms it dropped had read.
    optimize::branch::run(interner, cfg, types);
    optimize::reborrow::run(cfg, types);
    optimize::dse::run(cfg);
    // RFC-0081: after `ssa_pass`, which makes the counter a header
    // parameter, and after the fold, which settles a constant bound; before
    // `dce`, which sweeps the comparison the new terminator leaves unread.
    optimize::while_to_for::run(cfg, types, decls, laws);
    optimize::dce::run(cfg, types, decls, laws, functions);
    optimize::code_motion::run(cfg, types, decls);
    // RFC-0066 rule 7: every induction variable of a `for` that anything
    // besides its own step reads is computed from the counter, decided per
    // variable; after the hoist, which leaves each loop's invariants above
    // its header.
    optimize::iv_canon::run(cfg, types, decls, laws, functions);
    // RFC-0083: after IV canonicalization, whose arithmetic it simplifies
    // and merges; before a `dce` of its own, which sweeps what it leaves
    // unread, the header arguments the canonicalization removed included.
    optimize::gvn::run(cfg, types);
    optimize::dce::run(cfg, types, decls, laws, functions);
    // RFC-0088: after that `dce`, which sweeps the arithmetic the loop passes
    // left in a body that nothing reads, so a body that does nothing holds
    // no instruction; before `forward`, which collapses the header the
    // removal leaves only jumping.
    optimize::empty_loop::run(cfg, types, decls, laws, functions);
    // A block that only jumps is its target; before `reorder`, which
    // schedules within a block.
    optimize::forward::run(cfg);
    optimize::reorder::run(cfg, types, decls);
    // RFC-0066 rule 1: an exit that leaves a nested loop's parent moves into
    // the nested loop's own exit, so the parent is cut around an exit of its
    // own (RFC-0089 rule 5); before the cut, which reads the exits, and
    // after `forward`, which would take the exit block's plain jump for it.
    optimize::nested_exit::run(interner, cfg, types, decls);
    // RFC-0089 rule 6: after every pass that moves or merges a body's
    // instructions -- `code_motion` and `forward` across blocks, `gvn`
    // merging, `reorder` within one block -- since a stage is a set of
    // instructions and a pass that moved one across a stage's boundary
    // would leave a dependence cycle crossing it after the fact; after
    // `iv_canon`, so a canonicalized induction variable is no token.
    optimize::stages::run(cfg, types, decls, laws);
    // RFC-0056, RFC-0066 rule 7: after the stages, since it reduces only a
    // counter expression that one `InOrder` stage reads, and puts the
    // derived counter's step in that stage. It adds instructions to the
    // preheader, the end of that stage and nowhere else, and moves none.
    optimize::lsr::run(cfg, types, decls, laws);
    // Before `bce`, which then reads the chain the stage pass wrote as the
    // order `validate::bounds` reads again, and before the drops, which it
    // places in the stage that touches each value.
    // RFC-0047 rule 7: after the loop passes, which leave a range `for`
    // whose counter indexes and one hoisted `as_slice`, and after the last
    // pass that moves an instruction, so that `validate::bounds` reads the
    // order this pass read; what follows adds only `Drop`s, which the
    // interval domain does not read as a write.
    optimize::bce::run(cfg, types, laws);
}

/// Until this assertion replaced it, pass 2 reported these two rules to the
/// reader. `let v = [1, 2]; let r = &v[0]; v = [3, 4]; *r` was therefore
/// refused twice: once by pass 0, with the borrow and the use labelled, and
/// once here with no labels at all, because the rewrites had moved the
/// instructions the labels are read off.
///
/// The other half of what is asserted lives in the passes between the two.
/// Three of them move an instruction past another — `optimize::commute`,
/// `optimize::code_motion`, `optimize::reorder` — and each builds `Loans` and
/// takes a storage's writes as a dependency, `code_motion` moving a shared
/// borrow only where no block it would newly span writes the storage. A pass
/// that stops asking `Loans` compiles, and this is what fires.
#[cfg(debug_assertions)]
fn debug_rules_pass_0_held(qref: QualifiedRef, module: &MirModule, decls: &Declarations) {
    let broken: Vec<ValidationErrorKind> = validate::borrow_check::check_borrows(module, decls)
        .into_iter()
        .chain(validate::exhaustive::check_exhaustive(module))
        .map(|error| error.kind)
        .collect();
    assert!(
        broken.is_empty(),
        "optimizing {qref:?} broke a rule pass 0 held: {broken:?}"
    );
}

#[cfg(not(debug_assertions))]
fn debug_rules_pass_0_held(_qref: QualifiedRef, _module: &MirModule, _decls: &Declarations) {}

/// Validate CfgBody after optimization: check use-def integrity, SSA dominance, and type coverage.
/// Collects all violations and panics if any are found.
#[cfg(debug_assertions)]
fn debug_validate(cfg: &CfgBody) {
    use crate::analysis::domtree::DomTree;
    use crate::cfg::BlockIdx;
    use crate::ir::ValueId;
    use rustc_hash::{FxHashMap, FxHashSet};

    let mut errors: Vec<String> = Vec::new();

    // -- Build def set and def locations --
    let mut defs: FxHashSet<ValueId> = FxHashSet::default();
    let mut def_loc: FxHashMap<ValueId, (usize, usize)> = FxHashMap::default();

    // Function params/captures: defined "before" block 0.
    for v in cfg.entry_defs() {
        defs.insert(v);
        def_loc.insert(v, (0, usize::MAX));
    }

    for (bi, block) in cfg.blocks.iter().enumerate() {
        for &p in &block.params {
            defs.insert(p);
            def_loc.insert(p, (bi, usize::MAX));
        }
        for (ii, inst) in block.insts.iter().enumerate() {
            for d in crate::analysis::inst_info::defs(&inst.kind) {
                defs.insert(d);
                def_loc.insert(d, (bi, ii));
            }
        }
    }

    let domtree = DomTree::build(cfg);

    for (bi, block) in cfg.blocks.iter().enumerate() {
        // -- Check instructions --
        for (ii, inst) in block.insts.iter().enumerate() {
            // Type coverage: every def and use must have a type.
            for d in crate::analysis::inst_info::defs(&inst.kind) {
                if !cfg.val_types.contains_key(&d) {
                    errors.push(format!(
                        "B{bi}:{ii} DEF missing type: {d:?} in {:?}",
                        inst.kind
                    ));
                }
            }
            for u in crate::analysis::inst_info::uses(&inst.kind) {
                if !cfg.val_types.contains_key(&u) {
                    errors.push(format!(
                        "B{bi}:{ii} USE missing type: {u:?} in {:?}",
                        inst.kind
                    ));
                }
                // Use-def: every use must have a def.
                let Some(&(def_bi, def_ii)) = def_loc.get(&u) else {
                    errors.push(format!(
                        "B{bi}:{ii} use without def: {u:?} in {:?}",
                        inst.kind
                    ));
                    continue;
                };
                // SSA dominance.
                if def_bi == bi {
                    if def_ii != usize::MAX && def_ii >= ii {
                        errors.push(format!(
                            "B{bi}:{ii} ORDER VIOLATION (same block): \
                             use {u:?} at inst {ii}, def at inst {def_ii} in {:?}",
                            inst.kind
                        ));
                    }
                } else if !domtree.dominates(BlockIdx(def_bi), BlockIdx(bi)) {
                    errors.push(format!(
                        "B{bi}:{ii} DOMINANCE VIOLATION: \
                         use {u:?} in B{bi}, def in B{def_bi} (not dominator) in {:?}",
                        inst.kind
                    ));
                }
            }
        }

        // -- Check terminator uses --
        let term_uses = match &block.terminator {
            crate::cfg::Terminator::Return { value, order, .. } => {
                std::iter::once(*value).chain(*order).collect()
            }
            crate::cfg::Terminator::Jump { args, .. } => args.clone(),
            crate::cfg::Terminator::JumpIf {
                cond,
                then_args,
                else_args,
                ..
            }
            | crate::cfg::Terminator::Diamond {
                cond,
                then_args,
                else_args,
                ..
            } => {
                let mut v = vec![*cond];
                v.extend(then_args);
                v.extend(else_args);
                v
            }
            crate::cfg::Terminator::For {
                source, exit_args, ..
            } => {
                let mut v = source.uses().to_vec();
                v.extend(exit_args);
                v
            }
            crate::cfg::Terminator::While {
                cond, exit_args, ..
            } => {
                let mut v = vec![*cond];
                v.extend(exit_args);
                v
            }
            crate::cfg::Terminator::Switch { tag, arms, default } => {
                let mut v = vec![*tag];
                for (_, _, args) in arms {
                    v.extend(args);
                }
                if let Some((_, args)) = default {
                    v.extend(args);
                }
                v
            }
            crate::cfg::Terminator::Fallthrough | crate::cfg::Terminator::Diverge => vec![],
        };
        for u in &term_uses {
            if !defs.contains(u) {
                errors.push(format!(
                    "B{bi} TERM use without def: {u:?} in {:?}",
                    block.terminator
                ));
            }
        }
    }

    if !errors.is_empty() {
        let msg = errors.join("\n  ");
        let mut dump = String::new();
        for (bi, block) in cfg.blocks.iter().enumerate() {
            dump.push_str(&format!("B{bi} params={:?}\n", block.params));
            for (ii, inst) in block.insts.iter().enumerate() {
                dump.push_str(&format!("  {ii}: {:?}\n", inst.kind));
            }
            dump.push_str(&format!("  -> {:?}\n", block.terminator));
        }
        panic!(
            "CfgBody validation failed ({} errors):\n  {msg}\n{dump}",
            errors.len()
        );
    }
}

#[cfg(not(debug_assertions))]
fn debug_validate(_cfg: &CfgBody) {}
