//! Phase 5: Optimize
//!
//! Runs the full optimization pipeline on lowered MIR modules.

use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::inst_info;
use crate::cfg::{self, CfgBody};
use crate::graph::inliner;
use crate::graph::{ContextInfo, QualifiedRef};
use crate::ir::{Callee, InstKind, MirBody, MirModule, ValueId};
use crate::laws::LawTable;
use crate::optimize;
use crate::ty::Ty;

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
    pub inputs: FxHashMap<QualifiedRef, Vec<ContextInfo>>,
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
    modules: FxHashMap<QualifiedRef, MirModule>,
    opt: Opt,
) -> OptimizeResult {
    // -- Pass 0: moves, borrows and exhaustiveness as the source wrote
    // them (RFC-0029, RFC-0051) --

    let mut all_errors = Vec::new();
    let mut recursive: FxHashSet<QualifiedRef> = FxHashSet::default();
    for members in call_graph_sccs(&modules) {
        if Component::of(&members, &modules).is_cyclic() {
            recursive.extend(&members);
        }
        for qref in members {
            let module = &modules[&qref];
            let mut errors = validate::move_check::check_moves(module);
            errors.extend(validate::borrow_check::check_outputs_and_borrows(module));
            // A `match` is exhaustive (RFC-0051). Like the move check, it reads
            // the shape the source wrote, before any pass has moved it.
            errors.extend(validate::exhaustive::check_exhaustive(module));
            if !errors.is_empty() {
                all_errors.push((qref, errors));
            }
        }
    }

    // -- Pass 1: SSA (per-module) -> Inline (cross-module) -----

    let mut ssa_modules = modules;
    let inlined = match opt {
        Opt::None => inliner::InlineResult {
            modules: ssa_modules,
        },
        Opt::Full => {
            for module in ssa_modules.values_mut() {
                run_pass1_body(&mut module.main);
                for closure in module.closures.values_mut() {
                    run_pass1_body(closure);
                }
            }
            inliner::inline(&ssa_modules, &recursive)
        }
    };

    // -- Pass 2: Optimize + Validate (per-module, direct calls) ------

    let refused: FxHashSet<QualifiedRef> = all_errors.iter().map(|(qref, _)| *qref).collect();
    let mut result_modules = FxHashMap::default();
    let mut inputs = FxHashMap::default();

    for (qref, mut module) in inlined.modules {
        run_pass2_body(interner, laws, &mut module.main, opt);
        for closure in module.closures.values_mut() {
            run_pass2_body(interner, laws, closure, opt);
        }
        inputs.insert(qref, required_inputs(&module.main));

        let mut errors = validate::type_check::check_types(&module);
        errors.extend(validate::bounds::check_bounds(&module, laws));
        if !errors.is_empty() {
            all_errors.push((qref, errors));
        }
        if !refused.contains(&qref) {
            debug_rules_pass_0_held(qref, &module);
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
fn required_inputs(body: &MirBody) -> Vec<ContextInfo> {
    let mut read: FxHashSet<ValueId> = FxHashSet::default();
    for inst in &body.insts {
        read.extend(inst_info::uses(&inst.kind));
        if let InstKind::Ref { target, .. } | InstKind::Take { target, .. } = &inst.kind {
            read.extend(inst_info::storage(target));
        }
    }
    body.params
        .iter()
        .filter(|(_, slot)| read.contains(slot))
        .map(|(name, slot)| ContextInfo {
            name: QualifiedRef::root(*name),
            ty: input_ty(body, *slot),
        })
        .collect()
}

fn input_ty(body: &MirBody, slot: ValueId) -> Ty {
    body.val_types
        .get(&slot)
        .cloned()
        .expect("lowering gives every parameter of a body its type")
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
    crate::graph::infer::tarjan_scc(&ids, &edges)
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

fn run_pass1_body(body: &mut crate::ir::MirBody) {
    let mut cfg = cfg::promote(std::mem::take(body));
    optimize::ssa_pass::run(&mut cfg);
    optimize::string_copy::run(&mut cfg);
    optimize::dse::run(&mut cfg);
    optimize::dce::run(&mut cfg);
    *body = cfg::demote(cfg);
}

fn run_pass2_body(interner: &Interner, laws: &LawTable, body: &mut crate::ir::MirBody, opt: Opt) {
    let mut cfg = cfg::promote(std::mem::take(body));
    match opt {
        Opt::None => run_pass2_required(interner, &mut cfg),
        Opt::Full => run_pass2(interner, laws, &mut cfg),
    }
    *body = cfg::demote(cfg);
    optimize::rejoin::run(body);
}

/// Which inputs a body requires is a fact about the language and not an
/// optimization, so the two folds that decide it run at every level: a bound
/// `$` is a constant here as well, and the arms it decides against are gone
/// from both bodies alike (RFC-0071 rule 5).
fn run_pass2_required(interner: &Interner, cfg: &mut CfgBody) {
    optimize::ssa_pass::run(cfg);
    // A `String` copies (RFC-0018), and the copy is emitted here: without
    // it two names own one string and each drops it.
    optimize::string_copy::run(cfg);
    optimize::reborrow::run(cfg);
    optimize::fold::run(cfg);
    optimize::branch::run(interner, cfg);
    optimize::dce::run(cfg);
    debug_validate(cfg);
    optimize::drop_insertion::insert_drops(cfg, &cfg.val_types.clone());
}

fn run_pass2(interner: &Interner, laws: &LawTable, cfg: &mut CfgBody) {
    optimize::commute::run(cfg);
    optimize::spawn_split::run(cfg);
    // RFC-0050: an aggregate no use lets out of the body never exists.
    // Before `ssa_pass`, whose builder places the phis its parts need;
    // before `dce`, which sweeps the constructor left with no reader.
    optimize::sroa::run(cfg);
    optimize::ssa_pass::run(cfg);
    optimize::string_copy::run(cfg);
    // RFC-0055: after `ssa_pass`, which brings a constant and its reader
    // into one body; before `dse`/`dce`, which sweep the operands a fold
    // left with no reader.
    optimize::fold::run(cfg);
    // RFC-0071: after the fold, which is what makes a condition constant;
    // before `dse`/`dce`, which sweep what the arms it dropped had read.
    optimize::branch::run(interner, cfg);
    optimize::reborrow::run(cfg);
    optimize::dse::run(cfg);
    // RFC-0081: after `ssa_pass`, which makes the counter a header
    // parameter, and after the fold, which settles a constant bound; before
    // `dce`, which sweeps the comparison the new terminator leaves unread.
    optimize::while_to_for::run(cfg);
    optimize::dce::run(cfg);
    optimize::code_motion::run(cfg);
    // RFC-0066 rule 7: the weak loops' normal form, before `lsr` reduces
    // the strong ones; after the hoist, which leaves each loop's invariants
    // above its header.
    optimize::iv_canon::run(cfg, laws);
    // RFC-0056: after the hoist, which puts a loop's invariants above the
    // header and leaves the preheader a block of its own; before the
    // reorder, which schedules within a block.
    optimize::lsr::run(cfg, laws);
    // RFC-0083: after both loop passes, whose arithmetic it simplifies and
    // merges; before a `dce` of its own, which sweeps what it leaves unread.
    optimize::gvn::run(cfg);
    optimize::dce::run(cfg);
    // RFC-0088: after that `dce`, which sweeps the arithmetic the loop passes
    // left in a body that nothing reads, so a body that does nothing holds
    // no instruction; before `forward`, which collapses the header the
    // removal leaves only jumping.
    optimize::empty_loop::run(cfg);
    // A block that only jumps is its target: after `lsr`, which writes a
    // reduction into the preheader `code_motion` may have left empty;
    // before `reorder`, which schedules within a block.
    optimize::forward::run(cfg);
    optimize::reorder::run(cfg);
    // RFC-0047 rule 7: after the loop passes, which leave a range `for`
    // whose counter indexes and one hoisted `as_slice`, and after the last
    // pass that moves an instruction, so that `validate::bounds` reads the
    // order this pass read; what follows adds only `Drop`s, which the
    // interval domain does not read as a write.
    optimize::bce::run(cfg, laws);
    debug_validate(cfg);
    optimize::drop_insertion::insert_drops(cfg, &cfg.val_types.clone());
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
fn debug_rules_pass_0_held(qref: QualifiedRef, module: &MirModule) {
    let broken: Vec<ValidationErrorKind> = validate::borrow_check::check_borrows(module)
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
fn debug_rules_pass_0_held(_qref: QualifiedRef, _module: &MirModule) {}

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
                source,
                body_args,
                exit_args,
                ..
            } => {
                let mut v = source.uses().to_vec();
                v.extend(body_args);
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
