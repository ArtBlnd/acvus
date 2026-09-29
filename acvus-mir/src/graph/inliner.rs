//! Phase 4: Inliner
//!
//! Inlines local function calls into a single flat MIR body. Which call is
//! one: [`direct_target`] for `Callee::Direct`, [`ClosurePlan`] for
//! `Callee::Indirect`.
//!
//! After inlining, re-run SSABuilder to deduplicate context loads and
//! insert PHIs at merge points.

use acvus_utils::LocalIdOps;
use std::borrow::Cow;

use rustc_hash::{FxHashMap, FxHashSet};


use crate::analysis::{escape, inst_info};
use crate::ir::*;
use crate::ty::{Concrete, Mutability, ObjectTy, Ty, TyKind, TypeArg, Types};

use acvus_utils::QualifiedRef;

/// Result of the inlining pass.
#[derive(Debug)]
pub struct InlineResult {
    /// Inlined MIR per top-level function.
    pub modules: FxHashMap<QualifiedRef, MirModule>,
}

/// Inline all local function calls within each top-level function's MIR.
///
/// `modules`: per-function MIR from the lower phase.
/// `recursive_fns`: functions that should NOT be inlined (recursive / mutual recursive).
///
/// Every module's calls are inlined from the bodies the modules had on
/// entry. The types a spliced callee names are interned into the caller's
/// own table; interning only adds, so the caller's entry bodies stay
/// readable in it while the others are inlined.
pub fn inline(
    mut modules: FxHashMap<QualifiedRef, MirModule>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> InlineResult {
    let mut ids: Vec<QualifiedRef> = modules.keys().copied().collect();
    ids.sort_unstable();
    let mut inlined = Vec::with_capacity(ids.len());
    for id in ids {
        let mut module = modules.remove(&id).expect("each id is a key of the map");
        let bodies = inline_module(&mut module, &modules, recursive_fns);
        modules.insert(id, module);
        inlined.push((id, bodies));
    }
    for (id, bodies) in inlined {
        let module = modules.get_mut(&id).expect("each id is a key of the map");
        module.main = bodies.main;
        module.closures = bodies.closures;
    }
    InlineResult { modules }
}

/// A module's bodies with their calls inlined.
struct Inlined {
    main: MirBody,
    closures: FxHashMap<Label, MirBody>,
}

/// Inline all eligible calls within a single MirModule, interning what the
/// spliced callees name into its table. A call of the module itself is not
/// among `all_modules`, and is left standing: it is recursive.
fn inline_module(
    module: &mut MirModule,
    all_modules: &FxHashMap<QualifiedRef, MirModule>,
    recursive_fns: &FxHashSet<QualifiedRef>,
) -> Inlined {
    let types = &mut module.types;
    let main = inline_body(&module.main, all_modules, recursive_fns, &module.closures, types);

    // Inline within closures too.
    let mut closures = FxHashMap::default();
    for (label, closure) in &module.closures {
        let mut inlined_body = inline_body(closure, all_modules, recursive_fns, &module.closures, types);
        inlined_body.captures = closure.captures.clone();
        inlined_body.params = closure.params.clone();
        closures.insert(*label, inlined_body);
    }

    Inlined { main, closures }
}

/// The types of one module's table, re-interned into another table.
struct Import<'s> {
    from: &'s Types<Concrete>,
    imported: FxHashMap<Ty, Ty>,
}

impl<'s> Import<'s> {
    fn from(from: &'s Types<Concrete>) -> Self {
        Import {
            from,
            imported: FxHashMap::default(),
        }
    }

    /// `body` with every type it names imported into `to`.
    fn body(&mut self, body: &MirBody, to: &mut Types<Concrete>) -> MirBody {
        let mut body = body.clone();
        for ty in body.val_types.values_mut() {
            *ty = self.ty(*ty, to);
        }
        for inst in &mut body.insts {
            if let InstKind::FunctionCall { callee_ty, .. } | InstKind::Spawn { callee_ty, .. } = &mut inst.kind {
                *callee_ty = self.ty(*callee_ty, to);
            }
        }
        body
    }

    fn ty(&mut self, ty: Ty, to: &mut Types<Concrete>) -> Ty {
        if let Some(imported) = self.imported.get(&ty) {
            return *imported;
        }
        let imported = acvus_utils::grow(|| self.ty_level(ty, to));
        self.imported.insert(ty, imported);
        imported
    }

    fn arg(&mut self, arg: &TypeArg<Concrete>, to: &mut Types<Concrete>) -> TypeArg<Concrete> {
        match arg {
            TypeArg::Uniform(ty) => TypeArg::Uniform(self.ty(*ty, to)),
            TypeArg::Specialized(ty) => TypeArg::Specialized(self.ty(*ty, to)),
            TypeArg::Open(var, _) => match *var {},
        }
    }

    fn ty_level(&mut self, ty: Ty, to: &mut Types<Concrete>) -> Ty {
        let kind = match self.from.kind(ty).clone() {
            kind @ (TyKind::Int(_)
            | TyKind::Float
            | TyKind::Char
            | TyKind::String
            | TyKind::Bool
            | TyKind::Unit
            | TyKind::Never
            | TyKind::Order
            | TyKind::Str) => kind,
            TyKind::Handle(inner) => TyKind::Handle(self.ty(inner, to)),
            TyKind::Closures(signatures) => TyKind::Closures(signatures.iter().map(|signature| self.ty(*signature, to)).collect()),
            TyKind::Array(inner, len, _) => TyKind::Array(self.ty(inner, to), len, ()),
            TyKind::Slice(inner) => TyKind::Slice(self.ty(inner, to)),
            TyKind::Option(inner, _) => TyKind::Option(self.ty(inner, to), ()),
            TyKind::Result(ok, err, _) => TyKind::Result(self.ty(ok, to), self.ty(err, to), ()),
            TyKind::Tuple(items, _) => TyKind::Tuple(items.iter().map(|item| self.ty(*item, to)).collect(), ()),
            TyKind::Object(object) => {
                let fields = object
                    .fields()
                    .iter()
                    .map(|(name, field)| (*name, self.ty(*field, to)))
                    .collect();
                TyKind::Object(match object {
                    ObjectTy::Structural(_, ()) => ObjectTy::Structural(fields, ()),
                    ObjectTy::Declared(id, _) => ObjectTy::Declared(id, fields),
                })
            }
            TyKind::Enum { name, variants, label } => TyKind::Enum {
                name,
                variants: variants
                    .iter()
                    .map(|(tag, payload)| (*tag, payload.map(|payload| self.ty(payload, to))))
                    .collect(),
                label,
            },
            TyKind::Fn {
                params,
                ret,
                effect,
                flows,
                calls,
                holds,
            } => TyKind::Fn {
                params: params
                    .iter()
                    .map(|param| self.ty(*param, to))
                    .collect(),
                ret: self.ty(ret, to),
                effect,
                flows,
                calls,
                holds,
            },
            TyKind::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => TyKind::UserDefined {
                id,
                type_args: type_args.iter().map(|arg| self.arg(arg, to)).collect(),
                effect_args,
                identity_args,
            },
            TyKind::Ref(mutability, target) => TyKind::Ref(mutability, self.arg(&target, to)),
            TyKind::Var(var) => match var {},
        };
        to.intern(kind)
    }
}

/// Inline all eligible FunctionCall instructions within a MirBody.
/// Iterates until no more inlining opportunities remain (handles nested calls
/// where an inlined body itself contains calls to other local functions).
fn inline_body(
    body: &MirBody,
    all_modules: &FxHashMap<QualifiedRef, MirModule>,
    recursive_fns: &FxHashSet<QualifiedRef>,
    closures: &FxHashMap<Label, MirBody>,
    types: &mut Types<Concrete>,
) -> MirBody {
    let mut current = body.clone();

    loop {
        let mut changed = false;
        let mut new_insts = Vec::new();
        let mut val_remap: FxHashMap<ValueId, ValueId> = FxHashMap::default();

        let insts = std::mem::take(&mut current.insts);
        let plans = ClosurePlans::of_body(&insts, closures);
        let mut bound: FxHashMap<PlanId, Vec<CaptureBound>> = FxHashMap::default();

        for (idx, inst) in insts.iter().enumerate() {
            if let Some(plan) = plans.made_at(idx) {
                bound.insert(plan.id, plan.bind_captures(&val_remap));
                changed = true;
                continue;
            }
            if plans.is_residue(idx) {
                changed = true;
                continue;
            }

            let inline_target = match plans.called_at(idx) {
                Some(plan) => Some(plan.target(
                    &inst.kind,
                    bound.get(&plan.id).expect(
                        "a closure's captures are bound at its MakeClosure, \
                         which every call of it follows",
                    ),
                )),
                None => direct_target(&inst.kind, all_modules, recursive_fns, types),
            };

            if let Some(InlineTarget {
                dst,
                callee_body,
                args,
                captures,
                order,
            }) = inline_target
            {
                // Apply val_remap to args: earlier inlinings may have replaced
                // the original dst with a new value.
                let args: Vec<ValueId> = args.iter().map(|a| remap_one(*a, &val_remap)).collect();

                // Build ValueId remap: callee's ids -> fresh ids in caller.
                let mut callee_remap: FxHashMap<ValueId, ValueId> = FxHashMap::default();
                for i in 0..callee_body.val_factory.len() {
                    let old_id = ValueId::from_raw(i);
                    let new_id = current.val_factory.next();
                    callee_remap.insert(old_id, new_id);
                }

                // Build Label remap: callee's labels -> fresh labels in caller.
                let label_offset = current.label_count;
                current.label_count += callee_body.label_count;

                // Map callee capture_regs directly to caller args.
                // LLVM-style: direct SSA value substitution.
                let mut substituted_regs: FxHashSet<ValueId> = FxHashSet::default();
                for bound in &captures {
                    bound.substitute(&mut callee_remap, &mut substituted_regs);
                }
                // Spliced, a parameter the callee names as storage is a local
                // the argument fills, as `bind` makes a bound `$` one;
                // `validate::type_check` refuses a `RefTarget::Param` that
                // names anything but a parameter of its body.
                for ((_, param_reg), arg) in callee_body.params.iter().zip(args.iter()) {
                    if names_as_storage(&callee_body, *param_reg) {
                        new_insts.push(Inst {
                            span: inst.span,
                            kind: InstKind::Assign {
                                target: RefTarget::Var(remap_one(*param_reg, &callee_remap)),
                                path: Vec::new(),
                                value: *arg,
                                restores: false,
                            },
                        });
                    } else {
                        callee_remap.insert(*param_reg, *arg);
                        substituted_regs.insert(*param_reg);
                    }
                }
                // The callee's entry Order is the Order the call waited for.
                match (callee_body.order_param, order) {
                    (Some(order_param), Some(edge)) => {
                        callee_remap.insert(order_param, remap_one(edge.before, &val_remap));
                        substituted_regs.insert(order_param);
                    }
                    (None, None) => {}
                    (has_param, has_edge) => panic!(
                        "inline: callee order param {:?} does not match call order edge {:?}",
                        has_param, has_edge
                    ),
                }

                // Copy callee's val_types (remapped).
                // Skip types for substituted regs - caller already has types for those.
                for (&old_val, ty) in &callee_body.val_types {
                    if substituted_regs.contains(&old_val) {
                        continue;
                    }
                    if let Some(&new_val) = callee_remap.get(&old_val) {
                        current.val_types.insert(new_val, ty.clone());
                    }
                }

                // Copy callee's debug info (remapped).
                for (&old_val, origin) in &callee_body.debug.val_origins {
                    if substituted_regs.contains(&old_val) {
                        continue;
                    }
                    if let Some(&new_val) = callee_remap.get(&old_val) {
                        current.debug.val_origins.insert(new_val, origin.clone());
                    }
                }

                // Emit callee instructions with remapped ids.
                for callee_inst in &callee_body.insts {
                    match &callee_inst.kind {
                        InstKind::Return {
                            value,
                            order: returned_order,
                        } => {
                            let remapped_val = remap_one(*value, &callee_remap);
                            val_remap.insert(dst, remapped_val);
                            // The Order the call yields is the Order the callee returned.
                            if let (Some(edge), Some(ret)) = (order, returned_order) {
                                val_remap.insert(edge.after, remap_one(*ret, &callee_remap));
                            }
                        }

                        _ => {
                            let remapped = remap_inst(
                                &callee_inst.kind,
                                &callee_remap,
                                label_offset,
                                ParamTargets::BecomeLocals,
                            );
                            new_insts.push(Inst {
                                span: callee_inst.span,
                                kind: remapped,
                            });
                        }
                    }
                }

                changed = true;
            } else {
                // Non-inlineable: emit as-is, applying val_remap.
                let remapped = remap_inst(&inst.kind, &val_remap, 0, ParamTargets::StayParams);
                new_insts.push(Inst {
                    span: inst.span,
                    kind: remapped,
                });
            }
        }

        current.insts = new_insts;

        if !changed {
            break;
        }
    }

    current
}

/// A call the inliner replaces with its callee's body.
struct InlineTarget<'a> {
    dst: ValueId,
    /// The callee's body, its types in the caller's table.
    callee_body: Cow<'a, MirBody>,
    args: Vec<ValueId>,
    captures: Vec<CaptureBound>,
    order: Option<OrderEdge>,
}

/// A closure's `MirBody` lives in `MirModule::closures` of the module whose
/// body makes it, and `acvus_interpreter::prepare` looks a `MakeClosure` up in
/// the module it is preparing. So splicing a body that makes a closure would
/// leave the caller's module holding a `MakeClosure` naming a body that module
/// does not have, and `prepare` panics with "closure body not found". Carrying
/// the callee's closures across would need a label namespace the two modules
/// share; until there is one, such a callee is not spliced.
fn makes_a_closure(body: &MirBody) -> bool {
    body.insts
        .iter()
        .any(|inst| matches!(inst.kind, InstKind::MakeClosure { .. }))
}

/// The local function a `Callee::Direct` names, where this phase has its
/// body and the call is not part of a recursion.
fn direct_target<'a>(
    kind: &InstKind,
    all_modules: &'a FxHashMap<QualifiedRef, MirModule>,
    recursive_fns: &FxHashSet<QualifiedRef>,
    types: &mut Types<Concrete>,
) -> Option<InlineTarget<'a>> {
    let InstKind::FunctionCall {
        dst,
        callee: Callee::Direct(callee_id),
        args,
        order,
        ..
    } = kind
    else {
        return None;
    };
    if recursive_fns.contains(callee_id) {
        return None;
    }
    let callee = all_modules.get(callee_id)?;
    if makes_a_closure(&callee.main) {
        return None;
    }
    Some(InlineTarget {
        dst: *dst,
        callee_body: Cow::Owned(Import::from(&callee.types).body(&callee.main, types)),
        args: args.clone(),
        captures: Vec::new(),
        order: *order,
    })
}

// -- A closure the caller holds instead of calling -------------------

/// Inline a closure only when it is really small and pure. The bound is the
/// owner's, and it is a count rather than a measurement: what the inlined
/// call costs is a `Code::call` through the code's head word and the operand
/// space the chain entry builds (RFC-0052 rule 7, RFC-0060).
const INLINE_MAX_INSTS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlanId(usize);

/// What an inlined copy puts in place of one capture register: the value
/// `MakeClosure` passed for it, whose type is the register's (functions §2:
/// a capture by reference is a `&T` value the lowering passes).
#[derive(Debug, Clone, Copy)]
struct CaptureBound {
    reg: ValueId,
    value: ValueId,
}

impl CaptureBound {
    fn substitute(
        &self,
        remap: &mut FxHashMap<ValueId, ValueId>,
        substituted: &mut FxHashSet<ValueId>,
    ) {
        remap.insert(self.reg, self.value);
        substituted.insert(self.reg);
    }
}

/// One capture of the closure: the register the body reads it in, and the
/// value `MakeClosure` takes for it.
struct PlannedCapture {
    reg: ValueId,
    arg: ValueId,
}

/// A closure of one body, with every call of it and everything that exists
/// only to make it.
struct ClosurePlan<'a> {
    id: PlanId,
    body: &'a MirBody,
    captures: Vec<PlannedCapture>,
}

impl<'a> ClosurePlan<'a> {
    /// Each capture register bound to the value `MakeClosure` passed for
    /// it, as the caller holds that value where the closure was made.
    fn bind_captures(&self, val_remap: &FxHashMap<ValueId, ValueId>) -> Vec<CaptureBound> {
        self.captures
            .iter()
            .map(|capture| CaptureBound {
                reg: capture.reg,
                value: remap_one(capture.arg, val_remap),
            })
            .collect()
    }

    fn target(&self, kind: &InstKind, bound: &[CaptureBound]) -> InlineTarget<'a> {
        let InstKind::FunctionCall {
            dst, args, order, ..
        } = kind
        else {
            panic!("a closure plan's call site is a FunctionCall, not {kind:?}")
        };
        InlineTarget {
            dst: *dst,
            callee_body: Cow::Borrowed(self.body),
            args: args.clone(),
            captures: bound.to_vec(),
            order: *order,
        }
    }
}

/// Every closure of a body the inliner replaces with its own instructions,
/// indexed by the instructions it answers for.
struct ClosurePlans<'a> {
    plans: Vec<ClosurePlan<'a>>,
    made: FxHashMap<usize, PlanId>,
    called: FxHashMap<usize, PlanId>,
    residue: FxHashSet<usize>,
}

impl<'a> ClosurePlans<'a> {
    fn of_body(insts: &[Inst], closures: &'a FxHashMap<Label, MirBody>) -> Self {
        let mut this = Self {
            plans: Vec::new(),
            made: FxHashMap::default(),
            called: FxHashMap::default(),
            residue: FxHashSet::default(),
        };

        for (idx, inst) in insts.iter().enumerate() {
            let InstKind::MakeClosure {
                dst,
                body: label,
                captures,
            } = &inst.kind
            else {
                continue;
            };
            let Some(closure) = closures.get(label) else {
                continue;
            };
            if !small_pure_chain(closure) || closure.captures.len() != captures.len() {
                continue;
            }
            let Some(uses) = ClosureUses::of(insts, idx, *dst) else {
                continue;
            };
            let planned = plan_captures(closure, captures);
            if uses.overlaps(&this) {
                continue;
            }

            let id = PlanId(this.plans.len());
            this.made.insert(idx, id);
            this.residue
                .extend(uses.residue.iter().filter(|at| **at != idx));
            this.called.extend(uses.calls.iter().map(|at| (*at, id)));
            this.plans.push(ClosurePlan {
                id,
                body: closure,
                captures: planned,
            });
        }

        this
    }

    fn made_at(&self, idx: usize) -> Option<&ClosurePlan<'a>> {
        Some(&self.plans[self.made.get(&idx)?.0])
    }

    fn called_at(&self, idx: usize) -> Option<&ClosurePlan<'a>> {
        Some(&self.plans[self.called.get(&idx)?.0])
    }

    fn is_residue(&self, idx: usize) -> bool {
        self.residue.contains(&idx)
    }
}

/// Every instruction that exists only because a body made one closure, and
/// every call that names it.
struct ClosureUses {
    calls: FxHashSet<usize>,
    residue: FxHashSet<usize>,
}

impl ClosureUses {
    /// The uses of the closure made at `mk_idx`, or `None` where one of them
    /// is not a call of it: a closure that also escapes — stored, returned,
    /// passed — is one body, and an inlined copy beside it would be two.
    fn of(insts: &[Inst], mk_idx: usize, mk_dst: ValueId) -> Option<Self> {
        let mut names: FxHashSet<ValueId> = [mk_dst].into_iter().collect();
        let mut slots: FxHashSet<ValueId> = FxHashSet::default();
        let mut this = Self {
            calls: FxHashSet::default(),
            residue: [mk_idx].into_iter().collect(),
        };

        let mut growing = true;
        while growing {
            growing = false;
            for (idx, inst) in insts.iter().enumerate() {
                if this.residue.contains(&idx) || this.calls.contains(&idx) {
                    continue;
                }
                match &inst.kind {
                    InstKind::Assign {
                        target: RefTarget::Var(slot),
                        path,
                        value,
                        ..
                    } if path.is_empty() && names.contains(value) => {
                        growing |= slots.insert(*slot);
                        this.residue.insert(idx);
                    }
                    InstKind::Ref {
                        dst,
                        target: RefTarget::Var(slot),
                        path,
                        mutability: Mutability::Shared,
                    } if path.is_empty() && slots.contains(slot) => {
                        growing |= names.insert(*dst);
                        this.residue.insert(idx);
                    }
                    InstKind::Drop { src } if names.contains(src) || slots.contains(src) => {
                        this.residue.insert(idx);
                    }
                    InstKind::FunctionCall {
                        callee: Callee::Indirect(f),
                        args,
                        ..
                    } if idx > mk_idx
                        && names.contains(f)
                        && !args.iter().any(|a| names.contains(a)) =>
                    {
                        this.calls.insert(idx);
                    }
                    _ => {}
                }
            }
        }

        let clean = insts.iter().enumerate().all(|(idx, inst)| {
            this.residue.contains(&idx)
                || this.calls.contains(&idx)
                || (!inst_info::uses(&inst.kind)
                    .iter()
                    .any(|u| names.contains(u))
                    && !storage_of(&inst.kind).is_some_and(|s| slots.contains(&s)))
        });
        clean.then_some(this)
    }

    fn overlaps(&self, plans: &ClosurePlans<'_>) -> bool {
        self.residue
            .iter()
            .chain(self.calls.iter())
            .any(|at| plans.residue.contains(at) || plans.called.contains_key(at))
    }
}

/// The storage this instruction names directly, which `inst_info::uses` does
/// not count as a value read.
fn storage_of(kind: &InstKind) -> Option<ValueId> {
    match kind {
        InstKind::Ref { target, .. }
        | InstKind::Take { target, .. }
        | InstKind::Assign { target, .. } => Some(escape::named_storage(target)),
        _ => None,
    }
}

fn small_pure_chain(body: &MirBody) -> bool {
    let Some((last, rest)) = body.insts.split_last() else {
        return false;
    };
    body.order_param.is_none()
        && body.insts.len() <= INLINE_MAX_INSTS
        && matches!(last.kind, InstKind::Return { .. })
        && !rest.iter().any(|inst| starts_or_ends_a_block(&inst.kind))
}

fn starts_or_ends_a_block(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::BlockLabel { .. }
            | InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            | InstKind::Diamond { .. }
            | InstKind::Switch { .. }
            | InstKind::While { .. }
            | InstKind::Return { .. }
            | InstKind::Diverge
    )
}

/// Each capture register of `closure` against the value `MakeClosure` takes
/// for it.
fn plan_captures(closure: &MirBody, args: &[ValueId]) -> Vec<PlannedCapture> {
    closure
        .captures
        .iter()
        .zip(args)
        .map(|((_, reg), arg)| PlannedCapture { reg: *reg, arg: *arg })
        .collect()
}

/// Remap a single ValueId through a remap table. Returns original if not mapped.
fn remap_one(val: ValueId, remap: &FxHashMap<ValueId, ValueId>) -> ValueId {
    remap.get(&val).copied().unwrap_or(val)
}

fn names_as_storage(callee: &MirBody, param: ValueId) -> bool {
    callee.insts.iter().any(|inst| match &inst.kind {
        InstKind::Ref { target, .. }
        | InstKind::Take { target, .. }
        | InstKind::Assign { target, .. } => *target == RefTarget::Param(param),
        _ => false,
    })
}

#[derive(Debug, Clone, Copy)]
enum ParamTargets {
    StayParams,
    BecomeLocals,
}

/// Remap a Label with an offset.
fn remap_target(
    target: &crate::ir::RefTarget,
    remap: &FxHashMap<ValueId, ValueId>,
    params: ParamTargets,
) -> crate::ir::RefTarget {
    match (target, params) {
        (crate::ir::RefTarget::Var(slot), _)
        | (crate::ir::RefTarget::Param(slot), ParamTargets::BecomeLocals) => {
            crate::ir::RefTarget::Var(remap_one(*slot, remap))
        }
        (crate::ir::RefTarget::Param(slot), ParamTargets::StayParams) => {
            crate::ir::RefTarget::Param(remap_one(*slot, remap))
        }
        (crate::ir::RefTarget::Through(r), _) => {
            crate::ir::RefTarget::Through(remap_one(*r, remap))
        }
    }
}

fn remap_label(label: Label, offset: u32) -> Label {
    if offset == 0 {
        label
    } else {
        Label(label.0 + offset)
    }
}

/// Remap all ValueIds and Labels in an instruction.
fn remap_inst(
    kind: &InstKind,
    val_remap: &FxHashMap<ValueId, ValueId>,
    label_offset: u32,
    params: ParamTargets,
) -> InstKind {
    let r = |v: ValueId| -> ValueId { remap_one(v, val_remap) };
    let rl = |l: Label| -> Label { remap_label(l, label_offset) };
    let rv = |vals: &[ValueId]| -> Vec<ValueId> { vals.iter().map(|v| r(*v)).collect() };

    match kind {
        // Constants
        InstKind::Const { dst, value } => InstKind::Const {
            dst: r(*dst),
            value: value.clone(),
        },
        InstKind::ConstStr { dst, text } => InstKind::ConstStr {
            dst: r(*dst),
            text: text.clone(),
        },

        // Projection
        InstKind::Ref {
            dst,
            target,
            path,
            mutability,
        } => InstKind::Ref {
            dst: r(*dst),
            target: remap_target(target, val_remap, params),
            path: path.clone(),
            mutability: *mutability,
        },
        InstKind::Take {
            dst,
            target,
            path,
            taken_out,
        } => InstKind::Take {
            dst: r(*dst),
            target: remap_target(target, val_remap, params),
            path: path.clone(),
            taken_out: *taken_out,
        },
        InstKind::Assign {
            target,
            path,
            value,
            restores,
        } => InstKind::Assign {
            target: remap_target(target, val_remap, params),
            path: path.clone(),
            value: r(*value),
            restores: *restores,
        },
        InstKind::AsSlice {
            dst,
            container,
            mutability,
            r#impl,
        } => InstKind::AsSlice {
            dst: r(*dst),
            container: r(*container),
            mutability: *mutability,
            r#impl: *r#impl,
        },
        InstKind::Index {
            dst,
            slice,
            index,
            mode,
            bound,
        } => InstKind::Index {
            dst: r(*dst),
            slice: r(*slice),
            index: r(*index),
            mode: *mode,
            bound: *bound,
        },
        InstKind::IndexSet {
            slice,
            index,
            value,
            bound,
        } => InstKind::IndexSet {
            slice: r(*slice),
            index: r(*index),
            value: r(*value),
            bound: *bound,
        },
        InstKind::StringAppend { target, part } => InstKind::StringAppend {
            target: r(*target),
            part: r(*part),
        },
        InstKind::Fetch { dst, context } => InstKind::Fetch {
            dst: r(*dst),
            context: *context,
        },
        InstKind::Commit {
            context,
            value,
            wrote,
        } => InstKind::Commit {
            context: *context,
            value: r(*value),
            wrote: *wrote,
        },

        // Scalar field access
        InstKind::FieldGet {
            dst,
            object,
            field,
            rest,
        } => InstKind::FieldGet {
            dst: r(*dst),
            object: r(*object),
            field: *field,
            rest: rest.clone(),
        },
        InstKind::FieldSet {
            dst,
            object,
            field,
            rest,
            value,
        } => InstKind::FieldSet {
            dst: r(*dst),
            object: r(*object),
            field: *field,
            rest: rest.clone(),
            value: r(*value),
        },

        // Arithmetic
        InstKind::BinOp {
            dst,
            op,
            left,
            right,
        } => InstKind::BinOp {
            dst: r(*dst),
            op: *op,
            left: r(*left),
            right: r(*right),
        },
        InstKind::UnaryOp { dst, op, operand } => InstKind::UnaryOp {
            dst: r(*dst),
            op: *op,
            operand: r(*operand),
        },
        InstKind::Check { op, left, right } => InstKind::Check {
            op: *op,
            left: r(*left),
            right: r(*right),
        },
        InstKind::CheckSteps { from, step, count } => InstKind::CheckSteps {
            from: r(*from),
            step: r(*step),
            count: r(*count),
        },
        InstKind::Cast { dst, src, to } => InstKind::Cast {
            dst: r(*dst),
            src: r(*src),
            to: *to,
        },

        // Functions
        InstKind::LoadFunction { dst, id } => InstKind::LoadFunction {
            dst: r(*dst),
            id: *id,
        },
        InstKind::FunctionCall {
            dst,
            callee,
            callee_ty,
            args,
            order,
        } => {
            let callee = match callee {
                Callee::Direct(_) | Callee::Extern { .. } => callee.clone(),
                Callee::Indirect(v) => Callee::Indirect(r(*v)),
            };
            InstKind::FunctionCall {
                dst: r(*dst),
                callee,
                callee_ty: callee_ty.clone(),
                args: rv(args),
                order: order.map(|edge| OrderEdge {
                    before: r(edge.before),
                    after: r(edge.after),
                }),
            }
        }
        InstKind::Spawn {
            dst,
            callee,
            callee_ty,
            args,
            order,
        } => {
            let callee = match callee {
                Callee::Direct(_) | Callee::Extern { .. } => callee.clone(),
                Callee::Indirect(v) => Callee::Indirect(r(*v)),
            };
            InstKind::Spawn {
                dst: r(*dst),
                callee,
                callee_ty: callee_ty.clone(),
                args: rv(args),
                order: order.map(r),
            }
        }
        InstKind::TakeEffect { dst, src, order } => InstKind::TakeEffect {
            dst: r(*dst),
            src: r(*src),
            order: order.map(r),
        },
        InstKind::Merge { dst, orders } => InstKind::Merge {
            dst: r(*dst),
            orders: rv(orders),
        },

        // Composite constructors
        InstKind::ArrayBegin { dst, capacity } => InstKind::ArrayBegin {
            dst: r(*dst),
            capacity: *capacity,
        },
        InstKind::ArrayPush { dst, array, value } => InstKind::ArrayPush {
            dst: r(*dst),
            array: r(*array),
            value: r(*value),
        },
        InstKind::StringConcat { dst, parts } => InstKind::StringConcat {
            dst: r(*dst),
            parts: rv(parts),
        },
        InstKind::StringEq { dst, a, b } => InstKind::StringEq {
            dst: r(*dst),
            a: r(*a),
            b: r(*b),
        },
        InstKind::StringClone { dst, src } => InstKind::StringClone {
            dst: r(*dst),
            src: r(*src),
        },
        InstKind::StructuralEq { dst, a, b, leaves } => InstKind::StructuralEq {
            dst: r(*dst),
            a: r(*a),
            b: r(*b),
            leaves: leaves.clone(),
        },
        InstKind::StructuralClone { dst, src, leaves } => InstKind::StructuralClone {
            dst: r(*dst),
            src: r(*src),
            leaves: leaves.clone(),
        },
        InstKind::MakeObject { dst, fields } => InstKind::MakeObject {
            dst: r(*dst),
            fields: fields.iter().map(|(name, v)| (*name, r(*v))).collect(),
        },
        InstKind::MakeTuple { dst, elements } => InstKind::MakeTuple {
            dst: r(*dst),
            elements: rv(elements),
        },
        InstKind::TupleIndex { dst, tuple, index } => InstKind::TupleIndex {
            dst: r(*dst),
            tuple: r(*tuple),
            index: *index,
        },

        // Pattern matching
        InstKind::TestLiteral { dst, src, value } => InstKind::TestLiteral {
            dst: r(*dst),
            src: r(*src),
            value: value.clone(),
        },
        InstKind::TestObjectKey { dst, src, key } => InstKind::TestObjectKey {
            dst: r(*dst),
            src: r(*src),
            key: *key,
        },
        InstKind::ArrayIndex {
            dst,
            array: list,
            index,
        } => InstKind::ArrayIndex {
            dst: r(*dst),
            array: r(*list),
            index: *index,
        },
        InstKind::ObjectGet { dst, object, key } => InstKind::ObjectGet {
            dst: r(*dst),
            object: r(*object),
            key: *key,
        },

        // Closures
        //
        // `body` is a key of `MirModule::closures`, not a block of the body
        // being spliced, so the label offset does not apply to it. The only
        // callee that reaches here making a closure is a [`ClosurePlan`]'s,
        // whose closures the caller's module holds under the same keys
        // ([`makes_a_closure`] keeps every other one out).
        InstKind::MakeClosure {
            dst,
            body,
            captures,
        } => InstKind::MakeClosure {
            dst: r(*dst),
            body: *body,
            captures: rv(captures),
        },

        // Iterator

        // Variant
        InstKind::MakeVariant { dst, tag, payload } => InstKind::MakeVariant {
            dst: r(*dst),
            tag: *tag,
            payload: payload.map(&r),
        },
        InstKind::TestVariant { dst, src, tag } => InstKind::TestVariant {
            dst: r(*dst),
            src: r(*src),
            tag: *tag,
        },
        InstKind::UnwrapVariant { dst, src } => InstKind::UnwrapVariant {
            dst: r(*dst),
            src: r(*src),
        },

        // Control flow
        InstKind::BlockLabel { label, params } => InstKind::BlockLabel {
            label: rl(*label),
            params: rv(params),
        },
        InstKind::Jump { label, args } => InstKind::Jump {
            label: rl(*label),
            args: rv(args),
        },
        InstKind::JumpIf {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
        } => InstKind::JumpIf {
            cond: r(*cond),
            then_label: rl(*then_label),
            then_args: rv(then_args),
            else_label: rl(*else_label),
            else_args: rv(else_args),
        },
        InstKind::Diamond {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
            join,
        } => InstKind::Diamond {
            cond: r(*cond),
            then_label: rl(*then_label),
            then_args: rv(then_args),
            else_label: rl(*else_label),
            else_args: rv(else_args),
            join: rl(*join),
        },
        InstKind::For {
            source,
            stages,
            exit,
            exit_trip,
            exit_args,
        } => {
            let mut source = *source;
            source.for_each_use(|v| *v = r(*v));
            let mut stages = stages.clone();
            for entry in stages.entries_mut() {
                *entry = rl(*entry);
            }
            InstKind::For {
                source,
                stages,
                exit: rl(*exit),
                exit_trip: *exit_trip,
                exit_args: rv(exit_args),
            }
        }
        InstKind::While {
            cond,
            stages,
            exit,
            exit_args,
        } => {
            let mut stages = stages.clone();
            for entry in stages.entries_mut() {
                *entry = rl(*entry);
            }
            InstKind::While {
                cond: r(*cond),
                stages,
                exit: rl(*exit),
                exit_args: rv(exit_args),
            }
        }
        InstKind::Switch { tag, arms, default } => InstKind::Switch {
            tag: r(*tag),
            arms: arms
                .iter()
                .map(|(t, label, args)| (*t, rl(*label), rv(args)))
                .collect(),
            default: default.as_ref().map(|(label, args)| (rl(*label), rv(args))),
        },
        InstKind::Return { value, order } => InstKind::Return {
            value: r(*value),
            order: order.map(r),
        },
        InstKind::Nop => InstKind::Nop,
        InstKind::Diverge => InstKind::Diverge,

        // Cast

        // Drop
        InstKind::Drop { src } => InstKind::Drop { src: r(*src) },

        // Poison / Undef
        InstKind::Poison { dst } => InstKind::Poison { dst: r(*dst) },
        InstKind::Undef { dst } => InstKind::Undef { dst: r(*dst) },
    }
}

#[cfg(test)]
mod tests {
    use crate::ty::IntTy;

    use super::*;
    use acvus_ast::Span;
    use acvus_utils::LocalFactory;

    fn make_inst(kind: InstKind) -> Inst {
        Inst {
            span: Span::ZERO,
            kind,
        }
    }

    /// A body of `val_count` values, every one an `i64` of `types`: the
    /// table `insts` names it in, so a caller builds `types` and its
    /// `callee_ty`/etc. placeholders before this call, from the same
    /// `TyKind::Int(IntTy::I64)` interning re-reads.
    fn make_body(types: &mut Types<Concrete>, insts: Vec<InstKind>, val_count: usize) -> MirBody {
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let mut factory = LocalFactory::<ValueId>::new();
        let mut val_types = FxHashMap::default();
        for _ in 0..val_count {
            let v = factory.next();
            val_types.insert(v, i64_ty);
        }
        MirBody {
            demoted_diamonds: Default::default(),
            insts: insts.into_iter().map(make_inst).collect(),
            val_types,
            params: Vec::new(),
            captures: Vec::new(),
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
            order_param: None,
        }
    }

    /// `body`, wrapped as the one module owning `types`: `inline`'s doc
    /// says every module owns its own table.
    fn make_module(body: MirBody, mut types: Types<Concrete>) -> MirModule {
        let unit_ty = types.intern(TyKind::Unit);
        MirModule {
            declared_params: body.params.len(),
            main: body,
            closures: FxHashMap::default(),
            ret: unit_ty,
            flows: crate::ty::Flows::Every,
            types,
        }
    }

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    #[test]
    fn inline_simple_call() {
        // caller: r0 = const 1; r1 = call f(r0); yield r1
        // callee f: r0 = param; r1 = r0 + r0; return r1
        let i = acvus_utils::Interner::new();
        let callee_id = QualifiedRef::root(i.intern("callee"));

        let mut callee_types = Types::<Concrete>::new();
        let mut callee_body = make_body(
            &mut callee_types,
            vec![
                InstKind::BinOp {
                    dst: v(1),
                    op: crate::ir::BinOp::Add(crate::ir::Overflow::Trap),
                    left: v(0),
                    right: v(0),
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            2,
        );
        callee_body.params = vec![(i.intern("p0"), v(0))];

        let mut caller_types = Types::<Concrete>::new();
        let caller_i64 = caller_types.intern(TyKind::Int(IntTy::I64));
        let caller_body = make_body(
            &mut caller_types,
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(callee_id),
                    callee_ty: caller_i64,
                    args: vec![v(0)],
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            2,
        );

        let mut modules = FxHashMap::default();
        modules.insert(callee_id, make_module(callee_body, callee_types));
        let caller_id = QualifiedRef::root(i.intern("caller"));
        modules.insert(caller_id, make_module(caller_body, caller_types));

        let result = inline(modules, &FxHashSet::default());
        let inlined = &result.modules[&caller_id];

        // After inlining, there should be no FunctionCall instruction.
        let has_call = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::FunctionCall { .. }));
        assert!(!has_call, "FunctionCall should be inlined away");

        // Should have a BinOp (from callee) and a Yield.
        let has_binop = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::BinOp { .. }));
        assert!(has_binop, "callee's BinOp should be present after inlining");

        let has_yield = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::Return { .. }));
        assert!(has_yield, "Yield should remain");
    }

    #[test]
    fn inline_preserves_extern_call() {
        // call to an extern function (not in modules) should stay.
        let i = acvus_utils::Interner::new();
        let extern_id = QualifiedRef::root(i.intern("ext"));

        let mut types = Types::<Concrete>::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let caller_body = make_body(
            &mut types,
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Direct(extern_id),
                    callee_ty: i64_ty,
                    args: vec![v(0)],
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            2,
        );

        let caller_id = QualifiedRef::root(i.intern("caller"));
        let mut modules = FxHashMap::default();
        modules.insert(caller_id, make_module(caller_body, types));
        // extern_id is NOT in modules -> cannot be inlined.

        let result = inline(modules, &FxHashSet::default());
        let inlined = &result.modules[&caller_id];

        let has_call = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::FunctionCall { .. }));
        assert!(has_call, "extern call should remain");
    }

    #[test]
    fn inline_skips_recursive() {
        let i = acvus_utils::Interner::new();
        let rec_id = QualifiedRef::root(i.intern("rec"));

        let mut rec_types = Types::<Concrete>::new();
        let rec_body = make_body(
            &mut rec_types,
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(1),
                },
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            1,
        );

        let mut caller_types = Types::<Concrete>::new();
        let caller_i64 = caller_types.intern(TyKind::Int(IntTy::I64));
        let caller_body = make_body(
            &mut caller_types,
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(rec_id),
                    callee_ty: caller_i64,
                    args: vec![],
                    order: None,
                },
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            1,
        );

        let caller_id = QualifiedRef::root(i.intern("caller"));
        let mut modules = FxHashMap::default();
        modules.insert(rec_id, make_module(rec_body, rec_types));
        modules.insert(caller_id, make_module(caller_body, caller_types));

        let mut recursive = FxHashSet::default();
        recursive.insert(rec_id);

        let result = inline(modules, &recursive);
        let inlined = &result.modules[&caller_id];

        let has_call = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::FunctionCall { .. }));
        assert!(has_call, "recursive call should NOT be inlined");
    }

    #[test]
    fn inline_chain() {
        // g: return 42
        // f: return g()
        // main: yield f()
        // After inlining: main should have Const(42) + Yield, no calls.
        let i = acvus_utils::Interner::new();
        let g_id = QualifiedRef::root(i.intern("g"));
        let f_id = QualifiedRef::root(i.intern("f"));
        let main_id = QualifiedRef::root(i.intern("main"));

        let mut g_types = Types::<Concrete>::new();
        let g_body = make_body(
            &mut g_types,
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(42),
                },
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            1,
        );

        let mut f_types = Types::<Concrete>::new();
        let f_i64 = f_types.intern(TyKind::Int(IntTy::I64));
        let f_body = make_body(
            &mut f_types,
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(g_id),
                    callee_ty: f_i64,
                    args: vec![],
                    order: None,
                },
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            1,
        );

        let mut main_types = Types::<Concrete>::new();
        let main_i64 = main_types.intern(TyKind::Int(IntTy::I64));
        let main_body = make_body(
            &mut main_types,
            vec![
                InstKind::FunctionCall {
                    dst: v(0),
                    callee: Callee::Direct(f_id),
                    callee_ty: main_i64,
                    args: vec![],
                    order: None,
                },
                InstKind::Return {
                    value: v(0),
                    order: None,
                },
            ],
            1,
        );

        let mut modules = FxHashMap::default();
        modules.insert(g_id, make_module(g_body, g_types));
        modules.insert(f_id, make_module(f_body, f_types));
        modules.insert(main_id, make_module(main_body, main_types));

        let result = inline(modules, &FxHashSet::default());
        let inlined = &result.modules[&main_id];

        let has_call = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::FunctionCall { .. }));
        assert!(!has_call, "all calls should be inlined in chain");

        let has_const = inlined
            .main
            .insts
            .iter()
            .any(|i| matches!(i.kind, InstKind::Const { .. }));
        assert!(has_const, "g's Const(42) should be present");
    }

    /// A caller holding a closure of `len` instructions and one call of it.
    /// The caller's four instructions are the shape `lower.rs` writes for a
    /// `let`-bound lambda: `MakeClosure`, an `Assign` into the variable's
    /// slot, a `Ref` of that slot, and the call through the reference.
    fn closure_of(i: &acvus_utils::Interner, len: usize) -> MirModule {
        let mut types = Types::<Concrete>::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let adds = len - 1;
        let mut closure = make_body(
            &mut types,
            (0..adds)
                .map(|n| InstKind::BinOp {
                    dst: ValueId::from_raw(n + 1),
                    op: crate::ir::BinOp::Add(crate::ir::Overflow::Trap),
                    left: v(n),
                    right: v(0),
                })
                .chain([InstKind::Return {
                    value: ValueId::from_raw(adds),
                    order: None,
                }])
                .collect(),
            len,
        );
        closure.params = vec![(i.intern("x"), v(0))];

        let main = make_body(
            &mut types,
            vec![
                InstKind::MakeClosure {
                    dst: v(0),
                    body: Label(0),
                    captures: Vec::new(),
                },
                InstKind::Assign {
                    target: RefTarget::Var(v(3)),
                    path: Vec::new(),
                    value: v(0),
                    restores: false,
                },
                InstKind::Ref {
                    dst: v(1),
                    target: RefTarget::Var(v(3)),
                    path: Vec::new(),
                    mutability: Mutability::Shared,
                },
                InstKind::FunctionCall {
                    dst: v(2),
                    callee: Callee::Indirect(v(1)),
                    callee_ty: i64_ty,
                    args: vec![v(4)],
                    order: None,
                },
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            5,
        );

        let unit_ty = types.intern(TyKind::Unit);
        MirModule {
            declared_params: main.params.len(),
            main,
            closures: [(Label(0), closure)].into_iter().collect(),
            ret: unit_ty,
            flows: crate::ty::Flows::Every,
            types,
        }
    }

    fn inlines_body_of(len: usize) -> bool {
        let i = acvus_utils::Interner::new();
        let caller_id = QualifiedRef::root(i.intern("caller"));
        let mut modules = FxHashMap::default();
        modules.insert(caller_id, closure_of(&i, len));

        let result = inline(modules, &FxHashSet::default());
        !result.modules[&caller_id]
            .main
            .insts
            .iter()
            .any(|inst| matches!(inst.kind, InstKind::FunctionCall { .. }))
    }

    #[test]
    fn a_closure_body_of_inline_max_insts_is_inlined_and_one_more_is_not() {
        assert!(inlines_body_of(INLINE_MAX_INSTS));
        assert!(!inlines_body_of(INLINE_MAX_INSTS + 1));
    }

    #[test]
    fn an_indirect_call_to_no_make_closure_is_preserved() {
        let mut types = Types::<Concrete>::new();
        let i64_ty = types.intern(TyKind::Int(IntTy::I64));
        let caller_body = make_body(
            &mut types,
            vec![
                InstKind::FunctionCall {
                    dst: v(1),
                    callee: Callee::Indirect(v(0)),
                    callee_ty: i64_ty,
                    args: vec![],
                    order: None,
                },
                InstKind::Return {
                    value: v(1),
                    order: None,
                },
            ],
            2,
        );

        let i = acvus_utils::Interner::new();
        let caller_id = QualifiedRef::root(i.intern("caller"));
        let mut modules = FxHashMap::default();
        modules.insert(caller_id, make_module(caller_body, types));

        let result = inline(modules, &FxHashSet::default());
        let inlined = &result.modules[&caller_id];

        let has_call = inlined.main.insts.iter().any(|inst| {
            matches!(
                inst.kind,
                InstKind::FunctionCall {
                    callee: Callee::Indirect(_),
                    ..
                }
            )
        });
        assert!(has_call, "indirect call should remain");
    }
}
