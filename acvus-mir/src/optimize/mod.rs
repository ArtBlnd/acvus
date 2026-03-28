pub mod const_dedup;
pub mod reg_color;
pub mod reorder;
pub mod spawn_split;
pub mod ssa;
pub mod ssa_pass;

use crate::graph::QualifiedRef;
use crate::ir::MirModule;
use crate::pass::{
    AnalysisPass, Chain, PassManager, Transform, TransformMarker, TransformPass,
};
use crate::ty::Ty;
use crate::validate::{self, ValidationError};
use rustc_hash::FxHashMap;

// ── Shared dependency: fn_types ─────────────────────────────────────

/// Function type table. Stored in PassContext so passes can depend on it.
pub struct FnTypes(pub FxHashMap<QualifiedRef, Ty>);

// ── Transform passes ────────────────────────────────────────────────

pub struct ConstDedupPass;

impl TransformPass for ConstDedupPass {
    type Required<'a> = ();

    fn transform(&self, module: MirModule, _deps: ()) -> MirModule {
        const_dedup::dedup(module)
    }
}

pub struct SpawnSplitPass;

impl TransformPass for SpawnSplitPass {
    type Required<'a> = (&'a FnTypes,);

    fn transform(&self, mut module: MirModule, (fn_types,): (&FnTypes,)) -> MirModule {
        spawn_split::run(&mut module.main, &fn_types.0);
        for closure in module.closures.values_mut() {
            spawn_split::run(closure, &fn_types.0);
        }
        module
    }
}

pub struct ReorderPass;

impl TransformPass for ReorderPass {
    type Required<'a> = (&'a FnTypes, &'a TransformMarker<SpawnSplitPass>);

    fn transform(
        &self,
        mut module: MirModule,
        (fn_types, _): (&FnTypes, &TransformMarker<SpawnSplitPass>),
    ) -> MirModule {
        reorder::run(&mut module.main, &fn_types.0);
        for closure in module.closures.values_mut() {
            reorder::run(closure, &fn_types.0);
        }
        module
    }
}

pub struct SsaOptPass;

impl TransformPass for SsaOptPass {
    type Required<'a> = (&'a FnTypes, &'a TransformMarker<ReorderPass>);

    fn transform(
        &self,
        mut module: MirModule,
        (fn_types, _): (&FnTypes, &TransformMarker<ReorderPass>),
    ) -> MirModule {
        ssa_pass::run(&mut module.main, &fn_types.0);
        for closure in module.closures.values_mut() {
            ssa_pass::run(closure, &fn_types.0);
        }
        module
    }
}

pub struct RegColorPass;

impl TransformPass for RegColorPass {
    type Required<'a> = (&'a TransformMarker<SsaOptPass>,);

    fn transform(
        &self,
        module: MirModule,
        _: (&TransformMarker<SsaOptPass>,),
    ) -> MirModule {
        reg_color::color(module)
    }
}

// ── Validation (analysis — produces errors, doesn't transform) ──────

pub struct ValidateResult(pub Vec<ValidationError>);

pub struct ValidatePass;

impl AnalysisPass for ValidatePass {
    type Required<'a> = (&'a FnTypes, &'a TransformMarker<RegColorPass>);
    type Output = ValidateResult;

    fn run(
        &self,
        module: &MirModule,
        (fn_types, _): (&FnTypes, &TransformMarker<RegColorPass>),
    ) -> ValidateResult {
        ValidateResult(validate::validate(module, &fn_types.0))
    }
}

// ── Pipeline construction ───────────────────────────────────────────

/// Pass 2 pipeline: SpawnSplit → Reorder → SSA → RegColor → Validate.
///
/// Requires `FnTypes` to be pre-inserted into the PassContext.
pub type Pass2Pipeline = Chain<
    Transform<SpawnSplitPass>,
    Chain<
        Transform<ReorderPass>,
        Chain<
            Transform<SsaOptPass>,
            Chain<Transform<RegColorPass>, Chain<ValidatePass, ()>>,
        >,
    >,
>;

pub fn pass2_manager() -> PassManager<Pass2Pipeline> {
    PassManager::new(Chain(
        Transform(SpawnSplitPass),
        Chain(
            Transform(ReorderPass),
            Chain(
                Transform(SsaOptPass),
                Chain(Transform(RegColorPass), Chain(ValidatePass, ())),
            ),
        ),
    ))
}
