//! S, the structure rule (types §1, §3): a transparent structured type
//! taken apart by its kind's shape down to the parts that are not one, and
//! a rule applied there. Two rules reach it from typeck: a value coerced to
//! a slot written elsewhere (`fire_coerced`), and a generic function's
//! impl a structure has by its lift (`lift_at`, `reach_lifted`). Each part
//! is a node of its own, decided once its head is closed, as a comparison's
//! parts are (`op::fire_compare`).

use acvus_ast::{AstId, Span};

use super::ctxt::Ctxt;
use super::decision::{Reach, Reading};
use crate::decl::Lift;
use crate::infer::{Coerced, Term};
use crate::ty::{EffectTerm, Infer as InferPhase};

/// A structured value coerced at `site` (types §3): its node records the
/// shape and a node per part it keeps, and each part enters the slot's
/// part at that node, in the body whose effect is `body`.
pub(super) fn fire_coerced(cx: &mut Ctxt<'_>, site: AstId, span: Span, coerced: Coerced, body: EffectTerm<InferPhase>) {
    todo!("record a `CoercionNode` at `site`, and `enter_in` each kept part at a node of its own")
}

/// Where a reach's callee has a lift and the lift's type variable stands
/// at a transparent structured type, closed as `reading` admits: the lift,
/// and that type.
pub(super) struct LiftAt {
    pub lift: Lift,
    pub structured: Term,
}

/// The lift that answers `decision`, where no concrete impl did (types §1):
/// its callee's generic function has one, and the arguments at the lift's
/// type variable are closed at one transparent structured type.
pub(super) fn lift_at(cx: &Ctxt<'_>, decision: &Reach, reading: Reading) -> Option<LiftAt> {
    todo!("the callee's generic function's `Declarations::lift`, read at the arguments' closed type")
}

/// `decision` answered by the lifted impl `at` (types §1): the call reaches
/// `Impl::Structural`, its arguments enter the generic function's
/// parameters at the structured type, and each part is a `Callee::Part`
/// reach at a node of its own, stating the lift's law where it combines,
/// recorded as a `StructureNode` at the call's site.
pub(super) fn reach_lifted(cx: &mut Ctxt<'_>, decision: Reach, at: LiftAt) -> Result<(), Reach> {
    todo!("the call at `Impl::Structural`, and a part reach per part")
}
