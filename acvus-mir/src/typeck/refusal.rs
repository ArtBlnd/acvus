//! Every refusal typeck makes, in the program's terms (refusals): what was
//! written, where, and the demand that failed. A part never determined is
//! named by the value that left it open, never drawn as a type.

use acvus_ast::Span;
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use crate::infer::{FreezeMemo, Infer, MapMismatch, Mismatch, MismatchTerms, Solving, Unsettled};
use crate::ty::{Calls, CastTy, Concrete, Effect, EffectTerm, Holds, Infer as InferPhase, Ty, TyVarId, Types};

#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub kind: RefusalKind,
    pub span: Span,
}

/// Grouped by the dimension that refuses (typeck-structure section 2).
#[derive(Debug, Clone, PartialEq)]
pub enum RefusalKind {
    // A: a form the language does not admit.
    LambdaRecursion,
    InputAssigned { input: Astr },
    UnnamedVariant { tag: Astr },
    UnknownName { name: Astr },
    UnknownContext { context: QualifiedRef },
    /// A name of extern fns written where no call of it stands, by the
    /// name as written: an extern fn is called, never taken as a value, and
    /// a lambda calling it stands where a function value is wanted
    /// (extern-fn §1).
    ExternAsValue { name: QualifiedRef },
    /// A call by name of what is no unit: only a unit's parameters have
    /// names a call passes by (programs §1, §2).
    NamedArgumentsToNonUnit,
    /// A unit called by position: its parameters are passed by name
    /// (programs §1, §2).
    UnitCalledByPosition { unit: QualifiedRef },
    /// An argument named for no parameter of the unit called.
    UnknownArgument { unit: QualifiedRef, name: Astr },
    /// Two arguments named for one parameter.
    DuplicateArgument { unit: QualifiedRef, name: Astr },
    /// A parameter of the unit called that no argument names.
    MissingArgument { unit: QualifiedRef, name: Astr },
    /// A context's or a run's value whose type holds a loan (ownership §5).
    HoldsLoans,
    /// A store into a place reached through a shared reference, `*r = v` or
    /// a part's (operators §6, ownership §5).
    StoreThroughShared,
    /// A `&mut` of a place reached through a shared reference, exclusive
    /// only where every reference on its path is (ownership §5).
    LendMutThroughShared,
    /// `fn` in a script, which has none (programs §2).
    FnDeclInScript,
    /// A `return` or a `?` that would leave a template itself: a template's
    /// result is the `String` it builds, and what either leaves with is not
    /// that `String` (programs §1).
    ReturnInTemplate,
    /// `_` or `..` where a value is read: they are a pattern's.
    PatternInValue,
    /// A variant written with other than one payload.
    VariantArity { tag: Astr },
    /// `as` to a name that is not a number, `bool` or `char`
    /// (operators §5).
    UnknownCastTarget { name: Astr },
    // B: two types that do not relate.
    Mismatch(FrozenMismatch),
    /// A lambda needs more of its function type than a type it was made
    /// equal to states (D94): it holds a stronger loan (ownership §5), is
    /// called once where many calls are stated (functions §3), or has a
    /// larger effect (effects).
    ClosureExceedsSlot(Exceeding),
    NoConversion,
    /// A value carrying an identity made in a body instantiation, a call's
    /// or a loop's iteration, stored where that body did not make, with no
    /// ancestor carrying no identity to take instead (types §3).
    IdentityEscapes,
    /// `for` over a value that is no reference to a container, array or
    /// range (control-flow).
    NotIterable,
    /// `?` on a value that is neither a `Result` nor an `Option`
    /// (control-flow).
    NotTryable,
    /// `*r` where `r` is no reference (operators §6).
    NotAReference,
    /// `^` on a value that is neither a `bool` nor an integer
    /// (operators §2).
    NotBoolOrInteger,
    /// `!` on a value that is no `bool` (operators §2).
    NotBool,
    /// `&`, `<<` or `>>` on a value that is no integer (operators §2).
    NotInteger,
    /// `as` from a type Rust does not cast to `to` (operators §5); `from`
    /// is `None` where the source is no number, `bool` or `char`.
    CastNotAdmitted { from: Option<CastTy>, to: CastTy },
    /// `break` or `continue` with no loop around it before a lambda's
    /// boundary (control-flow).
    OutsideLoop,
    // C: growth into what may not grow.
    GrowsOpaque,
    /// A function value entering a function type that may not grow to be
    /// it, since the type is reached through a reference (types §1-2).
    NotOneOfFunctionValues { value: crate::ty::FnValue },
    // Decisions.
    NoDeclaration { name: Astr },
    SeveralDeclarations { name: Astr, left: Box<[QualifiedRef]> },
    /// A construct whose type never became known (types §6), named by the
    /// value that left it open.
    NeverKnown { open: Astr },
    /// A construct at the refusal's span whose type never became known,
    /// where no name the program wrote stands for the value left open.
    Undecided,
    /// A bounded variable no use answered (types §5).
    BoundUnanswered,
    /// `<`, `>`, `<=` or `>=` on an enum, an `Option` or a `Result`, which
    /// have equality but no order (operators §4).
    Unordered,
    /// `as_ref` or `as_mut` of a value that is no tuple, object, array,
    /// enum, `Option` or `Result`, whose parts it would lend (operators §2).
    LendsNoParts { name: Astr },
    /// `as_mut` through a `&`, which lends no `&mut` (operators §2).
    MutLendThroughShared { name: Astr },
    // Unstable: a form the language admits and this compiler does not
    // build yet. No rule refuses it; each is removed once it is built.
    Unimplemented { feature: Unimplemented },
}

/// A form the language admits that this compiler does not build yet. It is
/// not a rule of the language: a program refused so is a valid one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unimplemented {
    /// A requirement answered by an impl a structure has by its lift, or by
    /// the language's own answer (types §1, ownership §2, operators §2):
    /// there is no declared function to pass for it.
    LiftedImplAsRequirement,
}

/// What a lambda needs of its function type, and what the slot it was made
/// equal to states.
#[derive(Debug, Clone, PartialEq)]
pub enum Exceeding {
    Holds { needs: Holds, slot: Holds },
    Calls { needs: Calls, slot: Calls },
    Effect { needs: Effect, slot: Effect },
}

/// A mismatch's terms once the check is done: each frozen as far as it
/// settled, or, where a refusal left a variable in it open, that variable
/// as freeze reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frozen;

impl MismatchTerms for Frozen {
    type Ty = Result<Ty, Unsettled>;
    type TyVar = Result<Ty, Unsettled>;
    type Effect = Result<Effect, Unsettled>;
}

/// A mismatch whose terms were frozen at writeback, the one point where
/// every decision is made and the check's store is still alive. Its types
/// are interned in a table of its own, since a refusal outlives both the
/// store and the table the check froze its other types into.
pub struct FrozenMismatch {
    pub mismatch: Mismatch<Frozen>,
    pub types: Types<Concrete>,
}

impl FrozenMismatch {
    fn freeze(infer: &Infer, mismatch: Mismatch) -> FrozenMismatch {
        let mut types = Types::new();
        let mismatch = mismatch.map(&mut Freeze { infer, into: &mut types, memo: FreezeMemo::default() });
        FrozenMismatch { mismatch, types }
    }

    /// The mismatch with its types interned in `into`.
    fn transplanted(&self, into: &mut Types<Concrete>) -> Mismatch<Frozen> {
        self.mismatch.clone().map(&mut Transplant { from: &self.types, into })
    }

    /// Its shape, then each type or effect it holds, in the order it holds
    /// them.
    fn written(&self, names: &acvus_utils::Interner) -> String {
        let ty = |side: &Result<Ty, Unsettled>| match side {
            Ok(ty) => crate::ty::print::ty(&self.types, names, *ty),
            Err(open) => format!("{open:?}"),
        };
        let effect = |side: &Result<Effect, Unsettled>| match side {
            Ok(effect) => crate::ty::print::effect(names, effect),
            Err(open) => format!("{open:?}"),
        };
        let terms = match &self.mismatch {
            Mismatch::Heads { a, b } | Mismatch::Mark { a, b } => vec![ty(a), ty(b)],
            Mismatch::Occurs { var, in_term: term } | Mismatch::Bound { var, term } => vec![ty(var), ty(term)],
            Mismatch::OpaqueLacksField { object: term, .. }
            | Mismatch::OpaqueLacksVariant { r#enum: term, .. }
            | Mismatch::ReferenceToReference { target: term }
            | Mismatch::IdentityEscapes { term } => vec![ty(term)],
            Mismatch::Effects { a, b, .. } => vec![effect(a), effect(b)],
            Mismatch::TwoEnums { .. } | Mismatch::Conversions { .. } | Mismatch::NotOneOf { .. } | Mismatch::Values { .. } => Vec::new(),
        };
        std::iter::once(self.mismatch.shape().to_string()).chain(terms).collect::<Vec<_>>().join(" ")
    }
}

impl Clone for FrozenMismatch {
    fn clone(&self) -> Self {
        let mut types = Types::new();
        let mismatch = self.transplanted(&mut types);
        FrozenMismatch { mismatch, types }
    }
}

/// Equal where the shapes and the types they hold are, whichever tables
/// the types are interned in.
impl PartialEq for FrozenMismatch {
    fn eq(&self, other: &Self) -> bool {
        let mut types = Types::new();
        self.transplanted(&mut types) == other.transplanted(&mut types)
    }
}

impl std::fmt::Debug for FrozenMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kinds: Vec<_> = self.types.ids().map(|id| (id, self.types.kind(id))).collect();
        f.debug_struct("FrozenMismatch").field("mismatch", &self.mismatch).field("types", &kinds).finish()
    }
}

/// A mismatch's terms frozen by the check's store, into one table.
struct Freeze<'f> {
    infer: &'f Infer,
    into: &'f mut Types<Concrete>,
    memo: FreezeMemo,
}

impl MapMismatch<Solving, Frozen> for Freeze<'_> {
    fn ty(&mut self, ty: crate::infer::Term) -> Result<Ty, Unsettled> {
        self.infer.freeze(self.into, &mut self.memo, ty)
    }

    fn ty_var(&mut self, var: TyVarId) -> Result<Ty, Unsettled> {
        self.infer.freeze_var(self.into, &mut self.memo, var)
    }

    fn effect(&mut self, effect: EffectTerm<InferPhase>) -> Result<Effect, Unsettled> {
        self.infer.freeze_effect(&effect)
    }
}

/// A frozen mismatch's types moved from one table into another.
struct Transplant<'t> {
    from: &'t Types<Concrete>,
    into: &'t mut Types<Concrete>,
}

impl MapMismatch<Frozen, Frozen> for Transplant<'_> {
    fn ty(&mut self, ty: Result<Ty, Unsettled>) -> Result<Ty, Unsettled> {
        ty.map(|ty| self.into.transplant(self.from, ty))
    }

    fn ty_var(&mut self, var: Result<Ty, Unsettled>) -> Result<Ty, Unsettled> {
        self.ty(var)
    }

    fn effect(&mut self, effect: Result<Effect, Unsettled>) -> Result<Effect, Unsettled> {
        effect
    }
}

impl RefusalKind {
    /// The refusal in notation (no prose): its kind, and each name it
    /// holds as the program writes it; a mismatch by its shape and the
    /// types it holds.
    pub fn written(&self, names: &acvus_utils::Interner) -> String {
        let name = |astr: &Astr| names.resolve(*astr).to_string();
        let path = |unit: &QualifiedRef| crate::ty::print::qualified(names, *unit);
        match self {
            RefusalKind::InputAssigned { input } => format!("InputAssigned ${}", name(input)),
            RefusalKind::UnnamedVariant { tag } => format!("UnnamedVariant {}", name(tag)),
            RefusalKind::UnknownName { name: unknown } => format!("UnknownName {}", name(unknown)),
            RefusalKind::UnknownContext { context } => format!("UnknownContext @{}", path(context)),
            RefusalKind::ExternAsValue { name: written } => format!("ExternAsValue {}", path(written)),
            RefusalKind::UnitCalledByPosition { unit } => format!("UnitCalledByPosition {}", path(unit)),
            RefusalKind::UnknownArgument { unit, name: argument } => format!("UnknownArgument {}({}:)", path(unit), name(argument)),
            RefusalKind::DuplicateArgument { unit, name: argument } => format!("DuplicateArgument {}({}:)", path(unit), name(argument)),
            RefusalKind::MissingArgument { unit, name: argument } => format!("MissingArgument {}({}:)", path(unit), name(argument)),
            RefusalKind::VariantArity { tag } => format!("VariantArity {}", name(tag)),
            RefusalKind::UnknownCastTarget { name: target } => format!("UnknownCastTarget {}", name(target)),
            RefusalKind::Mismatch(mismatch) => format!("Mismatch {}", mismatch.written(names)),
            RefusalKind::ClosureExceedsSlot(exceeding) => format!("ClosureExceedsSlot {exceeding:?}"),
            RefusalKind::CastNotAdmitted { from, to } => match from {
                Some(from) => format!("CastNotAdmitted {from:?} as {to:?}"),
                None => format!("CastNotAdmitted as {to:?}"),
            },
            RefusalKind::NotOneOfFunctionValues { value } => match value {
                crate::ty::FnValue::Lambda(_) => "NotOneOfFunctionValues lambda".to_string(),
            },
            RefusalKind::NoDeclaration { name: missing } => format!("NoDeclaration {}", name(missing)),
            RefusalKind::SeveralDeclarations { name: several, left } => {
                format!("SeveralDeclarations {} [{}]", name(several), left.iter().map(path).collect::<Vec<_>>().join(", "))
            }
            RefusalKind::NeverKnown { open } => format!("NeverKnown {}", name(open)),
            RefusalKind::LendsNoParts { name: lent } => format!("LendsNoParts {}", name(lent)),
            RefusalKind::MutLendThroughShared { name: lent } => format!("MutLendThroughShared {}", name(lent)),
            RefusalKind::Unimplemented { feature } => format!("Unimplemented {feature:?}"),
            RefusalKind::LambdaRecursion
            | RefusalKind::NamedArgumentsToNonUnit
            | RefusalKind::HoldsLoans
            | RefusalKind::StoreThroughShared
            | RefusalKind::LendMutThroughShared
            | RefusalKind::FnDeclInScript
            | RefusalKind::ReturnInTemplate
            | RefusalKind::PatternInValue
            | RefusalKind::NoConversion
            | RefusalKind::IdentityEscapes
            | RefusalKind::NotIterable
            | RefusalKind::NotTryable
            | RefusalKind::NotAReference
            | RefusalKind::NotBoolOrInteger
            | RefusalKind::NotBool
            | RefusalKind::NotInteger
            | RefusalKind::OutsideLoop
            | RefusalKind::GrowsOpaque
            | RefusalKind::Undecided
            | RefusalKind::BoundUnanswered
            | RefusalKind::Unordered => format!("{self:?}"),
        }
    }
}

/// A refusal as the check makes it. A mismatch holds the check's terms
/// until writeback freezes them (`frozen`).
#[derive(Debug)]
pub(super) enum Made {
    Refusal(Refusal),
    Mismatch { mismatch: Mismatch, span: Span },
}

/// The one way a refusal is made (S4).
pub(super) fn push(cx: &mut Ctxt<'_>, kind: RefusalKind, span: Span) {
    cx.out.refusals.push(Made::Refusal(Refusal { kind, span }));
}

/// Every refusal the check made, in the order it made them, each
/// mismatch's terms frozen by `infer`, the check's store.
pub(super) fn frozen(infer: &Infer, made: Vec<Made>) -> Vec<Refusal> {
    made.into_iter()
        .map(|made| match made {
            Made::Refusal(refusal) => refusal,
            Made::Mismatch { mismatch, span } => {
                Refusal { kind: RefusalKind::Mismatch(FrozenMismatch::freeze(infer, mismatch)), span }
            }
        })
        .collect()
}

/// B's refusal in the dimension that refuses: growth into an opaque term is
/// C's (types §1), no conversion at all is its own, and every other is B's
/// relation failing. A lambda's effect that does not fit a function type it
/// meets is refused at the lambda as `ClosureExceedsSlot`, as it is when
/// its effect grows past that type later (D96, types §6: the kind does
/// not depend on which came first).
pub(super) fn mismatch(cx: &mut Ctxt<'_>, mismatch: Mismatch, span: Span) {
    if let Mismatch::Effects { a, b, a_is, b_is } = &mismatch {
        let lambda = |effect: &EffectTerm<InferPhase>| {
            cx.out
                .lambdas
                .iter()
                .find(|lambda| cx.infer.one_effect_variable(&lambda.effect, effect))
                .map(|lambda| lambda.span)
        };
        let exceeding = match (lambda(a), lambda(b)) {
            (Some(at), _) => Some((at, a_is, b_is)),
            (None, Some(at)) => Some((at, b_is, a_is)),
            (None, None) => None,
        };
        if let Some((at, own, slot)) = exceeding {
            let exceeding = Exceeding::Effect { needs: own.join(slot), slot: slot.clone() };
            push(cx, RefusalKind::ClosureExceedsSlot(exceeding), at);
            return;
        }
    }
    match mismatch {
        Mismatch::OpaqueLacksField { .. } | Mismatch::OpaqueLacksVariant { .. } => push(cx, RefusalKind::GrowsOpaque, span),
        Mismatch::NotOneOf { value, .. } => push(cx, RefusalKind::NotOneOfFunctionValues { value }, span),
        Mismatch::Conversions { candidates } if candidates.is_empty() => push(cx, RefusalKind::NoConversion, span),
        Mismatch::IdentityEscapes { .. } => push(cx, RefusalKind::IdentityEscapes, span),
        other => cx.out.refusals.push(Made::Mismatch { mismatch: other, span }),
    }
}
