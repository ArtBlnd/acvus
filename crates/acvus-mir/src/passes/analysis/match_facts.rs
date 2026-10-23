//! `MatchFacts`: every value's `MatchValue` over one body, conditional on
//! the arms its facts leave (the sparse engine, Wegman and Zadeck), for
//! `check::matchck` (patterns §3). Cost: the sparse engine's; the variants'
//! height is the body's variants, the constant's 3, the interval's bounded
//! by its widening.
//!
//! An instruction that is not a value operation (`transfer`):
//! - `MakeVariant { variant, .. }`: variants `Of({variant})`; constant and
//!   interval top.
//! - `TestVariant { src, variant }`: a `bool` whose constant is, from
//!   `src`'s variants: `true` where they are `Of({variant})`; `false` where
//!   they are `Of(s)` and `variant ∉ s`; bottom where they are `None`; top
//!   otherwise, `Any` included. A `src` that is a reference has variants
//!   `Any`.
//! - Every other: top.
//!
//! A value's top and its widening at a loop header are `IntervalValues`'
//! for the interval: an integer value's interval lies within its type's
//! range, and a growing end widens to its type's bound.
//!
//! Which successors run (`executable`):
//! - `Diamond`, `JumpIf`: by the condition's constant, as `ConstantValues`.
//! - `Switch`: by the fact of the value it switches on, which is its
//!   operand, or that operand's target where the operand is a reference
//!   (`ir::tested_ty`). The analysis tracks no storage's contents, so a
//!   target's fact is the top of its type, an integer's interval its type's
//!   range (undefined behaviour §2).
//!   - On `Variant` keys: the set is the value's variants, `Any` read as
//!     every variant of its type. An arm runs where its key is in the set;
//!     `default` where the set holds a variant no key names.
//!   - On `Bool` keys: by the constant; `default` where a `bool` value the
//!     constant allows has no key.
//!   - On `Int` or `Char` keys: by the interval; `default` where the
//!     interval holds a value no key names.
//! - Every other terminator: all its successors.

use std::borrow::Cow;
use std::collections::BTreeSet;

use acvus_ast::Literal;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use super::values::OperandTypes;
use crate::ir::cfg::{BlockId, BodyId, Inst, Module, Op, Terminator};
use crate::ir::{BinOp, SwitchKey, UnaryOp, ValueId, tested_ty};
use crate::passes::analysis::{Analysis, Executable, Queries, SparseAnalysis, SparseFacts, solve_sparse};
use crate::passes::domain::{
    Bound, Constant, ConstantDomain, Interval, IntervalDomain, JoinSemiLattice, ValueDomain, Variants, VariantsDomain, Widen,
};
use crate::ty::structural::{VariantKey, structure};
use crate::ty::{CastTy, TyKind};

/// This analysis's own product: a value's variants, its constant and its
/// interval, run together in one engine so that a fact of any of them may
/// rule an arm out. Each component is its own domain's; the product only
/// pairs them, reducing nothing between them.

#[derive(Debug, Clone, PartialEq)]
pub struct MatchValue {
    pub variants: Variants,
    pub constant: Constant,
    pub interval: Interval,
}

/// Each component joins, compares and widens by its own domain.
impl JoinSemiLattice for MatchValue {
    fn bottom() -> Self {
        MatchValue { variants: Variants::bottom(), constant: Constant::bottom(), interval: Interval::bottom() }
    }

    fn join(&mut self, other: &Self) -> bool {
        let variants = self.variants.join(&other.variants);
        let constant = self.constant.join(&other.constant);
        let interval = self.interval.join(&other.interval);
        variants || constant || interval
    }

    fn le(&self, other: &Self) -> bool {
        self.variants.le(&other.variants) && self.constant.le(&other.constant) && self.interval.le(&other.interval)
    }
}

impl Widen for MatchValue {
    fn widen(&mut self, next: &Self) -> bool {
        let variants = self.variants.widen(&next.variants);
        let constant = self.constant.widen(&next.constant);
        let interval = self.interval.widen(&next.interval);
        variants || constant || interval
    }
}

/// Each value operation is each component domain's.
pub struct MatchDomain {
    pub variants: VariantsDomain,
    pub constant: ConstantDomain,
    pub interval: IntervalDomain,
}

impl ValueDomain for MatchDomain {
    type Value = MatchValue;

    fn top(&self) -> MatchValue {
        MatchValue { variants: self.variants.top(), constant: self.constant.top(), interval: self.interval.top() }
    }

    fn constant(&self, literal: &Literal, ty: CastTy) -> MatchValue {
        MatchValue {
            variants: self.variants.constant(literal, ty),
            constant: self.constant.constant(literal, ty),
            interval: self.interval.constant(literal, ty),
        }
    }

    fn binary(&self, op: BinOp, ty: CastTy, left: &MatchValue, right: &MatchValue) -> MatchValue {
        MatchValue {
            variants: self.variants.binary(op, ty, &left.variants, &right.variants),
            constant: self.constant.binary(op, ty, &left.constant, &right.constant),
            interval: self.interval.binary(op, ty, &left.interval, &right.interval),
        }
    }

    fn unary(&self, op: UnaryOp, ty: CastTy, operand: &MatchValue) -> MatchValue {
        MatchValue {
            variants: self.variants.unary(op, ty, &operand.variants),
            constant: self.constant.unary(op, ty, &operand.constant),
            interval: self.interval.unary(op, ty, &operand.interval),
        }
    }

    fn cast(&self, from: CastTy, to: CastTy, operand: &MatchValue) -> MatchValue {
        MatchValue {
            variants: self.variants.cast(from, to, &operand.variants),
            constant: self.constant.cast(from, to, &operand.constant),
            interval: self.interval.cast(from, to, &operand.interval),
        }
    }
}

pub struct MatchFacts;

impl Analysis for MatchFacts {
    type Facts = SparseFacts<MatchDomain>;

    const READS: &'static [crate::ir::cfg::Aspect] = &[
        crate::ir::cfg::Aspect::Instructions,
        crate::ir::cfg::Aspect::CfgShape,
        crate::ir::cfg::Aspect::Types,
    ];

    fn compute(_: &mut Queries<'_, '_>, module: &Module, body: BodyId) -> Self::Facts {
        solve_sparse(&Matches::new(module, body), module.body(body))
    }
}

impl MatchFacts {
    /// The blocks of `body` the entry reaches through the edges `facts` let
    /// run, by the same `executable` the engine ran. Every edge the engine
    /// found executable is one these final facts let run, and no other,
    /// since a fact only rises and `executable` only grows with it. Cost:
    /// O(blocks + each terminator's operands and keys).
    pub fn reached(facts: &SparseFacts<MatchDomain>, module: &Module, body: BodyId) -> FxHashSet<BlockId> {
        let matches = Matches::new(module, body);
        let body = module.body(body);
        let mut reached = FxHashSet::default();
        let mut to_visit = vec![body.entry()];
        while let Some(block) = to_visit.pop() {
            if !reached.insert(block) {
                continue;
            }
            let terminator = body.block(block).terminator();
            let operands: SmallVec<[&MatchValue; 4]> = terminator
                .operands()
                .into_iter()
                .filter(|value| !body.is_token(*value))
                .map(|value| facts.of(value))
                .collect();
            let allowed = matches.executable(terminator, &operands);
            for edge in terminator.edges() {
                let runs = match &allowed {
                    Executable::All => true,
                    Executable::Only(targets) => targets.contains(&edge.to),
                };
                if runs {
                    to_visit.push(edge.to);
                }
            }
        }
        reached
    }
}

/// The sparse analysis over `MatchDomain`, by the module doc's tables.
struct Matches<'m> {
    domain: MatchDomain,
    operand_types: OperandTypes<'m>,
}

impl<'m> Matches<'m> {
    fn new(module: &'m Module, body: BodyId) -> Matches<'m> {
        Matches {
            domain: MatchDomain { variants: VariantsDomain, constant: ConstantDomain, interval: IntervalDomain },
            operand_types: OperandTypes { types: module.types(), body: module.body(body) },
        }
    }

    fn kind(&self, value: ValueId) -> &'m TyKind<crate::ty::Concrete> {
        let OperandTypes { types, body } = self.operand_types;
        types.kind(body.ty_of(value))
    }

    fn is_reference(&self, value: ValueId) -> bool {
        matches!(self.kind(value), TyKind::Ref(..))
    }

    /// Every variant of the type a `Switch` on `value` reads
    /// (`ir::tested_ty`).
    ///
    /// # Panics
    /// That type is no enum, `Option` or `Result`: lower switches on
    /// variants only of those.
    fn variants(&self, value: ValueId) -> BTreeSet<VariantKey> {
        let OperandTypes { types, body } = self.operand_types;
        let tested = tested_ty(types, body.ty_of(value));
        structure(types, tested)
            .and_then(|structure| structure.variants())
            .unwrap_or_else(|| panic!("a switch on variants reads an enum, an Option or a Result, not {:?}", types.kind(tested)))
            .iter()
            .map(|variant| variant.key)
            .collect()
    }

    /// Whether `src`, of variants `variants`, holds the variant `variant`.
    fn tested(&self, src: ValueId, variant: VariantKey, variants: &Variants) -> Constant {
        let variants = if self.is_reference(src) { &Variants::Any } else { variants };
        match variants {
            Variants::None => Constant::Bottom,
            Variants::Of(held) if !held.contains(&variant) => Constant::Known(Literal::Bool(false)),
            Variants::Of(held) if held.len() == 1 => Constant::Known(Literal::Bool(true)),
            Variants::Of(_) | Variants::Any => Constant::Top,
        }
    }

    /// The fact of the value a `Switch` on `tag` switches on: the operand's
    /// own, or, where it is a reference, the top of the type the `Switch`
    /// reads (`ir::tested_ty`).
    fn switched<'f>(&self, tag: ValueId, fact: &'f MatchValue) -> Cow<'f, MatchValue> {
        let OperandTypes { types, body } = self.operand_types;
        match self.is_reference(tag) {
            true => {
                let tested = tested_ty(types, body.ty_of(tag));
                Cow::Owned(MatchValue { interval: self.operand_types.interval_top_of(tested), ..self.domain.top() })
            }
            false => Cow::Borrowed(fact),
        }
    }

    fn switch(&self, tag: ValueId, arms: &[(SwitchKey, BlockId, Vec<ValueId>)], default: Option<BlockId>, fact: &MatchValue) -> Executable {
        let keys: Vec<SwitchKey> = arms.iter().map(|(key, _, _)| *key).collect();
        let fact = self.switched(tag, fact);
        let chosen = match keys.first() {
            Some(SwitchKey::Variant(_)) => self.variant_choice(tag, &keys, &fact.variants),
            Some(SwitchKey::Bool(_)) => bool_choice(&keys, &fact.constant),
            Some(SwitchKey::Int(_) | SwitchKey::Char(_)) => ordinal_choice(&keys, &fact.interval),
            Some(SwitchKey::Str(_)) | None => return Executable::All,
        };
        let arms_run = arms.iter().filter(|(key, _, _)| chosen.keys.contains(key)).map(|(_, to, _)| *to);
        let default_runs = default.filter(|_| chosen.unkeyed);
        Executable::Only(arms_run.chain(default_runs).collect())
    }

    /// The set is the operand's variants, `Any` read as every variant of
    /// the type the `Switch` reads.
    fn variant_choice(&self, tag: ValueId, keys: &[SwitchKey], variants: &Variants) -> Choice {
        let held = match variants {
            Variants::None => BTreeSet::new(),
            Variants::Of(held) => held.clone(),
            Variants::Any => self.variants(tag),
        };
        let named: BTreeSet<VariantKey> = keys.iter().map(|key| key_variant(*key)).collect();
        Choice {
            keys: held.iter().map(|variant| SwitchKey::Variant(*variant)).filter(|key| keys.contains(key)).collect(),
            unkeyed: !held.is_subset(&named),
        }
    }
}

/// Which keys of a `Switch` a value's fact lets run, and whether it lets
/// run a value no key names.
struct Choice {
    keys: Vec<SwitchKey>,
    unkeyed: bool,
}

fn key_variant(key: SwitchKey) -> VariantKey {
    match key {
        SwitchKey::Variant(variant) => variant,
        other => panic!("a switch on variants has only variant keys, not {other:?}"),
    }
}

/// By the constant: each `bool` it allows.
fn bool_choice(keys: &[SwitchKey], constant: &Constant) -> Choice {
    let allowed: &[bool] = match constant {
        Constant::Bottom => &[],
        Constant::Known(Literal::Bool(false)) => &[false],
        Constant::Known(Literal::Bool(true)) => &[true],
        Constant::Top => &[false, true],
        Constant::Known(other) => panic!("a switch on bool keys reads a bool, not {other:?}"),
    };
    let named: Vec<bool> = keys
        .iter()
        .map(|key| match key {
            SwitchKey::Bool(value) => *value,
            other => panic!("a switch on bool keys has only bool keys, not {other:?}"),
        })
        .collect();
    Choice {
        keys: allowed.iter().filter(|value| named.contains(value)).map(|value| SwitchKey::Bool(*value)).collect(),
        unkeyed: allowed.iter().any(|value| !named.contains(value)),
    }
}

/// By the interval: each key it holds, a `char` read at its code point.
fn ordinal_choice(keys: &[SwitchKey], interval: &Interval) -> Choice {
    let ordinal = |key: &SwitchKey| match key {
        SwitchKey::Int(value) => *value,
        SwitchKey::Char(value) => i128::from(u32::from(*value)),
        other => panic!("a switch on int or char keys has only such keys, not {other:?}"),
    };
    let Interval::Range { lo, hi } = interval else {
        return Choice { keys: Vec::new(), unkeyed: false };
    };
    let held = |value: i128| *lo <= Bound::Finite(value) && Bound::Finite(value) <= *hi;
    let named: BTreeSet<i128> = keys.iter().map(ordinal).filter(|value| held(*value)).collect();
    let unkeyed = match (lo, hi) {
        (Bound::Finite(lo), Bound::Finite(hi)) => named.len() as i128 != hi - lo + 1,
        _ => true,
    };
    Choice { keys: keys.iter().filter(|key| held(ordinal(key))).copied().collect(), unkeyed }
}

/// On `bottom` no arm runs yet, on a known truth only its arm, else both,
/// as `ConstantValues`.
fn branch(condition: &Constant, on_true: BlockId, on_false: BlockId) -> Executable {
    match condition {
        Constant::Bottom => Executable::Only(Vec::new()),
        Constant::Top => Executable::All,
        Constant::Known(Literal::Bool(true)) => Executable::Only(vec![on_true]),
        Constant::Known(Literal::Bool(false)) => Executable::Only(vec![on_false]),
        Constant::Known(other) => panic!("a branch's condition is a bool, not {other:?}"),
    }
}

impl SparseAnalysis for Matches<'_> {
    type Domain = MatchDomain;

    fn transfer(&self, inst: &Inst, operands: &[&MatchValue]) -> MatchValue {
        if let Some(value) = self.operand_types.value_operation(&self.domain, inst, operands) {
            return value;
        }
        let result = inst.result().expect("the engine evaluates an instruction that yields a value");
        match (inst.op(), operands) {
            (Op::MakeVariant { variant, .. }, _) => MatchValue { variants: Variants::Of(BTreeSet::from([*variant])), ..self.top(result) },
            (Op::TestVariant { src, variant }, [src_fact]) => {
                MatchValue { constant: self.tested(*src, *variant, &src_fact.variants), ..self.top(result) }
            }
            (Op::TestVariant { .. }, _) => panic!("a TestVariant is handed {} operand facts, not one", operands.len()),
            _ => self.top(result),
        }
    }

    fn executable(&self, terminator: &Terminator, operands: &[&MatchValue]) -> Executable {
        match terminator {
            Terminator::JumpIf { then_to, else_to, .. } | Terminator::Diamond { then_to, else_to, .. } => {
                branch(&operands[0].constant, *then_to, *else_to)
            }
            Terminator::Switch { tag, arms, default, .. } => {
                self.switch(*tag, arms, default.as_ref().map(|(to, _)| *to), operands[0])
            }
            Terminator::Jump { .. }
            | Terminator::For { .. }
            | Terminator::While { .. }
            | Terminator::Return { .. }
            | Terminator::Diverge
            | Terminator::NoArmMatched { .. } => Executable::All,
        }
    }

    /// What a terminator fills itself, a `For`'s counter or element, is no
    /// instruction the tables name: its top.
    fn supplied(&self, _terminator: &Terminator, to: BlockId, index: usize, _operands: &[&MatchValue]) -> MatchValue {
        self.top(self.operand_types.body.block(to).params()[index])
    }

    /// Each component's top, the interval's by `IntervalValues`' rule.
    fn top(&self, value: ValueId) -> MatchValue {
        MatchValue { interval: self.operand_types.interval_top(value), ..self.domain.top() }
    }

    /// Each component widens by its own domain, the interval by
    /// `IntervalValues`' rule.
    fn widen(&self, value: ValueId, fact: &mut MatchValue, next: &MatchValue) -> bool {
        let variants = fact.variants.widen(&next.variants);
        let constant = fact.constant.widen(&next.constant);
        let interval = self.operand_types.widen_interval(value, &mut fact.interval, &next.interval);
        variants || constant || interval
    }
}
