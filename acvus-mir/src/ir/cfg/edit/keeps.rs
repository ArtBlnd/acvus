//! Invariants an editor keeps, one checker each. They run after every edit
//! in debug and test builds; `passes::pipeline` reuses them after each
//! transform.

use acvus_utils::QualifiedRef;
use rustc_hash::{FxHashMap, FxHashSet};

use super::dominators::{Dominators, def_points};
use crate::ir::cfg::body::InstId;
use crate::ir::cfg::{BlockId, Body, Inst, Op, OrderUse, Site, Terminator};
use crate::ir::{BinOp, Callee, ForSource, IndexMode, Intrinsic, PathSeg, RefTarget, SwitchKey, UnaryOp, ValueId, known_len};
use crate::ty::structural;
use crate::ty::{CastTy, Concrete, IntTy, Mark, MarkTerm, Mutability, Ty, TyKind, Types};

/// A breach of an invariant: a compiler defect, never a refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub invariant: &'static str,
    pub detail: String,
}

pub trait Invariant {
    const NAME: &'static str;
    fn check(body: &Body, types: &Types<Concrete>) -> Result<(), Violation>;
}

/// A set of invariants, as a tuple.
pub trait Invariants {
    fn check(body: &Body, types: &Types<Concrete>) -> Result<(), Violation>;
}

impl Invariants for () {
    fn check(_: &Body, _: &Types<Concrete>) -> Result<(), Violation> {
        Ok(())
    }
}

macro_rules! tuple_invariants {
    ($($name:ident),+) => {
        impl<$($name),+> Invariants for ($($name,)+)
        where
            $($name: Invariants,)+
        {
            fn check(body: &Body, types: &Types<Concrete>) -> Result<(), Violation> {
                $($name::check(body, types)?;)+
                Ok(())
            }
        }
    };
}
tuple_invariants!(A, B);
tuple_invariants!(A, B, C, D);

macro_rules! invariant {
    ($name:ident, $checker:ident, $doc:literal) => {
        #[doc = $doc]
        pub struct $name;
        impl Invariant for $name {
            const NAME: &'static str = stringify!($name);
            fn check(body: &Body, types: &Types<Concrete>) -> Result<(), Violation> {
                $checker(body, types).map_err(|detail| Violation { invariant: Self::NAME, detail })
            }
        }
        impl Invariants for $name {
            fn check(body: &Body, types: &Types<Concrete>) -> Result<(), Violation> {
                <$name as Invariant>::check(body, types)
            }
        }
    };
}

invariant!(TypesAgree, types_agree, "Every operand's type is what its instruction takes, and every edge's arguments agree with its target's parameters.");
invariant!(DefDominatesUses, def_dominates_uses, "Every value's definition dominates each of its uses.");
invariant!(OrderChainWhole, order_chain_whole, "Every effectful instruction consumes exactly one live token, and every token but a Merge's reaches the body's return (effects §2-3); a Merge nothing consumes orders nothing and is a dead value (D65). A place operation on a volatile context stands on the chain, and one on a context that is not volatile does not (context §3).");
invariant!(StructureKept, structure_kept, "Every `Diamond`'s and `Switch`'s join and every loop's body and exit are blocks of the body (D45, D57).");
invariant!(NeverDiverges, never_diverges, "In every block the entry reaches, an instruction yielding `!` is the block's last, and the block ends in `Diverge`: no value of type `!` exists, so no run goes past one (types §4, control-flow).");
invariant!(SplitTakenOnce, split_taken_once, "Every start of a split call is taken exactly once on every path from it (effects §6).");

/// A checker's finding: what breaks the invariant, in words.
type Found = Result<(), String>;

fn found(holds: bool, detail: impl FnOnce() -> String) -> Found {
    match holds {
        true => Ok(()),
        false => Err(detail()),
    }
}

// -- NeverDiverges ------------------------------------------------------------

fn never_diverges(body: &Body, types: &Types<Concrete>) -> Found {
    let reached = reached_blocks(body);
    for (id, block) in body.blocks() {
        if !reached.contains(&id) {
            continue;
        }
        let insts: Vec<&Inst> = block.insts().collect();
        for (at, inst) in insts.iter().enumerate() {
            let Some(result) = inst.result() else {
                continue;
            };
            if !matches!(types.kind(body.ty_of(result)), TyKind::Never) {
                continue;
            }
            let last = at + 1 == insts.len();
            found(last && matches!(block.terminator(), Terminator::Diverge), || {
                format!("{id:?} {:?} {:?} yields `!` and is followed by {:?}", inst.id(), inst.op(), match last {
                    true => format!("{:?}", block.terminator()),
                    false => format!("{:?}", insts[at + 1].op()),
                })
            })?;
        }
    }
    Ok(())
}

/// The blocks some path from the entry reaches.
fn reached_blocks(body: &Body) -> FxHashSet<BlockId> {
    let mut reached = FxHashSet::default();
    let mut to_visit = vec![body.entry()];
    while let Some(block) = to_visit.pop() {
        if reached.insert(block) {
            to_visit.extend(body.block(block).terminator().edges().iter().map(|edge| edge.to));
        }
    }
    reached
}

// -- TypesAgree --------------------------------------------------------------

/// Checks each instruction's operands and result against what its operation
/// takes and yields, every edge against its target's parameters, every
/// switch key against its scrutinee, every loop source against what it
/// hands its body, and every return against the body's result type. A
/// function loaded or called by name, like a context, is typed by
/// declarations the body does not hold: those are checked only as far as
/// the body shows (a function type, a type without positions).
fn types_agree(body: &Body, types: &Types<Concrete>) -> Found {
    let check = TypeCheck { body, types };
    for (id, block) in body.blocks() {
        for inst in block.insts() {
            check
                .inst(inst)
                .map_err(|detail| format!("{id:?} {:?} {:?}: {detail}", inst.id(), inst.op()))?;
        }
        check.terminator(id, block.terminator())?;
    }
    Ok(())
}

struct TypeCheck<'a> {
    body: &'a Body,
    types: &'a Types<Concrete>,
}

impl TypeCheck<'_> {
    fn ty(&self, value: ValueId) -> Ty {
        self.body.ty_of(value)
    }

    fn kind(&self, ty: Ty) -> &TyKind<Concrete> {
        self.types.kind(ty)
    }

    fn is(&self, value: ValueId, expected: impl Fn(&TyKind<Concrete>) -> bool, what: &str) -> Found {
        let ty = self.ty(value);
        found(expected(self.kind(ty)), || format!("{value:?} is {:?}, not {what}", self.kind(ty)))
    }

    /// Whether `to` is `from` with marks raised: the same type, each mark
    /// either the same or marked in `from` and unmarked in `to` (types §8).
    fn raises_marks(&self, from: Ty, to: Ty) -> bool {
        if from == to {
            return true;
        }
        let (from_kind, to_kind) = (self.kind(from), self.kind(to));
        let raised = match (from_kind, to_kind) {
            (TyKind::UserDefined { marked: before, .. }, TyKind::UserDefined { marked: after, .. }) => {
                before == after || (*before == MarkTerm::Known(Mark::Marked) && *after == MarkTerm::Known(Mark::Unmarked))
            }
            _ => true,
        };
        let shell = |kind: &TyKind<Concrete>| match kind.with_children(&mut |_| from) {
            TyKind::UserDefined { id, type_args, effect, identity, .. } => {
                TyKind::UserDefined { id, type_args, effect, identity, marked: MarkTerm::Known(Mark::Unmarked) }
            }
            other => other,
        };
        let parts = |kind: &TyKind<Concrete>| {
            let mut parts = Vec::new();
            kind.for_each_child(|part| parts.push(part));
            parts
        };
        let (from_parts, to_parts) = (parts(from_kind), parts(to_kind));
        raised
            && shell(from_kind) == shell(to_kind)
            && from_parts.len() == to_parts.len()
            && from_parts.iter().zip(&to_parts).all(|(from, to)| self.raises_marks(*from, *to))
    }

    fn same(&self, value: ValueId, expected: Ty, what: &str) -> Found {
        let ty = self.ty(value);
        found(ty == expected, || {
            format!("{value:?} is {:?}, and {what} is {:?}", self.kind(ty), self.kind(expected))
        })
    }

    fn is_bool(&self, value: ValueId) -> Found {
        self.is(value, |kind| matches!(kind, TyKind::Bool), "Bool")
    }

    fn is_u64(&self, value: ValueId) -> Found {
        self.is(value, |kind| matches!(kind, TyKind::Int(IntTy::U64)), "u64")
    }

    fn is_int(&self, value: ValueId) -> Found {
        self.is(value, |kind| matches!(kind, TyKind::Int(_)), "an integer")
    }

    fn is_text(&self, value: ValueId) -> Found {
        self.is(
            value,
            |kind| match kind {
                TyKind::String => true,
                TyKind::Ref(_, lent) => matches!(self.kind(*lent), TyKind::String | TyKind::Str),
                _ => false,
            },
            "String, &String or &str",
        )
    }

    fn is_lent_text(&self, value: ValueId) -> Found {
        self.is(
            value,
            |kind| matches!(kind, TyKind::Ref(_, lent) if matches!(self.kind(*lent), TyKind::String | TyKind::Str)),
            "&String or &str",
        )
    }

    fn result(inst: &Inst) -> Result<ValueId, String> {
        inst.result().ok_or_else(|| "the instruction yields a value and has no result".to_string())
    }

    fn inst(&self, inst: &Inst) -> Found {
        found(inst.op().has_result() == inst.result().is_some(), || {
            "the instruction's result does not match whether its operation yields one".to_string()
        })?;
        if let Some(order) = inst.order() {
            for token in order.consumed().into_iter().chain(order.yielded()) {
                self.is(token, |kind| matches!(kind, TyKind::Order), "Order")?;
            }
        }
        match inst.op() {
            Op::Const { value } => {
                let result = Self::result(inst)?;
                let literal_fits = |kind: &TyKind<Concrete>| match value {
                    acvus_ast::Literal::Int(_) => matches!(kind, TyKind::Int(_)),
                    acvus_ast::Literal::IntOf(suffixed) => *kind == TyKind::Int(suffixed.width.into()),
                    acvus_ast::Literal::String(_) => matches!(kind, TyKind::String),
                    acvus_ast::Literal::Float(_) => matches!(kind, TyKind::Float),
                    acvus_ast::Literal::Char(_) => matches!(kind, TyKind::Char),
                    acvus_ast::Literal::Bool(_) => matches!(kind, TyKind::Bool),
                    acvus_ast::Literal::Unit => matches!(kind, TyKind::Unit),
                    acvus_ast::Literal::Bytes(bytes) => matches!(
                        kind,
                        TyKind::Array(byte, len, _)
                            if matches!(self.kind(*byte), TyKind::Int(IntTy::U8)) && known_len(len) == bytes.len()
                    ),
                    acvus_ast::Literal::List(_) => matches!(kind, TyKind::Array(..)),
                };
                self.is(result, literal_fits, "the literal's type")
            }
            Op::ConstStr { .. } => self.is(
                Self::result(inst)?,
                |kind| matches!(kind, TyKind::Ref(Mutability::Shared, lent) if matches!(self.kind(*lent), TyKind::Str)),
                "&str",
            ),
            Op::StringConcat { parts } => {
                self.is(Self::result(inst)?, |kind| matches!(kind, TyKind::String), "String")?;
                parts.iter().try_for_each(|part| self.is_text(*part))
            }
            Op::StringAppend { target, part } => {
                self.is(
                    *target,
                    |kind| matches!(kind, TyKind::Ref(Mutability::Mut, lent) if matches!(self.kind(*lent), TyKind::String)),
                    "&mut String",
                )?;
                self.is_text(*part)
            }
            Op::StringEq { a, b } => {
                self.is_bool(Self::result(inst)?)?;
                self.is_lent_text(*a)?;
                self.is_lent_text(*b)
            }
            Op::StringClone { src } => {
                self.is(Self::result(inst)?, |kind| matches!(kind, TyKind::String), "String")?;
                self.is(
                    *src,
                    |kind| match kind {
                        TyKind::String => true,
                        TyKind::Ref(_, lent) => matches!(self.kind(*lent), TyKind::String),
                        _ => false,
                    },
                    "String or &String",
                )
            }
            Op::Ref { target, path, mutability } => {
                let result = Self::result(inst)?;
                let TyKind::Ref(made, lent) = self.kind(self.ty(result)) else {
                    return Err(format!("{result:?} is not a reference"));
                };
                found(made == mutability, || format!("{result:?} is not a {mutability:?} reference"))?;
                let lent = *lent;
                let places = self.place(target, path)?;
                found(places.contains(&lent), || format!("{result:?} lends another type than the place holds"))
            }
            Op::Take { target, path, .. } => self.one_of(Self::result(inst)?, &self.place(target, path)?, "the place"),
            Op::Assign { target, path, value, .. } => self.one_of(*value, &self.place(target, path)?, "the place"),
            Op::AsSlice { container, mutability, .. } => {
                self.is(*container, |kind| matches!(kind, TyKind::Ref(..)), "a reference")?;
                self.is(
                    Self::result(inst)?,
                    |kind| matches!(kind, TyKind::Ref(made, lent)
                        if made == mutability && matches!(self.kind(*lent), TyKind::Slice(_) | TyKind::Str)),
                    "a slice reference of the stated mutability",
                )
            }
            Op::SliceLen { src } => {
                self.slice(*src)?;
                self.is_u64(Self::result(inst)?)
            }
            Op::ArrayView { src, mutability } => {
                let TyKind::Ref(held, lent) = self.kind(self.ty(*src)) else {
                    return Err(format!("{src:?} is not a reference"));
                };
                found(held == mutability, || format!("{src:?} is not a {mutability:?} reference"))?;
                let TyKind::Array(element, _, _) = self.kind(*lent) else {
                    return Err(format!("{src:?} lends {:?}, not an array", self.kind(*lent)));
                };
                let (made, viewed) = self.slice(Self::result(inst)?)?;
                found(made == *mutability && viewed == *element, || {
                    format!("the view is not a {mutability:?} slice of the array's elements")
                })
            }
            Op::Index { slice, index, mode, .. } => {
                self.is_u64(*index)?;
                let (held, element) = self.slice(*slice)?;
                let result = Self::result(inst)?;
                match mode {
                    IndexMode::Copy => self.same(result, element, "the element"),
                    IndexMode::Ref => self.is(
                        result,
                        |kind| matches!(kind, TyKind::Ref(made, lent) if *made == held && *lent == element),
                        "a reference to the element at the slice's mutability",
                    ),
                }
            }
            Op::IndexSet { slice, index, value, .. } => {
                self.is_u64(*index)?;
                let (held, element) = self.slice(*slice)?;
                found(held == Mutability::Mut, || format!("{slice:?} is not a &mut slice"))?;
                self.same(*value, element, "the element")
            }
            Op::FieldGet { object, field, rest } => {
                let at = self.field_path(self.ty(*object), *field, rest)?;
                self.same(Self::result(inst)?, at, "the field")
            }
            Op::FieldSet { object, field, rest, value } => {
                let object_ty = self.ty(*object);
                let at = self.field_path(object_ty, *field, rest)?;
                self.same(*value, at, "the field")?;
                self.same(Self::result(inst)?, object_ty, "the object")
            }
            Op::BinOp { op: BinOp::Shl | BinOp::Shr, left, right } => {
                self.is_int(*left)?;
                self.is_int(*right)?;
                self.same(Self::result(inst)?, self.ty(*left), "the shifted value")
            }
            Op::BinOp { op, left, right } => {
                let result = Self::result(inst)?;
                self.same(*right, self.ty(*left), "the left operand")?;
                match op {
                    BinOp::Min | BinOp::Max => {
                        self.is_int(*left)?;
                        self.same(result, self.ty(*left), "the operands")
                    }
                    BinOp::And | BinOp::Or | BinOp::Xor if matches!(self.kind(self.ty(*left)), TyKind::Bool) => {
                        self.is_bool(result)
                    }
                    BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => self.is_bool(result),
                    _ => self.same(result, self.ty(*left), "the operands"),
                }
            }
            Op::UnaryOp { op, operand } => {
                let result = Self::result(inst)?;
                match op {
                    UnaryOp::Not => {
                        self.is_bool(*operand)?;
                        self.is_bool(result)
                    }
                    UnaryOp::Neg => self.same(result, self.ty(*operand), "the operand"),
                }
            }
            Op::Cast { src, to } => {
                self.is(*src, |kind| CastTy::of_kind(kind).is_some(), "an integer width, f64, char or bool")?;
                self.is(Self::result(inst)?, |kind| CastTy::of_kind(kind) == Some(*to), "the cast's target")
            }
            Op::Unmark { src } => {
                let result = Self::result(inst)?;
                let (from, to) = (self.ty(*src), self.ty(result));
                found(from != to && self.raises_marks(from, to), || {
                    format!("{result:?} is {:?}, not {src:?}'s {:?} with marks raised", self.kind(to), self.kind(from))
                })
            }
            // One body at a function type, or one per signature, in order,
            // at a `Closures` type, of none or at least two (functions §2).
            Op::MakeClosure { bodies, .. } => self.is(
                Self::result(inst)?,
                |kind| match kind {
                    TyKind::Fn { .. } => bodies.len() == 1,
                    TyKind::Closures(signatures) => signatures.len() != 1 && signatures.len() == bodies.len(),
                    _ => false,
                },
                "a function of one body, or a closure of one body per signature of none or at least two",
            ),
            // The closure kept at some of its signatures: a function type
            // for one, and a closure of those, in order, otherwise.
            Op::Specialize { closure, signatures } => {
                let TyKind::Closures(has) = self.kind(self.ty(*closure)) else {
                    return Err(format!("{closure:?} is no closure of signatures"));
                };
                found(has.len() != 1, || format!("{closure:?} is a closure of one signature, which is written as a function type"))?;
                found(signatures.iter().all(|signature| has.contains(signature)), || {
                    format!("{closure:?} has no code at some of {signatures:?}")
                })?;
                let result = Self::result(inst)?;
                match &signatures[..] {
                    [one] => self.same(result, *one, "the one signature"),
                    _ => self.is(
                        result,
                        |kind| matches!(kind, TyKind::Closures(kept) if kept[..] == signatures[..]),
                        "a closure of the signatures kept, none or at least two",
                    ),
                }
            }
            Op::FunctionCall { callee, callee_ty, args } | Op::Spawn { callee, callee_ty, args } => {
                let TyKind::Fn { params, ret, .. } = self.kind(*callee_ty) else {
                    return Err(format!("the callee type {:?} is not a function", self.kind(*callee_ty)));
                };
                if let Callee::Indirect(function) = callee {
                    self.same(*function, *callee_ty, "the callee type")?;
                }
                found(params.len() == args.len(), || {
                    format!("{} arguments are passed to {} parameters", args.len(), params.len())
                })?;
                for (arg, param) in args.iter().zip(params.iter()) {
                    self.same(*arg, *param, "the parameter")?;
                }
                let result = Self::result(inst)?;
                match inst.op() {
                    Op::Spawn { .. } => self.is(
                        result,
                        |kind| matches!(kind, TyKind::Handle(held) if held == ret),
                        "a Handle of the callee's return type",
                    ),
                    _ => self.same(result, *ret, "the callee's return type"),
                }
            }
            Op::TakeEffect { src } => match self.kind(self.ty(*src)) {
                TyKind::Handle(held) => self.same(Self::result(inst)?, *held, "the handle's type"),
                other => Err(format!("{src:?} is {other:?}, not a Handle")),
            },
            Op::Merge { orders } => {
                self.is(Self::result(inst)?, |kind| matches!(kind, TyKind::Order), "Order")?;
                orders.iter().try_for_each(|order| self.is(*order, |kind| matches!(kind, TyKind::Order), "Order"))
            }
            Op::ArrayBegin { .. } => self.is(
                Self::result(inst)?,
                |kind| matches!(kind, TyKind::Array(_, len, _) if known_len(len) == 0),
                "an array of no elements",
            ),
            Op::ArrayPush { array, value } => {
                let result = Self::result(inst)?;
                let (TyKind::Array(element, len, _), TyKind::Array(held, held_len, _)) =
                    (self.kind(self.ty(result)), self.kind(self.ty(*array)))
                else {
                    return Err(format!("{result:?} and {array:?} are not both arrays"));
                };
                found(element == held && known_len(len) == known_len(held_len) + 1, || {
                    format!("{result:?} is not {array:?} one element longer")
                })?;
                self.same(*value, *element, "the element")
            }
            Op::MakeObject { fields } => {
                let TyKind::Object { fields: declared_fields, .. } = self.kind(self.ty(Self::result(inst)?)) else {
                    return Err("the result is not an object".to_string());
                };
                for (name, value) in fields {
                    let Some((_, field)) = declared_fields.iter().find(|(declared, _)| declared == name) else {
                        return Err(format!("the object type has no field {name:?}"));
                    };
                    self.same(*value, *field, "the field")?;
                }
                Ok(())
            }
            Op::MakeTuple { elements } => {
                let TyKind::Tuple(element_tys, _) = self.kind(self.ty(Self::result(inst)?)) else {
                    return Err("the result is not a tuple".to_string());
                };
                found(element_tys.len() == elements.len(), || "the tuple's arity differs".to_string())?;
                elements.iter().zip(element_tys.iter()).try_for_each(|(element, ty)| self.same(*element, *ty, "the element"))
            }
            Op::Intrinsic { intrinsic, args } => self.intrinsic(*intrinsic, args, Self::result(inst)?),
            Op::TupleIndex { tuple, index } => match self.kind(self.ty(*tuple)) {
                TyKind::Tuple(elements, _) => match elements.get(*index) {
                    Some(element) => self.same(Self::result(inst)?, *element, "the element"),
                    None => Err(format!("{tuple:?} has no element {index}")),
                },
                other => Err(format!("{tuple:?} is {other:?}, not a tuple")),
            },
            Op::ArrayIndex { array, .. } => match self.kind(self.ty(*array)) {
                TyKind::Array(element, _, _) => self.same(Self::result(inst)?, *element, "the element"),
                other => Err(format!("{array:?} is {other:?}, not an array")),
            },
            Op::ObjectGet { object, key } => {
                let at = self.field_path(self.ty(*object), *key, &[])?;
                self.same(Self::result(inst)?, at, "the field")
            }
            Op::TestLiteral { .. } | Op::TestObjectKey { .. } | Op::TestVariant { .. } => {
                self.is_bool(Self::result(inst)?)
            }
            Op::MakeVariant { tag, payload } => {
                let result = Self::result(inst)?;
                match (self.kind(self.ty(result)), payload) {
                    (TyKind::Option(held, _), Some(payload)) => self.same(*payload, *held, "the Option's payload"),
                    (TyKind::Enum { variants, .. }, payload) => {
                        let Some((_, declared)) = variants.iter().find(|(name, _)| name == tag) else {
                            return Err(format!("the enum has no variant {tag:?}"));
                        };
                        match (declared, payload) {
                            (Some(declared), Some(payload)) => self.same(*payload, *declared, "the variant's payload"),
                            (None, None) => Ok(()),
                            (Some(_), None) | (None, Some(_)) => {
                                Err(format!("the payload of {tag:?} does not match its declaration"))
                            }
                        }
                    }
                    (TyKind::Option(.., _), None) => Ok(()),
                    (TyKind::Result(ok, err, _), Some(payload)) => self.one_of(*payload, &[*ok, *err], "a Result's payload"),
                    (TyKind::Result(..), None) => Err("a Result's variant always carries a payload".to_string()),
                    (other, _) => Err(format!("the result is {other:?}, not a variant type")),
                }
            }
            Op::UnwrapVariant { src } => {
                let payloads = self.payloads(self.ty(*src))?;
                self.one_of(Self::result(inst)?, &payloads, "a payload of the variant type")
            }
            Op::Undef | Op::Drop { .. } => Ok(()),
        }
    }

    /// `intrinsic` over `args` yielding `result`, as `Intrinsic` states.
    fn intrinsic(&self, intrinsic: Intrinsic, args: &[ValueId], result: ValueId) -> Found {
        match (intrinsic, args) {
            // `&m T` from the structure of `&m` references to `T`'s parts,
            // one reference at each part of `T`.
            (Intrinsic::Gather, [structure]) => {
                let TyKind::Ref(mutability, target) = self.kind(self.ty(result)) else {
                    return Err(format!("{result:?} is not a reference"));
                };
                let types = self.types;
                let (Some(gathered), Some(whole)) = (structural::structure(types, self.ty(*structure)), structural::structure(types, *target)) else {
                    return Err(format!("{structure:?} and {result:?}'s target are not both transparent structured types"));
                };
                let (gathered, whole) = (gathered.parts(), whole.parts());
                found(gathered.len() == whole.len(), || format!("{structure:?} does not have the parts of {result:?}'s target"))?;
                for (part, of) in gathered.iter().zip(whole.iter()) {
                    let lends = matches!(self.kind(part.ty), TyKind::Ref(made, lent) if made == mutability && *lent == of.ty);
                    found(part.key == of.key && lends, || format!("{structure:?}'s part {:?} is not a {mutability:?} reference to the target's", part.key))?;
                }
                Ok(())
            }
            // `[a_1, …, a_k, f]`: `k` arrays of one length, or references
            // to them, mapped by `f` of `k` parameters, each taking its
            // array's element as the array holds it, or through a
            // reference of the reference's mutability; for `k` = 0 the
            // length is the result's.
            (Intrinsic::ArrayMap, [arrays @ .., f]) => {
                let TyKind::Fn { params, ret, .. } = self.kind(self.ty(*f)) else {
                    return Err(format!("{f:?} is not a function"));
                };
                found(params.len() == arrays.len(), || format!("{f:?} does not take one element of each of {} arrays", arrays.len()))?;
                let ret = *ret;
                let mut len = None;
                for (array, param) in arrays.iter().zip(params.iter()) {
                    let (element, array_len, lent) = match self.kind(self.ty(*array)) {
                        TyKind::Array(element, len, _) => (*element, known_len(len), None),
                        TyKind::Ref(mutability, target) => match self.kind(*target) {
                            TyKind::Array(element, len, _) => (*element, known_len(len), Some(*mutability)),
                            other => return Err(format!("{array:?} lends {other:?}, not an array")),
                        },
                        other => return Err(format!("{array:?} is {other:?}, not an array or a reference to one")),
                    };
                    found(len.is_none_or(|len| len == array_len), || format!("{array:?} is not of the others' length"))?;
                    len = Some(array_len);
                    let takes = match (lent, self.kind(element)) {
                        (None, _) => *param == element,
                        // A reference element, reborrowed at the weaker of
                        // the two mutabilities.
                        (Some(mutability), TyKind::Ref(held, target)) => {
                            let weaker = match (mutability, *held) {
                                (Mutability::Mut, Mutability::Mut) => Mutability::Mut,
                                _ => Mutability::Shared,
                            };
                            matches!(self.kind(*param), TyKind::Ref(made, taken) if *made == weaker && *taken == *target)
                        }
                        (Some(mutability), _) => {
                            matches!(self.kind(*param), TyKind::Ref(made, taken) if *made == mutability && *taken == element)
                        }
                    };
                    found(takes, || format!("{f:?} does not take the element of {array:?}"))?;
                }
                self.is(
                    result,
                    |kind| matches!(kind, TyKind::Array(mapped, mapped_len, _) if *mapped == ret && len.is_none_or(|len| known_len(mapped_len) == len)),
                    "the array of `f`'s results",
                )
            }
            // `Intrinsic`'s docs state no arguments for it: a clone of a
            // `String` is `Op::StringClone`.
            (Intrinsic::StringClone, _) => Err("`StringClone` is no `Op::Intrinsic`: `Intrinsic` states no arguments for it".to_string()),
            (intrinsic, args) => Err(format!("{intrinsic:?} takes other arguments than {} given", args.len())),
        }
    }

    /// The `&[T]` or `&mut [T]` `slice` is: its mutability and `T`.
    fn slice(&self, slice: ValueId) -> Result<(Mutability, Ty), String> {
        match self.kind(self.ty(slice)) {
            TyKind::Ref(held, lent) => match self.kind(*lent) {
                TyKind::Slice(element) => Ok((*held, *element)),
                other => Err(format!("{slice:?} lends {other:?}, not a slice")),
            },
            other => Err(format!("{slice:?} is {other:?}, not a slice reference")),
        }
    }

    /// Every type `path` may reach under `target`: one, but at a payload of
    /// a `Result` or an enum, which names no one variant, each payload's.
    fn place(&self, target: &RefTarget, path: &[PathSeg]) -> Result<Vec<Ty>, String> {
        let root = match target {
            RefTarget::Var(storage) | RefTarget::Param(storage) => self.ty(*storage),
            RefTarget::Through(reference) => match self.kind(self.ty(*reference)) {
                TyKind::Ref(_, lent) => *lent,
                other => return Err(format!("{reference:?} is {other:?}, not a reference")),
            },
            RefTarget::Context(context) => self.context(*context)?,
        };
        let mut reached = vec![root];
        for seg in path {
            let mut next = Vec::new();
            for at in reached {
                match (seg, self.kind(at)) {
                    (PathSeg::Field(field), TyKind::Object { fields, .. }) => {
                        next.extend(fields.iter().filter(|(name, _)| name == field).map(|(_, ty)| *ty))
                    }
                    (PathSeg::Index(_), TyKind::Array(element, _, _)) => next.push(*element),
                    (PathSeg::Index(index), TyKind::Tuple(elements, _)) => next.extend(elements.get(*index).copied()),
                    (PathSeg::Payload, TyKind::Option(.., _) | TyKind::Result(..) | TyKind::Enum { .. }) => {
                        next.extend(self.payloads(at)?)
                    }
                    _ => {}
                }
            }
            if next.is_empty() {
                return Err(format!("{seg:?} steps into none of the types the path reached"));
            }
            reached = next;
        }
        Ok(reached)
    }

    /// The payload types a variant of `ty` may carry.
    fn payloads(&self, ty: Ty) -> Result<Vec<Ty>, String> {
        match self.kind(ty) {
            TyKind::Option(payload, _) => Ok(vec![*payload]),
            TyKind::Result(ok, err, _) => Ok(vec![*ok, *err]),
            TyKind::Enum { variants, .. } => Ok(variants.iter().filter_map(|(_, payload)| *payload).collect()),
            other => Err(format!("{other:?} is not a variant type")),
        }
    }

    fn one_of(&self, value: ValueId, expected: &[Ty], what: &str) -> Found {
        let ty = self.ty(value);
        found(expected.contains(&ty), || {
            let expected: Vec<&TyKind<Concrete>> = expected.iter().map(|ty| self.kind(*ty)).collect();
            format!("{value:?} is {:?}, and {what} is one of {expected:?}", self.kind(ty))
        })
    }

    /// The type of `context`, which the body names, and whose value
    /// outlives the run, so that its type has no positions: no reference
    /// and no function value, at any depth (context §3, ownership §5).
    fn context(&self, context: QualifiedRef) -> Result<Ty, String> {
        let Some(named) = self.body.named_context(context) else {
            return Err(format!("{context:?} is not among the contexts the body names"));
        };
        let ty = named.ty;
        let mut pending = vec![ty];
        while let Some(held) = pending.pop() {
            match self.kind(held) {
                TyKind::Ref(..) | TyKind::Fn { .. } => {
                    return Err(format!("{context:?} outlives the run, and its type holds {:?}", self.kind(held)));
                }
                kind => kind.for_each_child(|child| pending.push(child)),
            }
        }
        Ok(ty)
    }

    fn field_path(&self, object: Ty, field: acvus_utils::Astr, rest: &[acvus_utils::Astr]) -> Result<Ty, String> {
        let mut at = object;
        for name in std::iter::once(&field).chain(rest) {
            let TyKind::Object { fields, .. } = self.kind(at) else {
                return Err(format!("{:?} is not an object", self.kind(at)));
            };
            let Some((_, ty)) = fields.iter().find(|(declared, _)| declared == name) else {
                return Err(format!("the object has no field {name:?}"));
            };
            at = *ty;
        }
        Ok(at)
    }

    fn terminator(&self, block: BlockId, terminator: &Terminator) -> Found {
        let at = |detail: String| format!("{block:?}'s terminator: {detail}");
        match terminator {
            Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. } | Terminator::While { cond, .. } => {
                self.is_bool(*cond).map_err(at)?
            }
            Terminator::Return { value, order, .. } => {
                self.same(*value, self.body.ret(), "the body's result type").map_err(at)?;
                if let Some(order) = order {
                    self.is(*order, |kind| matches!(kind, TyKind::Order), "Order").map_err(at)?;
                }
            }
            Terminator::Switch { tag, arms, .. } => {
                let scrutinee = match self.kind(self.ty(*tag)) {
                    TyKind::Ref(_, lent) => *lent,
                    _ => self.ty(*tag),
                };
                for (key, _, _) in arms {
                    self.key_fits(key, scrutinee).map_err(at)?;
                }
            }
            Terminator::For { source, body, .. } => {
                self.loop_source(source, *body).map_err(at)?;
            }
            Terminator::Jump { .. } | Terminator::Diverge | Terminator::NoArmMatched { .. } => {}
        }
        for edge in terminator.edges() {
            let params = self.body.block(edge.to).params();
            found(params.len() == edge.supplied + edge.args.len(), || {
                at(format!(
                    "the edge to {:?} supplies {} and passes {} arguments to {} parameters",
                    edge.to,
                    edge.supplied,
                    edge.args.len(),
                    params.len()
                ))
            })?;
            for (param, arg) in params[edge.supplied..].iter().zip(edge.args) {
                self.same(*arg, self.ty(*param), "the parameter").map_err(at)?;
            }
        }
        Ok(())
    }

    /// Whether a switch key names a value of the scrutinee's type. A tag of
    /// an `Option` or a `Result` is checked as a tag of a variant type: its
    /// name is text the interner holds.
    fn key_fits(&self, key: &SwitchKey, scrutinee: Ty) -> Found {
        let fits = match (key, self.kind(scrutinee)) {
            (SwitchKey::Tag(tag), TyKind::Enum { variants, .. }) => variants.iter().any(|(name, _)| name == tag),
            (SwitchKey::Tag(_), TyKind::Option(.., _) | TyKind::Result(..)) => true,
            (SwitchKey::Int(_), TyKind::Int(_)) | (SwitchKey::Bool(_), TyKind::Bool) | (SwitchKey::Char(_), TyKind::Char) => {
                true
            }
            (SwitchKey::Str(_), TyKind::String | TyKind::Str) => true,
            _ => false,
        };
        found(fits, || format!("the key {key:?} names no value of {:?}", self.kind(scrutinee)))
    }

    /// A `For`'s source against what it hands its body: the element, and
    /// the counter where the source is a slice or an array.
    fn loop_source(&self, source: &ForSource, body: BlockId) -> Found {
        let params = self.body.block(body).params();
        let (element, counted) = match source {
            ForSource::Slice(slice) | ForSource::SliceMut(slice) => {
                let wanted = match source {
                    ForSource::SliceMut(_) => Mutability::Mut,
                    _ => Mutability::Shared,
                };
                let (held, element) = self.slice(*slice)?;
                found(held == wanted, || format!("{slice:?} is not a {wanted:?} slice reference"))?;
                (Element::Lent(wanted, element), true)
            }
            ForSource::Array(array) => match self.kind(self.ty(*array)) {
                TyKind::Array(element, _, _) => (Element::Held(*element), true),
                other => return Err(format!("{array:?} is {other:?}, not an array")),
            },
            ForSource::Range { at, hi } => {
                self.is_int(*at)?;
                self.same(*hi, self.ty(*at), "the range's start")?;
                (Element::Held(self.ty(*at)), false)
            }
        };
        let [first, rest @ ..] = params else {
            return Err(format!("the loop body {body:?} takes no element"));
        };
        match element {
            Element::Held(element) => self.same(*first, element, "the element")?,
            Element::Lent(mutability, element) => self.is(
                *first,
                |kind| matches!(kind, TyKind::Ref(made, lent) if *made == mutability && *lent == element),
                "a reference to the element",
            )?,
        }
        match (counted, rest) {
            (true, [counter, ..]) => self.is_u64(*counter),
            (true, []) => Err(format!("the loop body {body:?} takes no counter")),
            (false, _) => Ok(()),
        }
    }
}

/// What a loop hands its body each iteration.
enum Element {
    Held(Ty),
    Lent(Mutability, Ty),
}

// -- DefDominatesUses --------------------------------------------------------

fn def_dominates_uses(body: &Body, _: &Types<Concrete>) -> Found {
    let dominators = Dominators::of(body);
    let points = def_points(body);
    let check = |value: ValueId, site: Site| {
        let Some(point) = points.get(&value) else {
            return Err(format!("{site:?} reads {value:?}, which nothing in the body defines"));
        };
        found(dominators.dominates_site(body, *point, site), || {
            format!("{value:?}, defined at {point:?}, does not dominate its use at {site:?}")
        })
    };
    for (id, block) in body.blocks() {
        for inst in block.insts() {
            inst.operands().into_iter().try_for_each(|value| check(value, Site::Inst(inst.id())))?;
        }
        block.terminator().operands().into_iter().try_for_each(|value| check(value, Site::Term(id)))?;
    }
    Ok(())
}

// -- OrderChainWhole ---------------------------------------------------------

/// Every token is consumed by one instruction, or by several whose yields
/// all reach one `Merge` joining them, an `anyorder` fork (effects §3). Each
/// token reaches the body's return: it flows on to the token an instruction
/// yields or merges, to the parameter an edge passes it to, or to a
/// `Return`; a token nothing uses is whole only where no path from its block
/// returns, since a run that dies owes nothing (effects §5).
fn order_chain_whole(body: &Body, _: &Types<Concrete>) -> Found {
    let ordered = body.order_param().is_some();
    let mut defined_in: FxHashMap<ValueId, BlockId> = FxHashMap::default();
    let mut flows: FxHashMap<ValueId, Vec<Flow>> = FxHashMap::default();
    let mut consumers: FxHashMap<ValueId, Vec<InstId>> = FxHashMap::default();
    let mut handles: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    let mut merged: FxHashSet<ValueId> = FxHashSet::default();
    if let Some(order_param) = body.order_param() {
        defined_in.insert(order_param, body.entry());
    }
    for (id, block) in body.blocks() {
        for param in block.params().iter().filter(|param| body.is_token(**param)) {
            defined_in.insert(*param, id);
        }
        for inst in block.insts() {
            let place = || format!("{id:?} {:?} {:?}", inst.id(), inst.op());
            if let Some((RefTarget::Context(context), _)) = inst.op().place() {
                let Some(named) = body.named_context(*context) else {
                    return Err(format!("{} names {context:?}, which the body does not name", place()));
                };
                found(named.volatile == inst.order().is_some(), || match named.volatile {
                    true => format!("{} is a place operation on the volatile {context:?}, off the order chain", place()),
                    false => format!("{} is a place operation on {context:?}, not volatile, on the order chain", place()),
                })?;
            }
            if let Some(order) = inst.order() {
                found(ordered, || format!("{} stands on the order chain of an unordered body", place()))?;
                for token in order.consumed().into_iter().chain(order.yielded()) {
                    found(body.is_token(token), || format!("{} names {token:?}, which is not a token", place()))?;
                }
                if let Some(token) = order.yielded() {
                    defined_in.insert(token, id);
                }
                match order {
                    OrderUse::Edge(edge) => {
                        consumers.entry(edge.before).or_default().push(inst.id());
                        flows.entry(edge.before).or_default().push(Flow::Yields(edge.after));
                    }
                    OrderUse::Start(before) => {
                        consumers.entry(before).or_default().push(inst.id());
                        let handle = inst.result().expect("a start yields its handle");
                        handles.insert(handle, before);
                    }
                    OrderUse::Take(_) => {}
                }
            }
            let tokens_in_op = match inst.op() {
                Op::Merge { orders } => {
                    let result = inst.result().expect("a Merge yields its token");
                    defined_in.insert(result, id);
                    merged.insert(result);
                    for order in orders {
                        flows.entry(*order).or_default().push(Flow::Merged(result));
                    }
                    orders.iter().all(|order| body.is_token(*order)) && body.is_token(result)
                }
                _ => {
                    let mut plain = true;
                    inst.op().for_each_operand(|value| plain &= !body.is_token(value));
                    plain && inst.result().is_none_or(|result| !body.is_token(result))
                }
            };
            found(tokens_in_op, || format!("{} reads or yields a token outside the order chain", place()))?;
        }
        let terminator = block.terminator();
        if let Terminator::Return { order, value, .. } = terminator {
            found(ordered == order.is_some(), || {
                format!("{id:?}'s return carries {order:?} in a body whose order parameter is {:?}", body.order_param())
            })?;
            found(!body.is_token(*value), || format!("{id:?} returns a token"))?;
            if let Some(order) = order {
                flows.entry(*order).or_default().push(Flow::Returned);
            }
        }
        for edge in terminator.edges() {
            let params = &body.block(edge.to).params()[edge.supplied..];
            for (param, arg) in params.iter().zip(edge.args) {
                found(body.is_token(*param) == body.is_token(*arg), || {
                    format!("{id:?} passes {arg:?} to {param:?}, and only one of them is a token")
                })?;
                if body.is_token(*arg) {
                    flows.entry(*arg).or_default().push(Flow::Passed(*param));
                }
            }
        }
    }
    // A start's token flows on to what its taking yields.
    let mut taken_yields: FxHashMap<ValueId, ValueId> = FxHashMap::default();
    for (_, block) in body.blocks() {
        for inst in block.insts() {
            if let (Op::TakeEffect { src }, Some(OrderUse::Take(after))) = (inst.op(), inst.order()) {
                taken_yields.entry(*src).or_insert(after);
                if let Some(before) = handles.get(src) {
                    flows.entry(*before).or_default().push(Flow::Yields(after));
                }
            }
        }
    }

    let yield_of = |id: InstId| -> Option<ValueId> {
        let inst = body.inst(id);
        match inst.order()? {
            OrderUse::Edge(edge) => Some(edge.after),
            OrderUse::Start(_) => {
                let handle = inst.result()?;
                taken_yields.get(&handle).copied()
            }
            OrderUse::Take(_) => None,
        }
    };
    let merges_reached = |token: ValueId| -> FxHashSet<ValueId> {
        let mut reached = FxHashSet::default();
        let mut pending = vec![token];
        let mut seen = FxHashSet::default();
        while let Some(token) = pending.pop() {
            if !seen.insert(token) {
                continue;
            }
            for flow in flows.get(&token).into_iter().flatten() {
                match flow {
                    Flow::Merged(merge) => {
                        reached.insert(*merge);
                        pending.push(*merge);
                    }
                    Flow::Passed(param) => pending.push(*param),
                    Flow::Yields(_) | Flow::Returned => {}
                }
            }
        }
        reached
    };
    let mut forks: Vec<(&ValueId, &Vec<InstId>)> = consumers.iter().filter(|(_, by)| by.len() > 1).collect();
    forks.sort();
    for (token, by) in forks {
        let mut joined: Option<FxHashSet<ValueId>> = None;
        for consumer in by {
            let reached = match yield_of(*consumer) {
                Some(yielded) => merges_reached(yielded),
                None => FxHashSet::default(),
            };
            joined = Some(match joined {
                Some(joined) => joined.intersection(&reached).copied().collect(),
                None => reached,
            });
        }
        found(joined.is_some_and(|joined| !joined.is_empty()), || {
            format!("{token:?} is consumed by {by:?}, and no Merge joins what they yield")
        })?;
    }

    let returning = blocks_reaching_return(body);
    let tokens: Vec<ValueId> = defined_in.keys().copied().collect();
    for token in &tokens {
        flows.entry(*token).or_default();
    }
    let flows_of = |token: &ValueId| &flows[token];
    let mut whole: FxHashSet<ValueId> = tokens
        .iter()
        .copied()
        .filter(|token| {
            let flows = flows_of(token);
            flows.contains(&Flow::Returned) || (flows.is_empty() && !returning.contains(&defined_in[token]))
        })
        .collect();
    // A token is whole where a flow of it reaches a whole token: the least
    // such set is what the seeds reach backward along the flows, one walk.
    let mut flows_into: FxHashMap<ValueId, Vec<ValueId>> = FxHashMap::default();
    for token in &tokens {
        for flow in flows_of(token) {
            if let Flow::Yields(next) | Flow::Merged(next) | Flow::Passed(next) = flow {
                flows_into.entry(*next).or_default().push(*token);
            }
        }
    }
    let mut pending: Vec<ValueId> = whole.iter().copied().collect();
    while let Some(token) = pending.pop() {
        for from in flows_into.get(&token).into_iter().flatten() {
            if whole.insert(*from) {
                pending.push(*from);
            }
        }
    }
    // A merge's own token need not reach the return: a merge nothing
    // consumes is a dead value (D65). What flows into it still must, so a
    // merge is never seeded as whole, only left out of the report.
    let mut broken: Vec<ValueId> =
        tokens.into_iter().filter(|token| !whole.contains(token) && !merged.contains(token)).collect();
    broken.sort();
    found(broken.is_empty(), || format!("the chain from {broken:?} does not reach the body's return"))
}

/// Where a token goes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    /// An instruction consumes it and yields this token.
    Yields(ValueId),
    /// A `Merge` joins it into this token.
    Merged(ValueId),
    /// An edge passes it to this parameter.
    Passed(ValueId),
    /// A `Return` carries it.
    Returned,
}

fn blocks_reaching_return(body: &Body) -> FxHashSet<BlockId> {
    let preds = body.predecessors();
    let mut reaching: FxHashSet<BlockId> = body
        .blocks()
        .filter(|(_, block)| matches!(block.terminator(), Terminator::Return { .. }))
        .map(|(id, _)| id)
        .collect();
    let mut pending: Vec<BlockId> = reaching.iter().copied().collect();
    while let Some(block) = pending.pop() {
        for pred in &preds[&block] {
            if reaching.insert(*pred) {
                pending.push(*pred);
            }
        }
    }
    reaching
}

// -- StructureKept -----------------------------------------------------------

fn structure_kept(body: &Body, _: &Types<Concrete>) -> Found {
    let count = body.block_count();
    let is_block = |block: BlockId| block.index() < count;
    for (id, block) in body.blocks() {
        match block.terminator() {
            Terminator::Diamond { join, .. } => {
                found(is_block(*join), || format!("{id:?}'s Diamond joins at {join:?}, which is not a block"))?
            }
            Terminator::Switch { join, .. } => {
                found(is_block(*join), || format!("{id:?}'s Switch joins at {join:?}, which is not a block"))?
            }
            Terminator::For { body, exit, .. } | Terminator::While { body, exit, .. } => {
                for named in [*body, *exit] {
                    found(is_block(named), || format!("{id:?}'s loop names {named:?}, which is not a block"))?;
                }
            }
            Terminator::Jump { .. } | Terminator::JumpIf { .. } | Terminator::Return { .. } | Terminator::Diverge | Terminator::NoArmMatched { .. } => {}
        }
    }
    Ok(())
}

// -- SplitTakenOnce ----------------------------------------------------------

/// Walks every path from each start: its handle is read only by its
/// taking, taken at most once, and taken on every path that returns. A path
/// that dies owes nothing (effects §5), and a path that comes back to the
/// start meets a new call.
fn split_taken_once(body: &Body, _: &Types<Concrete>) -> Found {
    for (_, block) in body.blocks() {
        for inst in block.insts() {
            let Op::Spawn { .. } = inst.op() else { continue };
            let handle = inst.result().expect("a start yields its handle");
            for site in body.users(handle) {
                let takes = match site {
                    Site::Inst(user) => matches!(body.inst(user).op(), Op::TakeEffect { .. }),
                    Site::Term(_) => false,
                };
                found(takes, || format!("the handle {handle:?} of {:?} is read at {site:?}, not taken", inst.id()))?;
            }
            walk_from_start(body, inst.id(), handle)?;
        }
    }
    Ok(())
}

/// A walk's place: in `block`, before the instruction `next` or at the
/// block's end where that is none, and whether the call was taken on the
/// way there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WalkState {
    block: BlockId,
    next: Option<InstId>,
    taken: bool,
}

fn walk_from_start(body: &Body, start: InstId, handle: ValueId) -> Found {
    let mut visited: FxHashSet<WalkState> = FxHashSet::default();
    let mut pending = vec![WalkState { block: body.block_of(start), next: body.next_in_block(start), taken: false }];
    while let Some(state) = pending.pop() {
        if !visited.insert(state) {
            continue;
        }
        let WalkState { block, next, taken } = state;
        if let Some(at) = next {
            if at == start {
                continue;
            }
            let takes = matches!(body.inst(at).op(), Op::TakeEffect { src } if *src == handle);
            found(!(takes && taken), || format!("{start:?}'s call is taken again at {at:?}"))?;
            pending.push(WalkState { block, next: body.next_in_block(at), taken: taken || takes });
            continue;
        }
        match body.block(block).terminator() {
            Terminator::Return { .. } => {
                found(taken, || format!("{start:?}'s call is not taken on a path that returns from {block:?}"))?
            }
            Terminator::Diverge | Terminator::NoArmMatched { .. } => {}
            terminator => pending.extend(
                terminator.edges().iter().map(|edge| WalkState { block: edge.to, next: body.first_inst(edge.to), taken }),
            ),
        }
    }
    Ok(())
}
