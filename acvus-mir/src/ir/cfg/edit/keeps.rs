//! Invariants an editor keeps, one checker each. They run after every edit
//! in debug and test builds; `validate` reuses them after each transform.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::dominators::{Dominators, def_points};
use crate::ir::cfg::{BlockId, Body, Inst, Op, OrderUse, Site, Terminator};
use crate::ir::{BinOp, Callee, IndexMode, PathSeg, RefTarget, UnaryOp, ValueId, known_len};
use crate::ty::{CastTy, Concrete, IntTy, Mutability, Ty, TyKind, Types};

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
invariant!(OrderChainWhole, order_chain_whole, "Every effectful instruction consumes exactly one live token, and the chain reaches the body's return (effects §2-3).");
invariant!(StructureKept, structure_kept, "Every `Diamond`'s join and every loop's stages and exit are blocks of the body (D45).");
invariant!(SplitTakenOnce, split_taken_once, "Every start of a split call is taken exactly once on every path from it (effects §6).");

/// A checker's finding: what breaks the invariant, in words.
type Found = Result<(), String>;

fn found(holds: bool, detail: impl FnOnce() -> String) -> Found {
    match holds {
        true => Ok(()),
        false => Err(detail()),
    }
}

// -- TypesAgree --------------------------------------------------------------

/// Checks each instruction's operands and result against what its operation
/// takes and yields, where the operation and its operands' types determine
/// that, and every edge against its target's parameters. A payload read of a
/// `Result` or an enum names no one variant, and a context's value, a
/// function loaded by name and a switch key are typed by declarations this
/// body does not hold; those are left to their definers.
fn types_agree(body: &Body, types: &Types<Concrete>) -> Found {
    let check = TypeCheck { body, types };
    for (id, block) in body.blocks() {
        for (index, inst) in block.insts().iter().enumerate() {
            check
                .inst(inst)
                .map_err(|detail| format!("{id:?}[{index}] {:?}: {detail}", inst.op()))?;
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
                TyKind::Ref(_, lent) => matches!(self.kind(lent.ty()), TyKind::String | TyKind::Str),
                _ => false,
            },
            "String, &String or &str",
        )
    }

    fn is_lent_text(&self, value: ValueId) -> Found {
        self.is(
            value,
            |kind| matches!(kind, TyKind::Ref(_, lent) if matches!(self.kind(lent.ty()), TyKind::String | TyKind::Str)),
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
                        TyKind::Array(byte, len)
                            if matches!(self.kind(*byte), TyKind::Int(IntTy::U8)) && known_len(len) == bytes.len()
                    ),
                    acvus_ast::Literal::List(_) => matches!(kind, TyKind::Array(..)),
                };
                self.is(result, literal_fits, "the literal's type")
            }
            Op::ConstStr { .. } => self.is(
                Self::result(inst)?,
                |kind| matches!(kind, TyKind::Ref(Mutability::Shared, lent) if matches!(self.kind(lent.ty()), TyKind::Str)),
                "&str",
            ),
            Op::StringConcat { parts } => {
                self.is(Self::result(inst)?, |kind| matches!(kind, TyKind::String), "String")?;
                parts.iter().try_for_each(|part| self.is_text(*part))
            }
            Op::StringAppend { target, part } => {
                self.is(
                    *target,
                    |kind| matches!(kind, TyKind::Ref(Mutability::Mut, lent) if matches!(self.kind(lent.ty()), TyKind::String)),
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
                        TyKind::Ref(_, lent) => matches!(self.kind(lent.ty()), TyKind::String),
                        _ => false,
                    },
                    "String or &String",
                )
            }
            Op::StructuralEq { a, b, .. } => {
                self.is_bool(Self::result(inst)?)?;
                let lent = |value: ValueId| match self.kind(self.ty(value)) {
                    TyKind::Ref(_, lent) => Ok(lent.ty()),
                    other => Err(format!("{value:?} is {other:?}, not a reference")),
                };
                let (a_lent, b_lent) = (lent(*a)?, lent(*b)?);
                found(a_lent == b_lent, || format!("{a:?} and {b:?} lend different types"))
            }
            Op::StructuralClone { src, .. } => match self.kind(self.ty(*src)) {
                TyKind::Ref(_, lent) => self.same(Self::result(inst)?, lent.ty(), "the lent type"),
                other => Err(format!("{src:?} is {other:?}, not a reference")),
            },
            Op::Ref { target, path, mutability } => {
                let result = Self::result(inst)?;
                let TyKind::Ref(made, lent) = self.kind(self.ty(result)) else {
                    return Err(format!("{result:?} is not a reference"));
                };
                found(made == mutability, || format!("{result:?} is not a {mutability:?} reference"))?;
                let lent = lent.ty();
                match self.place(target, path)? {
                    Some(at) => found(at == lent, || format!("{result:?} lends another type than the place holds")),
                    None => Ok(()),
                }
            }
            Op::Take { target, path, .. } => {
                let result = Self::result(inst)?;
                match self.place(target, path)? {
                    Some(at) => self.same(result, at, "the place"),
                    None => Ok(()),
                }
            }
            Op::Assign { target, path, value, .. } => match self.place(target, path)? {
                Some(at) => self.same(*value, at, "the place"),
                None => Ok(()),
            },
            Op::AsSlice { container, mutability, .. } => {
                self.is(*container, |kind| matches!(kind, TyKind::Ref(..)), "a reference")?;
                self.is(
                    Self::result(inst)?,
                    |kind| matches!(kind, TyKind::Ref(made, lent)
                        if made == mutability && matches!(self.kind(lent.ty()), TyKind::Slice(_) | TyKind::Str)),
                    "a slice reference of the stated mutability",
                )
            }
            Op::Index { slice, index, mode, .. } => {
                self.is_u64(*index)?;
                let (held, element) = self.slice(*slice)?;
                let result = Self::result(inst)?;
                match mode {
                    IndexMode::Copy => self.same(result, element, "the element"),
                    IndexMode::Ref => self.is(
                        result,
                        |kind| matches!(kind, TyKind::Ref(made, lent) if *made == held && lent.ty() == element),
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
                    UnaryOp::Neg(_) => self.same(result, self.ty(*operand), "the operand"),
                }
            }
            Op::Check { left, right, .. } => {
                self.is_int(*left)?;
                self.same(*right, self.ty(*left), "the left operand")
            }
            Op::CheckSteps { from, step, count } => {
                self.is_int(*from)?;
                self.same(*step, self.ty(*from), "the start")?;
                self.is_u64(*count)
            }
            Op::Cast { src, to } => {
                self.is(*src, |kind| CastTy::of_kind(kind).is_some(), "an integer width, f64, char or bool")?;
                self.is(Self::result(inst)?, |kind| CastTy::of_kind(kind) == Some(*to), "the cast's target")
            }
            Op::LoadFunction { .. } | Op::MakeClosure { .. } => {
                self.is(Self::result(inst)?, |kind| matches!(kind, TyKind::Fn { .. }), "a function")
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
                |kind| matches!(kind, TyKind::Array(_, len) if known_len(len) == 0),
                "an array of no elements",
            ),
            Op::ArrayPush { array, value } => {
                let result = Self::result(inst)?;
                let (TyKind::Array(element, len), TyKind::Array(held, held_len)) =
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
                let TyKind::Object(object) = self.kind(self.ty(Self::result(inst)?)) else {
                    return Err("the result is not an object".to_string());
                };
                for (name, value) in fields {
                    let Some((_, field)) = object.fields().iter().find(|(declared, _)| declared == name) else {
                        return Err(format!("the object type has no field {name:?}"));
                    };
                    self.same(*value, *field, "the field")?;
                }
                Ok(())
            }
            Op::MakeTuple { elements } => {
                let TyKind::Tuple(element_tys) = self.kind(self.ty(Self::result(inst)?)) else {
                    return Err("the result is not a tuple".to_string());
                };
                found(element_tys.len() == elements.len(), || "the tuple's arity differs".to_string())?;
                elements.iter().zip(element_tys.iter()).try_for_each(|(element, ty)| self.same(*element, *ty, "the element"))
            }
            Op::TupleIndex { tuple, index } => match self.kind(self.ty(*tuple)) {
                TyKind::Tuple(elements) => match elements.get(*index) {
                    Some(element) => self.same(Self::result(inst)?, *element, "the element"),
                    None => Err(format!("{tuple:?} has no element {index}")),
                },
                other => Err(format!("{tuple:?} is {other:?}, not a tuple")),
            },
            Op::ArrayIndex { array, .. } => match self.kind(self.ty(*array)) {
                TyKind::Array(element, _) => self.same(Self::result(inst)?, *element, "the element"),
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
                    (TyKind::Option(held), Some(payload)) => self.same(*payload, *held, "the Option's payload"),
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
                    (TyKind::Option(_) | TyKind::Result(..), _) => Ok(()),
                    (other, _) => Err(format!("the result is {other:?}, not a variant type")),
                }
            }
            Op::UnwrapVariant { src } => match self.kind(self.ty(*src)) {
                TyKind::Option(held) => self.same(Self::result(inst)?, *held, "the Option's payload"),
                TyKind::Result(..) | TyKind::Enum { .. } => Ok(()),
                other => Err(format!("{src:?} is {other:?}, not a variant type")),
            },
            Op::Fetch { .. } | Op::Commit { .. } | Op::Undef | Op::Drop { .. } => Ok(()),
        }
    }

    /// The `&[T]` or `&mut [T]` `slice` is: its mutability and `T`.
    fn slice(&self, slice: ValueId) -> Result<(Mutability, Ty), String> {
        match self.kind(self.ty(slice)) {
            TyKind::Ref(held, lent) => match self.kind(lent.ty()) {
                TyKind::Slice(element) => Ok((*held, *element)),
                other => Err(format!("{slice:?} lends {other:?}, not a slice")),
            },
            other => Err(format!("{slice:?} is {other:?}, not a slice reference")),
        }
    }

    /// The type `path` reaches under `target`, or `None` where a payload of
    /// a `Result` or an enum names no one variant, which only the
    /// instruction's own value types.
    fn place(&self, target: &RefTarget, path: &[PathSeg]) -> Result<Option<Ty>, String> {
        let mut at = match target {
            RefTarget::Var(storage) | RefTarget::Param(storage) => self.ty(*storage),
            RefTarget::Through(reference) => match self.kind(self.ty(*reference)) {
                TyKind::Ref(_, lent) => lent.ty(),
                other => return Err(format!("{reference:?} is {other:?}, not a reference")),
            },
        };
        for seg in path {
            at = match (seg, self.kind(at)) {
                (PathSeg::Field(field), TyKind::Object(object)) => match object.fields().iter().find(|(name, _)| name == field) {
                    Some((_, ty)) => *ty,
                    None => return Err(format!("the object has no field {field:?}")),
                },
                (PathSeg::Index(_), TyKind::Array(element, _)) => *element,
                (PathSeg::Index(index), TyKind::Tuple(elements)) => match elements.get(*index) {
                    Some(element) => *element,
                    None => return Err(format!("the tuple has no element {index}")),
                },
                (PathSeg::Payload, TyKind::Option(payload)) => *payload,
                (PathSeg::Payload, TyKind::Result(..) | TyKind::Enum { .. }) => return Ok(None),
                (seg, other) => return Err(format!("{seg:?} does not step into {other:?}")),
            };
        }
        Ok(Some(at))
    }

    fn field_path(&self, object: Ty, field: acvus_utils::Astr, rest: &[acvus_utils::Astr]) -> Result<Ty, String> {
        let mut at = object;
        for name in std::iter::once(&field).chain(rest) {
            let TyKind::Object(fields) = self.kind(at) else {
                return Err(format!("{:?} is not an object", self.kind(at)));
            };
            let Some((_, ty)) = fields.fields().iter().find(|(declared, _)| declared == name) else {
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
            Terminator::Return { order: Some(order), .. } => {
                self.is(*order, |kind| matches!(kind, TyKind::Order), "Order").map_err(at)?
            }
            Terminator::Jump { .. }
            | Terminator::Switch { .. }
            | Terminator::For { .. }
            | Terminator::Return { order: None, .. }
            | Terminator::Diverge => {}
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
}

// -- DefDominatesUses --------------------------------------------------------

fn def_dominates_uses(body: &Body, _: &Types<Concrete>) -> Found {
    let dominators = Dominators::of(body);
    let points = def_points(body);
    let check = |value: ValueId, site: Site| {
        let Some(point) = points.get(&value) else {
            return Err(format!("{site:?} reads {value:?}, which nothing in the body defines"));
        };
        found(dominators.dominates_site(*point, site), || {
            format!("{value:?}, defined at {point:?}, does not dominate its use at {site:?}")
        })
    };
    for (id, block) in body.blocks() {
        for (index, inst) in block.insts().iter().enumerate() {
            inst.operands().into_iter().try_for_each(|value| check(value, Site::Inst(id, index)))?;
        }
        block.terminator().operands().into_iter().try_for_each(|value| check(value, Site::Term(id)))?;
    }
    Ok(())
}

// -- OrderChainWhole ---------------------------------------------------------

/// Each token reaches the body's return: a use carries it on, to the token
/// an instruction yields or merges, to the parameter an edge passes it to,
/// or to a `Return`; a token nothing uses is whole only where no path from
/// its block returns, since a run that dies owes nothing (effects §5).
fn order_chain_whole(body: &Body, _: &Types<Concrete>) -> Found {
    let ordered = body.order_param().is_some();
    let mut defined_in: FxHashMap<ValueId, BlockId> = FxHashMap::default();
    for (id, block) in body.blocks() {
        for param in block.params().iter().filter(|param| body.is_token(**param)) {
            defined_in.insert(*param, id);
        }
        for (index, inst) in block.insts().iter().enumerate() {
            let place = || format!("{id:?}[{index}] {:?}", inst.op());
            if let Some(order) = inst.order() {
                found(ordered, || format!("{} stands on the order chain of an unordered body", place()))?;
                for token in order.consumed().into_iter().chain(order.yielded()) {
                    found(body.is_token(token), || format!("{} names {token:?}, which is not a token", place()))?;
                }
                if let Some(token) = order.yielded() {
                    defined_in.insert(token, id);
                }
            }
            let tokens_in_op = match inst.op() {
                Op::Merge { orders } => {
                    let result = inst.result().expect("a Merge yields its token");
                    defined_in.insert(result, id);
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
        match terminator {
            Terminator::Return { order, value, .. } => {
                found(ordered == order.is_some(), || {
                    format!("{id:?}'s return carries {order:?} in a body whose order parameter is {:?}", body.order_param())
                })?;
                found(!body.is_token(*value), || format!("{id:?} returns a token"))?;
            }
            Terminator::Diverge => {}
            _ => {
                for edge in terminator.edges() {
                    let params = &body.block(edge.to).params()[edge.supplied..];
                    for (param, arg) in params.iter().zip(edge.args) {
                        found(body.is_token(*param) == body.is_token(*arg), || {
                            format!("{id:?} passes {arg:?} to {param:?}, and only one of them is a token")
                        })?;
                    }
                }
            }
        }
    }
    if let Some(order_param) = body.order_param() {
        defined_in.insert(order_param, body.entry());
    }

    let returning = blocks_reaching_return(body);
    let carried = |token: ValueId| -> SmallVec<[Carried; 2]> {
        let mut carried = SmallVec::new();
        for site in body.users(token) {
            match site {
                Site::Inst(block, index) => {
                    let inst = &body.block(block).insts()[index];
                    match (inst.op(), inst.order()) {
                        (Op::Merge { .. }, _) => carried.extend(inst.result().map(Carried::To)),
                        (_, Some(OrderUse::Edge(edge))) => carried.push(Carried::To(edge.after)),
                        (Op::Spawn { .. }, Some(OrderUse::Start(_))) => {
                            let handle = inst.result().expect("a start yields its handle");
                            for taken in body.users(handle) {
                                let Site::Inst(block, index) = taken else { continue };
                                if let Some(after) = body.block(block).insts()[index].order().and_then(OrderUse::yielded) {
                                    carried.push(Carried::To(after));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Site::Term(block) => match body.block(block).terminator() {
                    Terminator::Return { order: Some(order), .. } if *order == token => carried.push(Carried::Returned),
                    terminator => {
                        for edge in terminator.edges() {
                            let params = &body.block(edge.to).params()[edge.supplied..];
                            for (param, arg) in params.iter().zip(edge.args) {
                                if *arg == token {
                                    carried.push(Carried::To(*param));
                                }
                            }
                        }
                    }
                },
            }
        }
        carried
    };
    let tokens: Vec<ValueId> = defined_in.keys().copied().collect();
    let flows: FxHashMap<ValueId, SmallVec<[Carried; 2]>> = tokens.iter().map(|token| (*token, carried(*token))).collect();
    let mut whole: FxHashSet<ValueId> = tokens
        .iter()
        .copied()
        .filter(|token| {
            let carried = &flows[token];
            carried.contains(&Carried::Returned) || (carried.is_empty() && !returning.contains(&defined_in[token]))
        })
        .collect();
    let mut grew = true;
    while grew {
        grew = false;
        for token in &tokens {
            if !whole.contains(token)
                && flows[token].iter().any(|carried| matches!(carried, Carried::To(next) if whole.contains(next)))
            {
                whole.insert(*token);
                grew = true;
            }
        }
    }
    let mut broken: Vec<ValueId> = tokens.into_iter().filter(|token| !whole.contains(token)).collect();
    broken.sort();
    found(broken.is_empty(), || format!("the chain from {broken:?} does not reach the body's return"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Carried {
    To(ValueId),
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
    let count = body.blocks().count();
    let is_block = |block: BlockId| block.index() < count;
    for (id, block) in body.blocks() {
        match block.terminator() {
            Terminator::Diamond { join, .. } => {
                found(is_block(*join), || format!("{id:?}'s Diamond joins at {join:?}, which is not a block"))?
            }
            Terminator::For { stages, exit, .. } | Terminator::While { stages, exit, .. } => {
                for stage in stages.entries().chain(std::iter::once(*exit)) {
                    found(is_block(stage), || format!("{id:?}'s loop names {stage:?}, which is not a block"))?;
                }
            }
            Terminator::Jump { .. }
            | Terminator::JumpIf { .. }
            | Terminator::Switch { .. }
            | Terminator::Return { .. }
            | Terminator::Diverge => {}
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
    for (id, block) in body.blocks() {
        for (index, inst) in block.insts().iter().enumerate() {
            let Op::Spawn { .. } = inst.op() else { continue };
            let handle = inst.result().expect("a start yields its handle");
            for site in body.users(handle) {
                let takes = match site {
                    Site::Inst(block, index) => matches!(body.block(block).insts()[index].op(), Op::TakeEffect { .. }),
                    Site::Term(_) => false,
                };
                found(takes, || format!("the handle {handle:?} of {id:?}[{index}] is read at {site:?}, not taken"))?;
            }
            walk_from_start(body, (id, index), handle)?;
        }
    }
    Ok(())
}

fn walk_from_start(body: &Body, start: (BlockId, usize), handle: ValueId) -> Found {
    let (start_block, start_index) = start;
    let mut visited: FxHashSet<(BlockId, usize, bool)> = FxHashSet::default();
    let mut pending = vec![(start_block, start_index + 1, false)];
    while let Some(state) = pending.pop() {
        if !visited.insert(state) {
            continue;
        }
        let (block, index, taken) = state;
        let insts = body.block(block).insts();
        if let Some(inst) = insts.get(index) {
            if (block, index) == start {
                continue;
            }
            let takes = matches!(inst.op(), Op::TakeEffect { src } if *src == handle);
            found(!(takes && taken), || format!("{start:?}'s call is taken again at {block:?}[{index}]"))?;
            pending.push((block, index + 1, taken || takes));
            continue;
        }
        match body.block(block).terminator() {
            Terminator::Return { .. } => {
                found(taken, || format!("{start:?}'s call is not taken on a path that returns from {block:?}"))?
            }
            Terminator::Diverge => {}
            terminator => pending.extend(terminator.edges().iter().map(|edge| (edge.to, 0, taken))),
        }
    }
    Ok(())
}
