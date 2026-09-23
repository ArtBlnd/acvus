use std::collections::BTreeSet;
use std::convert::Infallible;
use std::fmt;

use crate::graph::types::QualifiedRef;
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

// -- UserDefined type system ------------------------------------------

/// Declaration of a user-defined type - the **single source of truth**
/// for parameter count and constraints. Registered once, referenced by QualifiedRef everywhere.
#[derive(Debug, Clone)]
pub struct UserDefinedDecl {
    pub qref: QualifiedRef,
    pub type_params: Vec<TyVarBound>,
    pub effect_params: usize,
    /// Identity parameters, at most one: a value of a type with one is a
    /// distinct source, and the type is move-only.
    pub identity_params: usize,
    /// Region parameters: the lifetimes the declaring Rust type takes, one
    /// position each in front of its type arguments' (RFC-0079 rules 2 and
    /// 6). A type that holds a reference, a slice, a closure or an instance
    /// declares one.
    pub region_params: usize,
    /// Per type parameter, whether the type lays its storage out by that
    /// argument, so the argument is a specializing position
    /// (hash-types.md, R1). One entry per `type_params` entry.
    pub specializable: Vec<bool>,
}

/// A declaration under a name the registry already has a type for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DuplicateType(pub QualifiedRef);

/// The width and signedness of an integer type: `i8` to `i64`, `u8` to
/// `u64`. An integer literal takes the width its use demands and is `i64`
/// where nothing demands one (RFC-0037).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntTy {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntTy {
    pub const ALL: [IntTy; 8] = [
        IntTy::I8,
        IntTy::I16,
        IntTy::I32,
        IntTy::I64,
        IntTy::U8,
        IntTy::U16,
        IntTy::U32,
        IntTy::U64,
    ];

    pub fn bits(self) -> u32 {
        match self {
            IntTy::I8 | IntTy::U8 => 8,
            IntTy::I16 | IntTy::U16 => 16,
            IntTy::I32 | IntTy::U32 => 32,
            IntTy::I64 | IntTy::U64 => 64,
        }
    }

    pub fn bytes(self) -> usize {
        self.bits() as usize / 8
    }

    pub fn signed(self) -> bool {
        matches!(self, IntTy::I8 | IntTy::I16 | IntTy::I32 | IntTy::I64)
    }

    pub fn name(self) -> &'static str {
        match self {
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::U8 => "u8",
            IntTy::U16 => "u16",
            IntTy::U32 => "u32",
            IntTy::U64 => "u64",
        }
    }

    pub fn min(self) -> i128 {
        if self.signed() {
            -(1i128 << (self.bits() - 1))
        } else {
            0
        }
    }

    pub fn max(self) -> i128 {
        if self.signed() {
            (1i128 << (self.bits() - 1)) - 1
        } else {
            (1i128 << self.bits()) - 1
        }
    }

    /// Whether `value` is representable in this type.
    pub fn holds(self, value: i128) -> bool {
        (self.min()..=self.max()).contains(&value)
    }

    /// The value the low `bits()` of `bits` spell, sign- or zero-extended
    /// to `i128`.
    pub fn read(self, bits: u64) -> i128 {
        match self {
            IntTy::I8 => bits as u8 as i8 as i128,
            IntTy::I16 => bits as u16 as i16 as i128,
            IntTy::I32 => bits as u32 as i32 as i128,
            IntTy::I64 => bits as i64 as i128,
            IntTy::U8 => bits as u8 as i128,
            IntTy::U16 => bits as u16 as i128,
            IntTy::U32 => bits as u32 as i128,
            IntTy::U64 => bits as i128,
        }
    }
}

/// The type an `as` cast names, at either end of it (RFC-0049, RFC-0058).
///
/// No `f32`: RFC-0037 does not have the type, and `as` does not add one.
/// Which pairs a cast may join is `Cast::admits`, not this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CastTy {
    Int(IntTy),
    F64,
    Char,
}

impl CastTy {
    pub const NAMES: &'static str = "i8, i16, i32, i64, u8, u16, u32, u64, f64 or char";

    pub fn of_name(name: &str) -> Option<CastTy> {
        match name {
            "f64" => Some(CastTy::F64),
            "char" => Some(CastTy::Char),
            _ => IntTy::ALL
                .into_iter()
                .find(|k| k.name() == name)
                .map(CastTy::Int),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CastTy::Int(k) => k.name(),
            CastTy::F64 => "f64",
            CastTy::Char => "char",
        }
    }

    pub fn of_ty<V>(ty: &TyTerm<V>) -> Option<CastTy>
    where
        V: Phase,
    {
        match ty {
            TyTerm::Int(k) => Some(CastTy::Int(*k)),
            TyTerm::Float => Some(CastTy::F64),
            TyTerm::Char => Some(CastTy::Char),
            _ => None,
        }
    }

    /// The numeric type whose word this type's word is. A `char` is a
    /// Unicode scalar value, which is its `u32`, and every cast Rust
    /// admits at either end of a `char` has the value of the same cast
    /// through `u32`: `u8 as char` zero-extends, and `char as T`
    /// truncates or extends from 32 bits.
    pub fn word(self) -> WordTy {
        match self {
            CastTy::Int(k) => WordTy::Int(k),
            CastTy::F64 => WordTy::F64,
            CastTy::Char => WordTy::Int(IntTy::U32),
        }
    }

    /// Whether `self as to` is a cast Rust admits: every pair of numbers,
    /// a `char` to any integer, and `u8` to `char` and nothing else.
    pub fn admits(self, to: CastTy) -> bool {
        match (self, to) {
            (_, CastTy::Char) => self == CastTy::Int(IntTy::U8),
            (CastTy::Char, CastTy::F64) => false,
            (CastTy::Char, _) | (CastTy::Int(_) | CastTy::F64, _) => true,
        }
    }
}

/// The type a word is read and written at: RFC-0037's eight widths and
/// `f64`, which is every type the machine converts between. A `char` is
/// not one of them and has no word of its own — `CastTy::word`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WordTy {
    Int(IntTy),
    F64,
}

impl<V> From<CastTy> for TyTerm<V>
where
    V: Phase,
{
    fn from(num: CastTy) -> TyTerm<V> {
        match num {
            CastTy::Int(k) => TyTerm::Int(k),
            CastTy::F64 => TyTerm::Float,
            CastTy::Char => TyTerm::Char,
        }
    }
}

impl From<acvus_ast::IntWidth> for IntTy {
    fn from(width: acvus_ast::IntWidth) -> IntTy {
        match width {
            acvus_ast::IntWidth::I8 => IntTy::I8,
            acvus_ast::IntWidth::I16 => IntTy::I16,
            acvus_ast::IntWidth::I32 => IntTy::I32,
            acvus_ast::IntWidth::I64 => IntTy::I64,
            acvus_ast::IntWidth::U8 => IntTy::U8,
            acvus_ast::IntWidth::U16 => IntTy::U16,
            acvus_ast::IntWidth::U32 => IntTy::U32,
            acvus_ast::IntWidth::U64 => IntTy::U64,
        }
    }
}

/// What a declaration says about one of its type variables. The solver
/// carries it on the variable and verifies it when the variable freezes.
#[derive(Debug, Clone, PartialEq)]
pub enum TyVarBound {
    Any,
    /// The variable resolves to a type of one of these shapes (RFC-0011 rule 2).
    OneOf {
        shapes: Vec<PolyTy>,
    },
    /// The variable is an integer literal's type: one of `among`, which a
    /// use narrows, signed where the literal is negated (RFC-0037).
    Integer {
        signed: bool,
        among: Vec<IntTy>,
    },
}

impl TyVarBound {
    pub fn one_of(shapes: Vec<PolyTy>) -> Self {
        Self::OneOf { shapes }
    }

    /// Whether a variable under this bound may still become a reference.
    pub fn admits_a_reference(&self) -> bool {
        match self {
            TyVarBound::Any => true,
            TyVarBound::OneOf { shapes } => shapes
                .iter()
                .any(|shape| matches!(shape, TyTerm::Ref(..) | TyTerm::Var(_))),
            TyVarBound::Integer { .. } => false,
        }
    }

    /// An integer bound over the widths `among` yields, signed-only where
    /// `signed`; `None` when no width remains.
    pub fn integer(signed: bool, among: impl Iterator<Item = IntTy>) -> Option<Self> {
        let among: Vec<IntTy> = among.filter(|k| !signed || k.signed()).collect();
        (!among.is_empty()).then_some(Self::Integer { signed, among })
    }

    pub fn is_text(&self) -> bool {
        let Self::OneOf { shapes } = self else {
            return false;
        };
        shapes.len() == 2
            && shapes.iter().any(|s| matches!(s, TyTerm::String))
            && shapes.iter().any(|s| matches!(s, TyTerm::Str))
    }

    /// The width an integer literal takes where its uses left more than one:
    /// `i64` when it remains, the only one when one remains, none otherwise.
    pub fn integer_default(&self) -> Option<IntTy> {
        let Self::Integer { among, .. } = self else {
            return None;
        };
        if among.contains(&IntTy::I64) {
            Some(IntTy::I64)
        } else if let [only] = among.as_slice() {
            Some(*only)
        } else {
            None
        }
    }

    pub fn admits(&self, ty: &Ty) -> bool {
        match self {
            _ if matches!(ty, Ty::Error(_)) => true,
            Self::Any => true,
            Self::OneOf { shapes } => shapes.iter().any(|s| matches_poly(ty, s)),
            Self::Integer { signed, among } => {
                matches!(ty, Ty::Int(k) if among.contains(k) && (!*signed || k.signed()))
            }
        }
    }

    /// The bound either side satisfies (RFC-0043).
    pub fn union(self, other: Self) -> Self {
        let shapes = |bound: Self| -> Option<Vec<PolyTy>> {
            match bound {
                Self::Any => None,
                Self::OneOf { shapes, .. } => Some(shapes),
                Self::Integer { among, .. } => Some(among.into_iter().map(TyTerm::Int).collect()),
            }
        };
        match (shapes(self), shapes(other)) {
            (Some(mut a), Some(b)) => {
                a.extend(b);
                Self::OneOf { shapes: a }
            }
            (None, _) | (_, None) => Self::Any,
        }
    }

    /// The bound both sides satisfy: the pairwise unifiers of their shapes.
    /// `None` when no type does.
    pub fn meet(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::Any, b) | (b, Self::Any) => Some(b.clone()),
            (
                Self::Integer {
                    signed: sa,
                    among: aa,
                },
                Self::Integer {
                    signed: sb,
                    among: ab,
                },
            ) => Self::integer(*sa || *sb, aa.iter().filter(|k| ab.contains(k)).copied()),
            (Self::Integer { signed, among }, Self::OneOf { shapes, .. })
            | (Self::OneOf { shapes, .. }, Self::Integer { signed, among }) => Self::integer(
                *signed,
                shapes.iter().filter_map(|s| match s {
                    TyTerm::Int(k) if among.contains(k) => Some(*k),
                    _ => None,
                }),
            ),
            (Self::OneOf { shapes: a }, Self::OneOf { shapes: b }) => {
                let both: Vec<PolyTy> = a
                    .iter()
                    .flat_map(|x| b.iter().filter_map(move |y| unify_patterns(x, y)))
                    .collect();
                (!both.is_empty()).then_some(Self::OneOf { shapes: both })
            }
        }
    }
}

/// The instances of an Extern function, numbered as the runtime numbers
/// its handlers: the concrete ones in order, then the generic one
/// (RFC-0040). `acvus_extern::Instances` is the runtime's
/// half of that contract.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Instances {
    pub concrete: Vec<InstanceSig>,
    pub generic: Option<GenericSig>,
}

/// The generic instance, whose type is the function's own scheme.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GenericSig {
    pub laws: crate::laws::Laws,
    pub ensures: Vec<crate::laws::Postcondition>,
}

impl Instances {
    pub fn generic_index(&self) -> usize {
        self.concrete.len()
    }
}

/// One concrete instance of an extern: its signature, the greatest task it
/// admits a call at, which is a ceiling and not its own task (RFC-0046),
/// and the task its body runs at, which is what a requirement written at a
/// lower task cannot reach (RFC-0068).
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceSig {
    pub ty: PolyTy,
    pub admits: Task,
    pub task: Task,
    pub requires: Vec<RequirementSig>,
    /// The bound of each effect variable of `ty`, by position, as the
    /// `instance_of` declaration states it (RFC-0011 rule 5). A declaration
    /// that is no signature's instance states its bounds on its scheme, so
    /// its instances carry none.
    pub effect_bounds: Vec<EffectVarBound>,
    pub laws: crate::laws::Laws,
    pub ensures: Vec<crate::laws::Postcondition>,
}

impl InstanceSig {
    pub fn any_task(ty: PolyTy) -> Self {
        Self {
            ty,
            admits: Task::Heavy,
            task: Task::Sync,
            requires: Vec::new(),
            effect_bounds: Vec::new(),
            laws: crate::laws::Laws::None,
            ensures: Vec::new(),
        }
    }
}

pub fn effect_bound_at(bounds: &[EffectVarBound], var: u32) -> EffectVarBound {
    bounds
        .get(var as usize)
        .copied()
        .unwrap_or(EffectVarBound::Any)
}

/// What a declaration requires of its type variables: an instance of
/// `signature` whose type joins `pattern`, written at the declaration's own
/// variables, and called at no task above `calls` (RFC-0068 rule 5). The
/// checker decides it as it decides a call of the signature.
#[derive(Debug, Clone, PartialEq)]
pub struct RequirementSig {
    pub signature: QualifiedRef,
    pub pattern: PolyTy,
    pub calls: Task,
}

/// What a declaration says about one of its effect variables: a floor on
/// its task. The solver carries it on the variable and verifies it when the
/// variable freezes (RFC-0011 rule 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectVarBound {
    Any,
    /// The variable's task is `Async` or `Heavy`.
    Suspends,
}

impl EffectVarBound {
    /// The bound both sides satisfy: two unified variables share a floor.
    pub fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Self::Any, Self::Any) => Self::Any,
            (Self::Suspends, _) | (_, Self::Suspends) => Self::Suspends,
        }
    }

    pub fn admits(self, task: Task) -> bool {
        match (self, task) {
            (Self::Any, _) | (Self::Suspends, Task::Async | Task::Heavy) => true,
            (Self::Suspends, Task::Sync) => false,
        }
    }
}

/// A polymorphic type with the bounds its variables were declared with.
/// Type variable `i` of `ty` has bound `bounds[i]`, effect variable `i`
/// has `effect_bounds[i]`; a missing entry is `Any`. `instances` is `Some`
/// for an Extern function.
#[derive(Debug, Clone, PartialEq)]
pub struct Scheme {
    pub ty: PolyTy,
    pub bounds: Vec<TyVarBound>,
    pub effect_bounds: Vec<EffectVarBound>,
    pub instances: Option<Instances>,
    pub requires: Vec<Requirement>,
}

/// A `RequirementSig` with the instances of the signature it names, so
/// that the solver decides it without reaching for the environment.
#[derive(Debug, Clone, PartialEq)]
pub struct Requirement {
    pub signature: QualifiedRef,
    pub pattern: PolyTy,
    pub calls: Task,
    pub instances: Instances,
}

impl Scheme {
    pub fn unbounded(ty: PolyTy) -> Self {
        Self {
            ty,
            bounds: Vec::new(),
            effect_bounds: Vec::new(),
            instances: None,
            requires: Vec::new(),
        }
    }

    pub fn bound_of(&self, var: u32) -> TyVarBound {
        self.bounds
            .get(var as usize)
            .cloned()
            .unwrap_or(TyVarBound::Any)
    }

    pub fn effect_bound_of(&self, var: u32) -> EffectVarBound {
        effect_bound_at(&self.effect_bounds, var)
    }

    pub fn params(&self) -> &[ParamTerm<Poly>] {
        match &self.ty {
            TyTerm::Fn { params, .. } => params,
            _ => &[],
        }
    }

    pub fn ret(&self) -> &PolyTy {
        match &self.ty {
            TyTerm::Fn { ret, .. } => ret,
            other => other,
        }
    }

    /// The bound a call's parameter takes from this scheme's parameter
    /// (RFC-0043): every `OneOf`-bounded variable expanded to its shapes,
    /// shifted past the scheme's variables so the two sets stay apart.
    pub fn param_bound(&self, param: &PolyTy) -> TyVarBound {
        let scheme_span = var_span(&self.ty);
        let mut shapes = vec![param.clone()];
        while let Some((var, bound)) = shapes.iter().find_map(|s| self.first_bounded_var(s)) {
            shapes = shapes
                .iter()
                .flat_map(|shape| {
                    let by = scheme_span.max(var_span(shape));
                    bound
                        .iter()
                        .map(move |b| substitute_var(shape, var, &shift_vars(b, by)))
                })
                .collect();
        }
        if shapes.iter().any(|shape| matches!(shape, TyTerm::Var(_))) {
            return TyVarBound::Any;
        }
        TyVarBound::one_of(shapes)
    }

    fn first_bounded_var(&self, pattern: &PolyTy) -> Option<(u32, Vec<PolyTy>)> {
        let mut found = None;
        let visited = pattern.map::<Poly>(
            &mut |v| {
                if found.is_none()
                    && let TyVarBound::OneOf { shapes: bound, .. } = self.bound_of(v)
                {
                    found = Some((v, bound));
                }
                TyTerm::Var(v)
            },
            &mut IdentityTerm::Var,
            &mut EffectTerm::Var,
            &mut LenTerm::Var,
            &mut Repr::Var,
        );
        debug_assert_eq!(&visited, pattern, "the identity map rebuilds the pattern");
        found
    }
}

fn substitute_var(pattern: &PolyTy, var: u32, by: &PolyTy) -> PolyTy {
    pattern.map::<Poly>(
        &mut |v| if v == var { by.clone() } else { TyTerm::Var(v) },
        &mut IdentityTerm::Var,
        &mut EffectTerm::Var,
        &mut LenTerm::Var,
        &mut Repr::Var,
    )
}

/// Whether a concrete type has the shape of a polymorphic pattern.
pub fn matches_poly(ty: &Ty, pattern: &PolyTy) -> bool {
    matches_pattern(ty, pattern)
}

/// Whether a type has the shape of a polymorphic pattern. A pattern
/// variable stands for any one term, the same term wherever it recurs;
/// effect, length, and identity variables of the pattern stand for any
/// effect, length, or identity. A variable of `ty` matches only a pattern
/// variable: the pattern is the more general of the two.
pub fn matches_pattern<P>(ty: &TyTerm<P>, pattern: &PolyTy) -> bool
where
    P: Phase + PartialEq,
{
    matches_pattern_with(ty, pattern, Unknowns::Fixed, Reprs::Exact)
}

/// Whether every type of `special`'s shape is of `general`'s, the
/// representation of an argument aside: `Items<#i64>` is of the shape
/// `Items<T>`. A refusal lists shapes, and a specialized instance is not a
/// second shape to a reader.
pub fn subsumes(general: &PolyTy, special: &PolyTy) -> bool {
    matches_pattern_with(special, general, Unknowns::Fixed, Reprs::Alike)
}

/// Whether a type whose variables are still open could have the shape of
/// a pattern.
pub fn could_match_pattern<P>(ty: &TyTerm<P>, pattern: &PolyTy) -> bool
where
    P: Phase + PartialEq,
{
    matches_pattern_with(ty, pattern, Unknowns::Open, Reprs::Exact)
}

/// `pattern` with each type variable of `chosen` bound where `ty` fills
/// its first slot (RFC-0041): the slot's `ρ` binds the tree `ty` holds
/// there, and the variable binds that tree's type, everywhere in
/// `pattern`. A signature so bound states what its instance chose, and
/// `matches_pattern` compares the rest exactly. The tree's own variables
/// are renamed past `pattern`'s. A variable with no slot the walk
/// reaches is left as it is, and its `ρ` stands for uniform.
pub fn bind_chosen(pattern: &PolyTy, ty: &PolyTy, chosen: &[u32]) -> PolyTy {
    let mut found = FxHashMap::default();
    chosen_slots(pattern, ty, chosen, &mut found);
    let by = var_span(pattern);
    let found: FxHashMap<u32, ChosenSlot> = found
        .into_iter()
        .map(|(var, slot)| (var, slot.renamed_past(by)))
        .collect();
    let reprs: FxHashMap<u32, Repr<Poly>> = found
        .values()
        .map(|slot| (slot.repr, slot.arg.repr()))
        .collect();
    pattern.map::<Poly>(
        &mut |v| match found.get(&v) {
            Some(slot) => slot.arg.ty().into_owned(),
            None => TyTerm::Var(v),
        },
        &mut IdentityTerm::Var,
        &mut EffectTerm::Var,
        &mut LenTerm::Var,
        &mut |r| match reprs.get(&r) {
            Some(repr) => repr.clone(),
            None => Repr::Var(r),
        },
    )
}

/// A chosen variable's first slot in a pattern: the slot's `ρ`, and the
/// argument the matched type holds there.
struct ChosenSlot {
    repr: u32,
    arg: TypeArg<Poly>,
}

impl ChosenSlot {
    fn renamed_past(self, by: VarSpan) -> Self {
        let arg = self.arg.map::<Poly>(
            &mut |v| TyTerm::Var(v + by.ty),
            &mut |v| IdentityTerm::Var(v + by.identity),
            &mut |v| EffectTerm::Var(v + by.effect),
            &mut |v| LenTerm::Var(v + by.len),
            &mut |v| Repr::Var(v + by.repr),
        );
        Self { arg, ..self }
    }
}

/// A head with no arm here, an object or an enum, is not descended: a
/// slot under it stays unbound, and `matches_pattern` refuses an instance
/// that fills it with `#`. Two heads that differ are left to
/// `matches_pattern` the same way.
fn chosen_slots(
    pattern: &PolyTy,
    ty: &PolyTy,
    chosen: &[u32],
    found: &mut FxHashMap<u32, ChosenSlot>,
) {
    match (pattern, ty) {
        (TyTerm::Array(p, _), TyTerm::Array(t, _))
        | (TyTerm::Option(p), TyTerm::Option(t))
        | (TyTerm::Slice(p), TyTerm::Slice(t))
        | (TyTerm::Handle(p), TyTerm::Handle(t)) => chosen_slots(p, t, chosen, found),
        (TyTerm::Result(p_ok, p_err), TyTerm::Result(t_ok, t_err)) => {
            chosen_slots(p_ok, t_ok, chosen, found);
            chosen_slots(p_err, t_err, chosen, found);
        }
        (TyTerm::Tuple(ps), TyTerm::Tuple(ts)) => {
            for (p, t) in ps.iter().zip(ts) {
                chosen_slots(p, t, chosen, found);
            }
        }
        (
            TyTerm::Fn {
                params: pp,
                ret: pr,
                captures: pc,
                ..
            },
            TyTerm::Fn {
                params: tp,
                ret: tr,
                captures: tc,
                ..
            },
        ) => {
            for (p, t) in pp.iter().zip(tp) {
                chosen_slots(&p.ty, &t.ty, chosen, found);
            }
            chosen_slots(pr, tr, chosen, found);
            for (p, t) in pc.iter().zip(tc) {
                chosen_slots(p, t, chosen, found);
            }
        }
        (TyTerm::UserDefined { type_args: pa, .. }, TyTerm::UserDefined { type_args: ta, .. }) => {
            for (p, t) in pa.iter().zip(ta) {
                chosen_arg(p, t, chosen, found);
            }
        }
        (TyTerm::Ref(_, p), TyTerm::Ref(_, t)) => chosen_arg(p, t, chosen, found),
        _ => {}
    }
}

fn chosen_arg(
    pattern: &TypeArg<Poly>,
    ty: &TypeArg<Poly>,
    chosen: &[u32],
    found: &mut FxHashMap<u32, ChosenSlot>,
) {
    match (pattern, ty) {
        (TypeArg::Open(repr, TyTerm::Var(v)), _) if chosen.contains(v) => {
            found.entry(*v).or_insert_with(|| ChosenSlot {
                repr: *repr,
                arg: ty.clone(),
            });
        }
        (TypeArg::Specialized(p), TypeArg::Specialized(t)) => chosen_held(p, t, chosen, found),
        _ => chosen_slots(&pattern.ty(), &ty.ty(), chosen, found),
    }
}

fn chosen_held(
    pattern: &HeldTy<Poly>,
    ty: &HeldTy<Poly>,
    chosen: &[u32],
    found: &mut FxHashMap<u32, ChosenSlot>,
) {
    match (pattern, ty) {
        (HeldTy::Tuple(ps), HeldTy::Tuple(ts)) => {
            for (p, t) in ps.iter().zip(ts) {
                chosen_arg(p, t, chosen, found);
            }
        }
        (HeldTy::Option(p), HeldTy::Option(t))
        | (HeldTy::Array(p, _), HeldTy::Array(t, _))
        | (HeldTy::RustArray(p, _), HeldTy::RustArray(t, _)) => chosen_arg(p, t, chosen, found),
        (HeldTy::Result(p_ok, p_err), HeldTy::Result(t_ok, t_err)) => {
            chosen_arg(p_ok, t_ok, chosen, found);
            chosen_arg(p_err, t_err, chosen, found);
        }
        (HeldTy::Leaf(p), HeldTy::Leaf(t)) => chosen_slots(p.ty(), t.ty(), chosen, found),
        _ => {}
    }
}

/// What a variable of the matched type stands for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Unknowns {
    /// One term, not yet named: it matches only a pattern variable.
    Fixed,
    /// Anything: it matches every pattern.
    Open,
}

/// Whether an argument's representation takes part in the match.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reprs {
    /// A fixed representation matches itself (hash-types.md, R3).
    Exact,
    /// Any representation matches any: the shape alone is compared.
    Alike,
}

fn matches_pattern_with<P>(
    ty: &TyTerm<P>,
    pattern: &PolyTy,
    unknowns: Unknowns,
    reprs: Reprs,
) -> bool
where
    P: Phase + PartialEq,
{
    fn effect_matches<P>(
        effect: &EffectTerm<P>,
        pattern: &EffectTerm<Poly>,
        unknowns: Unknowns,
    ) -> bool
    where
        P: Phase + PartialEq,
    {
        match (effect, pattern) {
            (_, EffectTerm::Var(_)) => true,
            (EffectTerm::Var(_), _) => unknowns == Unknowns::Open,
            (EffectTerm::Known(e), EffectTerm::Known(p)) => e == p,
        }
    }
    fn effect_arg_matches<P>(
        arg: &EffectArg<P>,
        pat: &EffectArg<Poly>,
        unknowns: Unknowns,
        reprs: Reprs,
    ) -> bool
    where
        P: Phase + PartialEq,
    {
        (reprs == Reprs::Alike || arg.is_specialized() == pat.is_specialized())
            && effect_matches(arg.effect(), pat.effect(), unknowns)
    }
    fn len_matches<P>(len: &LenTerm<P>, pat: &LenTerm<Poly>, unknowns: Unknowns) -> bool
    where
        P: Phase + PartialEq,
    {
        match (len, pat) {
            (_, LenTerm::Var(_)) => true,
            (LenTerm::Var(_), _) => unknowns == Unknowns::Open,
            (LenTerm::Known(n), LenTerm::Known(k)) => n == k,
        }
    }
    /// A pattern's representation variable stands for the uniform
    /// representation: the generic instance is the uniform one
    /// (hash-types.md, R3). A fixed representation matches itself, part by
    /// part; a variable matches a pattern variable, and a fixed one only
    /// when unknowns are open.
    fn arg_matches<P>(
        arg: &TypeArg<P>,
        pat: &TypeArg<Poly>,
        seen: &mut FxHashMap<u32, TyTerm<P>>,
        unknowns: Unknowns,
        reprs: Reprs,
    ) -> bool
    where
        P: Phase + PartialEq,
    {
        match (arg, pat) {
            _ if reprs == Reprs::Alike => go(&arg.ty(), &pat.ty(), seen, unknowns, reprs),
            (TypeArg::Open(_, ty), TypeArg::Open(_, p)) => go(ty, p, seen, unknowns, reprs),
            (TypeArg::Open(..), _) => {
                unknowns == Unknowns::Open && go(&arg.ty(), &pat.ty(), seen, unknowns, reprs)
            }
            (TypeArg::Uniform(ty), TypeArg::Uniform(p) | TypeArg::Open(_, p)) => {
                go(ty, p, seen, unknowns, reprs)
            }
            (TypeArg::Specialized(held), TypeArg::Specialized(p)) => {
                held_matches(held, p, seen, unknowns, reprs)
            }
            (TypeArg::Specialized(_), TypeArg::Uniform(_) | TypeArg::Open(..))
            | (TypeArg::Uniform(_), TypeArg::Specialized(_)) => false,
        }
    }
    fn held_matches<P>(
        held: &HeldTy<P>,
        pat: &HeldTy<Poly>,
        seen: &mut FxHashMap<u32, TyTerm<P>>,
        unknowns: Unknowns,
        reprs: Reprs,
    ) -> bool
    where
        P: Phase + PartialEq,
    {
        match (held, pat) {
            (_, HeldTy::Held(v)) => {
                held.is_full(unknowns == Unknowns::Open)
                    && go(&held.ty(), &TyTerm::Var(*v), seen, unknowns, reprs)
            }
            (HeldTy::Held(_), _) => {
                unknowns == Unknowns::Open && go(&held.ty(), &pat.ty(), seen, unknowns, reprs)
            }
            (HeldTy::Tuple(parts), HeldTy::Tuple(ps)) => {
                parts.len() == ps.len()
                    && parts
                        .iter()
                        .zip(ps)
                        .all(|(a, p)| arg_matches(a, p, seen, unknowns, reprs))
            }
            (HeldTy::Option(a), HeldTy::Option(p)) => arg_matches(a, p, seen, unknowns, reprs),
            (HeldTy::Result(a_ok, a_err), HeldTy::Result(p_ok, p_err)) => {
                arg_matches(a_ok, p_ok, seen, unknowns, reprs)
                    && arg_matches(a_err, p_err, seen, unknowns, reprs)
            }
            (HeldTy::Array(a, n), HeldTy::Array(p, pn)) => {
                len_matches(n, pn, unknowns) && arg_matches(a, p, seen, unknowns, reprs)
            }
            (HeldTy::RustArray(a, n), HeldTy::RustArray(p, pn)) => {
                n == pn && arg_matches(a, p, seen, unknowns, reprs)
            }
            (HeldTy::Leaf(a), HeldTy::Leaf(p)) => go(a.ty(), p.ty(), seen, unknowns, reprs),
            (
                HeldTy::Tuple(_)
                | HeldTy::Option(_)
                | HeldTy::Result(..)
                | HeldTy::Array(..)
                | HeldTy::RustArray(..)
                | HeldTy::Leaf(_),
                _,
            ) => false,
        }
    }
    fn go<P>(
        ty: &TyTerm<P>,
        pat: &PolyTy,
        seen: &mut FxHashMap<u32, TyTerm<P>>,
        unknowns: Unknowns,
        reprs: Reprs,
    ) -> bool
    where
        P: Phase + PartialEq,
    {
        let open = |t: &TyTerm<P>| unknowns == Unknowns::Open && matches!(t, TyTerm::Var(_));
        match (ty, pat) {
            (_, TyTerm::Var(v)) => match seen.get(v) {
                Some(bound) => bound == ty || open(bound) || open(ty),
                None => {
                    seen.insert(*v, ty.clone());
                    true
                }
            },
            (TyTerm::Var(_), _) => unknowns == Unknowns::Open,
            (TyTerm::Int(a), TyTerm::Int(b)) => a == b,
            (TyTerm::Float, TyTerm::Float)
            | (TyTerm::Char, TyTerm::Char)
            | (TyTerm::String, TyTerm::String)
            | (TyTerm::Str, TyTerm::Str)
            | (TyTerm::Bool, TyTerm::Bool)
            | (TyTerm::Unit, TyTerm::Unit)
            | (TyTerm::Never, TyTerm::Never)
            | (TyTerm::Order, TyTerm::Order) => true,
            (TyTerm::Array(e, n), TyTerm::Array(pe, pn)) => {
                len_matches(n, pn, unknowns) && go(e, pe, seen, unknowns, reprs)
            }
            (TyTerm::Option(i), TyTerm::Option(pi)) => go(i, pi, seen, unknowns, reprs),
            (TyTerm::Result(t, e), TyTerm::Result(pt, pe)) => {
                go(t, pt, seen, unknowns, reprs) && go(e, pe, seen, unknowns, reprs)
            }
            (TyTerm::Handle(i), TyTerm::Handle(pi)) | (TyTerm::Slice(i), TyTerm::Slice(pi)) => {
                go(i, pi, seen, unknowns, reprs)
            }
            (TyTerm::Ref(m, i), TyTerm::Ref(pm, pi)) => {
                m == pm && arg_matches(i, pi, seen, unknowns, reprs)
            }
            (TyTerm::Tuple(es), TyTerm::Tuple(ps)) => {
                es.len() == ps.len()
                    && es
                        .iter()
                        .zip(ps)
                        .all(|(e, p)| go(e, p, seen, unknowns, reprs))
            }
            (TyTerm::Object(fs), TyTerm::Object(pfs)) => {
                fs.declaration() == pfs.declaration()
                    && fs.len() == pfs.len()
                    && fs.iter().all(|(k, v)| {
                        pfs.get(k)
                            .is_some_and(|pv| go(v, pv, seen, unknowns, reprs))
                    })
            }
            (
                TyTerm::Fn {
                    params,
                    ret,
                    effect,
                    ..
                },
                TyTerm::Fn {
                    params: pp,
                    ret: pr,
                    effect: pe,
                    ..
                },
            ) => {
                params.len() == pp.len()
                    && effect_matches(effect, pe, unknowns)
                    && params
                        .iter()
                        .zip(pp)
                        .all(|(a, b)| go(&a.ty, &b.ty, seen, unknowns, reprs))
                    && go(ret, pr, seen, unknowns, reprs)
            }
            (
                TyTerm::UserDefined {
                    id,
                    type_args,
                    effect_args,
                    identity_args,
                    region_params,
                },
                TyTerm::UserDefined {
                    id: pid,
                    type_args: pargs,
                    effect_args: peffects,
                    identity_args: pidentities,
                    region_params: pregions,
                },
            ) => {
                id == pid
                    && region_params == pregions
                    && type_args.len() == pargs.len()
                    && effect_args.len() == peffects.len()
                    && identity_args.len() == pidentities.len()
                    && effect_args
                        .iter()
                        .zip(peffects)
                        .all(|(a, b)| effect_arg_matches(a, b, unknowns, reprs))
                    && identity_args
                        .iter()
                        .zip(pidentities)
                        .all(|(a, b)| match (a, b) {
                            (_, IdentityTerm::Var(_)) => true,
                            (IdentityTerm::Var(_), _) => unknowns == Unknowns::Open,
                            (IdentityTerm::Known(a), IdentityTerm::Known(k)) => a == k,
                        })
                    && type_args
                        .iter()
                        .zip(pargs)
                        .all(|(a, b)| arg_matches(a, b, seen, unknowns, reprs))
            }
            (
                TyTerm::Enum { name, variants, .. },
                TyTerm::Enum {
                    name: pn,
                    variants: pv,
                    ..
                },
            ) => {
                name == pn
                    && variants.len() == pv.len()
                    && variants.iter().all(|(k, v)| match (v, pv.get(k)) {
                        (Some(v), Some(Some(p))) => go(v, p, seen, unknowns, reprs),
                        (None, Some(None)) => true,
                        _ => false,
                    })
            }
            _ => false,
        }
    }
    go(ty, pattern, &mut FxHashMap::default(), unknowns, reprs)
}

// -- Pattern unification ----------------------------------------------

/// The variables of a pattern, numbered per kind, as the one past the
/// highest of each.
#[derive(Default, Clone, Copy)]
struct VarSpan {
    ty: u32,
    identity: u32,
    effect: u32,
    len: u32,
    repr: u32,
}

impl VarSpan {
    fn max(self, other: Self) -> Self {
        Self {
            ty: self.ty.max(other.ty),
            identity: self.identity.max(other.identity),
            effect: self.effect.max(other.effect),
            len: self.len.max(other.len),
            repr: self.repr.max(other.repr),
        }
    }
}

fn var_span(pattern: &PolyTy) -> VarSpan {
    let mut span = VarSpan::default();
    let _ = pattern.map::<Poly>(
        &mut |v| {
            span.ty = span.ty.max(v + 1);
            TyTerm::Var(v)
        },
        &mut |v| {
            span.identity = span.identity.max(v + 1);
            IdentityTerm::Var(v)
        },
        &mut |v| {
            span.effect = span.effect.max(v + 1);
            EffectTerm::Var(v)
        },
        &mut |v| {
            span.len = span.len.max(v + 1);
            LenTerm::Var(v)
        },
        &mut |v| {
            span.repr = span.repr.max(v + 1);
            Repr::Var(v)
        },
    );
    span
}

fn shift_vars(pattern: &PolyTy, by: VarSpan) -> PolyTy {
    pattern.map::<Poly>(
        &mut |v| TyTerm::Var(v + by.ty),
        &mut |v| IdentityTerm::Var(v + by.identity),
        &mut |v| EffectTerm::Var(v + by.effect),
        &mut |v| LenTerm::Var(v + by.len),
        &mut |v| Repr::Var(v + by.repr),
    )
}

#[derive(Default)]
struct PatternSubst {
    ty: FxHashMap<u32, PolyTy>,
    repr: FxHashMap<u32, Repr<Poly>>,
    identity: FxHashMap<u32, IdentityTerm<Poly>>,
    effect: FxHashMap<u32, EffectTerm<Poly>>,
    len: FxHashMap<u32, LenTerm<Poly>>,
}

impl PatternSubst {
    fn walk(&self, t: &PolyTy) -> PolyTy {
        match t {
            TyTerm::Var(v) => match self.ty.get(v) {
                Some(bound) => self.walk(bound),
                None => t.clone(),
            },
            other => other.clone(),
        }
    }

    fn occurs(&self, v: u32, t: &PolyTy) -> bool {
        let mut found = false;
        let _ = t.map::<Poly>(
            &mut |x| {
                if x == v || self.ty.get(&x).is_some_and(|b| self.occurs(v, b)) {
                    found = true;
                }
                TyTerm::Var(x)
            },
            &mut IdentityTerm::Var,
            &mut EffectTerm::Var,
            &mut LenTerm::Var,
            &mut Repr::Var,
        );
        found
    }

    fn unify_arg(&mut self, a: &TypeArg<Poly>, b: &TypeArg<Poly>) -> bool {
        self.unify(&a.ty(), &b.ty()) && self.meet_reprs(&self.apply_arg(a), &self.apply_arg(b))
    }

    /// Two arguments whose types have unified, their representations met
    /// part by part.
    fn meet_reprs(&mut self, a: &TypeArg<Poly>, b: &TypeArg<Poly>) -> bool {
        match (a, b) {
            (TypeArg::Open(x, _), TypeArg::Open(y, _)) if x == y => true,
            (TypeArg::Open(v, _), other) | (other, TypeArg::Open(v, _)) => {
                self.repr.insert(*v, other.repr());
                true
            }
            (TypeArg::Uniform(_), TypeArg::Uniform(_)) => true,
            (TypeArg::Specialized(x), TypeArg::Specialized(y)) => self.meet_held(x, y),
            (TypeArg::Uniform(_), TypeArg::Specialized(_))
            | (TypeArg::Specialized(_), TypeArg::Uniform(_)) => false,
        }
    }

    fn meet_held(&mut self, a: &HeldTy<Poly>, b: &HeldTy<Poly>) -> bool {
        match (a, b) {
            (HeldTy::Tuple(xs), HeldTy::Tuple(ys)) => {
                xs.len() == ys.len() && xs.iter().zip(ys).all(|(x, y)| self.meet_reprs(x, y))
            }
            (HeldTy::Option(x), HeldTy::Option(y))
            | (HeldTy::Array(x, _), HeldTy::Array(y, _))
            | (HeldTy::RustArray(x, _), HeldTy::RustArray(y, _)) => self.meet_reprs(x, y),
            (HeldTy::Result(xo, xe), HeldTy::Result(yo, ye)) => {
                self.meet_reprs(xo, yo) && self.meet_reprs(xe, ye)
            }
            (HeldTy::Leaf(_), HeldTy::Leaf(_)) => true,
            (HeldTy::Held(_), _) | (_, HeldTy::Held(_)) => a.is_full(false) && b.is_full(false),
            (
                HeldTy::Tuple(_)
                | HeldTy::Option(_)
                | HeldTy::Result(..)
                | HeldTy::Array(..)
                | HeldTy::RustArray(..)
                | HeldTy::Leaf(_),
                _,
            ) => false,
        }
    }

    fn unify_effect_arg(&mut self, a: &EffectArg<Poly>, b: &EffectArg<Poly>) -> bool {
        a.is_specialized() == b.is_specialized() && self.unify_effect(a.effect(), b.effect())
    }

    fn unify_effect(&mut self, a: &EffectTerm<Poly>, b: &EffectTerm<Poly>) -> bool {
        let resolve = |s: &Self, e: &EffectTerm<Poly>| match e {
            EffectTerm::Var(v) => s.effect.get(v).cloned().unwrap_or(e.clone()),
            known => known.clone(),
        };
        match (resolve(self, a), resolve(self, b)) {
            (EffectTerm::Known(x), EffectTerm::Known(y)) => x == y,
            (EffectTerm::Var(x), EffectTerm::Var(y)) if x == y => true,
            (EffectTerm::Var(v), other) | (other, EffectTerm::Var(v)) => {
                self.effect.insert(v, other);
                true
            }
        }
    }

    fn unify_len(&mut self, a: &LenTerm<Poly>, b: &LenTerm<Poly>) -> bool {
        let resolve = |s: &Self, l: &LenTerm<Poly>| match l {
            LenTerm::Var(v) => s.len.get(v).copied().unwrap_or(*l),
            known => *known,
        };
        match (resolve(self, a), resolve(self, b)) {
            (LenTerm::Known(x), LenTerm::Known(y)) => x == y,
            (LenTerm::Var(x), LenTerm::Var(y)) if x == y => true,
            (LenTerm::Var(v), other) | (other, LenTerm::Var(v)) => {
                self.len.insert(v, other);
                true
            }
        }
    }

    fn unify_identity(&mut self, a: &IdentityTerm<Poly>, b: &IdentityTerm<Poly>) -> bool {
        let resolve = |s: &Self, i: &IdentityTerm<Poly>| match i {
            IdentityTerm::Var(v) => s.identity.get(v).copied().unwrap_or(*i),
            known => *known,
        };
        match (resolve(self, a), resolve(self, b)) {
            (IdentityTerm::Known(x), IdentityTerm::Known(y)) => x == y,
            (IdentityTerm::Var(x), IdentityTerm::Var(y)) if x == y => true,
            (IdentityTerm::Var(v), other) | (other, IdentityTerm::Var(v)) => {
                self.identity.insert(v, other);
                true
            }
        }
    }

    fn unify(&mut self, a: &PolyTy, b: &PolyTy) -> bool {
        let (a, b) = (self.walk(a), self.walk(b));
        match (&a, &b) {
            (TyTerm::Var(x), TyTerm::Var(y)) if x == y => true,
            (TyTerm::Var(v), other) | (other, TyTerm::Var(v)) => {
                if self.occurs(*v, other) {
                    return false;
                }
                self.ty.insert(*v, other.clone());
                true
            }
            (TyTerm::Int(a), TyTerm::Int(b)) => a == b,
            (TyTerm::Float, TyTerm::Float)
            | (TyTerm::Char, TyTerm::Char)
            | (TyTerm::String, TyTerm::String)
            | (TyTerm::Str, TyTerm::Str)
            | (TyTerm::Bool, TyTerm::Bool)
            | (TyTerm::Unit, TyTerm::Unit)
            | (TyTerm::Never, TyTerm::Never)
            | (TyTerm::Order, TyTerm::Order) => true,
            (TyTerm::Error(_), _) | (_, TyTerm::Error(_)) => false,
            (TyTerm::Array(ea, la), TyTerm::Array(eb, lb)) => {
                self.unify_len(la, lb) && self.unify(ea, eb)
            }
            (TyTerm::Option(ia), TyTerm::Option(ib))
            | (TyTerm::Handle(ia), TyTerm::Handle(ib))
            | (TyTerm::Slice(ia), TyTerm::Slice(ib)) => self.unify(ia, ib),
            (TyTerm::Result(ta, ea), TyTerm::Result(tb, eb)) => {
                self.unify(ta, tb) && self.unify(ea, eb)
            }
            (TyTerm::Ref(ma, ia), TyTerm::Ref(mb, ib)) => ma == mb && self.unify_arg(ia, ib),
            (TyTerm::Tuple(ea), TyTerm::Tuple(eb)) => {
                ea.len() == eb.len() && ea.iter().zip(eb).all(|(x, y)| self.unify(x, y))
            }
            (TyTerm::Object(fa), TyTerm::Object(fb)) => {
                fa.declaration() == fb.declaration()
                    && fa.len() == fb.len()
                    && fa
                        .iter()
                        .all(|(k, v)| fb.get(k).is_some_and(|w| self.unify(v, w)))
            }
            (
                TyTerm::Fn {
                    params: pa,
                    ret: ra,
                    captures: ca,
                    effect: ea,
                },
                TyTerm::Fn {
                    params: pb,
                    ret: rb,
                    captures: cb,
                    effect: eb,
                },
            ) => {
                pa.len() == pb.len()
                    && ca.len() == cb.len()
                    && self.unify_effect(ea, eb)
                    && pa.iter().zip(pb).all(|(x, y)| self.unify(&x.ty, &y.ty))
                    && ca.iter().zip(cb).all(|(x, y)| self.unify(x, y))
                    && self.unify(ra, rb)
            }
            (
                TyTerm::UserDefined {
                    id: ia,
                    type_args: ta,
                    effect_args: ea,
                    identity_args: na,
                    region_params: ra,
                },
                TyTerm::UserDefined {
                    id: ib,
                    type_args: tb,
                    effect_args: eb,
                    identity_args: nb,
                    region_params: rb,
                },
            ) => {
                ia == ib
                    && ra == rb
                    && ta.len() == tb.len()
                    && ea.len() == eb.len()
                    && na.len() == nb.len()
                    && ea.iter().zip(eb).all(|(x, y)| self.unify_effect_arg(x, y))
                    && na.iter().zip(nb).all(|(x, y)| self.unify_identity(x, y))
                    && ta.iter().zip(tb).all(|(x, y)| self.unify_arg(x, y))
            }
            (
                TyTerm::Enum {
                    name: na,
                    variants: va,
                    ..
                },
                TyTerm::Enum {
                    name: nb,
                    variants: vb,
                    ..
                },
            ) => {
                na == nb
                    && va.len() == vb.len()
                    && va.iter().all(|(k, v)| match (v, vb.get(k)) {
                        (Some(v), Some(Some(w))) => self.unify(v, w),
                        (None, Some(None)) => true,
                        _ => false,
                    })
            }
            _ => false,
        }
    }

    fn apply(&self, t: &PolyTy) -> PolyTy {
        t.map::<Poly>(
            &mut |v| match self.ty.get(&v) {
                Some(bound) => self.apply(bound),
                None => TyTerm::Var(v),
            },
            &mut |v| {
                self.identity
                    .get(&v)
                    .copied()
                    .unwrap_or(IdentityTerm::Var(v))
            },
            &mut |v| self.effect.get(&v).cloned().unwrap_or(EffectTerm::Var(v)),
            &mut |v| self.len.get(&v).copied().unwrap_or(LenTerm::Var(v)),
            &mut |v| self.apply_repr(v),
        )
    }

    fn apply_repr(&self, v: u32) -> Repr<Poly> {
        match self.repr.get(&v) {
            None => Repr::Var(v),
            Some(Repr::Var(w)) => self.apply_repr(*w),
            Some(Repr::Uniform) => Repr::Uniform,
            Some(Repr::Specialized(held)) => Repr::Specialized(self.apply_held(held)),
        }
    }

    fn apply_arg(&self, arg: &TypeArg<Poly>) -> TypeArg<Poly> {
        match arg {
            TypeArg::Uniform(ty) => TypeArg::Uniform(self.apply(ty)),
            TypeArg::Open(v, ty) => self.apply_repr(*v).at(self.apply(ty)),
            TypeArg::Specialized(held) => TypeArg::Specialized(self.apply_held(held)),
        }
    }

    fn apply_held(&self, held: &HeldTy<Poly>) -> HeldTy<Poly> {
        held.map::<Poly>(
            &mut |v| match self.ty.get(&v) {
                Some(bound) => self.apply(bound),
                None => TyTerm::Var(v),
            },
            &mut |v| {
                self.identity
                    .get(&v)
                    .copied()
                    .unwrap_or(IdentityTerm::Var(v))
            },
            &mut |v| self.effect.get(&v).cloned().unwrap_or(EffectTerm::Var(v)),
            &mut |v| self.len.get(&v).copied().unwrap_or(LenTerm::Var(v)),
            &mut |v| self.apply_repr(v),
        )
    }
}

/// The anti-unifier of two patterns. An effect argument has no
/// representation variable, so two user-defined types that disagree on an
/// effect's `#` generalize to a type variable.
pub fn generalize_patterns(a: &PolyTy, b: &PolyTy) -> PolyTy {
    fn fresh(next: &mut u32) -> u32 {
        let var = *next;
        *next += 1;
        var
    }
    fn walk(a: &PolyTy, b: &PolyTy, next: &mut u32) -> PolyTy {
        match (a, b) {
            (TyTerm::Ref(ma, x), TyTerm::Ref(mb, y)) if ma == mb => {
                TyTerm::Ref(*ma, Box::new(walk_arg(x, y, next)))
            }
            (TyTerm::Option(x), TyTerm::Option(y)) => TyTerm::Option(Box::new(walk(x, y, next))),
            (TyTerm::Handle(x), TyTerm::Handle(y)) => TyTerm::Handle(Box::new(walk(x, y, next))),
            (TyTerm::Slice(x), TyTerm::Slice(y)) => TyTerm::Slice(Box::new(walk(x, y, next))),
            (TyTerm::Result(xa, xb), TyTerm::Result(ya, yb)) => {
                TyTerm::Result(Box::new(walk(xa, ya, next)), Box::new(walk(xb, yb, next)))
            }
            (TyTerm::Array(x, la), TyTerm::Array(y, lb)) => {
                let len = if la == lb {
                    la.clone()
                } else {
                    LenTerm::Var(fresh(next))
                };
                TyTerm::Array(Box::new(walk(x, y, next)), len)
            }
            (TyTerm::Tuple(xs), TyTerm::Tuple(ys)) if xs.len() == ys.len() => {
                TyTerm::Tuple(xs.iter().zip(ys).map(|(x, y)| walk(x, y, next)).collect())
            }
            (
                TyTerm::UserDefined {
                    id: ia,
                    type_args: ta,
                    effect_args: ea,
                    identity_args: ida,
                    region_params: ra,
                },
                TyTerm::UserDefined {
                    id: ib,
                    type_args: tb,
                    effect_args: eb,
                    identity_args: idb,
                    region_params: rb,
                },
            ) if ia == ib
                && ra == rb
                && ta.len() == tb.len()
                && ea.len() == eb.len()
                && ida.len() == idb.len()
                && ea
                    .iter()
                    .zip(eb)
                    .all(|(x, y)| x.is_specialized() == y.is_specialized()) =>
            {
                let type_args = ta
                    .iter()
                    .zip(tb)
                    .map(|(x, y)| walk_arg(x, y, next))
                    .collect();
                let effect_args = ea
                    .iter()
                    .zip(eb)
                    .map(|(x, y)| {
                        x.with_effect(if x.effect() == y.effect() {
                            x.effect().clone()
                        } else {
                            EffectTerm::Var(fresh(next))
                        })
                    })
                    .collect();
                let identity_args = ida
                    .iter()
                    .zip(idb)
                    .map(|(x, y)| {
                        if x == y {
                            *x
                        } else {
                            IdentityTerm::Var(fresh(next))
                        }
                    })
                    .collect();
                TyTerm::UserDefined {
                    id: *ia,
                    type_args,
                    effect_args,
                    identity_args,
                    region_params: *ra,
                }
            }
            (x, y) if x == y => x.clone(),
            _ => TyTerm::Var(fresh(next)),
        }
    }
    fn walk_arg(x: &TypeArg<Poly>, y: &TypeArg<Poly>, next: &mut u32) -> TypeArg<Poly> {
        match (x, y) {
            (TypeArg::Uniform(a), TypeArg::Uniform(b)) => TypeArg::Uniform(walk(a, b, next)),
            (TypeArg::Open(r, a), TypeArg::Open(s, b)) if r == s => {
                TypeArg::Open(*r, walk(a, b, next))
            }
            (TypeArg::Specialized(a), TypeArg::Specialized(b)) => match walk_held(a, b, next) {
                Some(held) => TypeArg::Specialized(held),
                None => TypeArg::Open(fresh(next), walk(&x.ty(), &y.ty(), next)),
            },
            _ => TypeArg::Open(fresh(next), walk(&x.ty(), &y.ty(), next)),
        }
    }
    fn walk_held(a: &HeldTy<Poly>, b: &HeldTy<Poly>, next: &mut u32) -> Option<HeldTy<Poly>> {
        match (a, b) {
            (HeldTy::Tuple(xs), HeldTy::Tuple(ys)) if xs.len() == ys.len() => Some(HeldTy::Tuple(
                xs.iter()
                    .zip(ys)
                    .map(|(x, y)| walk_arg(x, y, next))
                    .collect(),
            )),
            (HeldTy::Option(x), HeldTy::Option(y)) => {
                Some(HeldTy::Option(Box::new(walk_arg(x, y, next))))
            }
            (HeldTy::Result(xo, xe), HeldTy::Result(yo, ye)) => {
                let ok = walk_arg(xo, yo, next);
                Some(HeldTy::Result(
                    Box::new(ok),
                    Box::new(walk_arg(xe, ye, next)),
                ))
            }
            (HeldTy::Array(x, la), HeldTy::Array(y, lb)) => {
                let elem = walk_arg(x, y, next);
                let len = if la == lb {
                    *la
                } else {
                    LenTerm::Var(fresh(next))
                };
                Some(HeldTy::Array(Box::new(elem), len))
            }
            (HeldTy::RustArray(x, la), HeldTy::RustArray(y, lb)) if la == lb => {
                Some(HeldTy::RustArray(Box::new(walk_arg(x, y, next)), *la))
            }
            _ if a.is_full(false) && b.is_full(false) => {
                Some(HeldTy::of(walk(&a.ty(), &b.ty(), next)))
            }
            _ => None,
        }
    }
    walk(a, b, &mut 0)
}

/// The most general pattern that has the shape of both, with the two
/// patterns' variables kept apart; `None` when no type has.
pub fn unify_patterns(a: &PolyTy, b: &PolyTy) -> Option<PolyTy> {
    let b = shift_vars(b, var_span(a));
    let mut subst = PatternSubst::default();
    subst.unify(a, &b).then(|| subst.apply(a))
}

/// Immutable registry of all UserDefined type declarations and ExternCast rules.
/// Built once at setup, then frozen via `Freeze<TypeRegistry>` and shared everywhere.
///
/// Contains:
/// - `decls`: UserDefined type declarations (source of truth for params/constraints).
/// - `cast_rules`: ExternCast coercion rules (UserDefined -> other type).
#[derive(Debug, Clone, Default)]
pub struct TypeRegistry {
    decls: FxHashMap<QualifiedRef, UserDefinedDecl>,
    /// ExternCast coercion rules, indexed by source UserDefined QualifiedRef.
    /// `from_rules`: keyed by the `from` type's QualifiedRef.
    /// `to_rules`: keyed by the `to` type's QualifiedRef (when target is UserDefined).
    // pub(crate) for test access.
    pub(crate) from_rules: FxHashMap<QualifiedRef, Vec<CastRule>>,
    pub(crate) to_rules: FxHashMap<QualifiedRef, Vec<CastRule>>,
    machine_views: FxHashMap<QualifiedRef, Viewed>,
    sliced: FxHashSet<SlicedStorage>,
}

/// A coercion rule: `from` can be implicitly converted to `to`.
/// Both `from` and `to` share positional Var placeholders (Poly phase),
/// so instantiating them together links corresponding parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct CastRule {
    /// Source type pattern (must be UserDefined). May contain positional Var placeholders.
    pub from: PolyTy,
    /// Target type pattern (Var placeholders shared with `from`).
    pub to: PolyTy,
    /// The pure ExternFn that performs the conversion.
    pub fn_ref: QualifiedRef,
}

/// Head constructor of a type - used for duplicate cast rule detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TyHead {
    Int(IntTy),
    Float,
    Char,
    String,
    Bool,
    Unit,
    Never,
    Order,
    Array,
    Object,
    Tuple,
    Fn,
    Option,
    Result,
    Enum,
    Handle,
    Ref,
    Slice,
    Str,
    /// A user-defined type with the representation of each argument: two
    /// cast rules between `Vec<#T>` and `Vec<T>` have distinct heads.
    UserDefined(QualifiedRef, Vec<ArgHead>),
    Error,
}

/// An argument's tree of representations without its types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ArgHead {
    Uniform,
    Open,
    Tuple(Vec<ArgHead>),
    Option(Box<ArgHead>),
    Result(Box<ArgHead>, Box<ArgHead>),
    Array(Box<ArgHead>),
    RustArray(Box<ArgHead>),
    Leaf,
    Held,
}

fn arg_head(arg: &TypeArg<Poly>) -> ArgHead {
    match arg {
        TypeArg::Uniform(_) => ArgHead::Uniform,
        TypeArg::Open(..) => ArgHead::Open,
        TypeArg::Specialized(held) => match held {
            HeldTy::Tuple(parts) => ArgHead::Tuple(parts.iter().map(arg_head).collect()),
            HeldTy::Option(part) => ArgHead::Option(Box::new(arg_head(part))),
            HeldTy::Result(ok, err) => {
                ArgHead::Result(Box::new(arg_head(ok)), Box::new(arg_head(err)))
            }
            HeldTy::Array(part, _) => ArgHead::Array(Box::new(arg_head(part))),
            HeldTy::RustArray(part, _) => ArgHead::RustArray(Box::new(arg_head(part))),
            HeldTy::Leaf(_) => ArgHead::Leaf,
            HeldTy::Held(_) => ArgHead::Held,
        },
    }
}

fn ty_head(ty: &PolyTy) -> TyHead {
    match ty {
        TyTerm::Int(k) => TyHead::Int(*k),
        TyTerm::Float => TyHead::Float,
        TyTerm::Char => TyHead::Char,
        TyTerm::String => TyHead::String,
        TyTerm::Bool => TyHead::Bool,
        TyTerm::Unit => TyHead::Unit,
        TyTerm::Never => TyHead::Never,
        TyTerm::Order => TyHead::Order,
        TyTerm::Array(..) => TyHead::Array,
        TyTerm::Object(_) => TyHead::Object,
        TyTerm::Tuple(_) => TyHead::Tuple,
        TyTerm::Fn { .. } => TyHead::Fn,
        TyTerm::Option(_) => TyHead::Option,
        TyTerm::Result(..) => TyHead::Result,
        TyTerm::Enum { .. } => TyHead::Enum,
        TyTerm::Handle(..) => TyHead::Handle,
        TyTerm::Ref(..) => TyHead::Ref,
        TyTerm::Slice(_) => TyHead::Slice,
        TyTerm::Str => TyHead::Str,
        TyTerm::UserDefined { id, type_args, .. } => {
            TyHead::UserDefined(*id, type_args.iter().map(arg_head).collect())
        }
        TyTerm::Error(_) => TyHead::Error,
        TyTerm::Var(_) => TyHead::Error,
    }
}

impl TypeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    // -- Type declarations -------------------------------------------

    /// Register a declaration. A second declaration under a name already
    /// registered is refused: one name is one type.
    pub fn register(&mut self, decl: UserDefinedDecl) -> Result<(), DuplicateType> {
        let qref = decl.qref;
        assert!(
            decl.identity_params <= 1,
            "{qref:?}: a type has at most one identity parameter; a value is one source"
        );
        assert_eq!(
            decl.specializable.len(),
            decl.type_params.len(),
            "{qref:?}: one specializable flag per type parameter"
        );
        match self.decls.entry(qref) {
            std::collections::hash_map::Entry::Occupied(_) => Err(DuplicateType(qref)),
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(decl);
                Ok(())
            }
        }
    }

    /// Look up a declaration by qref. Panics if not found - missing decl is a bug.
    pub fn get(&self, qref: QualifiedRef) -> &UserDefinedDecl {
        self.decls
            .get(&qref)
            .unwrap_or_else(|| panic!("unknown UserDefined type: {qref:?}"))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&QualifiedRef, &UserDefinedDecl)> {
        self.decls.iter()
    }

    /// Whether the `index`-th type argument of `qref` is a specializing
    /// position (hash-types.md, R1). Panics on an unknown type or index:
    /// both are bugs of the caller.
    pub fn specializes(&self, qref: QualifiedRef, index: usize) -> bool {
        let decl = self.get(qref);
        *decl.specializable.get(index).unwrap_or_else(|| {
            panic!(
                "{qref:?}: type argument {index} of {} declared",
                decl.type_params.len()
            )
        })
    }

    // -- Cast rules --------------------------------------------------

    /// Register a cast rule. Indexes by `from`'s QualifiedRef (if UserDefined)
    /// and by `to`'s QualifiedRef (if UserDefined). At least one side must be UserDefined.
    pub fn register_cast(&mut self, rule: CastRule) {
        let from_qref = match &rule.from {
            TyTerm::UserDefined { id, .. } => Some(*id),
            _ => None,
        };
        let to_qref = match &rule.to {
            TyTerm::UserDefined { id, .. } => Some(*id),
            _ => None,
        };
        assert!(
            from_qref.is_some() || to_qref.is_some(),
            "CastRule: at least one side must be UserDefined"
        );

        // Duplicate check (same from head + same to head).
        let from_head = ty_head(&rule.from);
        let to_head = ty_head(&rule.to);
        if let Some(fq) = from_qref {
            for existing in self.from_rules.get(&fq).into_iter().flatten() {
                assert!(
                    !(ty_head(&existing.from) == from_head && ty_head(&existing.to) == to_head),
                    "duplicate CastRule: same from and to head constructor"
                );
            }
        }

        // Index by from (if UserDefined).
        if let Some(fq) = from_qref {
            self.from_rules.entry(fq).or_default().push(rule.clone());
        }
        // Index by to (if UserDefined).
        if let Some(tq) = to_qref {
            self.to_rules.entry(tq).or_default().push(rule);
        }
    }

    /// Get all cast rules where `from` matches the given QualifiedRef.
    pub fn rules_from(&self, qref: QualifiedRef) -> &[CastRule] {
        self.from_rules.get(&qref).map_or(&[], |v| v.as_slice())
    }

    /// Get all cast rules where `to` is a UserDefined matching the given QualifiedRef.
    pub fn rules_to(&self, qref: QualifiedRef) -> &[CastRule] {
        self.to_rules.get(&qref).map_or(&[], |v| v.as_slice())
    }

    // -- Machine coercions -------------------------------------------

    pub fn register_machine_view(
        &mut self,
        qref: QualifiedRef,
        viewed: Viewed,
        referent_taken: &PolyTy,
    ) {
        let prev = self.machine_views.insert(qref, viewed);
        assert!(prev.is_none(), "duplicate machine coercion: {qref:?}");
        if let (View::Slice, Some(head)) = (viewed.view, SliceableHead::of(referent_taken)) {
            self.sliced.insert(SlicedStorage {
                head,
                mutability: viewed.mutability,
            });
        }
    }

    /// Whether a storage of this head lends a slice at this mutability.
    pub fn lends_a_slice(&self, head: SliceableHead, mutability: Mutability) -> bool {
        self.sliced.contains(&SlicedStorage { head, mutability })
    }

    /// There is deliberately no way to ask this question of a name or of a
    /// type. A declaration whose shape is a view is still an ordinary
    /// function a script may call, and the machine claims one only where
    /// the registry declared it a coercion, so the answer lives with the
    /// registration and nowhere else.
    pub fn machine_view(&self, qref: QualifiedRef) -> Option<Viewed> {
        self.machine_views.get(&qref).copied()
    }
}

/// A named, typed function parameter.
pub type Param = ParamTerm<Concrete>;

/// Token for `Ty::Error` construction.
///
/// `Ty::Error` is a **poison type** - it suppresses cascading errors by unifying
/// with anything. Permitted uses:
///
/// - **Type checker / compiler**: After reporting a type error, return `Ty::error()`
///   so compilation continues and collects all errors (not just the first one).
/// - **Deserialization recovery**: When loading a persisted type that can't be parsed.
///
/// **Forbidden uses**:
///
/// - As a "don't know" placeholder (use the actual type instead).
/// - As a default/fallback when you're too lazy to propagate the real type.
/// - In runtime code paths - Error must never appear in a running program's types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ErrorToken(());

impl ErrorToken {
    pub(crate) fn new() -> Self {
        Self(())
    }
}

/// Whether a call may be issued again (RFC-0014). The
/// derived order is the chain `Pure < Idempotent < Opaque`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Reissue {
    Pure,
    Idempotent,
    Opaque,
}

/// What a call costs the scheduler (RFC-0046).
///
/// RFC-0046 rejected deriving synchrony from purity instead of carrying
/// this field. A `heavy` pure extern - a regex match, a hash of a large
/// buffer - is offloaded to a blocking pool and awaited, so `Pure` would
/// have been an unsound claim about synchrony that nothing ever checked.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum Task {
    #[default]
    Sync,
    Async,
    Heavy,
}

impl Task {
    pub fn join(self, other: Task) -> Task {
        self.max(other)
    }

    pub fn meet(self, other: Task) -> Task {
        self.min(other)
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// The contexts a call may touch (RFC-0025 rule 4), as a set of context names.
pub type Contexts = BTreeSet<QualifiedRef>;

/// The effect of a call (RFC-0013, RFC-0025): the reissue chain, whether
/// two calls commute, and the contexts the call may read and write. A
/// Pure call that writes nothing commutes by definition; a call that
/// writes a context never commutes, since a second call may read it.
/// Every constructor keeps both invariants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Effect {
    pub reissue: Reissue,
    pub task: Task,
    pub commutes: bool,
    pub reads: Contexts,
    pub writes: Contexts,
}

impl Effect {
    pub const PURE: Effect = Effect {
        reissue: Reissue::Pure,
        task: Task::Sync,
        commutes: true,
        reads: BTreeSet::new(),
        writes: BTreeSet::new(),
    };
    pub const IDEMPOTENT: Effect = Effect {
        reissue: Reissue::Idempotent,
        task: Task::Sync,
        commutes: false,
        reads: BTreeSet::new(),
        writes: BTreeSet::new(),
    };
    pub const OPAQUE: Effect = Effect {
        reissue: Reissue::Opaque,
        task: Task::Sync,
        commutes: false,
        reads: BTreeSet::new(),
        writes: BTreeSet::new(),
    };
    pub const TOP: Effect = Effect {
        reissue: Reissue::Opaque,
        task: Task::Heavy,
        commutes: false,
        reads: BTreeSet::new(),
        writes: BTreeSet::new(),
    };

    pub fn new(reissue: Reissue, commutes: bool) -> Effect {
        Effect::with_contexts(reissue, commutes, Contexts::new(), Contexts::new())
    }

    pub fn with_contexts(
        reissue: Reissue,
        commutes: bool,
        reads: Contexts,
        writes: Contexts,
    ) -> Effect {
        Effect {
            reissue,
            task: Task::Sync,
            commutes: (commutes || reissue == Reissue::Pure) && writes.is_empty(),
            reads,
            writes,
        }
    }

    pub fn at_task(&self, task: Task) -> Effect {
        Effect {
            task,
            ..self.clone()
        }
    }

    /// The effect of reading one context: Pure, and a read of it.
    pub fn read(context: QualifiedRef) -> Effect {
        Effect::with_contexts(
            Reissue::Pure,
            true,
            Contexts::from([context]),
            Contexts::new(),
        )
    }

    /// The effect of writing one context: Pure on the chain, and a write
    /// of it, which does not commute.
    pub fn write(context: QualifiedRef) -> Effect {
        Effect::with_contexts(
            Reissue::Pure,
            true,
            Contexts::new(),
            Contexts::from([context]),
        )
    }

    /// The same level, declared to commute.
    pub fn commutative(&self) -> Effect {
        Effect::with_contexts(self.reissue, true, self.reads.clone(), self.writes.clone())
            .at_task(self.task)
    }

    pub fn is_pure(&self) -> bool {
        self.reissue == Reissue::Pure
    }

    pub fn runs_apart(&self) -> bool {
        self.task > Task::Sync || !self.is_pure()
    }

    /// No level above Pure and no context touched: the effect a type
    /// display leaves out.
    pub fn is_empty(&self) -> bool {
        self.is_pure() && self.task == Task::Sync && self.reads.is_empty() && self.writes.is_empty()
    }

    /// Whether the call may touch `context` at all.
    pub fn touches(&self, context: QualifiedRef) -> bool {
        self.reads.contains(&context) || self.writes.contains(&context)
    }

    /// The product order on the chain and commutativity: `self` is no
    /// more effectful than `other` when it is no higher on the chain and
    /// commutes whenever `other` does. The context sets are not part of
    /// this order: a bound on an effect bounds its level, and the sets
    /// only accumulate through `join`. `touches_at_most` orders the sets.
    pub fn at_most(&self, other: &Effect) -> bool {
        self.reissue <= other.reissue
            && self.task <= other.task
            && (self.commutes || !other.commutes)
    }

    /// Whether `self` touches no context `other` does not.
    pub fn touches_at_most(&self, other: &Effect) -> bool {
        self.reads.is_subset(&other.reads) && self.writes.is_subset(&other.writes)
    }

    /// The least effect above both: the higher level, commutative only if
    /// both are, touching what either does.
    pub fn join(&self, other: &Effect) -> Effect {
        Effect::with_contexts(
            self.reissue.max(other.reissue),
            self.commutes && other.commutes,
            self.reads.union(&other.reads).copied().collect(),
            self.writes.union(&other.writes).copied().collect(),
        )
        .at_task(self.task.join(other.task))
    }

    /// The greatest effect below both.
    pub fn meet(&self, other: &Effect) -> Effect {
        Effect::with_contexts(
            self.reissue.min(other.reissue),
            self.commutes || other.commutes,
            self.reads.intersection(&other.reads).copied().collect(),
            self.writes.intersection(&other.writes).copied().collect(),
        )
        .at_task(self.task.meet(other.task))
    }
}

impl PartialOrd for Effect {
    fn partial_cmp(&self, other: &Effect) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (self.at_most(other), other.at_most(self)) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.reissue)?;
        if self.task != Task::Sync {
            write!(f, "/{}", self.task)?;
        }
        if self.commutes && !self.is_pure() {
            write!(f, "+commutative")?;
        }
        Ok(())
    }
}

/// A required effect level that exceeds the level allowed at that point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectConflict {
    pub required: Effect,
    pub allowed: Effect,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectTerm<V: Phase> {
    Known(Effect),
    Var(V::EffectVar),
}

impl<V: Phase> From<Effect> for EffectTerm<V> {
    fn from(effect: Effect) -> Self {
        EffectTerm::Known(effect)
    }
}

impl<V: Phase> EffectTerm<V> {
    pub fn map<W: Phase>(
        &self,
        on_effect: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>,
    ) -> EffectTerm<W> {
        match self {
            EffectTerm::Known(e) => EffectTerm::Known(e.clone()),
            EffectTerm::Var(v) => on_effect(*v),
        }
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_effect: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
    ) -> Result<EffectTerm<W>, E> {
        match self {
            EffectTerm::Known(e) => Ok(EffectTerm::Known(e.clone())),
            EffectTerm::Var(v) => on_effect(*v),
        }
    }
}

impl EffectTerm<Concrete> {
    pub fn get(&self) -> &Effect {
        match self {
            EffectTerm::Known(e) => e,
            EffectTerm::Var(v) => match *v {},
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LenTerm<V: Phase> {
    Known(usize),
    Var(V::LenVar),
}

impl<V: Phase> LenTerm<V> {
    pub fn map<W: Phase>(&self, on_len: &mut impl FnMut(V::LenVar) -> LenTerm<W>) -> LenTerm<W> {
        match self {
            LenTerm::Known(n) => LenTerm::Known(*n),
            LenTerm::Var(v) => on_len(*v),
        }
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_len: &mut impl FnMut(V::LenVar) -> Result<LenTerm<W>, E>,
    ) -> Result<LenTerm<W>, E> {
        match self {
            LenTerm::Known(n) => Ok(LenTerm::Known(*n)),
            LenTerm::Var(v) => on_len(*v),
        }
    }
}

impl LenTerm<Concrete> {
    pub fn get(&self) -> usize {
        match self {
            LenTerm::Known(n) => *n,
            LenTerm::Var(v) => match *v {},
        }
    }
}

// -- Identity system --------------------------------------------------

acvus_utils::declare_local_id!(pub IdentityId);

/// An identity argument of a user-defined type: the source a value came
/// from. Two values unify only when their identities are the same, so a
/// value from one source never mixes with a value from another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityTerm<V: Phase> {
    Known(IdentityId),
    Var(V::IdentityVar),
}

impl<V: Phase> IdentityTerm<V> {
    pub fn map<W: Phase>(
        &self,
        on_identity: &mut impl FnMut(V::IdentityVar) -> IdentityTerm<W>,
    ) -> IdentityTerm<W> {
        match self {
            IdentityTerm::Known(id) => IdentityTerm::Known(*id),
            IdentityTerm::Var(v) => on_identity(*v),
        }
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_identity: &mut impl FnMut(V::IdentityVar) -> Result<IdentityTerm<W>, E>,
    ) -> Result<IdentityTerm<W>, E> {
        match self {
            IdentityTerm::Known(id) => Ok(IdentityTerm::Known(*id)),
            IdentityTerm::Var(v) => on_identity(*v),
        }
    }
}

impl IdentityTerm<Concrete> {
    pub fn get(&self) -> IdentityId {
        match self {
            IdentityTerm::Known(id) => *id,
            IdentityTerm::Var(v) => match *v {},
        }
    }
}

impl std::fmt::Display for IdentityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Identity({self:?})")
    }
}

/// Concrete type - always fully resolved. `Var(Infallible)` is uninhabitable.
pub type Ty = TyTerm<Concrete>;

impl TyTerm<Concrete> {
    /// Create an `Error` (poison) type. See [`ErrorToken`] for permitted uses.
    pub fn error() -> Self {
        Ty::Error(ErrorToken::new())
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Ty::Error(_))
    }

    pub fn effect(&self) -> Option<Effect> {
        match self {
            Ty::Fn { effect, .. } => Some(effect.get().clone()),
            _ => None,
        }
    }

    /// Extract the element type from a collection type.
    pub fn elem_of(&self) -> Option<&Ty> {
        match self {
            Ty::Array(elem, _) => Some(elem),
            _ => None,
        }
    }

    /// Whether a value of this type is data: something a host can keep
    /// from one run to the next (RFC-0014). A function, a spawn handle, an
    /// order, or a reference is not data, and neither is a type that holds
    /// one. An extension type is data; how it is written down is the
    /// host's declaration, not the checker's.
    pub fn is_data(&self) -> bool {
        match self {
            Ty::Int(_) | Ty::Float | Ty::Char | Ty::String | Ty::Bool | Ty::Unit | Ty::Never => {
                true
            }
            Ty::Array(inner, _) | Ty::Option(inner) => inner.is_data(),
            Ty::Result(ok, err) => ok.is_data() && err.is_data(),
            Ty::Tuple(elems) => elems.iter().all(Ty::is_data),
            Ty::Object(fields) => fields.values().all(Ty::is_data),
            Ty::Enum { variants, .. } => variants
                .values()
                .all(|p| p.as_ref().is_none_or(|ty| ty.is_data())),
            Ty::UserDefined { type_args, .. } => type_args.iter().all(|a| a.ty().is_data()),
            Ty::Fn { .. }
            | Ty::Handle(..)
            | Ty::Order
            | Ty::Ref(..)
            | Ty::Slice(_)
            | Ty::Str
            | Ty::Error(_) => false,
            Ty::Var(v) => match *v {},
        }
    }
}

impl<V> TyTerm<V>
where
    V: Phase,
{
    pub fn display<'a>(&'a self, interner: &'a Interner) -> TyDisplay<'a, V> {
        TyDisplay {
            ty: self,
            interner,
            never: NeverAs::Diverges,
        }
    }

    /// The type as a refusal shows it. `Solver::written_ty` closes a
    /// variable the solve never bound to `Never`, and `solver.rs` is the
    /// only place `acvus-mir` builds one, so a `Never` here is a type the
    /// solve never settled and `_` is what a reader can act on.
    pub fn shown<'a>(&'a self, interner: &'a Interner) -> TyDisplay<'a, V> {
        TyDisplay {
            ty: self,
            interner,
            never: NeverAs::Unsettled,
        }
    }
}

/// What a `Never` is where it is printed: the type of an expression that
/// does not return, or a variable the solve never bound.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NeverAs {
    Diverges,
    Unsettled,
}

impl NeverAs {
    fn written(self) -> &'static str {
        match self {
            NeverAs::Diverges => "!",
            NeverAs::Unsettled => "_",
        }
    }
}

pub struct TyDisplay<'a, V>
where
    V: Phase,
{
    ty: &'a TyTerm<V>,
    interner: &'a Interner,
    never: NeverAs,
}

impl<'a, V> TyDisplay<'a, V>
where
    V: Phase,
{
    /// A type inside this one, printed with the same reading of `Never`.
    fn nested(&self, ty: &'a TyTerm<V>) -> Self {
        TyDisplay {
            ty,
            interner: self.interner,
            never: self.never,
        }
    }

    fn nested_arg(&self, arg: &'a TypeArg<V>) -> ArgDisplay<'a, V> {
        ArgDisplay {
            arg,
            interner: self.interner,
            never: self.never,
        }
    }
}

impl<V> TypeArg<V>
where
    V: Phase,
{
    pub fn display<'a>(&'a self, interner: &'a Interner) -> ArgDisplay<'a, V> {
        ArgDisplay {
            arg: self,
            interner,
            never: NeverAs::Diverges,
        }
    }
}

/// An argument as `τ`, as `#?τ` while its representation is open, or with
/// `#` on each specialized part, as `#(#i64, T)`.
pub struct ArgDisplay<'a, V>
where
    V: Phase,
{
    arg: &'a TypeArg<V>,
    interner: &'a Interner,
    never: NeverAs,
}

impl<'a, V> ArgDisplay<'a, V>
where
    V: Phase,
{
    fn ty(&self, ty: &'a TyTerm<V>) -> TyDisplay<'a, V> {
        TyDisplay {
            ty,
            interner: self.interner,
            never: self.never,
        }
    }

    fn part(&self, arg: &'a TypeArg<V>) -> Self {
        ArgDisplay {
            arg,
            interner: self.interner,
            never: self.never,
        }
    }

    fn held(&self, held: &'a HeldTy<V>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match held {
            HeldTy::Tuple(parts) => {
                write!(f, "(")?;
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.part(part))?;
                }
                write!(f, ")")
            }
            HeldTy::Option(part) => write!(f, "Option<{}>", self.part(part)),
            HeldTy::Result(ok, err) => {
                write!(f, "Result<{}, {}>", self.part(ok), self.part(err))
            }
            HeldTy::Array(part, len) => {
                write!(f, "Array<{}, ", self.part(part))?;
                match len {
                    LenTerm::Known(n) => write!(f, "{n}>"),
                    LenTerm::Var(v) => {
                        V::spell_len_var(v, f)?;
                        write!(f, ">")
                    }
                }
            }
            HeldTy::RustArray(part, len) => write!(f, "[{}; {len}]", self.part(part)),
            HeldTy::Leaf(leaf) => write!(f, "{}", self.ty(leaf.ty())),
            HeldTy::Held(v) => V::spell_ty_var(v, f),
        }
    }
}

impl<'a, V> fmt::Display for ArgDisplay<'a, V>
where
    V: Phase,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.arg {
            TypeArg::Uniform(ty) => write!(f, "{}", self.ty(ty)),
            TypeArg::Open(_, ty) => {
                V::spell_open_repr(f)?;
                write!(f, "{}", self.ty(ty))
            }
            TypeArg::Specialized(held) => {
                write!(f, "#")?;
                self.held(held, f)
            }
        }
    }
}

impl<'a, V> fmt::Display for TyDisplay<'a, V>
where
    V: Phase,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty {
            TyTerm::Int(k) => write!(f, "{}", k.name()),
            TyTerm::Order => write!(f, "Order"),
            TyTerm::Slice(elem) => write!(f, "[{}]", self.nested(elem)),
            TyTerm::Str => write!(f, "str"),
            TyTerm::Float => write!(f, "Float"),
            TyTerm::Char => write!(f, "char"),
            TyTerm::String => write!(f, "String"),
            TyTerm::Bool => write!(f, "Bool"),
            TyTerm::Unit => write!(f, "Unit"),
            TyTerm::Never => f.write_str(self.never.written()),
            TyTerm::Object(object) => {
                let mut sorted: Vec<_> = object.iter().collect();
                sorted.sort_by_key(|(k, _)| self.interner.resolve(**k).to_string());
                if let Some(name) = object.declaration() {
                    write!(f, "{}", self.interner.resolve(name.name))?;
                }
                write!(f, "{{")?;
                for (i, (k, v)) in sorted.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", self.interner.resolve(**k), self.nested(v))?;
                }
                write!(f, "}}")
            }
            TyTerm::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.nested(e))?;
                }
                write!(f, ")")
            }
            TyTerm::Fn {
                params,
                ret,
                captures: _,
                effect,
            } => {
                write!(f, "Fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.nested(&p.ty))?;
                }
                write!(f, ") -> {}", self.nested(ret))?;
                let effect = match effect {
                    EffectTerm::Known(effect) => effect,
                    EffectTerm::Var(v) => {
                        write!(f, " with ")?;
                        return V::spell_effect_var(v, f);
                    }
                };
                if effect.is_empty() {
                    return Ok(());
                }
                write!(f, " with")?;
                if !effect.is_pure() || effect.task != Task::Sync {
                    write!(f, " {effect}")?;
                }
                for (label, set) in [("reads", &effect.reads), ("writes", &effect.writes)] {
                    if set.is_empty() {
                        continue;
                    }
                    write!(f, " {label} {{")?;
                    for (i, ctx) in set.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "@{}", self.interner.resolve(ctx.name))?;
                    }
                    write!(f, "}}")?;
                }
                Ok(())
            }
            TyTerm::Array(inner, len) => {
                write!(f, "Array<{}, ", self.nested(inner))?;
                match len {
                    LenTerm::Known(n) => write!(f, "{n}>"),
                    LenTerm::Var(v) => {
                        V::spell_len_var(v, f)?;
                        write!(f, ">")
                    }
                }
            }
            TyTerm::Handle(inner) => {
                write!(f, "Handle<{}>", self.nested(inner))
            }
            TyTerm::Option(inner) => write!(f, "Option<{}>", self.nested(inner)),
            TyTerm::Result(ok, err) => {
                write!(f, "Result<{}, {}>", self.nested(ok), self.nested(err))
            }
            // Identity arguments are not printed. An identity is a source,
            // and its number names that source to nobody: a reader told
            // `Iterator<i64, Pure, #1>` learns only that there is a `#1`.
            // Where the difference between two sources is what the compiler
            // must say, it says it by pointing at the two expressions that
            // minted them, which `MirErrorKind::IdentityMismatch` does.
            TyTerm::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args: _,
                region_params: _,
            } => {
                let name = self.interner.resolve(id.name);
                write!(f, "{name}")?;
                if !type_args.is_empty() || !effect_args.is_empty() {
                    write!(f, "<")?;
                    let mut first = true;
                    for arg in type_args {
                        if !first {
                            write!(f, ", ")?;
                        }
                        first = false;
                        write!(f, "{}", self.nested_arg(arg))?;
                    }
                    for arg in effect_args {
                        if !first {
                            write!(f, ", ")?;
                        }
                        first = false;
                        if arg.is_specialized() {
                            write!(f, "#")?;
                        }
                        match arg.effect() {
                            EffectTerm::Known(e) => write!(f, "{e}")?,
                            EffectTerm::Var(v) => V::spell_effect_var(v, f)?,
                        }
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            TyTerm::Enum { name, variants, .. } => {
                write!(f, "{}{{", self.interner.resolve(name.name))?;
                let mut sorted: Vec<_> = variants.iter().collect();
                sorted.sort_by_key(|(tag, _)| self.interner.resolve(**tag).to_string());
                for (i, (tag, payload)) in sorted.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", self.interner.resolve(**tag))?;
                    if let Some(payload) = payload {
                        write!(f, "({})", self.nested(payload))?;
                    }
                }
                write!(f, "}}")
            }
            TyTerm::Ref(m, inner) => write!(f, "{}{}", m.prefix(), self.nested_arg(inner)),
            TyTerm::Error(_) => write!(f, "<error>"),
            TyTerm::Var(v) => V::spell_ty_var(v, f),
        }
    }
}

impl<'a, V> fmt::Debug for TyDisplay<'a, V>
where
    V: Phase,
    TyDisplay<'a, V>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// -- TypeEnv ----------------------------------------------------------

/// Unified type environment for the type checker.
///
/// Replaces `ContextTypeRegistry` + internal `BuiltinRegistry`.
/// The type checker receives this as its sole external input -
/// it does not know whether a function is a builtin, extern, or user-defined.
/// All keys are QualifiedRef - the canonical identifier.
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Context variable types - may contain inference variables (Solver-scoped).
    pub contexts: FxHashMap<QualifiedRef, InferTy>,
    /// Function type schemes - polymorphic, instantiated per call site.
    /// Every name a script can write resolves here and nowhere else.
    pub functions: FxHashMap<QualifiedRef, Scheme>,
    /// These are kept out of `functions` rather than marked inside it so
    /// that `resolve_fn` cannot reach them at all: a script's bare name
    /// resolves over `functions` alone, and the only way one of these
    /// reaches a script is `signature_set` offering it back deliberately.
    pub machine: FxHashMap<QualifiedRef, MachineCoercion>,
}

#[derive(Debug, Clone)]
pub struct MachineCoercion {
    pub viewed: Viewed,
    pub scheme: Scheme,
}

/// The run behind a reference: a container's elements (RFC-0047 rule 6),
/// or a `String`'s bytes (RFC-0062 rule 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Slice,
    Str,
}

impl View {
    pub fn of<V>(ty: &TyTerm<V>) -> Option<View>
    where
        V: Phase,
    {
        match ty {
            TyTerm::Slice(_) => Some(View::Slice),
            TyTerm::Str => Some(View::Str),
            _ => None,
        }
    }

    fn taken_of<V>(self, storage: &TyTerm<V>) -> bool
    where
        V: Phase,
    {
        match self {
            View::Slice => matches!(storage, TyTerm::UserDefined { .. } | TyTerm::Array(..)),
            View::Str => matches!(storage, TyTerm::String),
        }
    }
}

/// A container the machine can slice, named by its head alone: two `Vec`s
/// of different elements have one `as_slice` between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceableHead {
    Array,
    Named(QualifiedRef),
}

impl SliceableHead {
    pub fn of<V>(ty: &TyTerm<V>) -> Option<Self>
    where
        V: Phase,
    {
        match ty {
            TyTerm::UserDefined { id, .. } => Some(SliceableHead::Named(*id)),
            TyTerm::Array(..) => Some(SliceableHead::Array),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SlicedStorage {
    head: SliceableHead,
    mutability: Mutability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewed {
    pub view: View,
    pub mutability: Mutability,
}

impl Viewed {
    pub fn of_declaration<V>(ty: &TyTerm<V>) -> Option<Viewed>
    where
        V: Phase,
    {
        let TyTerm::Fn { params, ret, .. } = ty else {
            return None;
        };
        let [
            ParamTerm {
                ty: TyTerm::Ref(takes, storage),
                ..
            },
        ] = params.as_slice()
        else {
            return None;
        };
        let TyTerm::Ref(lends, run) = ret.as_ref() else {
            return None;
        };
        let view = View::of(&run.ty())?;
        (takes == lends && view.taken_of(&storage.ty())).then_some(Viewed {
            view,
            mutability: *takes,
        })
    }

    /// These three words are printed, not resolved: a refusal that could
    /// not settle a coercion names the function the container was missing,
    /// and `acvus-cli/tests/refusals/36-index-into-what-has-no-slice.expected`
    /// pins the sentence byte for byte. Changing one moves that golden.
    pub fn spelling(self) -> &'static str {
        match (self.view, self.mutability) {
            (View::Str, _) => "as_str",
            (View::Slice, Mutability::Shared) => "as_slice",
            (View::Slice, Mutability::Mut) => "as_slice_mut",
        }
    }
}

/// What a name resolves to among the environment's functions.
pub enum FnLookup<'a> {
    Found(QualifiedRef, &'a Scheme),
    /// A bare name several namespaces declare (RFC-0043).
    Overloaded(Vec<(QualifiedRef, &'a Scheme)>),
    Missing,
}

impl TypeEnv {
    /// A script's bare name is its own function if it has one, else the
    /// functions of that name under any namespace (RFC-0021, RFC-0043).
    pub fn resolve_fn(&self, name: QualifiedRef) -> FnLookup<'_> {
        if let Some(scheme) = self.functions.get(&name) {
            return FnLookup::Found(name, scheme);
        }
        if name.namespace.is_some() {
            return FnLookup::Missing;
        }
        let mut candidates: Vec<QualifiedRef> = self
            .functions
            .keys()
            .filter(|q| q.name == name.name && q.namespace.is_some())
            .copied()
            .collect();
        candidates.sort();
        match candidates.as_slice() {
            [] => FnLookup::Missing,
            [one] => FnLookup::Found(*one, &self.functions[one]),
            _ => FnLookup::Overloaded(
                candidates
                    .into_iter()
                    .map(|qref| (qref, &self.functions[&qref]))
                    .collect(),
            ),
        }
    }

    /// Every machine coercion, in a stable order.
    pub fn machine_coercions(&self) -> Vec<(QualifiedRef, &MachineCoercion)> {
        let mut found: Vec<QualifiedRef> = self.machine.keys().copied().collect();
        found.sort();
        found
            .into_iter()
            .map(|qref| (qref, &self.machine[&qref]))
            .collect()
    }

    /// Every declaration that is this coercion, in a stable order.
    pub fn machine_views(&self, viewed: Viewed) -> Vec<(QualifiedRef, &Scheme)> {
        self.machine_coercions()
            .into_iter()
            .filter(|(_, coercion)| coercion.viewed == viewed)
            .map(|(qref, coercion)| (qref, &coercion.scheme))
            .collect()
    }

    pub fn new() -> Self {
        Self {
            contexts: FxHashMap::default(),
            functions: FxHashMap::default(),
            machine: FxHashMap::default(),
        }
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

// -- Phase-parameterized type system ---------------------------------
//
// `TyTerm<V>` is a type term parameterized over inference variables.
// Two phases:
//   - `Concrete`: no inference variables (Var = Infallible). Post-inference.
//   - `Infer`:    may contain inference variables (Var = TypeBoundId). During inference.
//
// `type Ty = TyTerm<Concrete>` - always fully resolved. Compiler enforces this.
// `type InferTy = TyTerm<Infer>` - may have holes. Solver fills them in.

/// Phase marker trait - determines what can appear in inference variable slots.
pub trait Phase: 'static + Clone {
    /// Type inference variable. `Infallible` for concrete (uninhabitable).
    type TyVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
    /// Effect inference variable. `Infallible` for concrete (uninhabitable).
    type EffectVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
    /// Array length inference variable. `Infallible` for concrete (uninhabitable).
    type LenVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
    /// Identity inference variable. `Infallible` for concrete (uninhabitable).
    type IdentityVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
    /// Representation variable: the `ρ` of a type variable's binding
    /// (hash-types.md). `Infallible` for concrete (uninhabitable).
    type ReprVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;

    /// How a shown type spells a type variable of this phase, so that two
    /// occurrences of one variable read as one.
    fn spell_ty_var(var: &Self::TyVar, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    fn spell_effect_var(var: &Self::EffectVar, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    fn spell_len_var(var: &Self::LenVar, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// How a shown type marks an argument whose representation is open.
    fn spell_open_repr(f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// Post-inference phase - all types fully resolved.
/// `TyVar = Infallible` makes `TyTerm::Var` uninhabitable at type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Concrete;

impl Phase for Concrete {
    type TyVar = Infallible;
    type EffectVar = Infallible;
    type LenVar = Infallible;
    type IdentityVar = Infallible;
    type ReprVar = Infallible;

    fn spell_ty_var(var: &Infallible, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *var {}
    }

    fn spell_effect_var(var: &Infallible, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *var {}
    }

    fn spell_len_var(var: &Infallible, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *var {}
    }

    fn spell_open_repr(_: &mut fmt::Formatter<'_>) -> fmt::Result {
        unreachable!("a concrete type has no open representation")
    }
}

/// Polymorphic declaration phase - type templates stored in the graph.
/// `TyVar = u32` is a positional placeholder, not tied to any Solver instance.
/// Instantiated to `Infer` per call site during type checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Poly;

impl Phase for Poly {
    type TyVar = u32;
    type EffectVar = u32;
    type LenVar = u32;
    type IdentityVar = u32;
    type ReprVar = u32;

    /// A placeholder is spelled as a declaration would name it, `T` and
    /// `N` and `E`, so `Fn(Array<T, N>) -> Vec<T>` reads as the shape it
    /// is; the letters continue as `T7` past the alphabet each kind has.
    fn spell_ty_var(var: &u32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        spell_placeholder(*var, "TUVWXYZ", f)
    }

    fn spell_effect_var(var: &u32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        spell_placeholder(*var, "EF", f)
    }

    fn spell_len_var(var: &u32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        spell_placeholder(*var, "NM", f)
    }

    /// A declaration whose argument's representation is open takes
    /// either, and that is nothing to show.
    fn spell_open_repr(_: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

/// The `index`-th placeholder of a kind: its letter among `letters`, or the
/// kind's first letter and the index past them.
fn spell_placeholder(index: u32, letters: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match letters.chars().nth(index as usize) {
        Some(letter) => write!(f, "{letter}"),
        None => {
            let first = letters
                .chars()
                .next()
                .expect("every placeholder kind has at least one letter");
            write!(f, "{first}{index}")
        }
    }
}

/// During-inference phase - types may contain unresolved variables.
/// `TyVar = TypeBoundId` is scoped to a specific `Solver` instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Infer;

impl Phase for Infer {
    type TyVar = TypeBoundId;
    type EffectVar = EffectVarId;
    type LenVar = LenVarId;
    type IdentityVar = IdentityVarId;
    type ReprVar = ReprVarId;

    fn spell_ty_var(var: &TypeBoundId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'{}", var.0)
    }

    fn spell_effect_var(var: &EffectVarId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'{}", var.0)
    }

    fn spell_len_var(var: &LenVarId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'{}", var.0)
    }

    fn spell_open_repr(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#?")
    }
}

/// Polymorphic type - template with positional placeholders.
pub type PolyTy = TyTerm<Poly>;
/// Polymorphic function parameter.
pub type PolyParam = ParamTerm<Poly>;

// -- Solver types ----------------------------------------------------

/// Index into `Solver::ty_bounds`. Identifies a type inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeBoundId(pub u32);

/// Index into `Solver::effect_vars`. Identifies an effect inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectVarId(pub u32);

/// Index into `Solver::len_vars`. Identifies an array length inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LenVarId(pub u32);

/// Index into `Solver::identity_vars`. Identifies an identity inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IdentityVarId(pub u32);

/// Index into `Solver::repr_vars`. Identifies a representation variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReprVarId(pub u32);

// Re-export solver types - these were historically in ty.rs.
pub use crate::solver::{FreezeError, Solver, Sources, TypeBound};

/// Type alias - always concrete, no inference variables.
pub type InferTy = TyTerm<Infer>;

/// What a representation variable stands for (hash-types.md). A variable
/// that meets a `#` tree stands for the whole tree (RFC-0041).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Repr<V: Phase> {
    Uniform,
    Var(V::ReprVar),
    Specialized(HeldTy<V>),
}

/// A slot whose storage may be laid out by its argument: a type argument
/// of a user-defined type, and the target of a reference. `#` lives here
/// and on an `EffectArg` and nowhere else (hash-types.md, R1): a position
/// that holds a bare `TyTerm` cannot carry a representation.
///
/// The representation follows the type part by part (RFC-0041). A part
/// whose Rust type is a type variable's run-time instantiation is uniform;
/// a part whose Rust type is itself is `#`, and a `#` composite states the
/// representation of each of its parts again, because the runtime reads a
/// value by its parts and a box's Rust type differs part by part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeArg<V: Phase> {
    /// The representation every polymorphic position needs.
    Uniform(TyTerm<V>),
    /// A type variable's `ρ`, open until a specializing position fixes it;
    /// frozen open, it is `Uniform`.
    Open(V::ReprVar, TyTerm<V>),
    Specialized(HeldTy<V>),
}

/// A `#` node: a Rust head written out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeldTy<V: Phase> {
    Tuple(Vec<TypeArg<V>>),
    Option(Box<TypeArg<V>>),
    Result(Box<TypeArg<V>>, Box<TypeArg<V>>),
    /// The language's array, whose Rust head is the runtime's `Arr`: its
    /// elements in a buffer of their own.
    Array(Box<TypeArg<V>>, LenTerm<V>),
    /// A Rust `[T; N]`, its `N` elements in place: the same acvus type as
    /// `Array`, another box.
    RustArray(Box<TypeArg<V>>, usize),
    Leaf(LeafTy<V>),
    /// `α` with every part `#`: once `α` resolves, the node is the fully
    /// specialized tree of `α`'s type. What a family cast's pattern writes
    /// (RFC-0041), which a member's type fills: a member is a written Rust
    /// type with no part a run-time instantiation fills.
    Held(V::TyVar),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafTy<V: Phase>(TyTerm<V>);

impl<V: Phase> LeafTy<V> {
    pub fn ty(&self) -> &TyTerm<V> {
        &self.0
    }
}

impl<V: Phase> Repr<V> {
    /// This representation over `ty`. A tree drops `ty`: the solver binds a
    /// variable to a tree only after joining the tree's type with the
    /// position's (`Terms::join`).
    pub fn at(self, ty: TyTerm<V>) -> TypeArg<V> {
        match self {
            Repr::Uniform => TypeArg::Uniform(ty),
            Repr::Var(v) => TypeArg::Open(v, ty),
            Repr::Specialized(held) => TypeArg::Specialized(held),
        }
    }
}

impl<V: Phase> TypeArg<V> {
    pub fn uniform(ty: TyTerm<V>) -> Self {
        TypeArg::Uniform(ty)
    }

    /// `#ty` with every part `#`: the argument whose Rust type is `ty`
    /// written out to its leaves.
    pub fn specialized(ty: TyTerm<V>) -> Self {
        TypeArg::Specialized(HeldTy::of(ty))
    }

    pub fn ty(&self) -> std::borrow::Cow<'_, TyTerm<V>> {
        match self {
            TypeArg::Uniform(ty) | TypeArg::Open(_, ty) => std::borrow::Cow::Borrowed(ty),
            TypeArg::Specialized(held) => std::borrow::Cow::Owned(held.ty()),
        }
    }

    pub fn into_ty(self) -> TyTerm<V> {
        match self {
            TypeArg::Uniform(ty) | TypeArg::Open(_, ty) => ty,
            TypeArg::Specialized(held) => held.ty(),
        }
    }

    /// The type this argument names where the argument holds it whole: a
    /// uniform or open argument's, or a `#` leaf's. `None` for a `#`
    /// composite or a `#` variable, whose type is built from the tree.
    pub fn whole(&self) -> Option<&TyTerm<V>> {
        match self {
            TypeArg::Uniform(ty) | TypeArg::Open(_, ty) => Some(ty),
            TypeArg::Specialized(HeldTy::Leaf(leaf)) => Some(leaf.ty()),
            TypeArg::Specialized(
                HeldTy::Tuple(_)
                | HeldTy::Option(_)
                | HeldTy::Result(..)
                | HeldTy::Array(..)
                | HeldTy::RustArray(..)
                | HeldTy::Held(_),
            ) => None,
        }
    }

    pub fn is_uniform(&self) -> bool {
        matches!(self, TypeArg::Uniform(_))
    }

    /// `rewrite` on each type this argument holds: a `#` leaf or a `Held`
    /// node is rewritten as its type, and becomes the rewritten type's node.
    pub fn rewrite_types(&mut self, rewrite: &mut impl FnMut(&mut TyTerm<V>)) {
        match self {
            TypeArg::Uniform(ty) | TypeArg::Open(_, ty) => rewrite(ty),
            TypeArg::Specialized(held) => held.rewrite_types(rewrite),
        }
    }

    /// What a representation variable meeting this argument binds to.
    pub fn repr(&self) -> Repr<V> {
        match self {
            TypeArg::Uniform(_) => Repr::Uniform,
            TypeArg::Open(v, _) => Repr::Var(*v),
            TypeArg::Specialized(held) => Repr::Specialized(held.clone()),
        }
    }

    /// Whether this argument is `#` at every part, or could still be where
    /// `open` lets an open representation stand for `#`.
    fn is_full(&self, open: bool) -> bool {
        match self {
            TypeArg::Uniform(_) => false,
            TypeArg::Open(..) => open,
            TypeArg::Specialized(held) => held.is_full(open),
        }
    }

    /// Whether two arguments have one tree of representations and `same`
    /// holds of each pair of types the tree leaves to them.
    pub fn same_by(
        &self,
        other: &Self,
        same: &mut impl FnMut(&TyTerm<V>, &TyTerm<V>) -> bool,
    ) -> bool {
        match (self, other) {
            (TypeArg::Uniform(a), TypeArg::Uniform(b)) => same(a, b),
            (TypeArg::Open(x, a), TypeArg::Open(y, b)) => x == y && same(a, b),
            (TypeArg::Specialized(a), TypeArg::Specialized(b)) => a.same_by(b, same),
            (TypeArg::Uniform(_) | TypeArg::Open(..) | TypeArg::Specialized(_), _) => false,
        }
    }

    pub fn map<W: Phase>(
        &self,
        on_var: &mut impl FnMut(V::TyVar) -> TyTerm<W>,
        on_identity: &mut impl FnMut(V::IdentityVar) -> IdentityTerm<W>,
        on_effect: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>,
        on_len: &mut impl FnMut(V::LenVar) -> LenTerm<W>,
        on_repr: &mut impl FnMut(V::ReprVar) -> Repr<W>,
    ) -> TypeArg<W> {
        match self {
            TypeArg::Uniform(ty) => {
                TypeArg::Uniform(ty.map(on_var, on_identity, on_effect, on_len, on_repr))
            }
            TypeArg::Open(v, ty) => {
                let ty = ty.map(on_var, on_identity, on_effect, on_len, on_repr);
                on_repr(*v).at(ty)
            }
            TypeArg::Specialized(held) => {
                TypeArg::Specialized(held.map(on_var, on_identity, on_effect, on_len, on_repr))
            }
        }
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_var: &mut impl FnMut(V::TyVar) -> Result<TyTerm<W>, E>,
        on_identity: &mut impl FnMut(V::IdentityVar) -> Result<IdentityTerm<W>, E>,
        on_effect: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
        on_len: &mut impl FnMut(V::LenVar) -> Result<LenTerm<W>, E>,
        on_repr: &mut impl FnMut(V::ReprVar) -> Result<Repr<W>, E>,
    ) -> Result<TypeArg<W>, E> {
        Ok(match self {
            TypeArg::Uniform(ty) => {
                TypeArg::Uniform(ty.try_map(on_var, on_identity, on_effect, on_len, on_repr)?)
            }
            TypeArg::Open(v, ty) => {
                let ty = ty.try_map(on_var, on_identity, on_effect, on_len, on_repr)?;
                on_repr(*v)?.at(ty)
            }
            TypeArg::Specialized(held) => TypeArg::Specialized(held.try_map(
                on_var,
                on_identity,
                on_effect,
                on_len,
                on_repr,
            )?),
        })
    }
}

impl<V: Phase> HeldTy<V> {
    /// The fully specialized tree of `ty`: every part `#`, a variable's
    /// part `Held`.
    pub fn of(ty: TyTerm<V>) -> Self {
        match ty {
            TyTerm::Tuple(elems) => {
                HeldTy::Tuple(elems.into_iter().map(TypeArg::specialized).collect())
            }
            TyTerm::Option(inner) => HeldTy::Option(Box::new(TypeArg::specialized(*inner))),
            TyTerm::Result(ok, err) => HeldTy::Result(
                Box::new(TypeArg::specialized(*ok)),
                Box::new(TypeArg::specialized(*err)),
            ),
            TyTerm::Array(elem, len) => HeldTy::Array(Box::new(TypeArg::specialized(*elem)), len),
            TyTerm::Var(v) => HeldTy::Held(v),
            leaf @ (TyTerm::Int(_)
            | TyTerm::Float
            | TyTerm::Char
            | TyTerm::String
            | TyTerm::Bool
            | TyTerm::Unit
            | TyTerm::Never
            | TyTerm::Order
            | TyTerm::Object(_)
            | TyTerm::Fn { .. }
            | TyTerm::UserDefined { .. }
            | TyTerm::Enum { .. }
            | TyTerm::Slice(_)
            | TyTerm::Str
            | TyTerm::Handle(_)
            | TyTerm::Ref(..)
            | TyTerm::Error(_)) => HeldTy::Leaf(LeafTy(leaf)),
        }
    }

    pub fn ty(&self) -> TyTerm<V> {
        match self {
            HeldTy::Tuple(parts) => {
                TyTerm::Tuple(parts.iter().map(|p| p.ty().into_owned()).collect())
            }
            HeldTy::Option(part) => TyTerm::Option(Box::new(part.ty().into_owned())),
            HeldTy::Result(ok, err) => TyTerm::Result(
                Box::new(ok.ty().into_owned()),
                Box::new(err.ty().into_owned()),
            ),
            HeldTy::Array(part, len) => {
                TyTerm::Array(Box::new(part.ty().into_owned()), len.clone())
            }
            HeldTy::RustArray(part, len) => {
                TyTerm::Array(Box::new(part.ty().into_owned()), LenTerm::Known(*len))
            }
            HeldTy::Leaf(leaf) => leaf.0.clone(),
            HeldTy::Held(v) => TyTerm::Var(*v),
        }
    }

    pub fn parts(&self) -> Vec<&TypeArg<V>> {
        match self {
            HeldTy::Tuple(parts) => parts.iter().collect(),
            HeldTy::Option(part) | HeldTy::Array(part, _) | HeldTy::RustArray(part, _) => {
                vec![part]
            }
            HeldTy::Result(ok, err) => vec![ok, err],
            HeldTy::Leaf(_) | HeldTy::Held(_) => Vec::new(),
        }
    }

    fn rewrite_types(&mut self, rewrite: &mut impl FnMut(&mut TyTerm<V>)) {
        match self {
            HeldTy::Tuple(parts) => {
                for part in parts {
                    part.rewrite_types(rewrite);
                }
            }
            HeldTy::Option(part) | HeldTy::Array(part, _) | HeldTy::RustArray(part, _) => {
                part.rewrite_types(rewrite)
            }
            HeldTy::Result(ok, err) => {
                ok.rewrite_types(rewrite);
                err.rewrite_types(rewrite);
            }
            HeldTy::Leaf(_) | HeldTy::Held(_) => {
                let mut ty = self.ty();
                rewrite(&mut ty);
                *self = HeldTy::of(ty);
            }
        }
    }

    /// Whether every part is `#`, or could still be where `open` lets an
    /// open representation stand for `#`: the tree `of` writes for this
    /// type. A Rust array is not, since `of` writes an array as the
    /// language's.
    pub fn is_full(&self, open: bool) -> bool {
        match self {
            HeldTy::RustArray(..) => false,
            HeldTy::Tuple(_)
            | HeldTy::Option(_)
            | HeldTy::Result(..)
            | HeldTy::Array(..)
            | HeldTy::Leaf(_)
            | HeldTy::Held(_) => self.parts().into_iter().all(|part| part.is_full(open)),
        }
    }

    fn same_by(&self, other: &Self, same: &mut impl FnMut(&TyTerm<V>, &TyTerm<V>) -> bool) -> bool {
        match (self, other) {
            (HeldTy::Tuple(a), HeldTy::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same_by(y, same))
            }
            (HeldTy::Option(a), HeldTy::Option(b)) => a.same_by(b, same),
            (HeldTy::Result(a_ok, a_err), HeldTy::Result(b_ok, b_err)) => {
                a_ok.same_by(b_ok, same) && a_err.same_by(b_err, same)
            }
            (HeldTy::Array(a, an), HeldTy::Array(b, bn)) => {
                let same_len = match (an, bn) {
                    (LenTerm::Known(x), LenTerm::Known(y)) => x == y,
                    (LenTerm::Var(x), LenTerm::Var(y)) => x == y,
                    (LenTerm::Known(_), LenTerm::Var(_)) | (LenTerm::Var(_), LenTerm::Known(_)) => {
                        false
                    }
                };
                same_len && a.same_by(b, same)
            }
            (HeldTy::RustArray(a, an), HeldTy::RustArray(b, bn)) => an == bn && a.same_by(b, same),
            (HeldTy::Leaf(a), HeldTy::Leaf(b)) => same(&a.0, &b.0),
            (HeldTy::Held(a), HeldTy::Held(b)) => a == b,
            (
                HeldTy::Tuple(_)
                | HeldTy::Option(_)
                | HeldTy::Result(..)
                | HeldTy::Array(..)
                | HeldTy::RustArray(..)
                | HeldTy::Leaf(_)
                | HeldTy::Held(_),
                _,
            ) => false,
        }
    }

    pub fn map<W: Phase>(
        &self,
        on_var: &mut impl FnMut(V::TyVar) -> TyTerm<W>,
        on_identity: &mut impl FnMut(V::IdentityVar) -> IdentityTerm<W>,
        on_effect: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>,
        on_len: &mut impl FnMut(V::LenVar) -> LenTerm<W>,
        on_repr: &mut impl FnMut(V::ReprVar) -> Repr<W>,
    ) -> HeldTy<W> {
        let mut part = |p: &TypeArg<V>| p.map(on_var, on_identity, on_effect, on_len, on_repr);
        match self {
            HeldTy::Tuple(parts) => HeldTy::Tuple(parts.iter().map(&mut part).collect()),
            HeldTy::Option(p) => HeldTy::Option(Box::new(part(p))),
            HeldTy::Result(ok, err) => {
                let ok = part(ok);
                HeldTy::Result(Box::new(ok), Box::new(part(err)))
            }
            HeldTy::Array(p, len) => {
                let p = part(p);
                HeldTy::Array(Box::new(p), len.map(on_len))
            }
            HeldTy::RustArray(p, len) => HeldTy::RustArray(Box::new(part(p)), *len),
            HeldTy::Leaf(leaf) => {
                HeldTy::of(leaf.0.map(on_var, on_identity, on_effect, on_len, on_repr))
            }
            HeldTy::Held(v) => HeldTy::of(on_var(*v)),
        }
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_var: &mut impl FnMut(V::TyVar) -> Result<TyTerm<W>, E>,
        on_identity: &mut impl FnMut(V::IdentityVar) -> Result<IdentityTerm<W>, E>,
        on_effect: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
        on_len: &mut impl FnMut(V::LenVar) -> Result<LenTerm<W>, E>,
        on_repr: &mut impl FnMut(V::ReprVar) -> Result<Repr<W>, E>,
    ) -> Result<HeldTy<W>, E> {
        let mut part = |p: &TypeArg<V>| p.try_map(on_var, on_identity, on_effect, on_len, on_repr);
        Ok(match self {
            HeldTy::Tuple(parts) => {
                HeldTy::Tuple(parts.iter().map(&mut part).collect::<Result<_, _>>()?)
            }
            HeldTy::Option(p) => HeldTy::Option(Box::new(part(p)?)),
            HeldTy::Result(ok, err) => {
                let ok = part(ok)?;
                HeldTy::Result(Box::new(ok), Box::new(part(err)?))
            }
            HeldTy::Array(p, len) => {
                let p = part(p)?;
                HeldTy::Array(Box::new(p), len.try_map(on_len)?)
            }
            HeldTy::RustArray(p, len) => HeldTy::RustArray(Box::new(part(p)?), *len),
            HeldTy::Leaf(leaf) => {
                HeldTy::of(
                    leaf.0
                        .try_map(on_var, on_identity, on_effect, on_len, on_repr)?,
                )
            }
            HeldTy::Held(v) => HeldTy::of(on_var(*v)?),
        })
    }
}

/// An effect argument of a user-defined type. `#e` is an effect the
/// declaration wrote, which the value's Rust type carries as itself;
/// `e` is the effect variable's, which the runtime fills with nothing. Two
/// values whose Rust types differ in it are two types here, as `#τ` and
/// `τ` are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectArg<V: Phase> {
    Uniform(EffectTerm<V>),
    Specialized(EffectTerm<V>),
}

impl<V: Phase> EffectArg<V> {
    pub fn uniform(effect: EffectTerm<V>) -> Self {
        EffectArg::Uniform(effect)
    }

    pub fn specialized(effect: EffectTerm<V>) -> Self {
        EffectArg::Specialized(effect)
    }

    pub fn effect(&self) -> &EffectTerm<V> {
        match self {
            EffectArg::Uniform(effect) | EffectArg::Specialized(effect) => effect,
        }
    }

    pub fn is_specialized(&self) -> bool {
        matches!(self, EffectArg::Specialized(_))
    }

    pub fn with_effect<W: Phase>(&self, effect: EffectTerm<W>) -> EffectArg<W> {
        match self {
            EffectArg::Uniform(_) => EffectArg::Uniform(effect),
            EffectArg::Specialized(_) => EffectArg::Specialized(effect),
        }
    }

    pub fn map<W: Phase>(
        &self,
        on_effect: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>,
    ) -> EffectArg<W> {
        self.with_effect(self.effect().map(on_effect))
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_effect: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
    ) -> Result<EffectArg<W>, E> {
        Ok(self.with_effect(self.effect().try_map(on_effect)?))
    }
}

/// How an object type's field set relates to the values it admits
/// (RFC-0042).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldSet {
    /// The fields the named struct declares; a value of the type has exactly
    /// these. The registries hold the name to one Rust type (RFC-0021 rule 7).
    Declared(QualifiedRef),
    /// The fields an object literal wrote. A field store adds to them.
    Written,
    /// At least the fields a read or a pattern named, which is what asking
    /// an object for a field says about it.
    AtLeast,
}

/// An object type: the type of each field, and what the field set is
/// (RFC-0042).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectTy<V>
where
    V: Phase,
{
    set: FieldSet,
    fields: FxHashMap<Astr, TyTerm<V>>,
    home: Home<V>,
}

/// The variable a structural term was read from, when the solver resolved
/// it from one. A term that carries it is that variable wherever it is
/// joined again: a resolved copy keeps the identity of what it copies
/// (RFC-0042 rule 1). Two types are equal whatever their homes, since a
/// home is an identity and not part of the type.
pub struct Home<V>(Option<V::TyVar>)
where
    V: Phase;

impl<V> Clone for Home<V>
where
    V: Phase,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for Home<V> where V: Phase {}

impl<V> Home<V>
where
    V: Phase,
{
    pub const NONE: Self = Home(None);

    pub fn of(var: V::TyVar) -> Self {
        Home(Some(var))
    }

    pub fn var(self) -> Option<V::TyVar> {
        self.0
    }
}

impl<V> fmt::Debug for Home<V>
where
    V: Phase,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some(var) => write!(f, "@{var:?}"),
            None => write!(f, "@-"),
        }
    }
}

impl<V> PartialEq for Home<V>
where
    V: Phase,
{
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<V> Eq for Home<V> where V: Phase {}

impl<V> std::hash::Hash for Home<V>
where
    V: Phase,
{
    fn hash<H: std::hash::Hasher>(&self, _: &mut H) {}
}

/// The join of two object types (RFC-0042).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectMeet<V>
where
    V: Phase,
{
    Joined {
        ty: ObjectTy<V>,
    },
    /// A field `declared` names and an object of its type lacks.
    Lacks {
        declared: QualifiedRef,
        field: Astr,
    },
    /// A field an object has and `declared` does not name.
    Undeclared {
        declared: QualifiedRef,
        field: Astr,
    },
    /// Two declarations: a value has the fields of one declared struct and
    /// of no other.
    TwoDeclarations {
        a: QualifiedRef,
        b: QualifiedRef,
    },
    /// The union has more fields than [`ObjectTy::MAX_FIELDS`].
    TooWide {
        fields: usize,
    },
}

impl<V> ObjectTy<V>
where
    V: Phase,
{
    /// The most fields an object type has. A field's position crosses the
    /// boundary as an `acvus_extern::FieldAt`, which is a `u16`, so an object
    /// wider than this has positions the machine cannot name. The checker
    /// refuses one where the width is made — an object literal and the union
    /// two field sets join to — and nothing downstream converts.
    pub const MAX_FIELDS: usize = u16::MAX as usize;

    /// The field count, when it is over [`Self::MAX_FIELDS`], for the refusal
    /// that names it. This is the one comparison the bound is made of.
    pub fn too_wide<W>(fields: &FxHashMap<Astr, TyTerm<W>>) -> Option<usize>
    where
        W: Phase,
    {
        (fields.len() > Self::MAX_FIELDS).then_some(fields.len())
    }

    /// The type of a struct `name` declares.
    pub fn declared(name: QualifiedRef, fields: FxHashMap<Astr, TyTerm<V>>) -> Self {
        Self {
            set: FieldSet::Declared(name),
            fields,
            home: Home::NONE,
        }
    }

    /// The type of an object a literal wrote.
    pub fn written(fields: FxHashMap<Astr, TyTerm<V>>) -> Self {
        Self {
            set: FieldSet::Written,
            fields,
            home: Home::NONE,
        }
    }

    /// What reading these fields asks of the object read.
    pub fn at_least(fields: FxHashMap<Astr, TyTerm<V>>) -> Self {
        Self {
            set: FieldSet::AtLeast,
            fields,
            home: Home::NONE,
        }
    }

    pub fn field_set(&self) -> FieldSet {
        self.set
    }

    pub fn declaration(&self) -> Option<QualifiedRef> {
        match self.set {
            FieldSet::Declared(name) => Some(name),
            FieldSet::Written | FieldSet::AtLeast => None,
        }
    }

    /// This object's field set over `fields`: what a walk over an object
    /// type rebuilds.
    pub fn with_fields<W>(&self, fields: FxHashMap<Astr, TyTerm<W>>) -> ObjectTy<W>
    where
        W: Phase,
    {
        ObjectTy {
            set: self.set,
            fields,
            home: Home::NONE,
        }
    }

    pub fn home(&self) -> Home<V> {
        self.home
    }

    pub fn with_home(self, home: Home<V>) -> Self {
        Self { home, ..self }
    }

    /// A field of this object the other lacks. The one that is interned
    /// first, so that two runs of one program refuse it in the same words.
    pub fn missing_from(&self, other: &Self) -> Option<Astr> {
        self.only_in(other)
    }

    fn only_in(&self, other: &Self) -> Option<Astr> {
        self.fields
            .keys()
            .filter(|k| !other.fields.contains_key(k))
            .min()
            .copied()
    }

    /// The union of the two field sets, which the side that lacks a field
    /// of it grows into.
    fn union(a: &Self, b: &Self, set: FieldSet) -> ObjectMeet<V> {
        let mut fields = a.fields.clone();
        for (name, ty) in &b.fields {
            fields.entry(*name).or_insert_with(|| ty.clone());
        }
        if let Some(fields) = Self::too_wide(&fields) {
            return ObjectMeet::TooWide { fields };
        }
        ObjectMeet::Joined {
            ty: ObjectTy {
                set,
                fields,
                home: Home::NONE,
            },
        }
    }

    /// This is deliberately not a case of `meet`. `meet` joins a `Written`
    /// set with an `AtLeast` one by their union, which is RFC-0042's rule
    /// that a field store adds to a literal's set, and that rule stays.
    /// A projection parameter (RFC-0050 rule 6) is matched at least
    /// instead: it writes no field into its argument, so a literal that
    /// lacks a field it borrows is refused rather than widened.
    ///
    /// `Declared` is absent from the match because `meet` already refuses
    /// it by name, answering `Undeclared` for a declaration that does not
    /// name a field the projection does.
    pub fn borrowed_field_missing_from(&self, argument: &Self) -> Option<Astr> {
        match (self.set, argument.set) {
            (FieldSet::AtLeast, FieldSet::Written) => self.only_in(argument),
            _ => None,
        }
    }

    /// The field on which `other` disagrees with the declaration `of`
    /// carries. A value of the declared type has every field the struct
    /// names, so an object that is one and lacks a field is refused; what
    /// only asks an object for fields is not.
    fn disagreement(declared: QualifiedRef, of: &Self, other: &Self) -> Option<ObjectMeet<V>> {
        if let Some(field) = other.only_in(of) {
            return Some(ObjectMeet::Undeclared { declared, field });
        }
        match other.set {
            FieldSet::AtLeast => None,
            FieldSet::Declared(_) | FieldSet::Written => of
                .only_in(other)
                .map(|field| ObjectMeet::Lacks { declared, field }),
        }
    }

    /// What the two join to (RFC-0042): two undeclared field sets join to
    /// their union, and a declared field set is the value's own, so an
    /// object that is one lacking a field, or carrying a field the struct
    /// does not name, is refused by the field's name.
    pub fn meet(a: &Self, b: &Self) -> ObjectMeet<V> {
        let joined = |ty: &Self| ObjectMeet::Joined { ty: ty.clone() };
        match (a.set, b.set) {
            (FieldSet::Declared(x), FieldSet::Declared(y)) if x != y => {
                ObjectMeet::TwoDeclarations { a: x, b: y }
            }
            (FieldSet::Declared(x), FieldSet::Declared(_)) => {
                Self::disagreement(x, a, b).unwrap_or_else(|| joined(a))
            }
            (FieldSet::Declared(x), _) => Self::disagreement(x, a, b).unwrap_or_else(|| joined(a)),
            (_, FieldSet::Declared(y)) => Self::disagreement(y, b, a).unwrap_or_else(|| joined(b)),
            (FieldSet::Written, _) | (_, FieldSet::Written) => Self::union(a, b, FieldSet::Written),
            (FieldSet::AtLeast, FieldSet::AtLeast) => Self::union(a, b, FieldSet::AtLeast),
        }
    }
}

impl<V> std::ops::Deref for ObjectTy<V>
where
    V: Phase,
{
    type Target = FxHashMap<Astr, TyTerm<V>>;

    fn deref(&self) -> &Self::Target {
        &self.fields
    }
}

impl<'a, V> IntoIterator for &'a ObjectTy<V>
where
    V: Phase,
{
    type Item = (&'a Astr, &'a TyTerm<V>);
    type IntoIter = std::collections::hash_map::Iter<'a, Astr, TyTerm<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.iter()
    }
}

/// A type term parameterized over inference phase.
///
/// When `V = Concrete`: `Var(Infallible)` is uninhabitable - type is always concrete.
/// When `V = Infer`: `Var(TypeBoundId)` references the solver's bound table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TyTerm<V: Phase> {
    // Primitives
    Int(IntTy),
    Float,
    /// One Unicode scalar value, a word of `u32` bits (RFC-0058). Rust's
    /// `char`, and it crosses as one.
    Char,
    String,
    Bool,
    Unit,
    /// The type with no value: what an unconstrained type variable
    /// resolves to, and the error side of a `Result` nothing fails into
    /// (RFC-0038). A value of it never exists, so it is below every type.
    Never,
    /// A dependency between effectful calls (RFC-0007). IR-only: no script
    /// names it, and no value of it exists at runtime.
    Order,
    // Containers
    Array(Box<TyTerm<V>>, LenTerm<V>),
    Object(ObjectTy<V>),
    Tuple(Vec<TyTerm<V>>),
    Option(Box<TyTerm<V>>),
    Result(Box<TyTerm<V>>, Box<TyTerm<V>>),
    // Functions
    Fn {
        params: Vec<ParamTerm<V>>,
        ret: Box<TyTerm<V>>,
        captures: Vec<TyTerm<V>>,
        effect: EffectTerm<V>,
    },
    // Nominal
    UserDefined {
        id: QualifiedRef,
        type_args: Vec<TypeArg<V>>,
        effect_args: Vec<EffectArg<V>>,
        identity_args: Vec<IdentityTerm<V>>,
        /// The declaration's `region_params`, carried on the term as
        /// `identity_args` mirrors `identity_params`: a pass holding no
        /// registry reads a value's positions off its type (RFC-0079 rule 2).
        region_params: usize,
    },
    Enum {
        /// A script's `A::B` names `A` at the root; a derived enum's name is
        /// held to one Rust type (RFC-0021 rule 7).
        name: QualifiedRef,
        variants: FxHashMap<Astr, Option<Box<TyTerm<V>>>>,
        home: Home<V>,
    },
    /// `[T]`: the run of elements a container lends. Unsized — no value
    /// has this type and no storage holds one; it appears only under a
    /// `Ref`, as `&[T]` and `&mut [T]` (RFC-0047).
    Slice(Box<TyTerm<V>>),
    /// `str`: the run of UTF-8 bytes a `String` lends (RFC-0062).
    ///
    /// There is no `&mut str`, and that is a decision rather than an
    /// omission: a write through one could leave the bytes invalid UTF-8,
    /// and every operation that wants to write has the owned `String` to
    /// write into. `Mutability::Mut` over this type is refused where a
    /// type is written.
    Str,
    // Resources
    Handle(Box<TyTerm<V>>),
    /// `&T` or `&mut T`: a second name for a storage holding a `T`
    /// (RFC-0018). A word at runtime; never data. The target's
    /// representation is the storage's (hash-types.md, R1).
    Ref(Mutability, Box<TypeArg<V>>),
    // Special
    Error(ErrorToken),
    /// Inference variable - only inhabitable when `V = Infer`.
    /// For `V = Concrete`, this is `Var(Infallible)` which cannot be constructed.
    Var(V::TyVar),
}

/// Whether a reference may write through to its storage (RFC-0018).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Mutability {
    Shared,
    Mut,
}

impl Mutability {
    /// Whether a reference of this mutability reaches a position of
    /// `wanted`'s: a `&mut` reaches a `&` by the reborrow `&r` gives
    /// (RFC-0029 rule 3), and a `&` never reaches a `&mut`.
    pub fn reaches(self, wanted: Mutability) -> bool {
        self == wanted || wanted == Mutability::Shared
    }

    pub fn prefix(self) -> &'static str {
        match self {
            Mutability::Shared => "&",
            Mutability::Mut => "&mut ",
        }
    }
}

/// Named, typed function parameter - parameterized over phase. A parameter
/// that borrows has a reference type; there is no mode beside the type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamTerm<V: Phase> {
    pub name: Astr,
    pub ty: TyTerm<V>,
}

impl<V: Phase> ParamTerm<V> {
    pub fn new(name: Astr, ty: TyTerm<V>) -> Self {
        Self { name, ty }
    }

    /// The same parameter with another type.
    pub fn retyped<W: Phase>(&self, ty: TyTerm<W>) -> ParamTerm<W> {
        ParamTerm {
            name: self.name,
            ty,
        }
    }
}

// -- Generic phase traversal ----------------------------------------
//
// `map` and `try_map` provide the single recursive traversal over
// `TyTerm<V>`. All phase-to-phase transformations (lift, freeze,
// resolve, instantiate) are specializations of these two operations.

impl<V: Phase> TyTerm<V> {
    pub const I8: Self = TyTerm::Int(IntTy::I8);
    pub const I16: Self = TyTerm::Int(IntTy::I16);
    pub const I32: Self = TyTerm::Int(IntTy::I32);
    pub const I64: Self = TyTerm::Int(IntTy::I64);
    pub const U8: Self = TyTerm::Int(IntTy::U8);
    pub const U16: Self = TyTerm::Int(IntTy::U16);
    pub const U32: Self = TyTerm::Int(IntTy::U32);
    pub const U64: Self = TyTerm::Int(IntTy::U64);

    /// A type with no parts, held as one word.
    pub fn is_scalar(&self) -> bool {
        match self {
            TyTerm::Int(_)
            | TyTerm::Float
            | TyTerm::Char
            | TyTerm::Bool
            | TyTerm::Unit
            | TyTerm::Never
            | TyTerm::Order => true,
            TyTerm::String
            | TyTerm::Array(..)
            | TyTerm::Object(_)
            | TyTerm::Tuple(_)
            | TyTerm::Option(_)
            | TyTerm::Result(..)
            | TyTerm::Fn { .. }
            | TyTerm::UserDefined { .. }
            | TyTerm::Enum { .. }
            | TyTerm::Slice(_)
            | TyTerm::Str
            | TyTerm::Handle(_)
            | TyTerm::Ref(..)
            | TyTerm::Error(_)
            | TyTerm::Var(_) => false,
        }
    }

    /// Whether a value of this type is a word (RFC-0018 rule 1). `None`
    /// where a variable or an error leaves the answer open.
    ///
    /// The answer for an option is one half of a contract with the runtime's
    /// representation of one, written in RFC-0039 and `docs/runtime-value.md`:
    /// a host gives an option no storage of its own, so `Some(v)` is `v` and a
    /// `None` is a word counting the `Some`s around it. A host that gave an
    /// option a box would have to move this arm with it.
    pub fn is_word(&self) -> Option<bool> {
        match self {
            TyTerm::Ref(..) => Some(true),
            TyTerm::Option(payload) => payload.is_word(),
            TyTerm::Var(_) | TyTerm::Error(_) => None,
            other => Some(other.is_scalar()),
        }
    }

    /// Whether a value of this type is copied where it is read, not moved:
    /// a word, or a `String` (RFC-0018 rule 2).
    pub fn copies(&self) -> Option<bool> {
        match self {
            TyTerm::String => Some(true),
            other => other.is_word(),
        }
    }

    /// Two types that print the same and whose source lists differ are one
    /// type held by values from two different sources, which is what
    /// `MirErrorKind::IdentityMismatch` reports: this order is the pairing
    /// that refusal reads, and `typeck` compares the two lists position by
    /// position.
    pub fn children(&self) -> Vec<std::borrow::Cow<'_, TyTerm<V>>> {
        use std::borrow::Cow;
        match self {
            TyTerm::Int(_)
            | TyTerm::Float
            | TyTerm::Char
            | TyTerm::String
            | TyTerm::Bool
            | TyTerm::Unit
            | TyTerm::Never
            | TyTerm::Order
            | TyTerm::Str
            | TyTerm::Error(_)
            | TyTerm::Var(_) => Vec::new(),
            TyTerm::Array(inner, _)
            | TyTerm::Slice(inner)
            | TyTerm::Handle(inner)
            | TyTerm::Option(inner) => vec![Cow::Borrowed(&**inner)],
            TyTerm::Ref(_, inner) => vec![inner.ty()],
            TyTerm::Result(ok, err) => vec![Cow::Borrowed(&**ok), Cow::Borrowed(&**err)],
            TyTerm::Tuple(elems) => elems.iter().map(Cow::Borrowed).collect(),
            TyTerm::Object(object) => object.values().map(Cow::Borrowed).collect(),
            TyTerm::Enum { variants, .. } => variants
                .values()
                .flatten()
                .map(|b| Cow::Borrowed(&**b))
                .collect(),
            TyTerm::Fn {
                params,
                ret,
                captures,
                ..
            } => params
                .iter()
                .map(|param| &param.ty)
                .chain(std::iter::once(&**ret))
                .chain(captures.iter())
                .map(Cow::Borrowed)
                .collect(),
            TyTerm::UserDefined { type_args, .. } => type_args.iter().map(TypeArg::ty).collect(),
        }
    }

    /// `rewrite` on each type directly below this one. A type inside a `#`
    /// tree is rewritten where the tree holds it, and the node it sits at is
    /// the rewritten type's.
    pub fn rewrite_children(&mut self, rewrite: &mut impl FnMut(&mut TyTerm<V>)) {
        match self {
            TyTerm::Int(_)
            | TyTerm::Float
            | TyTerm::Char
            | TyTerm::String
            | TyTerm::Bool
            | TyTerm::Unit
            | TyTerm::Never
            | TyTerm::Order
            | TyTerm::Str
            | TyTerm::Error(_)
            | TyTerm::Var(_) => {}
            TyTerm::Array(inner, _)
            | TyTerm::Slice(inner)
            | TyTerm::Handle(inner)
            | TyTerm::Option(inner) => rewrite(inner),
            TyTerm::Ref(_, inner) => inner.rewrite_types(rewrite),
            TyTerm::Result(ok, err) => {
                rewrite(ok);
                rewrite(err);
            }
            TyTerm::Tuple(elems) => elems.iter_mut().for_each(rewrite),
            TyTerm::Object(object) => object.fields.values_mut().for_each(rewrite),
            TyTerm::Enum { variants, .. } => variants
                .values_mut()
                .flatten()
                .for_each(|payload| rewrite(payload)),
            TyTerm::Fn {
                params,
                ret,
                captures,
                ..
            } => {
                for param in params {
                    rewrite(&mut param.ty);
                }
                rewrite(ret);
                captures.iter_mut().for_each(rewrite);
            }
            TyTerm::UserDefined { type_args, .. } => {
                for arg in type_args {
                    arg.rewrite_types(rewrite);
                }
            }
        }
    }

    /// The variable a structural term was resolved from.
    pub fn home(&self) -> Option<V::TyVar> {
        match self {
            TyTerm::Object(object) => object.home.var(),
            TyTerm::Enum { home, .. } => home.var(),
            _ => None,
        }
    }

    pub fn set_home(&mut self, to: Home<V>) {
        match self {
            TyTerm::Object(object) => object.home = to,
            TyTerm::Enum { home, .. } => *home = to,
            _ => {}
        }
    }

    pub fn mentions_error(&self) -> bool {
        matches!(self, TyTerm::Error(_)) || self.children().iter().any(|c| c.mentions_error())
    }

    pub fn for_each_source(&self, on_source: &mut impl FnMut(IdentityId)) {
        self.for_each_identity(&mut |identity| {
            if let IdentityTerm::Known(id) = identity {
                on_source(*id);
            }
        });
    }

    /// Every identity argument in the type, in `for_each_source`'s order.
    pub fn for_each_identity(&self, on_identity: &mut impl FnMut(&IdentityTerm<V>)) {
        match self {
            TyTerm::Int(_)
            | TyTerm::Float
            | TyTerm::Char
            | TyTerm::String
            | TyTerm::Bool
            | TyTerm::Unit
            | TyTerm::Never
            | TyTerm::Order
            | TyTerm::Str
            | TyTerm::Error(_)
            | TyTerm::Var(_) => {}
            TyTerm::Array(inner, _) => inner.for_each_identity(on_identity),
            TyTerm::Slice(elem) => elem.for_each_identity(on_identity),
            TyTerm::Handle(inner) => inner.for_each_identity(on_identity),
            TyTerm::Option(inner) => inner.for_each_identity(on_identity),
            TyTerm::Ref(_, inner) => inner.ty().for_each_identity(on_identity),
            TyTerm::Result(ok, err) => {
                ok.for_each_identity(on_identity);
                err.for_each_identity(on_identity);
            }
            TyTerm::Tuple(elems) => {
                for elem in elems {
                    elem.for_each_identity(on_identity);
                }
            }
            TyTerm::Object(object) => {
                let mut fields: Vec<_> = object.iter().collect();
                fields.sort_by_key(|(name, _)| **name);
                for (_, ty) in fields {
                    ty.for_each_identity(on_identity);
                }
            }
            TyTerm::Enum { variants, .. } => {
                let mut tags: Vec<_> = variants.iter().collect();
                tags.sort_by_key(|(tag, _)| **tag);
                for (_, payload) in tags {
                    if let Some(ty) = payload {
                        ty.for_each_identity(on_identity);
                    }
                }
            }
            TyTerm::Fn {
                params,
                ret,
                captures,
                ..
            } => {
                for param in params {
                    param.ty.for_each_identity(on_identity);
                }
                ret.for_each_identity(on_identity);
                for capture in captures {
                    capture.for_each_identity(on_identity);
                }
            }
            TyTerm::UserDefined {
                type_args,
                identity_args,
                ..
            } => {
                for arg in type_args {
                    arg.ty().for_each_identity(on_identity);
                }
                for arg in identity_args {
                    on_identity(arg);
                }
            }
        }
    }

    /// Equality with identity arguments ignored. Two values minted from
    /// two sources have the same type and different identities, which is
    /// what `MirErrorKind::OneTypeTwoSources` reports: the funnel asks
    /// this, and `TyDisplay` -- which drops identity arguments -- stays a
    /// rendering.
    pub fn same_erased(&self, other: &Self) -> bool
    where
        V: PartialEq,
    {
        match (self, other) {
            (TyTerm::Int(a), TyTerm::Int(b)) => a == b,
            (TyTerm::Float, TyTerm::Float)
            | (TyTerm::Char, TyTerm::Char)
            | (TyTerm::String, TyTerm::String)
            | (TyTerm::Bool, TyTerm::Bool)
            | (TyTerm::Unit, TyTerm::Unit)
            | (TyTerm::Never, TyTerm::Never)
            | (TyTerm::Order, TyTerm::Order)
            | (TyTerm::Str, TyTerm::Str) => true,
            (TyTerm::Error(a), TyTerm::Error(b)) => a == b,
            (TyTerm::Var(a), TyTerm::Var(b)) => a == b,
            (TyTerm::Array(a, an), TyTerm::Array(b, bn)) => an == bn && a.same_erased(b),
            (TyTerm::Slice(a), TyTerm::Slice(b))
            | (TyTerm::Option(a), TyTerm::Option(b))
            | (TyTerm::Handle(a), TyTerm::Handle(b)) => a.same_erased(b),
            (TyTerm::Result(a_ok, a_err), TyTerm::Result(b_ok, b_err)) => {
                a_ok.same_erased(b_ok) && a_err.same_erased(b_err)
            }
            (TyTerm::Tuple(a), TyTerm::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same_erased(y))
            }
            (TyTerm::Ref(a_mut, a), TyTerm::Ref(b_mut, b)) => {
                a_mut == b_mut && a.same_by(b, &mut |x, y| x.same_erased(y))
            }
            (TyTerm::Object(a), TyTerm::Object(b)) => {
                a.set == b.set
                    && a.fields.len() == b.fields.len()
                    && a.fields.iter().all(|(name, ty)| {
                        b.fields
                            .get(name)
                            .is_some_and(|other| ty.same_erased(other))
                    })
            }
            (
                TyTerm::Enum {
                    name: a_name,
                    variants: a,
                    ..
                },
                TyTerm::Enum {
                    name: b_name,
                    variants: b,
                    ..
                },
            ) => {
                a_name == b_name
                    && a.len() == b.len()
                    && a.iter().all(|(tag, payload)| {
                        b.get(tag).is_some_and(|other| match (payload, other) {
                            (Some(x), Some(y)) => x.same_erased(y),
                            (None, None) => true,
                            _ => false,
                        })
                    })
            }
            (
                TyTerm::Fn {
                    params: a_params,
                    ret: a_ret,
                    captures: a_caps,
                    effect: a_effect,
                },
                TyTerm::Fn {
                    params: b_params,
                    ret: b_ret,
                    captures: b_caps,
                    effect: b_effect,
                },
            ) => {
                a_effect == b_effect
                    && a_ret.same_erased(b_ret)
                    && a_params.len() == b_params.len()
                    && a_params
                        .iter()
                        .zip(b_params)
                        .all(|(x, y)| x.name == y.name && x.ty.same_erased(&y.ty))
                    && a_caps.len() == b_caps.len()
                    && a_caps.iter().zip(b_caps).all(|(x, y)| x.same_erased(y))
            }
            (
                TyTerm::UserDefined {
                    id: a_id,
                    type_args: a_args,
                    effect_args: a_effects,
                    identity_args: _,
                    region_params: a_regions,
                },
                TyTerm::UserDefined {
                    id: b_id,
                    type_args: b_args,
                    effect_args: b_effects,
                    identity_args: _,
                    region_params: b_regions,
                },
            ) => {
                a_id == b_id
                    && a_regions == b_regions
                    && a_effects == b_effects
                    && a_args.len() == b_args.len()
                    && a_args
                        .iter()
                        .zip(b_args)
                        .all(|(x, y)| x.same_by(y, &mut |x, y| x.same_erased(y)))
            }
            (TyTerm::Int(_), _)
            | (TyTerm::Float, _)
            | (TyTerm::Char, _)
            | (TyTerm::String, _)
            | (TyTerm::Bool, _)
            | (TyTerm::Unit, _)
            | (TyTerm::Never, _)
            | (TyTerm::Order, _)
            | (TyTerm::Str, _)
            | (TyTerm::Error(_), _)
            | (TyTerm::Var(_), _)
            | (TyTerm::Array(..), _)
            | (TyTerm::Slice(_), _)
            | (TyTerm::Option(_), _)
            | (TyTerm::Handle(_), _)
            | (TyTerm::Result(..), _)
            | (TyTerm::Tuple(_), _)
            | (TyTerm::Ref(..), _)
            | (TyTerm::Object(_), _)
            | (TyTerm::Enum { .. }, _)
            | (TyTerm::Fn { .. }, _)
            | (TyTerm::UserDefined { .. }, _) => false,
        }
    }

    /// Map this type term from phase `V` to phase `W`.
    ///
    /// Structural recursion is automatic - only variable slots and
    /// identity slots need custom handling via the provided closures.
    pub fn map<W: Phase>(
        &self,
        on_var: &mut impl FnMut(V::TyVar) -> TyTerm<W>,
        on_identity: &mut impl FnMut(V::IdentityVar) -> IdentityTerm<W>,
        on_effect: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>,
        on_len: &mut impl FnMut(V::LenVar) -> LenTerm<W>,
        on_repr: &mut impl FnMut(V::ReprVar) -> Repr<W>,
    ) -> TyTerm<W> {
        match self {
            TyTerm::Int(k) => TyTerm::Int(*k),
            TyTerm::Float => TyTerm::Float,
            TyTerm::Char => TyTerm::Char,
            TyTerm::String => TyTerm::String,
            TyTerm::Bool => TyTerm::Bool,
            TyTerm::Unit => TyTerm::Unit,
            TyTerm::Never => TyTerm::Never,
            TyTerm::Order => TyTerm::Order,
            TyTerm::Array(inner, len) => TyTerm::Array(
                Box::new(inner.map(on_var, on_identity, on_effect, on_len, on_repr)),
                len.map(on_len),
            ),
            TyTerm::Slice(elem) => TyTerm::Slice(Box::new(elem.map(
                on_var,
                on_identity,
                on_effect,
                on_len,
                on_repr,
            ))),
            TyTerm::Str => TyTerm::Str,
            TyTerm::Object(object) => TyTerm::Object(
                object.with_fields(
                    object
                        .iter()
                        .map(|(k, v)| (*k, v.map(on_var, on_identity, on_effect, on_len, on_repr)))
                        .collect(),
                ),
            ),
            TyTerm::Tuple(elems) => TyTerm::Tuple(
                elems
                    .iter()
                    .map(|e| e.map(on_var, on_identity, on_effect, on_len, on_repr))
                    .collect(),
            ),
            TyTerm::Option(inner) => TyTerm::Option(Box::new(inner.map(
                on_var,
                on_identity,
                on_effect,
                on_len,
                on_repr,
            ))),
            TyTerm::Result(ok, err) => TyTerm::Result(
                Box::new(ok.map(on_var, on_identity, on_effect, on_len, on_repr)),
                Box::new(err.map(on_var, on_identity, on_effect, on_len, on_repr)),
            ),
            TyTerm::Fn {
                params,
                ret,
                captures,
                effect,
            } => TyTerm::Fn {
                params: params
                    .iter()
                    .map(|p| p.retyped(p.ty.map(on_var, on_identity, on_effect, on_len, on_repr)))
                    .collect(),
                ret: Box::new(ret.map(on_var, on_identity, on_effect, on_len, on_repr)),
                captures: captures
                    .iter()
                    .map(|c| c.map(on_var, on_identity, on_effect, on_len, on_repr))
                    .collect(),
                effect: effect.map(on_effect),
            },
            TyTerm::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
                region_params,
            } => TyTerm::UserDefined {
                id: *id,
                type_args: type_args
                    .iter()
                    .map(|t| t.map(on_var, on_identity, on_effect, on_len, on_repr))
                    .collect(),
                effect_args: effect_args.iter().map(|e| e.map(on_effect)).collect(),
                identity_args: identity_args.iter().map(|i| i.map(on_identity)).collect(),
                region_params: *region_params,
            },
            TyTerm::Enum { name, variants, .. } => TyTerm::Enum {
                name: *name,
                variants: variants
                    .iter()
                    .map(|(tag, payload)| {
                        (
                            *tag,
                            payload.as_ref().map(|ty| {
                                Box::new(ty.map(on_var, on_identity, on_effect, on_len, on_repr))
                            }),
                        )
                    })
                    .collect(),
                home: Home::NONE,
            },
            TyTerm::Handle(inner) => TyTerm::Handle(Box::new(inner.map(
                on_var,
                on_identity,
                on_effect,
                on_len,
                on_repr,
            ))),
            TyTerm::Ref(m, inner) => TyTerm::Ref(
                *m,
                Box::new(inner.map(on_var, on_identity, on_effect, on_len, on_repr)),
            ),
            TyTerm::Error(token) => TyTerm::Error(*token),
            TyTerm::Var(v) => on_var(*v),
        }
    }

    /// Fallible version of `map` - short-circuits on first error.
    pub fn try_map<W: Phase, E>(
        &self,
        on_var: &mut impl FnMut(V::TyVar) -> Result<TyTerm<W>, E>,
        on_identity: &mut impl FnMut(V::IdentityVar) -> Result<IdentityTerm<W>, E>,
        on_effect: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
        on_len: &mut impl FnMut(V::LenVar) -> Result<LenTerm<W>, E>,
        on_repr: &mut impl FnMut(V::ReprVar) -> Result<Repr<W>, E>,
    ) -> Result<TyTerm<W>, E> {
        match self {
            TyTerm::Int(k) => Ok(TyTerm::Int(*k)),
            TyTerm::Float => Ok(TyTerm::Float),
            TyTerm::Char => Ok(TyTerm::Char),
            TyTerm::String => Ok(TyTerm::String),
            TyTerm::Bool => Ok(TyTerm::Bool),
            TyTerm::Unit => Ok(TyTerm::Unit),
            TyTerm::Never => Ok(TyTerm::Never),
            TyTerm::Order => Ok(TyTerm::Order),
            TyTerm::Array(inner, len) => Ok(TyTerm::Array(
                Box::new(inner.try_map(on_var, on_identity, on_effect, on_len, on_repr)?),
                len.try_map(on_len)?,
            )),
            TyTerm::Slice(elem) => Ok(TyTerm::Slice(Box::new(elem.try_map(
                on_var,
                on_identity,
                on_effect,
                on_len,
                on_repr,
            )?))),
            TyTerm::Str => Ok(TyTerm::Str),
            TyTerm::Object(object) => {
                let mapped: Result<FxHashMap<_, _>, E> = object
                    .iter()
                    .map(|(k, v)| {
                        v.try_map(on_var, on_identity, on_effect, on_len, on_repr)
                            .map(|mv| (*k, mv))
                    })
                    .collect();
                Ok(TyTerm::Object(object.with_fields(mapped?)))
            }
            TyTerm::Tuple(elems) => Ok(TyTerm::Tuple(
                elems
                    .iter()
                    .map(|e| e.try_map(on_var, on_identity, on_effect, on_len, on_repr))
                    .collect::<Result<_, _>>()?,
            )),
            TyTerm::Option(inner) => Ok(TyTerm::Option(Box::new(inner.try_map(
                on_var,
                on_identity,
                on_effect,
                on_len,
                on_repr,
            )?))),
            TyTerm::Result(ok, err) => Ok(TyTerm::Result(
                Box::new(ok.try_map(on_var, on_identity, on_effect, on_len, on_repr)?),
                Box::new(err.try_map(on_var, on_identity, on_effect, on_len, on_repr)?),
            )),
            TyTerm::Fn {
                params,
                ret,
                captures,
                effect,
            } => Ok(TyTerm::Fn {
                params: params
                    .iter()
                    .map(|p| {
                        p.ty.try_map(on_var, on_identity, on_effect, on_len, on_repr)
                            .map(|ty| p.retyped(ty))
                    })
                    .collect::<Result<_, _>>()?,
                ret: Box::new(ret.try_map(on_var, on_identity, on_effect, on_len, on_repr)?),
                captures: captures
                    .iter()
                    .map(|c| c.try_map(on_var, on_identity, on_effect, on_len, on_repr))
                    .collect::<Result<_, _>>()?,
                effect: effect.try_map(on_effect)?,
            }),
            TyTerm::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
                region_params,
            } => Ok(TyTerm::UserDefined {
                region_params: *region_params,
                id: *id,
                type_args: type_args
                    .iter()
                    .map(|t| t.try_map(on_var, on_identity, on_effect, on_len, on_repr))
                    .collect::<Result<_, _>>()?,
                effect_args: effect_args
                    .iter()
                    .map(|e| e.try_map(on_effect))
                    .collect::<Result<_, _>>()?,
                identity_args: identity_args
                    .iter()
                    .map(|i| i.try_map(on_identity))
                    .collect::<Result<_, _>>()?,
            }),
            TyTerm::Enum { name, variants, .. } => {
                let mapped: Result<FxHashMap<_, _>, E> = variants
                    .iter()
                    .map(|(tag, payload)| {
                        let mp = match payload {
                            Some(ty) => Some(Box::new(ty.try_map(
                                on_var,
                                on_identity,
                                on_effect,
                                on_len,
                                on_repr,
                            )?)),
                            None => None,
                        };
                        Ok((*tag, mp))
                    })
                    .collect();
                Ok(TyTerm::Enum {
                    name: *name,
                    variants: mapped?,
                    home: Home::NONE,
                })
            }
            TyTerm::Handle(inner) => Ok(TyTerm::Handle(Box::new(inner.try_map(
                on_var,
                on_identity,
                on_effect,
                on_len,
                on_repr,
            )?))),
            TyTerm::Ref(m, inner) => Ok(TyTerm::Ref(
                *m,
                Box::new(inner.try_map(on_var, on_identity, on_effect, on_len, on_repr)?),
            )),
            TyTerm::Error(token) => Ok(TyTerm::Error(*token)),
            TyTerm::Var(v) => on_var(*v),
        }
    }
}

// -- Lift: Concrete -> any Phase --------------------------------------

/// Lift a concrete `Ty` into any phase (mechanical, zero information change).
/// Infallible because `Concrete` has `TyVar = Infallible` (uninhabitable).
pub fn lift_ty<W: Phase>(ty: &Ty) -> TyTerm<W> {
    ty.map(
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
    )
}

/// Lift a concrete argument into any phase.
pub fn lift_arg<W: Phase>(arg: &TypeArg<Concrete>) -> TypeArg<W> {
    arg.map(
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
    )
}

/// Alias: lift `Ty` to `PolyTy`. Kept for call-site compatibility.
pub fn lift_to_poly(ty: &Ty) -> PolyTy {
    lift_ty(ty)
}

/// Lift a concrete type as a declaration: a host that declares a context
/// or a parameter names no source, so every identity argument becomes a
/// variable the compilation mints a source for (RFC-0012). A source
/// number in a `Ty` is meaningful only inside the compilation that
/// minted it, never in a declaration.
pub fn lift_declaration(ty: &Ty, builder: &mut PolyBuilder) -> PolyTy {
    fn arg(a: &TypeArg<Concrete>, builder: &mut PolyBuilder) -> TypeArg<Poly> {
        match a {
            TypeArg::Uniform(ty) => TypeArg::Uniform(go(ty, builder)),
            TypeArg::Open(v, _) => match *v {},
            TypeArg::Specialized(held) => TypeArg::Specialized(held_arg(held, builder)),
        }
    }
    fn held_arg(held: &HeldTy<Concrete>, builder: &mut PolyBuilder) -> HeldTy<Poly> {
        match held {
            HeldTy::Tuple(parts) => HeldTy::Tuple(parts.iter().map(|p| arg(p, builder)).collect()),
            HeldTy::Option(p) => HeldTy::Option(Box::new(arg(p, builder))),
            HeldTy::Result(ok, err) => {
                let ok = Box::new(arg(ok, builder));
                HeldTy::Result(ok, Box::new(arg(err, builder)))
            }
            HeldTy::Array(p, len) => HeldTy::Array(Box::new(arg(p, builder)), lift_ty_len(len)),
            HeldTy::RustArray(p, len) => HeldTy::RustArray(Box::new(arg(p, builder)), *len),
            HeldTy::Leaf(leaf) => HeldTy::of(go(leaf.ty(), builder)),
            HeldTy::Held(v) => match *v {},
        }
    }
    fn effect_arg(a: &EffectArg<Concrete>) -> EffectArg<Poly> {
        a.map(&mut |v: Infallible| match v {})
    }
    fn go(ty: &Ty, builder: &mut PolyBuilder) -> PolyTy {
        match ty {
            Ty::Int(k) => TyTerm::Int(*k),
            Ty::Float => TyTerm::Float,
            Ty::Char => TyTerm::Char,
            Ty::String => TyTerm::String,
            Ty::Bool => TyTerm::Bool,
            Ty::Unit => TyTerm::Unit,
            Ty::Never => TyTerm::Never,
            Ty::Order => TyTerm::Order,
            Ty::Array(inner, len) => TyTerm::Array(Box::new(go(inner, builder)), lift_ty_len(len)),
            Ty::Slice(elem) => TyTerm::Slice(Box::new(go(elem, builder))),
            Ty::Str => TyTerm::Str,
            Ty::Object(object) => TyTerm::Object(
                object.with_fields(object.iter().map(|(k, v)| (*k, go(v, builder))).collect()),
            ),
            Ty::Tuple(elems) => TyTerm::Tuple(elems.iter().map(|e| go(e, builder)).collect()),
            Ty::Option(inner) => TyTerm::Option(Box::new(go(inner, builder))),
            Ty::Result(ok, err) => {
                TyTerm::Result(Box::new(go(ok, builder)), Box::new(go(err, builder)))
            }
            Ty::Fn {
                params,
                ret,
                captures,
                effect,
            } => TyTerm::Fn {
                params: params
                    .iter()
                    .map(|p| ParamTerm {
                        name: p.name,
                        ty: go(&p.ty, builder),
                    })
                    .collect(),
                ret: Box::new(go(ret, builder)),
                captures: captures.iter().map(|c| go(c, builder)).collect(),
                effect: lift_ty_effect(effect),
            },
            Ty::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
                region_params,
            } => TyTerm::UserDefined {
                id: *id,
                region_params: *region_params,
                type_args: type_args.iter().map(|t| arg(t, builder)).collect(),
                effect_args: effect_args.iter().map(effect_arg).collect(),
                identity_args: identity_args
                    .iter()
                    .map(|_| builder.fresh_identity_var())
                    .collect(),
            },
            Ty::Enum { name, variants, .. } => TyTerm::Enum {
                name: *name,
                variants: variants
                    .iter()
                    .map(|(k, v)| (*k, v.as_ref().map(|t| Box::new(go(t, builder)))))
                    .collect(),
                home: Home::NONE,
            },
            Ty::Handle(inner) => TyTerm::Handle(Box::new(go(inner, builder))),
            Ty::Ref(m, inner) => TyTerm::Ref(*m, Box::new(arg(inner, builder))),
            Ty::Error(token) => TyTerm::Error(*token),
            Ty::Var(v) => match *v {},
        }
    }
    go(ty, builder)
}

fn lift_ty_len(len: &LenTerm<Concrete>) -> LenTerm<Poly> {
    len.map(&mut |v: Infallible| match v {})
}

fn lift_ty_effect(effect: &EffectTerm<Concrete>) -> EffectTerm<Poly> {
    effect.map(&mut |v: Infallible| match v {})
}

/// Try to convert a `PolyTy` to a concrete `Ty`.
/// Returns `None` if the poly type contains any Var placeholders.
pub fn try_freeze_poly(ty: &PolyTy) -> Option<Ty> {
    ty.try_map(
        &mut |_: u32| Err(()),
        &mut |_: u32| Err(()),
        &mut |_: u32| Err(()),
        &mut |_: u32| Err(()),
        &mut |_: u32| Err(()),
    )
    .ok()
}

// -- PolyBuilder -----------------------------------------------------

/// Builder for polymorphic type templates. No Solver dependency.
/// Creates positional placeholders (Var(0), Var(1), ...) for type variables.
pub struct PolyBuilder {
    next_ty: u32,
    next_effect: u32,
    next_len: u32,
    next_identity: u32,
}

impl PolyBuilder {
    pub fn new() -> Self {
        Self {
            next_ty: 0,
            next_effect: 0,
            next_len: 0,
            next_identity: 0,
        }
    }

    pub fn fresh_len_var(&mut self) -> LenTerm<Poly> {
        let id = self.next_len;
        self.next_len += 1;
        LenTerm::Var(id)
    }

    pub fn fresh_effect_var(&mut self) -> EffectTerm<Poly> {
        let id = self.next_effect;
        self.next_effect += 1;
        EffectTerm::Var(id)
    }

    /// Create a fresh type placeholder.
    pub fn fresh_ty_var(&mut self) -> PolyTy {
        let id = self.next_ty;
        self.next_ty += 1;
        TyTerm::Var(id)
    }

    pub fn fresh_identity_var(&mut self) -> IdentityTerm<Poly> {
        let id = self.next_identity;
        self.next_identity += 1;
        IdentityTerm::Var(id)
    }
}

impl Default for PolyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::QualifiedRef;
    use crate::solver::{Answer, Conversion, Decision};
    use acvus_utils::Interner;

    #[test]
    fn a_pure_effect_commutes_by_definition() {
        assert_eq!(Effect::new(Reissue::Pure, false), Effect::PURE);
        assert!(Effect::PURE.commutes);
        assert_eq!(Effect::PURE.to_string(), "Pure");
        assert_eq!(
            Effect::IDEMPOTENT.commutative().to_string(),
            "Idempotent+commutative"
        );
    }

    #[test]
    fn effects_join_on_both_axes() {
        let c = Effect::IDEMPOTENT.commutative();
        assert_eq!(c.join(&Effect::IDEMPOTENT), Effect::IDEMPOTENT);
        assert_eq!(c.join(&Effect::PURE), c);
        assert_eq!(
            c.join(&Effect::OPAQUE.commutative()),
            Effect::OPAQUE.commutative()
        );
        assert_eq!(Effect::OPAQUE.meet(&c), c);
    }

    #[test]
    fn the_product_order_leaves_the_axes_incomparable() {
        let c = Effect::OPAQUE.commutative();
        assert!(Effect::PURE.at_most(&c));
        assert!(c.at_most(&Effect::OPAQUE));
        assert!(!Effect::IDEMPOTENT.at_most(&c));
        assert!(!c.at_most(&Effect::IDEMPOTENT));
        assert_eq!(Effect::IDEMPOTENT.partial_cmp(&c), None);
    }

    fn arr<V: Phase>(elem: TyTerm<V>, n: usize) -> TyTerm<V> {
        TyTerm::Array(Box::new(elem), LenTerm::Known(n))
    }

    /// Test helper: create a unique `QualifiedRef` for each call.
    /// Uses a thread-local counter to ensure uniqueness across tests.
    fn fresh_qref() -> QualifiedRef {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        thread_local! {
            static INTERNER: Interner = Interner::new();
        }
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        INTERNER.with(|i| QualifiedRef::root(i.intern(&format!("TestType{n}"))))
    }

    /// Test helper: create a `TyTerm::UserDefined` with a fresh id and no type args.
    fn test_user_defined() -> Ty {
        TyTerm::UserDefined {
            id: fresh_qref(),
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        }
    }

    #[test]
    fn unify_same_concrete() {
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        assert!(s.unify(&TyTerm::I64, &TyTerm::I64).is_ok());
        assert!(s.unify(&TyTerm::Float, &TyTerm::Float).is_ok());
        assert!(s.unify(&TyTerm::String, &TyTerm::String).is_ok());
        assert!(s.unify(&TyTerm::Bool, &TyTerm::Bool).is_ok());
        assert!(s.unify(&TyTerm::Unit, &TyTerm::Unit).is_ok());
    }

    #[test]
    fn unify_different_concrete_fails() {
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        assert!(s.unify(&TyTerm::I64, &TyTerm::Float).is_err());
        assert!(s.unify(&TyTerm::String, &TyTerm::Bool).is_err());
    }

    #[test]
    fn unify_var_with_concrete() {
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let t = s.fresh_ty_var();
        assert!(s.unify(&t, &TyTerm::I64).is_ok());
        assert_eq!(s.resolve_ty(&t), TyTerm::I64);
    }

    #[test]
    fn unify_object() {
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let interner = Interner::new();
        let t = s.fresh_ty_var();
        let obj1 = TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([
            (interner.intern("name"), TyTerm::String),
            (interner.intern("age"), t.clone()),
        ])));
        let obj2 = TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([
            (interner.intern("name"), TyTerm::String),
            (interner.intern("age"), TyTerm::I64),
        ])));
        assert!(s.unify(&obj1, &obj2).is_ok());
        assert_eq!(s.resolve_ty(&t), TyTerm::I64);
    }

    #[test]
    fn two_objects_join_to_the_union_of_their_fields() {
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let interner = Interner::new();
        let obj1 = TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([(
            interner.intern("name"),
            TyTerm::String,
        )])));
        let obj2 = TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([(
            interner.intern("age"),
            TyTerm::I64,
        )])));
        let (obj1, obj2) = (s.construct(obj1), s.construct(obj2));
        let home = s.fresh_ty_var();
        assert!(s.unify(&obj1, &home).is_ok());
        assert!(s.unify(&obj2, &home).is_ok());
        assert_eq!(
            s.resolve_ty(&home),
            TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([
                (interner.intern("name"), TyTerm::String),
                (interner.intern("age"), TyTerm::I64),
            ])))
        );
    }

    #[test]
    fn transitive_resolution() {
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let t1 = s.fresh_ty_var();
        let t2 = s.fresh_ty_var();
        assert!(s.unify(&t1, &t2).is_ok());
        assert!(s.unify(&t2, &TyTerm::String).is_ok());
        assert_eq!(s.resolve_ty(&t1), TyTerm::String);
    }

    // -- Object merge tests --

    /// The join of an object type with a declared struct's (RFC-0042).
    fn meet_of(
        i: &Interner,
        a: ObjectTy<Concrete>,
        b: ObjectTy<Concrete>,
    ) -> Result<Vec<String>, String> {
        let named = |name: Astr, field: Astr| format!("{} {}", i.resolve(name), i.resolve(field));
        match ObjectTy::meet(&a, &b) {
            ObjectMeet::Joined { ty, .. } => {
                let mut fields: Vec<String> =
                    ty.keys().map(|k| i.resolve(*k).to_string()).collect();
                fields.sort();
                Ok(fields)
            }
            ObjectMeet::Lacks { declared, field } => {
                Err(format!("lacks {}", named(declared.name, field)))
            }
            ObjectMeet::Undeclared { declared, field } => {
                Err(format!("undeclared {}", named(declared.name, field)))
            }
            ObjectMeet::TooWide { fields } => Err(format!("too wide {fields}")),
            ObjectMeet::TwoDeclarations { a, b } => {
                Err(format!("two {} {}", i.resolve(a.name), i.resolve(b.name)))
            }
        }
    }

    fn fields_of(i: &Interner, names: &[&str]) -> FxHashMap<Astr, Ty> {
        names.iter().map(|n| (i.intern(n), Ty::I64)).collect()
    }

    #[test]
    fn two_written_objects_join_to_the_union_of_their_fields() {
        let i = Interner::new();
        assert_eq!(
            meet_of(
                &i,
                ObjectTy::written(fields_of(&i, &["a"])),
                ObjectTy::written(fields_of(&i, &["b"])),
            ),
            Ok(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn a_written_object_lacking_a_declared_field_is_refused_by_its_name() {
        let i = Interner::new();
        let flags = ObjectTy::declared(
            QualifiedRef::root(i.intern("Flags")),
            fields_of(&i, &["a", "b"]),
        );
        assert_eq!(
            meet_of(&i, flags, ObjectTy::written(fields_of(&i, &["a"]))),
            Err("lacks Flags b".to_string())
        );
    }

    #[test]
    fn an_object_carrying_a_field_no_declaration_names_is_refused_by_its_name() {
        let i = Interner::new();
        let flags = ObjectTy::declared(
            QualifiedRef::root(i.intern("Flags")),
            fields_of(&i, &["a", "b"]),
        );
        assert_eq!(
            meet_of(
                &i,
                flags,
                ObjectTy::written(fields_of(&i, &["a", "b", "c"]))
            ),
            Err("undeclared Flags c".to_string())
        );
    }

    #[test]
    fn the_join_of_a_declaration_and_a_written_object_of_its_fields_is_the_declared_type() {
        let i = Interner::new();
        let flags = ObjectTy::declared(
            QualifiedRef::root(i.intern("Flags")),
            fields_of(&i, &["a", "b"]),
        );
        let written = ObjectTy::written(fields_of(&i, &["a", "b"]));
        let ObjectMeet::Joined { ty } = ObjectTy::meet(&flags, &written) else {
            panic!("the field sets agree")
        };
        assert_eq!(
            ty.declaration(),
            Some(QualifiedRef::root(i.intern("Flags")))
        );
    }

    /// Reading a field asks the object for it; the object a declaration
    /// fixed answers with its own type.
    #[test]
    fn asking_a_declared_object_for_one_of_its_fields_joins_to_the_declared_type() {
        let i = Interner::new();
        let flags = ObjectTy::declared(
            QualifiedRef::root(i.intern("Flags")),
            fields_of(&i, &["a", "b"]),
        );
        let ObjectMeet::Joined { ty } =
            ObjectTy::meet(&flags, &ObjectTy::at_least(fields_of(&i, &["a"])))
        else {
            panic!("a read of `a` is within the declaration")
        };
        assert_eq!(
            ty.declaration(),
            Some(QualifiedRef::root(i.intern("Flags")))
        );
    }

    #[test]
    fn asking_a_declared_object_for_a_field_it_does_not_have_is_refused() {
        let i = Interner::new();
        let flags = ObjectTy::declared(
            QualifiedRef::root(i.intern("Flags")),
            fields_of(&i, &["a", "b"]),
        );
        assert_eq!(
            meet_of(&i, flags, ObjectTy::at_least(fields_of(&i, &["c"]))),
            Err("undeclared Flags c".to_string())
        );
    }

    #[test]
    fn a_value_has_the_fields_of_one_declared_struct_and_of_no_other() {
        let i = Interner::new();
        assert_eq!(
            meet_of(
                &i,
                ObjectTy::declared(QualifiedRef::root(i.intern("Flags")), fields_of(&i, &["a"])),
                ObjectTy::declared(QualifiedRef::root(i.intern("Other")), fields_of(&i, &["a"])),
            ),
            Err("two Flags Other".to_string())
        );
    }

    /// A declaration's name is `ns::name`: two structs of one name in two
    /// namespaces are two declarations, fields alike or not.
    #[test]
    fn two_declarations_of_one_name_in_two_namespaces_are_two() {
        let i = Interner::new();
        let at = |ns: &str| QualifiedRef::qualified(i.intern(ns), i.intern("P"));
        assert_eq!(
            meet_of(
                &i,
                ObjectTy::declared(at("a"), fields_of(&i, &["x"])),
                ObjectTy::declared(at("b"), fields_of(&i, &["x"])),
            ),
            Err("two P P".to_string())
        );
    }

    #[test]
    fn a_resolved_copy_of_a_construction_is_the_construction() {
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let i = Interner::new();
        let built = s.construct(TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("a"),
            TyTerm::I64,
        )]))));
        let copy = s.resolve_ty(&built);
        let store = TyTerm::Object(ObjectTy::at_least(FxHashMap::from_iter([(
            i.intern("b"),
            TyTerm::I64,
        )])));
        assert!(
            s.unify(&copy, &store).is_ok(),
            "the copy grows as the construction"
        );
        let TyTerm::Object(grown) = s.resolve_ty(&built) else {
            panic!("an object")
        };
        assert!(grown.contains_key(&i.intern("b")), "{grown:?}");
    }

    #[test]
    fn unify_object_disjoint_via_var() {
        // Var -> {a} then Var -> {b} should merge to {a, b}
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let i = Interner::new();
        let v = s.fresh_ty_var();
        let obj_a = TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("a"),
            TyTerm::I64,
        )])));
        let obj_b = TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("b"),
            TyTerm::String,
        )])));
        let (obj_a, obj_b) = (s.construct(obj_a), s.construct(obj_b));
        assert!(s.unify(&v, &obj_a).is_ok());
        assert!(s.unify(&v, &obj_b).is_ok());
        let resolved = s.resolve_ty(&v);
        match &resolved {
            TyTerm::Object(fields) => {
                assert_eq!(fields.len(), 2, "expected {{a, b}}, got {fields:?}");
                assert_eq!(fields.get(&i.intern("a")), Some(&TyTerm::I64));
                assert_eq!(fields.get(&i.intern("b")), Some(&TyTerm::String));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn unify_object_overlapping_via_var() {
        // Var -> {a, b} then Var -> {b, c} should merge to {a, b, c}
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let i = Interner::new();
        let v = s.fresh_ty_var();
        let obj_ab = TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([
            (i.intern("a"), TyTerm::I64),
            (i.intern("b"), TyTerm::String),
        ])));
        let obj_bc = TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([
            (i.intern("b"), TyTerm::String),
            (i.intern("c"), TyTerm::Bool),
        ])));
        let (obj_ab, obj_bc) = (s.construct(obj_ab), s.construct(obj_bc));
        assert!(s.unify(&v, &obj_ab).is_ok());
        assert!(s.unify(&v, &obj_bc).is_ok());
        let resolved = s.resolve_ty(&v);
        match &resolved {
            TyTerm::Object(fields) => {
                assert_eq!(fields.len(), 3, "expected {{a, b, c}}, got {fields:?}");
                assert_eq!(fields.get(&i.intern("a")), Some(&TyTerm::I64));
                assert_eq!(fields.get(&i.intern("b")), Some(&TyTerm::String));
                assert_eq!(fields.get(&i.intern("c")), Some(&TyTerm::Bool));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn unify_object_overlap_type_conflict_fails() {
        // {b: Int} and {b: String} via same Var should fail
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let i = Interner::new();
        let v = s.fresh_ty_var();
        let obj1 = TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("b"),
            TyTerm::I64,
        )])));
        let obj2 = TyTerm::Object(ObjectTy::written(FxHashMap::from_iter([(
            i.intern("b"),
            TyTerm::String,
        )])));
        assert!(s.unify(&v, &obj1).is_ok());
        assert!(s.unify(&v, &obj2).is_err());
    }

    #[test]
    fn fresh_param_produces_unique_ids() {
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let o1 = s.fresh_ty_var();
        let o2 = s.fresh_ty_var();
        let o3 = s.fresh_ty_var();
        assert_ne!(o1, o2);
        assert_ne!(o2, o3);
        assert_ne!(o1, o3);
    }

    // -- Variance unsoundness edge case tests --

    // ================================================================
    // Var chain + coercion interaction
    // ================================================================

    // ================================================================
    // Occurs check + polarity
    // ================================================================

    #[test]
    fn occurs_check_through_list_covariant() {
        // Var = List<Var> should fail (occurs) regardless of polarity.
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let v = s.fresh_ty_var();
        let cyclic = arr(v.clone(), 3);
        assert!(s.unify(&v, &cyclic).is_err());
    }

    // ================================================================
    // Deep nesting coercion
    // ================================================================

    // ================================================================
    // Object merge + coercion at the same time
    // ================================================================

    // ================================================================
    // Snapshot/rollback isolation
    // ================================================================

    // ================================================================
    // Join symmetry
    // ================================================================

    #[test]
    fn invariant_same_types_both_directions() {
        // Same concrete type: Invariant must succeed regardless of order.
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let l1 = arr(TyTerm::I64, 3);
        let l2 = arr(TyTerm::I64, 3);
        assert!(s.unify(&l1, &l2).is_ok());
        assert!(s.unify(&l2, &l1).is_ok());
    }

    // ================================================================
    // Unresolved Var containers + coercion
    // ================================================================

    // ================================================================
    // Bidirectional Var binding + coercion
    // ================================================================

    // ================================================================
    // N-way demotion (large fan-out)
    // ================================================================

    // ================================================================
    // Mixed concrete/param identities
    // ================================================================

    // ================================================================
    // Error / Param + polarity (poison / unification absorption)
    // ================================================================

    // ================================================================
    // Transitive coercion chains
    // ================================================================

    // ================================================================
    // Inner type mismatch under coercion (must not be masked)
    // ================================================================

    // ================================================================
    // Coercion does NOT propagate across unrelated type constructors
    // ================================================================

    #[test]
    fn list_vs_tuple_fails_any_polarity() {
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let l = arr(TyTerm::I64, 3);
        let t = TyTerm::Tuple(vec![TyTerm::I64]);
        assert!(s.unify(&l, &t).is_err());
        assert!(s.unify(&l, &t).is_err());
    }

    // ================================================================
    // Triple flip (Fn<Fn<Fn<...>>>)
    // ================================================================

    // ================================================================
    // Regression: same identity must not trigger demotion
    // ================================================================

    // -- Sequence identity tracking ---------------------------------

    // -- UserDefined unification tests -------------------------------

    fn ud(id: QualifiedRef, type_args: Vec<InferTy>) -> InferTy {
        TyTerm::UserDefined {
            id,
            type_args: type_args.into_iter().map(TypeArg::uniform).collect(),
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        }
    }

    // -- Completeness: valid UserDefined unifications --

    #[test]
    fn user_defined_same_id_empty_args_unifies() {
        let mut sources = Sources::new();
        let id = fresh_qref();
        let mut registry = TypeRegistry::new();
        registry
            .register(UserDefinedDecl {
                qref: id,
                type_params: vec![TyVarBound::Any],
                effect_params: 0,
                identity_params: 0,
                region_params: 0,
                specializable: vec![false],
            })
            .expect("one declaration per name");
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        assert!(s.unify(&ud(id, vec![]), &ud(id, vec![])).is_ok());
    }

    #[test]
    fn user_defined_same_id_concrete_type_args_unifies() {
        let mut sources = Sources::new();
        let id = fresh_qref();
        let mut registry = TypeRegistry::new();
        registry
            .register(UserDefinedDecl {
                qref: id,
                type_params: vec![TyVarBound::Any],
                effect_params: 0,
                identity_params: 0,
                region_params: 0,
                specializable: vec![false],
            })
            .expect("one declaration per name");
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        assert!(
            s.unify(&ud(id, vec![TyTerm::I64]), &ud(id, vec![TyTerm::I64]))
                .is_ok()
        );
    }

    #[test]
    fn user_defined_param_type_arg_resolved_via_unify() {
        let mut sources = Sources::new();
        let id = fresh_qref();
        let mut registry = TypeRegistry::new();
        registry
            .register(UserDefinedDecl {
                qref: id,
                type_params: vec![TyVarBound::Any],
                effect_params: 0,
                identity_params: 0,
                region_params: 0,
                specializable: vec![false],
            })
            .expect("one declaration per name");
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let p = s.fresh_ty_var();
        assert!(
            s.unify(&ud(id, vec![p.clone()]), &ud(id, vec![TyTerm::I64]))
                .is_ok()
        );
        assert_eq!(s.resolve_ty(&p), TyTerm::I64);
    }

    #[test]
    fn user_defined_nested_type_arg_unifies() {
        // UserDefined<List<Param>> vs UserDefined<List<Int>> -> resolves Param to Int
        let mut sources = Sources::new();
        let id = fresh_qref();
        let mut registry = TypeRegistry::new();
        registry
            .register(UserDefinedDecl {
                qref: id,
                type_params: vec![TyVarBound::Any],
                effect_params: 0,
                identity_params: 0,
                region_params: 0,
                specializable: vec![false],
            })
            .expect("one declaration per name");
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let p = s.fresh_ty_var();
        assert!(
            s.unify(
                &ud(id, vec![arr(p.clone(), 3)]),
                &ud(id, vec![arr(TyTerm::I64, 3)])
            )
            .is_ok()
        );
        assert_eq!(s.resolve_ty(&p), TyTerm::I64);
    }

    // -- Soundness: invalid UserDefined unifications --

    #[test]
    fn user_defined_different_id_fails() {
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let id_a = fresh_qref();
        let id_b = fresh_qref();
        assert!(s.unify(&ud(id_a, vec![]), &ud(id_b, vec![])).is_err());
    }

    #[test]
    fn user_defined_type_arg_mismatch_fails() {
        let mut sources = Sources::new();
        let id = fresh_qref();
        let mut registry = TypeRegistry::new();
        registry
            .register(UserDefinedDecl {
                qref: id,
                type_params: vec![TyVarBound::Any],
                effect_params: 0,
                identity_params: 0,
                region_params: 0,
                specializable: vec![false],
            })
            .expect("one declaration per name");
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        assert!(
            s.unify(&ud(id, vec![TyTerm::I64]), &ud(id, vec![TyTerm::String]))
                .is_err()
        );
    }

    #[test]
    fn user_defined_vs_other_ty_fails() {
        let mut sources = Sources::new();
        let id = fresh_qref();
        let mut registry = TypeRegistry::new();
        registry
            .register(UserDefinedDecl {
                qref: id,
                type_params: vec![TyVarBound::Any],
                effect_params: 0,
                identity_params: 0,
                region_params: 0,
                specializable: vec![false],
            })
            .expect("one declaration per name");
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        assert!(s.unify(&ud(id, vec![]), &TyTerm::I64).is_err());
        assert!(s.unify(&TyTerm::String, &ud(id, vec![])).is_err());
    }

    // -- Resolve --

    #[test]
    fn user_defined_inside_list_resolves() {
        let mut sources = Sources::new();
        let id = fresh_qref();
        let mut registry = TypeRegistry::new();
        registry
            .register(UserDefinedDecl {
                qref: id,
                type_params: vec![TyVarBound::Any],
                effect_params: 0,
                identity_params: 0,
                region_params: 0,
                specializable: vec![false],
            })
            .expect("one declaration per name");
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);
        let p = s.fresh_ty_var();
        let ty = arr(ud(id, vec![p.clone()]), 3);
        assert!(s.unify(&p, &TyTerm::I64).is_ok());
        match s.resolve_ty(&ty) {
            TyTerm::Array(inner, _) => match *inner {
                TyTerm::UserDefined { type_args, .. } => {
                    assert_eq!(type_args, vec![TypeArg::uniform(TyTerm::I64)])
                }
                other => panic!("expected UserDefined, got {other:?}"),
            },
            other => panic!("expected Array, got {other:?}"),
        }
    }

    // -- TypeRegistry --

    #[test]
    fn type_registry_register_and_get() {
        let mut reg = TypeRegistry::new();
        let id = fresh_qref();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
            region_params: 0,
            specializable: vec![false],
        })
        .expect("one declaration per name");
        let decl = reg.get(id);
        assert_eq!(decl.qref, id);
        assert_eq!(decl.type_params.len(), 1);
    }

    #[test]
    fn type_registry_refuses_a_second_declaration_under_one_name() {
        let mut reg = TypeRegistry::new();
        let id = fresh_qref();
        let decl = || UserDefinedDecl {
            qref: id,
            type_params: vec![],
            effect_params: 0,
            identity_params: 0,
            region_params: 0,
            specializable: vec![],
        };
        reg.register(decl()).expect("one declaration per name");
        assert_eq!(reg.register(decl()), Err(DuplicateType(id)));
    }

    #[test]
    #[should_panic(expected = "unknown")]
    fn type_registry_unknown_id_panics() {
        let reg = TypeRegistry::new();
        let id = fresh_qref();
        reg.get(id);
    }

    // -- ExternCast tests --------------------------------------------

    /// A registry with one cast rule `UserDefined(A, [T..]) -> to`.
    struct CastSetup {
        from_id: QualifiedRef,
        registry: TypeRegistry,
    }

    fn make_cast_registry(
        type_param_count: usize,
        build_to: impl FnOnce(&[PolyTy]) -> PolyTy,
    ) -> CastSetup {
        let id = fresh_qref();
        let i = acvus_utils::Interner::new();
        let fn_id = QualifiedRef::root(i.intern("cast_fn"));
        let mut builder = PolyBuilder::new();
        let params: Vec<PolyTy> = (0..type_param_count)
            .map(|_| builder.fresh_ty_var())
            .collect();
        let from = TyTerm::UserDefined {
            id,
            type_args: params.iter().cloned().map(TypeArg::uniform).collect(),
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        let to = build_to(&params);
        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![TyVarBound::Any; type_param_count],
            effect_params: 0,
            identity_params: 0,
            region_params: 0,
            specializable: vec![false; type_param_count],
        })
        .expect("one declaration per name");
        reg.register_cast(CastRule {
            from,
            to,
            fn_ref: fn_id,
        });
        CastSetup {
            from_id: id,
            registry: reg,
        }
    }

    // -- Completeness: valid ExternCast coercions --

    #[test]
    fn extern_cast_basic_coercion() {
        // UserDefined(A, [T]) -> List<T>
        let CastSetup {
            from_id: id,
            registry,
        } = make_cast_registry(1, |p| arr(p[0].clone(), 3));
        let mut sources = Sources::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TypeArg::uniform(TyTerm::I64)],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        let to = arr(TyTerm::I64, 3);
        let conversion = s.decide(Decision::conversion(&from, &to));
        assert!(s.settle().is_empty());
        assert!(matches!(
            s.answer(conversion),
            Some(Answer::Conversion(Conversion::Cast(_)))
        ));
    }

    #[test]
    fn extern_cast_with_param_resolution() {
        // UserDefined(A, [T]) -> List<T>, where T is a fresh param on the consumer side
        let CastSetup {
            from_id: id,
            registry,
        } = make_cast_registry(1, |p| arr(p[0].clone(), 3));
        let mut sources = Sources::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TypeArg::uniform(TyTerm::I64)],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        let consumer_param = s.fresh_ty_var();
        let to = arr(consumer_param.clone(), 3);
        let conversion = s.decide(Decision::conversion(&from, &to));
        assert!(s.settle().is_empty());
        assert!(matches!(
            s.answer(conversion),
            Some(Answer::Conversion(Conversion::Cast(_)))
        ));
        assert_eq!(s.resolve_ty(&consumer_param), TyTerm::I64);
    }

    #[test]
    fn extern_cast_no_type_params() {
        // UserDefined(A, []) -> Int
        let CastSetup {
            from_id: id,
            registry,
        } = make_cast_registry(0, |_| TyTerm::I64);
        let mut sources = Sources::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        let conversion = s.decide(Decision::conversion(&from, &TyTerm::I64));
        assert!(s.settle().is_empty());
        assert!(matches!(
            s.answer(conversion),
            Some(Answer::Conversion(Conversion::Cast(_)))
        ));
    }

    // -- Soundness: invalid ExternCast --

    #[test]
    fn extern_cast_wrong_target_fails() {
        // Rule: A -> List<T>, but expected String
        let CastSetup {
            from_id: id,
            registry,
        } = make_cast_registry(1, |p| arr(p[0].clone(), 3));
        let mut sources = Sources::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TypeArg::uniform(TyTerm::I64)],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        assert!(s.unify(&from, &TyTerm::String).is_err());
    }

    #[test]
    fn extern_cast_no_rule_fails() {
        // No cast rules registered
        let id = fresh_qref();
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        assert!(s.unify(&from, &TyTerm::I64).is_err());
    }

    #[test]
    fn extern_cast_invariant_not_attempted() {
        // ExternCast only works in covariant/contravariant, not invariant
        let CastSetup {
            from_id: id,
            registry,
        } = make_cast_registry(0, |_| TyTerm::I64);
        let mut sources = Sources::new();
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &registry, &signatures);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        assert!(s.unify(&from, &TyTerm::I64).is_err());
    }

    // -- Ambiguity --

    #[test]
    fn extern_cast_ambiguity_rejected() {
        // Bypass TypeRegistry duplicate check - inject two rules with same to head
        // directly into the type_registry to test try_extern_cast ambiguity detection.
        let i = acvus_utils::Interner::new();
        let id = fresh_qref();
        let fn_id_a = QualifiedRef::root(i.intern("cast_a"));
        let fn_id_b = QualifiedRef::root(i.intern("cast_b"));

        // Use a PolyBuilder for the CastRule variables, separate Solver for unification.
        let mut sources = Sources::new();
        let mut builder = PolyBuilder::new();
        let t1 = builder.fresh_ty_var();
        let rule_a = CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![TypeArg::uniform(t1.clone())],
                effect_args: vec![],
                identity_args: vec![],
                region_params: 0,
            },
            to: arr(t1, 3),
            fn_ref: fn_id_a,
        };
        let t2 = builder.fresh_ty_var();
        let rule_b = CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![TypeArg::uniform(t2.clone())],
                effect_args: vec![],
                identity_args: vec![],
                region_params: 0,
            },
            to: arr(t2, 3),
            fn_ref: fn_id_b,
        };

        // Build registry manually (bypassing register_cast duplicate check)
        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
            region_params: 0,
            specializable: vec![false],
        })
        .expect("one declaration per name");
        reg.from_rules.entry(id).or_default().push(rule_a);
        reg.from_rules.entry(id).or_default().push(rule_b);
        let signatures = FxHashMap::default();
        let mut s = Solver::new(&mut sources, &reg, &signatures);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TypeArg::uniform(TyTerm::I64)],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        assert!(s.unify(&from, &arr(TyTerm::I64, 3)).is_err());
    }

    // -- TypeRegistry cast rules --

    #[test]
    #[should_panic(expected = "duplicate")]
    fn cast_registry_duplicate_panics() {
        let i = acvus_utils::Interner::new();
        let id = fresh_qref();
        let fn_id_a = QualifiedRef::root(i.intern("cast_a"));
        let fn_id_b = QualifiedRef::root(i.intern("cast_b"));
        let mut builder = PolyBuilder::new();
        let t = builder.fresh_ty_var();

        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
            region_params: 0,
            specializable: vec![false],
        })
        .expect("one declaration per name");
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![TypeArg::uniform(t.clone())],
                effect_args: vec![],
                identity_args: vec![],
                region_params: 0,
            },
            to: arr(t.clone(), 3),
            fn_ref: fn_id_a,
        });
        // Same from_id + same to head (List) -> panic
        let mut builder2 = PolyBuilder::new();
        let t2 = builder2.fresh_ty_var();
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![TypeArg::uniform(t2.clone())],
                effect_args: vec![],
                identity_args: vec![],
                region_params: 0,
            },
            to: arr(t2, 3),
            fn_ref: fn_id_b,
        });
    }

    #[test]
    fn cast_registry_different_to_head_ok() {
        let i = acvus_utils::Interner::new();
        let id = fresh_qref();
        let fn_id_a = QualifiedRef::root(i.intern("cast_a"));
        let fn_id_b = QualifiedRef::root(i.intern("cast_b"));

        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
            region_params: 0,
            specializable: vec![false],
        })
        .expect("one declaration per name");
        let mut builder1 = PolyBuilder::new();
        let t1 = builder1.fresh_ty_var();
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![TypeArg::uniform(t1.clone())],
                effect_args: vec![],
                identity_args: vec![],
                region_params: 0,
            },
            to: arr(t1, 3),
            fn_ref: fn_id_a,
        });
        // Different to head (Option vs List) -> ok
        let mut builder2 = PolyBuilder::new();
        let t2 = builder2.fresh_ty_var();
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![TypeArg::uniform(t2.clone())],
                effect_args: vec![],
                identity_args: vec![],
                region_params: 0,
            },
            to: TyTerm::Option(Box::new(t2)),
            fn_ref: fn_id_b,
        });
        assert_eq!(reg.rules_from(id).len(), 2);
    }

    // -- is_data ---------------------------------------------------------

    #[test]
    fn data_scalars_and_containers_of_scalars() {
        assert!(Ty::I64.is_data());
        assert!(Ty::String.is_data());
        assert!(arr(Ty::I64, 3).is_data());
        assert!(Ty::Option(Box::new(Ty::String)).is_data());
        assert!(Ty::Tuple(vec![Ty::I64, Ty::String]).is_data());
        assert!(arr(Ty::Option(Box::new(arr(Ty::I64, 2))), 3).is_data());
    }

    #[test]
    fn data_user_defined_is_data() {
        assert!(test_user_defined().is_data());
        assert!(arr(test_user_defined(), 3).is_data());
    }

    #[test]
    fn data_user_defined_over_fn_is_not_data() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::I64),
            captures: vec![],
            effect: Effect::OPAQUE.into(),
        };
        let Ty::UserDefined {
            id,
            effect_args,
            identity_args,
            ..
        } = test_user_defined()
        else {
            unreachable!()
        };
        let over_fn = Ty::UserDefined {
            id,
            type_args: vec![TypeArg::uniform(fn_ty)],
            effect_args,
            identity_args,
            region_params: 0,
        };
        assert!(!over_fn.is_data());
    }

    #[test]
    fn data_fn_handle_order_ref_are_not_data() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::I64),
            captures: vec![],
            effect: Effect::PURE.into(),
        };
        assert!(!fn_ty.is_data());
        assert!(!Ty::Handle(Box::new(Ty::I64)).is_data());
        assert!(!Ty::Order.is_data());
        assert!(!Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::I64))).is_data());
        assert!(!Ty::error().is_data());
    }

    #[test]
    fn data_container_holding_fn_is_not_data() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::I64),
            captures: vec![],
            effect: Effect::PURE.into(),
        };
        assert!(!arr(fn_ty.clone(), 3).is_data());
        assert!(!Ty::Option(Box::new(fn_ty.clone())).is_data());
        assert!(!Ty::Tuple(vec![Ty::I64, fn_ty.clone()]).is_data());
        let interner = Interner::new();
        let obj = Ty::Object(ObjectTy::written(FxHashMap::from_iter([(
            interner.intern("cb"),
            fn_ty,
        )])));
        assert!(!obj.is_data());
    }
}
