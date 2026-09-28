//! The relations A speaks in: `eq`, `join`, `enter` (typeck-structure
//! section 2). Each either records what it learned or refuses in B's terms.
//!
//! `eq` and `join` share one walk. It returns the term both sides are made
//! into, and rebinds the last variable on each side's chain to it: a node
//! is never changed in place, so a variable is where a term grows or
//! becomes opaque, and two variables made one are linked, so that what one
//! learns later the other learns too.

use std::cmp::Ordering;

use acvus_utils::{Astr, QualifiedRef};

use super::cells::{End as CellEnd, MarkTerm, Unequal};
use super::{End, Infer, Mark, Term, VarState};
use super::closed::poly_head;
use super::Closed;
use crate::decl::{Base, Declarations, Head, VerifiedExternCast};
use crate::ty::{
    Calls, Effect, EffectArg, EffectTerm, Flows, Holds, IdentityTerm, Infer as InferPhase, Label, ObjectTy, ParamTerm,
    TyKind, TyVarId, TypeArg,
};

/// Why two terms do not relate: B's refusals (refusals).
#[derive(Debug, Clone, PartialEq)]
pub enum Mismatch {
    /// Two heads that are not one head.
    Heads { a: Term, b: Term },
    /// A variable would contain itself (occurs check).
    Occurs { var: TyVarId, in_term: Term },
    /// A field an opaque object lacks, which a transparent one would add.
    OpaqueLacksField { object: Term, field: Astr },
    /// A variant a closed enum lacks (patterns §2).
    OpaqueLacksVariant { r#enum: Term, tag: Astr },
    /// Two enums of two names (types §2).
    TwoEnums { a: QualifiedRef, b: QualifiedRef },
    /// A variable's bound does not admit the term.
    Bound { var: TyVarId, term: Term },
    /// `#` on one side only (types §8).
    Mark { a: Term, b: Term },
}

/// What an `enter` did (types §3, §7).
#[derive(Debug, Clone, PartialEq)]
pub enum Entered {
    /// The slot took the value's type, or folded it up the tree.
    Folded,
    /// The value converts to the slot through one declared conversion.
    Converted(VerifiedExternCast),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Relation {
    Eq,
    Join,
}

/// A member of two sorted member lists: on both sides, or on one.
enum Member<T> {
    Both(Astr, T, T),
    OnlyA(Astr, T),
    OnlyB(Astr, T),
}

fn members<T: Copy>(a: &[(Astr, T)], b: &[(Astr, T)]) -> Vec<Member<T>> {
    let (mut a, mut b) = (a.iter().peekable(), b.iter().peekable());
    let mut out = Vec::new();
    loop {
        let member = match (a.peek(), b.peek()) {
            (None, None) => return out,
            (Some(&&(name, x)), None) => {
                a.next();
                Member::OnlyA(name, x)
            }
            (None, Some(&&(name, y))) => {
                b.next();
                Member::OnlyB(name, y)
            }
            (Some(&&(name_a, x)), Some(&&(name_b, y))) => match name_a.cmp(&name_b) {
                Ordering::Less => {
                    a.next();
                    Member::OnlyA(name_a, x)
                }
                Ordering::Greater => {
                    b.next();
                    Member::OnlyB(name_b, y)
                }
                Ordering::Equal => {
                    a.next();
                    b.next();
                    Member::Both(name_a, x, y)
                }
            },
        };
        out.push(member);
    }
}

/// `head` and every head up the tree from it, in order.
fn heads_up(decls: &Declarations, head: Head) -> Vec<Head> {
    let mut heads = vec![head];
    let mut at = head;
    while let Some(step) = decls.casts().step_up(at) {
        at = poly_head(decls.types(), step.rule().to);
        heads.push(at);
    }
    heads
}

/// Whether a value of `head` carries an identity (types §3): its type, or
/// a reference's target, is an extern-defined type with an identity
/// parameter.
fn carries_identity(decls: &Declarations, head: Head) -> bool {
    match head {
        Head::Base(Base::Named(id)) | Head::Ref(_, Base::Named(id)) => {
            decls.user_type(id).is_some_and(|declared| declared.identity_params > 0)
        }
        Head::Base(_) | Head::Ref(..) => false,
    }
}

/// Two shapes made one are opaque where either is (types §1).
fn either_opaque(a: Label, b: Label) -> Label {
    match (a, b) {
        (Label::Transparent, Label::Transparent) => Label::Transparent,
        (Label::Opaque, _) | (_, Label::Opaque) => Label::Opaque,
    }
}

fn mark_term(arg: &TypeArg<InferPhase>) -> MarkTerm {
    match arg {
        TypeArg::Uniform(_) => MarkTerm::Known(Mark::Uniform),
        TypeArg::Specialized(_) => MarkTerm::Known(Mark::Specialized),
        TypeArg::Open(var, _) => MarkTerm::Var(*var),
    }
}

impl Infer {
    /// Unification with the occurs check; every constructor invariant
    /// (types §4).
    pub fn eq(&mut self, a: Term, b: Term) -> Result<(), Mismatch> {
        self.relate(a, b, Relation::Eq).map(drop)
    }

    /// The join of two terms (types §2): a transparent side grows by union
    /// toward the other; an opaque side never grows; afterwards the two are
    /// `eq`. Function types join their effects, flows, calls and split.
    pub fn join(&mut self, a: Term, b: Term) -> Result<(), Mismatch> {
        let joined = self.relate(a, b, Relation::Join)?;
        for side in [a, b] {
            if let End::Node { holder: None, node } = self.end(side)
                && !matches!(self.types.kind(node), TyKind::Never)
            {
                self.relate(side, joined, Relation::Eq)?;
            }
        }
        Ok(())
    }

    /// A value entering a slot (types §3, §7): once the value's head is
    /// closed, the slot folds it by ⊔ along the cast tree, or the value
    /// converts by exactly one declared conversion.
    /// The value is closed: a decision fires only on closed terms (types §6),
    /// and the caller checked with `closed`.
    ///
    /// Every declared cast is a step of the tree (extern-fn §7, `verify`),
    /// so two heads no step path relates have no declared conversion
    /// between them either: that is a mismatch.
    pub fn enter(&mut self, decls: &Declarations, value: Closed, slot: Term) -> Result<Entered, Mismatch> {
        let (holder, slot_node) = match self.end(slot) {
            End::Unbound(var, _) => {
                self.bind(var, None, value.term)?;
                return Ok(Entered::Folded);
            }
            End::Node { holder, node } => (holder, node),
        };
        let mismatch = Mismatch::Heads {
            a: value.term,
            b: slot_node,
        };
        let never = Head::Base(Base::Never);
        let target = match self.head_of(slot_node) {
            None => None,
            Some(slot_head) if value.head == never || slot_head == never => None,
            Some(slot_head) => {
                let target = self.nearest_common_ancestor(decls, value, slot_node, slot_head);
                Some((slot_head, target.ok_or(mismatch.clone())?))
            }
        };
        let (value_at, slot_at) = match target {
            None => (value.term, slot_node),
            Some((slot_head, target)) => (
                self.climb(decls, value.term, value.head, target)?,
                self.climb(decls, slot_node, slot_head, target)?,
            ),
        };
        let made = self.relate(slot_at, value_at, Relation::Join)?;
        match holder {
            Some((var, _)) => self.rebind(var, made)?,
            None => {
                self.relate(slot_node, made, Relation::Eq).map_err(|_| mismatch)?;
            }
        }
        Ok(Entered::Folded)
    }

    /// Where `value` and a slot of `slot_head` meet (types §3): the first
    /// head up from the value's that is also up from the slot's; for two
    /// identities, the first such head that carries none. `None` where
    /// there is no such head.
    fn nearest_common_ancestor(
        &mut self,
        decls: &Declarations,
        value: Closed,
        slot_node: Term,
        slot_head: Head,
    ) -> Option<Head> {
        let value_chain = heads_up(decls, value.head);
        let slot_chain = heads_up(decls, slot_head);
        let common = value_chain.iter().copied().filter(|head| slot_chain.contains(head));
        let two_identities = match (self.identity_of(value.term), self.identity_of(slot_node)) {
            (Some(x), Some(y)) => self.identity_vars.unify(&x, &y).is_err(),
            _ => false,
        };
        let mut meeting = common.filter(|head| !two_identities || !carries_identity(decls, *head));
        meeting.next()
    }

    /// The identity a value of `node` carries: an extern-defined type's
    /// one identity argument, through references.
    fn identity_of(&self, node: Term) -> Option<IdentityTerm<InferPhase>> {
        match self.types.kind(node) {
            TyKind::UserDefined { identity_args, .. } => identity_args.first().copied(),
            TyKind::Ref(_, target) => match self.end(target.ty()) {
                End::Node { node, .. } => self.identity_of(node),
                End::Unbound(..) => None,
            },
            _ => None,
        }
    }

    /// `node`, of head `from`, stepped up the tree to `to` through each
    /// step's cast: the cast's `from` pattern is made one with the type
    /// reached, and its `to` pattern is the type above.
    ///
    /// # Panics
    /// `to` is not up from `from`: callers climb to a common ancestor.
    fn climb(&mut self, decls: &Declarations, node: Term, from: Head, to: Head) -> Result<Term, Mismatch> {
        let (mut at, mut head) = (node, from);
        while head != to {
            let step = decls
                .casts()
                .step_up(head)
                .expect("climb: the target is up the tree from the start");
            let (pattern_from, pattern_to) = self.instantiate_cast(decls.types(), step.rule());
            self.relate(pattern_from, at, Relation::Eq)?;
            at = pattern_to;
            head = poly_head(decls.types(), step.rule().to);
        }
        Ok(at)
    }

    /// The join of two effects (types §2).
    pub fn join_effects(
        &mut self,
        a: &EffectTerm<InferPhase>,
        b: &EffectTerm<InferPhase>,
    ) -> EffectTerm<InferPhase> {
        self.effect_vars.join(a, b, Effect::join)
    }

    fn relate(&mut self, a: Term, b: Term, how: Relation) -> Result<Term, Mismatch> {
        acvus_utils::grow(|| self.relate_level(a, b, how))
    }

    fn relate_level(&mut self, a: Term, b: Term, how: Relation) -> Result<Term, Mismatch> {
        if a == b {
            return Ok(a);
        }
        match (self.end(a), self.end(b)) {
            (End::Unbound(v, _), End::Unbound(w, w_term)) => {
                if v != w {
                    self.merge_unbound(v, w, w_term)?;
                }
                Ok(w_term)
            }
            (End::Unbound(var, _), End::Node { holder, node }) | (End::Node { holder, node }, End::Unbound(var, _)) => {
                self.bind(var, holder, node)
            }
            (End::Node { holder: holder_a, node: a }, End::Node { holder: holder_b, node: b }) => {
                let made = match (how, self.types.kind(a), self.types.kind(b)) {
                    (Relation::Join, TyKind::Never, _) => b,
                    (Relation::Join, _, TyKind::Never) => a,
                    _ => self.relate_nodes(a, b, how)?,
                };
                self.settle(holder_a, holder_b, made)
            }
        }
    }

    /// Makes two unbound variables one: `v` is linked to `w`, which takes
    /// what both bounds admit.
    fn merge_unbound(&mut self, v: TyVarId, w: TyVarId, w_term: Term) -> Result<(), Mismatch> {
        let (VarState::Unbound(bound_v), VarState::Unbound(bound_w)) =
            (&self.ty_vars[v.0 as usize], &self.ty_vars[w.0 as usize])
        else {
            panic!("merge_unbound: both ends are unbound variables")
        };
        let bound = bound_v
            .meet(bound_w)
            .ok_or(Mismatch::Bound { var: v, term: w_term })?;
        self.ty_vars[w.0 as usize] = VarState::Unbound(bound);
        self.ty_vars[v.0 as usize] = VarState::Bound(w_term);
        Ok(())
    }

    /// Binds the unbound `var` to `node`, through the variable that holds
    /// it where one does.
    fn bind(&mut self, var: TyVarId, holder: Option<(TyVarId, Term)>, node: Term) -> Result<Term, Mismatch> {
        if self.occurs(var, node) {
            return Err(Mismatch::Occurs { var, in_term: node });
        }
        let VarState::Unbound(bound) = &self.ty_vars[var.0 as usize] else {
            panic!("bind: the variable is an unbound end")
        };
        if !bound.admits(self.types.kind(node)) {
            return Err(Mismatch::Bound { var, term: node });
        }
        let target = holder.map_or(node, |(_, held)| held);
        self.ty_vars[var.0 as usize] = VarState::Bound(target);
        Ok(target)
    }

    /// Records `made` as what both sides are now: each side's last variable
    /// is rebound to it, and the second linked to the first.
    fn settle(
        &mut self,
        holder_a: Option<(TyVarId, Term)>,
        holder_b: Option<(TyVarId, Term)>,
        made: Term,
    ) -> Result<Term, Mismatch> {
        match (holder_a, holder_b) {
            (Some((a, a_term)), Some((b, _))) => {
                self.rebind(a, made)?;
                if a != b {
                    self.rebind(b, a_term)?;
                }
                Ok(a_term)
            }
            (Some((var, term)), None) | (None, Some((var, term))) => {
                self.rebind(var, made)?;
                Ok(term)
            }
            (None, None) => Ok(made),
        }
    }

    fn rebind(&mut self, var: TyVarId, to: Term) -> Result<(), Mismatch> {
        if self.ty_vars[var.0 as usize] == VarState::Bound(to) {
            return Ok(());
        }
        if self.occurs(var, to) {
            return Err(Mismatch::Occurs { var, in_term: to });
        }
        self.ty_vars[var.0 as usize] = VarState::Bound(to);
        Ok(())
    }

    /// Whether `var` occurs in `term`, bound variables followed.
    fn occurs(&self, var: TyVarId, term: Term) -> bool {
        let mut seen = rustc_hash::FxHashSet::default();
        let mut stack = vec![term];
        while let Some(at) = stack.pop() {
            if !seen.insert(at) {
                continue;
            }
            match self.types.kind(at) {
                TyKind::Var(found) if *found == var => return true,
                TyKind::Var(found) => {
                    if let VarState::Bound(to) = &self.ty_vars[found.0 as usize] {
                        stack.push(*to);
                    }
                }
                kind => kind.for_each_child(|child| stack.push(child)),
            }
        }
        false
    }

    /// The node both nodes are made into. Under `join` only a node's own
    /// member set grows, and a function type's effect, flows, calls and
    /// holds join (types §2); every part is related by `eq`, since every
    /// constructor is invariant (types §4).
    fn relate_nodes(&mut self, a: Term, b: Term, how: Relation) -> Result<Term, Mismatch> {
        if a == b {
            return Ok(a);
        }
        let heads = Mismatch::Heads { a, b };
        let kind = match (self.types.kind(a).clone(), self.types.kind(b).clone()) {
            (TyKind::Handle(x), TyKind::Handle(y)) => TyKind::Handle(self.relate(x, y, Relation::Eq)?),
            (TyKind::Array(x, len_x), TyKind::Array(y, len_y)) => {
                let element = self.relate(x, y, Relation::Eq)?;
                let len = self.len_vars.unify(&len_x, &len_y).map_err(|Unequal| heads)?;
                TyKind::Array(element, len)
            }
            (TyKind::Slice(x), TyKind::Slice(y)) => TyKind::Slice(self.relate(x, y, Relation::Eq)?),
            (TyKind::Option(x), TyKind::Option(y)) => TyKind::Option(self.relate(x, y, Relation::Eq)?),
            (TyKind::Result(ok_x, err_x), TyKind::Result(ok_y, err_y)) => {
                TyKind::Result(self.relate(ok_x, ok_y, Relation::Eq)?, self.relate(err_x, err_y, Relation::Eq)?)
            }
            (TyKind::Tuple(xs), TyKind::Tuple(ys)) => {
                if xs.len() != ys.len() {
                    return Err(heads);
                }
                let items = xs
                    .iter()
                    .zip(ys.iter())
                    .map(|(x, y)| self.relate(*x, *y, Relation::Eq))
                    .collect::<Result<_, _>>()?;
                TyKind::Tuple(items)
            }
            (TyKind::Object(x), TyKind::Object(y)) => TyKind::Object(self.relate_objects(a, b, x, y, how)?),
            (
                TyKind::Enum {
                    name: name_a,
                    variants: variants_a,
                    label: label_a,
                },
                TyKind::Enum {
                    name: name_b,
                    variants: variants_b,
                    label: label_b,
                },
            ) => {
                if name_a != name_b {
                    return Err(Mismatch::TwoEnums { a: name_a, b: name_b });
                }
                let variants = self.relate_variants((a, b), (&variants_a, label_a), (&variants_b, label_b), how)?;
                TyKind::Enum {
                    name: name_a,
                    variants,
                    label: either_opaque(label_a, label_b),
                }
            }
            (
                TyKind::Fn {
                    params: params_a,
                    ret: ret_a,
                    effect: effect_a,
                    flows: flows_a,
                    calls: calls_a,
                    holds: holds_a,
                },
                TyKind::Fn {
                    params: params_b,
                    ret: ret_b,
                    effect: effect_b,
                    flows: flows_b,
                    calls: calls_b,
                    holds: holds_b,
                },
            ) => {
                if params_a.len() != params_b.len() {
                    return Err(heads);
                }
                let params = params_a
                    .iter()
                    .zip(params_b.iter())
                    .map(|(x, y)| {
                        Ok(ParamTerm {
                            name: x.name,
                            ty: self.relate(x.ty, y.ty, Relation::Eq)?,
                        })
                    })
                    .collect::<Result<_, Mismatch>>()?;
                let ret = self.relate(ret_a, ret_b, Relation::Eq)?;
                let (effect, flows, calls, holds) = match how {
                    Relation::Eq => (
                        self.effect_vars.unify(&effect_a, &effect_b).map_err(|Unequal| heads.clone())?,
                        self.flow_vars.unify(&flows_a, &flows_b).map_err(|Unequal| heads.clone())?,
                        Self::same(calls_a, calls_b).ok_or(heads.clone())?,
                        Self::same(holds_a, holds_b).ok_or(heads)?,
                    ),
                    Relation::Join => (
                        self.join_effects(&effect_a, &effect_b),
                        self.flow_vars.join(&flows_a, &flows_b, Flows::join),
                        Calls::join(calls_a, calls_b),
                        Holds::join(holds_a, holds_b),
                    ),
                };
                TyKind::Fn {
                    params,
                    ret,
                    effect,
                    flows,
                    calls,
                    holds,
                }
            }
            (
                TyKind::UserDefined {
                    id: id_a,
                    type_args: type_args_a,
                    effect_args: effect_args_a,
                    identity_args: identity_args_a,
                },
                TyKind::UserDefined {
                    id: id_b,
                    type_args: type_args_b,
                    effect_args: effect_args_b,
                    identity_args: identity_args_b,
                },
            ) => {
                if id_a != id_b
                    || type_args_a.len() != type_args_b.len()
                    || effect_args_a.len() != effect_args_b.len()
                    || identity_args_a.len() != identity_args_b.len()
                {
                    return Err(heads);
                }
                let type_args = type_args_a
                    .iter()
                    .zip(type_args_b.iter())
                    .map(|(x, y)| self.relate_type_arg(a, b, x, y))
                    .collect::<Result<_, _>>()?;
                let effect_args = effect_args_a
                    .iter()
                    .zip(effect_args_b.iter())
                    .map(|(x, y)| self.relate_effect_arg(a, b, x, y))
                    .collect::<Result<_, _>>()?;
                let identity_args = identity_args_a
                    .iter()
                    .zip(identity_args_b.iter())
                    .map(|(x, y)| self.identity_vars.unify(x, y).map_err(|Unequal| heads.clone()))
                    .collect::<Result<_, _>>()?;
                TyKind::UserDefined {
                    id: id_a,
                    type_args,
                    effect_args,
                    identity_args,
                }
            }
            (TyKind::Ref(mutability_a, x), TyKind::Ref(mutability_b, y)) => {
                if mutability_a != mutability_b {
                    return Err(heads);
                }
                TyKind::Ref(mutability_a, self.relate_type_arg(a, b, &x, &y)?)
            }
            (TyKind::Var(_), _) | (_, TyKind::Var(_)) => {
                unreachable!("relate_nodes: an end is never a variable")
            }
            _ => return Err(heads),
        };
        Ok(self.intern(kind))
    }

    fn same<T: PartialEq>(a: T, b: T) -> Option<T> {
        (a == b).then_some(a)
    }

    fn relate_objects(
        &mut self,
        a: Term,
        b: Term,
        x: ObjectTy<InferPhase>,
        y: ObjectTy<InferPhase>,
        how: Relation,
    ) -> Result<ObjectTy<InferPhase>, Mismatch> {
        match (x, y) {
            (ObjectTy::Structural(fields_a, label_a), ObjectTy::Structural(fields_b, label_b)) => {
                let mut fields = Vec::new();
                for member in members(&fields_a, &fields_b) {
                    let field = match member {
                        Member::Both(name, x, y) => (name, self.relate(x, y, Relation::Eq)?),
                        Member::OnlyA(name, x) => {
                            Self::may_grow(how, (a, b), (b, label_b), |object| Mismatch::OpaqueLacksField {
                                object,
                                field: name,
                            })?;
                            (name, x)
                        }
                        Member::OnlyB(name, y) => {
                            Self::may_grow(how, (a, b), (a, label_a), |object| Mismatch::OpaqueLacksField {
                                object,
                                field: name,
                            })?;
                            (name, y)
                        }
                    };
                    fields.push(field);
                }
                Ok(ObjectTy::Structural(fields.into(), either_opaque(label_a, label_b)))
            }
            (ObjectTy::Declared(id_a, fields_a), ObjectTy::Declared(id_b, fields_b)) => {
                if id_a != id_b
                    || fields_a.len() != fields_b.len()
                    || fields_a.iter().zip(fields_b.iter()).any(|(x, y)| x.0 != y.0)
                {
                    return Err(Mismatch::Heads { a, b });
                }
                let fields = fields_a
                    .iter()
                    .zip(fields_b.iter())
                    .map(|((name, x), (_, y))| Ok((*name, self.relate(*x, *y, Relation::Eq)?)))
                    .collect::<Result<_, Mismatch>>()?;
                Ok(ObjectTy::Declared(id_a, fields))
            }
            (ObjectTy::Structural(..), ObjectTy::Declared(..)) | (ObjectTy::Declared(..), ObjectTy::Structural(..)) => {
                Err(Mismatch::Heads { a, b })
            }
        }
    }

    /// Whether `lacking`, labelled `label`, may grow by a member the other
    /// side has: only under `join`, and only where transparent (types §2).
    /// Under `eq` the two are one shape or not one head.
    fn may_grow(
        how: Relation,
        (a, b): (Term, Term),
        (lacking, label): (Term, Label),
        opaque_lacks: impl FnOnce(Term) -> Mismatch,
    ) -> Result<(), Mismatch> {
        match (how, label) {
            (Relation::Eq, _) => Err(Mismatch::Heads { a, b }),
            (Relation::Join, Label::Opaque) => Err(opaque_lacks(lacking)),
            (Relation::Join, Label::Transparent) => Ok(()),
        }
    }

    fn relate_variants(
        &mut self,
        (a, b): (Term, Term),
        (variants_a, label_a): (&[(Astr, Option<Term>)], Label),
        (variants_b, label_b): (&[(Astr, Option<Term>)], Label),
        how: Relation,
    ) -> Result<Box<[(Astr, Option<Term>)]>, Mismatch> {
        let mut variants = Vec::new();
        for member in members(variants_a, variants_b) {
            let variant = match member {
                Member::Both(tag, Some(x), Some(y)) => (tag, Some(self.relate(x, y, Relation::Eq)?)),
                Member::Both(tag, None, None) => (tag, None),
                Member::Both(_, Some(_), None) | Member::Both(_, None, Some(_)) => {
                    return Err(Mismatch::Heads { a, b });
                }
                Member::OnlyA(tag, x) => {
                    Self::may_grow(how, (a, b), (b, label_b), |r#enum| Mismatch::OpaqueLacksVariant { r#enum, tag })?;
                    (tag, x)
                }
                Member::OnlyB(tag, y) => {
                    Self::may_grow(how, (a, b), (a, label_a), |r#enum| Mismatch::OpaqueLacksVariant { r#enum, tag })?;
                    (tag, y)
                }
            };
            variants.push(variant);
        }
        Ok(variants.into())
    }

    fn relate_type_arg(
        &mut self,
        a: Term,
        b: Term,
        x: &TypeArg<InferPhase>,
        y: &TypeArg<InferPhase>,
    ) -> Result<TypeArg<InferPhase>, Mismatch> {
        let mark = self
            .repr_vars
            .unify(&mark_term(x), &mark_term(y))
            .map_err(|Unequal| Mismatch::Mark { a, b })?;
        let ty = self.relate(x.ty(), y.ty(), Relation::Eq)?;
        Ok(match self.repr_vars.end(&mark) {
            CellEnd::Known(Mark::Uniform) | CellEnd::Var { value: Some(Mark::Uniform), .. } => TypeArg::Uniform(ty),
            CellEnd::Known(Mark::Specialized) | CellEnd::Var { value: Some(Mark::Specialized), .. } => {
                TypeArg::Specialized(ty)
            }
            CellEnd::Var { var, value: None } => TypeArg::Open(var, ty),
        })
    }

    fn relate_effect_arg(
        &mut self,
        a: Term,
        b: Term,
        x: &EffectArg<InferPhase>,
        y: &EffectArg<InferPhase>,
    ) -> Result<EffectArg<InferPhase>, Mismatch> {
        let unify = |infer: &mut Infer, x, y| {
            infer
                .effect_vars
                .unify(x, y)
                .map_err(|Unequal| Mismatch::Heads { a, b })
        };
        match (x, y) {
            (EffectArg::Uniform(x), EffectArg::Uniform(y)) => Ok(EffectArg::Uniform(unify(self, x, y)?)),
            (EffectArg::Specialized(x), EffectArg::Specialized(y)) => Ok(EffectArg::Specialized(unify(self, x, y)?)),
            (EffectArg::Uniform(_), EffectArg::Specialized(_)) | (EffectArg::Specialized(_), EffectArg::Uniform(_)) => {
                Err(Mismatch::Mark { a, b })
            }
        }
    }
}
