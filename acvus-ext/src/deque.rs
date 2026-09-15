//! `Deque<T>`: a sequence changed only at its ends. Every change is in
//! place, so a context holding one is never consumed
//! (docs/context-model.md), and the deque keeps the net change since it was
//! last settled as counts over its own storage: an append-only journal
//! persists that record instead of the whole value.

use std::collections::VecDeque;

use acvus_extern::{
    Decode, EffectVar, Encode, ExternError, ExternTypeDecl, IdentityVar, Interner, Journaled,
    NodeHash, PolyTy, PolyVars, QualifiedRef, Ref, RefMut, Registry, Runtime, SpaceError,
    SpaceHooks, SpaceResult, TyArg, TyVar, TyVarBound, UserDefinedDecl, Visit, extern_fn,
    extern_registry,
};
use acvus_mir::ty::Ty;

use crate::container::{checked_index, sig as container};
use crate::iter_pipeline::Iter;
use crate::iterator::{lent_iter, sig};
use crate::list::{List, list};

#[derive(Debug, Clone, PartialEq)]
pub struct Deque<T>
where
    T: TyVar,
{
    items: VecDeque<T>,
    front: usize,
    back: usize,
    settled: usize,
    dropped_front: usize,
    dropped_back: usize,
    /// The log node this deque was loaded from or last committed as
    /// (RFC-0033).
    head: Option<NodeHash>,
}

/// The net change since the last `settle`.
pub struct Record<'a, T> {
    pub dropped_front: usize,
    pub dropped_back: usize,
    pub pushed_front: Box<dyn Iterator<Item = &'a T> + 'a>,
    pub pushed_back: Box<dyn Iterator<Item = &'a T> + 'a>,
}

impl<T> Default for Deque<T>
where
    T: TyVar,
{
    fn default() -> Self {
        Self {
            items: VecDeque::new(),
            front: 0,
            back: 0,
            settled: 0,
            dropped_front: 0,
            dropped_back: 0,
            head: None,
        }
    }
}

impl<T> Deque<T>
where
    T: TyVar,
{
    fn settled_remaining(&self) -> usize {
        self.settled - self.dropped_front - self.dropped_back
    }

    pub fn push_front(&mut self, item: T) {
        self.items.push_front(item);
        self.front += 1;
    }

    pub fn push_back(&mut self, item: T) {
        self.items.push_back(item);
        self.back += 1;
    }

    pub fn pop_front(&mut self) -> Option<T> {
        let item = self.items.pop_front()?;
        if self.front > 0 {
            self.front -= 1;
        } else if self.settled_remaining() > 0 {
            self.dropped_front += 1;
        } else {
            self.back -= 1;
        }
        Some(item)
    }

    pub fn pop_back(&mut self) -> Option<T> {
        let item = self.items.pop_back()?;
        if self.back > 0 {
            self.back -= 1;
        } else if self.settled_remaining() > 0 {
            self.dropped_back += 1;
        } else {
            self.front -= 1;
        }
        Some(item)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.items.get_mut(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    pub fn record(&self) -> Record<'_, T> {
        Record {
            dropped_front: self.dropped_front,
            dropped_back: self.dropped_back,
            pushed_front: Box::new(self.items.range(..self.front).rev()),
            pushed_back: Box::new(self.items.range(self.items.len() - self.back..)),
        }
    }

    /// The journal has taken the record; the deque starts a new one.
    pub fn settle(&mut self) {
        self.front = 0;
        self.back = 0;
        self.settled = self.items.len();
        self.dropped_front = 0;
        self.dropped_back = 0;
    }
}

impl<T> TyArg for Deque<T>
where
    T: TyArg + TyVar,
{
    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::UserDefined {
            id: QualifiedRef::root(i.intern("Deque")),
            type_args: vec![T::poly_ty(i, vars)],
            effect_args: vec![],
            identity_args: vec![],
        }
    }
}

impl<T> ExternTypeDecl for Deque<T>
where
    T: TyVar,
{
    fn type_decl(i: &Interner) -> UserDefinedDecl {
        UserDefinedDecl {
            qref: QualifiedRef::root(i.intern("Deque")),
            type_params: vec![TyVarBound::Any],
            effect_params: 0,
            identity_params: 0,
        }
    }

    fn space<R>() -> Option<SpaceHooks<R>>
    where
        R: Runtime,
    {
        Some(SpaceHooks::of::<Deque<R::Value>>())
    }
}

/// The deque's op tags in its log (RFC-0033).
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Op {
    PushFront = 0,
    PushBack = 1,
    PopFront = 2,
    PopBack = 3,
}

impl Op {
    fn from_byte(b: u8) -> SpaceResult<Self> {
        Ok(match b {
            0 => Op::PushFront,
            1 => Op::PushBack,
            2 => Op::PopFront,
            3 => Op::PopBack,
            other => return Err(SpaceError::new(format!("Deque: unknown op tag {other}"))),
        })
    }
}

fn element_of<'a>(type_args: &'a [Ty]) -> SpaceResult<&'a Ty> {
    match type_args {
        [elem] => Ok(elem),
        other => Err(SpaceError::new(format!(
            "Deque has one type argument, got {}",
            other.len()
        ))),
    }
}

fn read_u64(input: &mut &[u8]) -> SpaceResult<u64> {
    let (head, rest) = input
        .split_first_chunk::<8>()
        .ok_or_else(|| SpaceError::new("Deque: truncated count"))?;
    *input = rest;
    Ok(u64::from_le_bytes(*head))
}

/// State: `u64` count, then the items front to back. Op: one tag byte,
/// then the element for a push. Pushes at one end are recorded in push
/// order; a pop of a settled item is a pop op, a pop of an item pushed
/// since the last take cancels that push.
impl<Rt> Journaled<Rt> for Deque<Rt::Value>
where
    Rt: Runtime,
{
    fn encode_state(&self, _: &Rt, type_args: &[Ty], elem: &Encode<'_, Rt>, out: &mut Vec<u8>) -> SpaceResult<()> {
        let ty = element_of(type_args)?;
        out.extend_from_slice(&(self.items.len() as u64).to_le_bytes());
        for item in &self.items {
            elem(ty, item, out)?;
        }
        Ok(())
    }

    fn decode_state(_: &Rt, type_args: &[Ty], elem: &Decode<'_, Rt>, input: &mut &[u8]) -> SpaceResult<Self> {
        let ty = element_of(type_args)?;
        let count = read_u64(input)?;
        let mut d = Deque::default();
        for _ in 0..count {
            d.items.push_back(elem(ty, input)?);
        }
        d.settle();
        Ok(d)
    }

    fn take_ops(&mut self, _: &Rt, type_args: &[Ty], elem: &Encode<'_, Rt>) -> SpaceResult<Vec<Vec<u8>>> {
        let ty = element_of(type_args)?;
        let mut ops = Vec::new();
        for _ in 0..self.dropped_front {
            ops.push(vec![Op::PopFront as u8]);
        }
        for _ in 0..self.dropped_back {
            ops.push(vec![Op::PopBack as u8]);
        }
        {
            let record = self.record();
            for item in record.pushed_front {
                let mut op = vec![Op::PushFront as u8];
                elem(ty, item, &mut op)?;
                ops.push(op);
            }
            for item in record.pushed_back {
                let mut op = vec![Op::PushBack as u8];
                elem(ty, item, &mut op)?;
                ops.push(op);
            }
        }
        self.settle();
        Ok(ops)
    }

    fn apply_op(&mut self, _: &Rt, type_args: &[Ty], elem: &Decode<'_, Rt>, op: &mut &[u8]) -> SpaceResult<()> {
        let ty = element_of(type_args)?;
        let (tag, rest) = op
            .split_first()
            .ok_or_else(|| SpaceError::new("Deque: empty op"))?;
        *op = rest;
        let empty = || SpaceError::new("Deque: a pop on an empty deque in the log");
        match Op::from_byte(*tag)? {
            Op::PushFront => self.items.push_front(elem(ty, op)?),
            Op::PushBack => self.items.push_back(elem(ty, op)?),
            Op::PopFront => drop(self.items.pop_front().ok_or_else(empty)?),
            Op::PopBack => drop(self.items.pop_back().ok_or_else(empty)?),
        }
        self.settle();
        Ok(())
    }

    fn children(&mut self, type_args: &[Ty], visit: &mut Visit<'_, Rt>) -> SpaceResult<()> {
        let ty = element_of(type_args)?;
        if !matches!(ty, Ty::UserDefined { .. }) {
            return Ok(());
        }
        for item in self.items.iter_mut() {
            visit(ty, item)?;
        }
        Ok(())
    }

    fn head(&self) -> Option<NodeHash> {
        self.head
    }

    fn set_head(&mut self, head: NodeHash) {
        self.head = Some(head);
    }
}

#[extern_fn(effect = pure)]
fn deque<T, R>(_: &R) -> Deque<T>
where
    T: TyVar,
    R: Runtime,
{
    Deque::default()
}

#[extern_fn(effect = pure)]
fn push_front<T, R>(_: &R, d: &mut Deque<T>, item: T)
where
    T: TyVar,
    R: Runtime,
{
    d.push_front(item);
}

#[extern_fn(effect = pure)]
fn push_back<T, R>(_: &R, d: &mut Deque<T>, item: T)
where
    T: TyVar,
    R: Runtime,
{
    d.push_back(item);
}

#[extern_fn(effect = pure)]
fn pop_front<T, R>(_: &R, d: &mut Deque<T>) -> Option<T>
where
    T: TyVar,
    R: Runtime,
{
    d.pop_front()
}

#[extern_fn(effect = pure)]
fn pop_back<T, R>(_: &R, d: &mut Deque<T>) -> Option<T>
where
    T: TyVar,
    R: Runtime,
{
    d.pop_back()
}

/// A deque demotes to a list: the record is dropped with the deque.
#[extern_fn(instance_of = list, effect = pure)]
#[extern_cast]
fn list_deque<T, R>(_: &R, d: Deque<T>) -> List<T>
where
    T: TyVar,
    R: Runtime,
{
    List(d.items.into())
}

#[extern_fn(instance_of = sig::into_iter, effect = pure)]
fn into_iter_deque<T, E, I, Rt>(_: &Rt, d: Deque<T>) -> Iter<T, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    Iter::from_items(d.items.into())
}

#[extern_fn(instance_of = sig::as_iter, effect = pure)]
fn as_iter_deque<T, E, I, Rt>(_: &Rt, d: Ref<Deque<T>, Rt>) -> Iter<Ref<T, Rt>, E, I, Rt>
where
    T: TyVar,
    E: EffectVar,
    I: IdentityVar,
    Rt: Runtime,
{
    lent_iter(d, Deque::get)
}

#[extern_fn(instance_of = container::len, effect = pure)]
fn len_deque<T, Rt>(_: &Rt, d: &Deque<T>) -> i64
where
    T: TyVar,
    Rt: Runtime,
{
    d.len() as i64
}

#[extern_fn(instance_of = container::get, effect = pure)]
fn get_deque<T, Rt>(rt: &Rt, d: Ref<Deque<T>, Rt>, index: i64) -> Result<Ref<T, Rt>, ExternError>
where
    T: TyVar,
    Rt: Runtime,
{
    let i = d.with(rt, |d| checked_index("get", d.len(), index))?;
    Ok(d.map(rt, |d| &d.items[i]))
}

#[extern_fn(instance_of = container::get_mut, effect = pure)]
fn get_mut_deque<T, Rt>(
    rt: &Rt,
    d: RefMut<Deque<T>, Rt>,
    index: i64,
) -> Result<RefMut<T, Rt>, ExternError>
where
    T: TyVar,
    Rt: Runtime,
{
    let i = d.with_mut(rt, |d| checked_index("get_mut", d.len(), index))?;
    Ok(d.map_mut(rt, |d| &mut d.items[i]))
}

#[extern_fn(instance_of = container::first, effect = pure)]
fn first_deque<T, Rt>(rt: &Rt, d: Ref<Deque<T>, Rt>) -> Option<Ref<T, Rt>>
where
    T: TyVar,
    Rt: Runtime,
{
    d.try_map(rt, |d| d.items.front())
}

#[extern_fn(instance_of = container::last, effect = pure)]
fn last_deque<T, Rt>(rt: &Rt, d: Ref<Deque<T>, Rt>) -> Option<Ref<T, Rt>>
where
    T: TyVar,
    Rt: Runtime,
{
    d.try_map(rt, |d| d.items.back())
}

pub fn deque_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        types: [Deque<_>],
        fns: [
            deque, push_front, push_back, pop_front, pop_back,
            list_deque, into_iter_deque, as_iter_deque,
            len_deque, get_deque, get_mut_deque, first_deque, last_deque,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Taken {
        dropped_front: usize,
        dropped_back: usize,
        pushed_front: Vec<i32>,
        pushed_back: Vec<i32>,
    }

    fn taken(d: &Deque<i32>) -> Taken {
        let r = d.record();
        Taken {
            dropped_front: r.dropped_front,
            dropped_back: r.dropped_back,
            pushed_front: r.pushed_front.copied().collect(),
            pushed_back: r.pushed_back.copied().collect(),
        }
    }

    fn change(
        dropped_front: usize,
        dropped_back: usize,
        pushed_front: &[i32],
        pushed_back: &[i32],
    ) -> Taken {
        Taken {
            dropped_front,
            dropped_back,
            pushed_front: pushed_front.to_vec(),
            pushed_back: pushed_back.to_vec(),
        }
    }

    fn settled(items: impl IntoIterator<Item = i32>) -> Deque<i32> {
        let mut d = Deque::default();
        for x in items {
            d.push_back(x);
        }
        d.settle();
        d
    }

    #[test]
    fn each_end_records_its_pushes_in_push_order() {
        let mut d = Deque::default();
        d.push_front(1);
        d.push_back(2);
        d.push_front(3);
        assert_eq!(d.iter().copied().collect::<Vec<_>>(), [3, 1, 2]);
        assert_eq!(taken(&d), change(0, 0, &[1, 3], &[2]));
    }

    #[test]
    fn a_pop_of_a_pushed_item_cancels_its_push() {
        let mut d = settled([10]);
        d.push_front(1);
        d.push_back(2);
        assert_eq!(d.pop_front(), Some(1));
        assert_eq!(d.pop_back(), Some(2));
        assert_eq!(taken(&d), change(0, 0, &[], &[]));
    }

    #[test]
    fn a_pop_of_a_settled_item_is_a_drop() {
        let mut d = settled([10, 20, 30]);
        assert_eq!(d.pop_front(), Some(10));
        assert_eq!(d.pop_back(), Some(30));
        d.push_back(40);
        assert_eq!(taken(&d), change(1, 1, &[], &[40]));
    }

    #[test]
    fn a_pop_past_the_settled_items_takes_from_the_other_end_s_pushes() {
        let mut d = settled([10]);
        d.push_back(1);
        d.push_back(2);
        assert_eq!(d.pop_front(), Some(10));
        assert_eq!(d.pop_front(), Some(1));
        assert_eq!(taken(&d), change(1, 0, &[], &[2]));
        assert_eq!(d.pop_back(), Some(2));
        assert_eq!(d.pop_back(), None);
        assert_eq!(taken(&d), change(1, 0, &[], &[]));
    }

    #[test]
    fn settle_starts_a_new_record_and_keeps_the_items() {
        let mut d = settled([1]);
        d.push_back(2);
        d.push_front(0);
        d.settle();
        assert_eq!(taken(&d), change(0, 0, &[], &[]));
        assert_eq!(d.pop_front(), Some(0));
        assert_eq!(d.iter().copied().collect::<Vec<_>>(), [1, 2]);
        assert_eq!(taken(&d), change(1, 0, &[], &[]));
    }
}
