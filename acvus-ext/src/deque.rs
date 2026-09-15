//! `Deque<T>`: a sequence changed only at its ends. Every change is in
//! place, so a context holding one is never consumed
//! (docs/context-model.md), and the deque keeps the net change since it was
//! last settled as counts over its own storage: an append-only journal
//! persists that record instead of the whole value.

use std::collections::VecDeque;

use acvus_extern::{
    ExternError, ExternTypeDecl, Interner, PolyTy, PolyVars, QualifiedRef, Registry, Runtime,
    TyArg, TyVar, TyVarBound, UserDefinedDecl, extern_fn, extern_registry,
};

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

#[extern_fn(effect = pure)]
fn deque_len<T, R>(_: &R, d: &Deque<T>) -> i64
where
    T: TyVar,
    R: Runtime,
{
    d.len() as i64
}

/// Consumes the deque: an extern fn returns no reference and no clone of an
/// erased `T` exists yet (RFC-0027), so an element cannot leave a borrowed
/// deque.
#[extern_fn(effect = pure)]
fn deque_get<T, R>(_: &R, d: Deque<T>, index: i64) -> Result<T, ExternError>
where
    T: TyVar,
    R: Runtime,
{
    let len = d.len();
    let i = usize::try_from(index)
        .ok()
        .filter(|i| *i < len)
        .ok_or_else(|| ExternError::call("deque_get", format!("index {index} out of {len}")))?;
    Ok(d.items.into_iter().nth(i).expect("index checked against len"))
}

pub fn deque_registry<R: Runtime>() -> Registry<R> {
    extern_registry! {
        ns: "std",
        types: [Deque<_>],
        fns: [deque, push_front, push_back, pop_front, pop_back, deque_len, deque_get],
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
