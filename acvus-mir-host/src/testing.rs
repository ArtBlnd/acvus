//! Test utilities for acvus-mir-host.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::{Callable, EffectParam};
use crate::error::HostError;
use crate::ity::{Hosted, ITy};
use crate::registrar::Registrar;
use crate::scope::{CallArgs, Scope};

// ── DummyOwned: opaque value for testing ───────────────────────────

/// Opaque runtime value for DummyScope. Wraps any value as Box<dyn Any>.
pub struct DummyOwned(pub Box<dyn Any>);

impl ITy for DummyOwned {
    fn ty(_: &acvus_utils::Interner, _: &[acvus_mir::ty::Ty], _: &[acvus_mir::ty::Effect]) -> acvus_mir::ty::Ty {
        panic!("DummyOwned::ty should not be called — runtime only")
    }
}

// SAFETY: DummyOwned is the runtime opaque type for DummyScope.
unsafe impl Hosted for DummyOwned {}

// ── DummyCallable: type-erased function for testing ────────────────

/// A callable stored inside DummyOwned.
/// Receives arg slots as Reprs, dispatches via DummyScope, stores result into dst.
struct DummyCallable {
    /// (scope, arg_slots, dst_slot) → Result
    dispatch: Box<dyn Fn(&mut DummyScope, &[DummyRepr], DummyRepr) -> Result<(), HostError>>,
}

// ── DummyScope ─────────────────────────────────────────────────────

pub struct DummyScope {
    slots: HashMap<u32, (TypeId, Box<dyn Any>)>,
    next_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DummyRepr(pub u32);

impl DummyScope {
    pub fn new() -> Self {
        Self { slots: HashMap::new(), next_id: 0 }
    }

    /// Allocate an empty slot (simulates SSA pre-allocation).
    pub fn alloc(&mut self) -> DummyRepr {
        let id = self.next_id;
        self.next_id += 1;
        DummyRepr(id)
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Create a DummyOwned wrapping a 1-arg function.
    pub fn make_fn_1<A: ITy + 'static, R: ITy + 'static>(
        f: impl Fn(A) -> R + 'static,
    ) -> DummyOwned {
        DummyOwned(Box::new(DummyCallable {
            dispatch: Box::new(move |scope: &mut DummyScope, arg_slots: &[DummyRepr], dst: DummyRepr| {
                let a: A = scope.raw_take(arg_slots[0]);
                let r = f(a);
                scope.raw_store(dst, r);
                Ok(())
            }),
        }))
    }

    /// Create a DummyOwned wrapping a 2-arg function.
    pub fn make_fn_2<A: ITy + 'static, B: ITy + 'static, R: ITy + 'static>(
        f: impl Fn(A, B) -> R + 'static,
    ) -> DummyOwned {
        DummyOwned(Box::new(DummyCallable {
            dispatch: Box::new(move |scope: &mut DummyScope, arg_slots: &[DummyRepr], dst: DummyRepr| {
                let a: A = scope.raw_take(arg_slots[0]);
                let b: B = scope.raw_take(arg_slots[1]);
                let r = f(a, b);
                scope.raw_store(dst, r);
                Ok(())
            }),
        }))
    }

    // Internal: store without ITy bound (for DummyCallable dispatch).
    fn raw_store<T: 'static>(&mut self, repr: DummyRepr, val: T) {
        self.slots.insert(repr.0, (TypeId::of::<T>(), Box::new(val)));
    }

    // Internal: take without ITy bound (for DummyCallable dispatch).
    fn raw_take<T: 'static>(&mut self, repr: DummyRepr) -> T {
        let (type_id, val) = self.slots.remove(&repr.0).expect("slot not found");
        assert_eq!(type_id, TypeId::of::<T>(), "type mismatch in raw_take");
        *val.downcast::<T>().expect("downcast failed")
    }
}

impl Scope for DummyScope {
    type Repr = DummyRepr;
    type Owned = DummyOwned;

    fn store<T: ITy>(&mut self, repr: DummyRepr, val: T) {
        self.slots.insert(repr.0, (TypeId::of::<T>(), Box::new(val)));
    }

    fn clone<T: ITy + Clone>(&self, repr: &DummyRepr) -> T {
        let (type_id, val) = self.slots.get(&repr.0).expect("slot not found");
        assert_eq!(*type_id, TypeId::of::<T>(), "type mismatch in clone");
        val.downcast_ref::<T>().expect("downcast failed").clone()
    }

    fn take<T: ITy>(&mut self, repr: DummyRepr) -> T {
        let (type_id, val) = self.slots.remove(&repr.0).expect("slot not found (double take?)");
        assert_eq!(type_id, TypeId::of::<T>(), "type mismatch in take");
        *val.downcast::<T>().expect("downcast failed")
    }

    fn alloc_tmp(&mut self) -> DummyRepr {
        self.alloc()
    }

    async fn call_n<F, Args, R, E>(&mut self, f: &F, args: Args) -> Result<R, HostError>
    where
        F: Callable<Args, R, E>,
        Args: CallArgs,
        R: ITy,
        E: EffectParam,
    {
        // F is DummyOwned at runtime. Downcast via Any.
        let f_any = f as &dyn Any;
        let owned = f_any.downcast_ref::<DummyOwned>()
            .ok_or_else(|| HostError::CallFailed("F is not DummyOwned".into()))?;
        let callable = owned.0.downcast_ref::<DummyCallable>()
            .ok_or_else(|| HostError::CallFailed("DummyOwned does not contain a callable".into()))?;

        // Allocate temp slots for args + result.
        let arity = Args::arity();
        let arg_slots: Vec<DummyRepr> = (0..arity).map(|_| self.alloc()).collect();
        let dst = self.alloc();

        // Store args into slots.
        args.store_all(self, &arg_slots);

        // Dispatch: callable reads from arg_slots, writes to dst.
        // Need unsafe reborrow because dispatch takes &mut DummyScope.
        let dispatch = &callable.dispatch;
        let dispatch_ptr = dispatch.as_ref() as *const dyn Fn(&mut DummyScope, &[DummyRepr], DummyRepr) -> Result<(), HostError>;
        unsafe { (*dispatch_ptr)(self, &arg_slots, dst)? };

        // Take result.
        Ok(self.raw_take(dst))
    }
}

// ── DummyRegistrar ─────────────────────────────────────────────────

pub struct DummyRegistrar {
    pub drops: Vec<TypeId>,
    pub clones: Vec<TypeId>,
    pub copies: Vec<TypeId>,
}

impl DummyRegistrar {
    pub fn new() -> Self {
        Self { drops: Vec::new(), clones: Vec::new(), copies: Vec::new() }
    }
    pub fn has_drop<T: 'static>(&self) -> bool { self.drops.contains(&TypeId::of::<T>()) }
    pub fn has_clone<T: 'static>(&self) -> bool { self.clones.contains(&TypeId::of::<T>()) }
    pub fn has_copy<T: 'static>(&self) -> bool { self.copies.contains(&TypeId::of::<T>()) }
}

impl Registrar for DummyRegistrar {
    fn register_drop<T: ITy>(&mut self, _: fn(T)) {
        let id = TypeId::of::<T>();
        if !self.drops.contains(&id) { self.drops.push(id); }
    }
    fn register_clone<T: ITy + Clone>(&mut self, _: fn(&T) -> T) {
        let id = TypeId::of::<T>();
        if !self.clones.contains(&id) { self.clones.push(id); }
    }
    fn register_copy<T: ITy + Copy>(&mut self) {
        let id = TypeId::of::<T>();
        if !self.copies.contains(&id) { self.copies.push(id); }
    }
}
