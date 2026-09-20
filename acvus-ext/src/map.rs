//! `HashMap<K, V>` and `HashSet<K>`, with the key's
//! hash and equality passed at construction, as a `BuildHasher` and a
//! comparator are in Rust. The map stores the two closures and every
//! operation that has to find a key calls them, so a lookup carries the
//! join of their effects.
//!
//! Iteration is in insertion order. Rust leaves `HashMap`'s order
//! unspecified; the language's programs are run twice and compared
//! (`differential`), so an order that varied between runs would be a
//! failing corpus rather than a licence.
//!
//! A map is not journaled, and that is a decision rather than an omission.
//! `Journaled::decode_state` builds the whole value back from canonical
//! bytes; a closure has no canonical bytes, since `acvus-interpreter`'s
//! layout answers that a value of a function type is not held by a space;
//! and a map that lost its hasher on reload would answer every lookup
//! wrongly. The map a space can hold is the one whose key operations come
//! from the key's own instances at each call site (RFC-0067), which stores
//! no closure. It arrives with that RFC, and `docs/std/map.md` names it.
//!
//! **The element contract.** A handler takes `m: Ref<HashMap<K, V, E, Rt>, ..>`
//! or a `HashSet` as a declared parameter, so the checker unified `K` and `V`
//! with the key and value types of the table the argument names: every value
//! a table hands back was erased from that `V` (or `K`). Each `unsafe
//! { V::from_value(..) }` below cites this fact and nothing else.

use std::marker::PhantomData;

use acvus_extern::{
    Borrowable, BorrowableSpecialized, Closure, ClosureFn, Cross, Ctx, ExternTypeDecl, FromValue,
    FxHashMap, Interner, Mut, One, OneValue, Owned, PolyTy, PolyVars, QualifiedRef, Ref, Registry,
    Runtime, Shared, Specialized, Stored, Term, TransparentOver, TyArg, TyVarBound, TypeArg,
    UserDefinedDecl, Var, borrowed_as_self, extern_fn, extern_registry, kind,
};

use crate::iter::Iter;
use crate::iterator::sig;

/// The hash and the comparator as the declaration sees them: at the key
/// type and the effect the map's own type carries.
type HashOf<K, E, Rt> = Closure<(Ref<K, Shared, Rt>,), i64, E, Rt>;
type EqOf<K, E, Rt> = Closure<(Ref<K, Shared, Rt>, Ref<K, Shared, Rt>), bool, E, Rt>;

type Key<Rt> = Ref<Owned<Rt>, Shared, Rt>;
type Hasher<Rt> = Closure<(Key<Rt>,), i64, (), Rt>;
type Comparator<Rt> = Closure<(Key<Rt>, Key<Rt>), bool, (), Rt>;

/// A key's hash and equality as the table keeps them: at the erased key
/// every reference value is at run time, and at no effect, an effect being
/// a fact of the declaration that admitted the closure rather than
/// anything a lookup reads.
///
/// Concrete, and not `Box<dyn>`. `ClosureFn::call_now` takes the context as
/// `&mut Ctx<'_, Rt>`, a pointer to the one the machine owns; behind a
/// vtable that pointer would be to `Op::run`'s own slot, lent to a callee
/// LLVM must assume keeps it, and its sibling-call rule then refuses the
/// tail jump every operation owes its successor (RFC-0052, enforced by
/// `acvus-interpreter-test/benches/asm_probe.rs`).
struct Keying<Rt>
where
    Rt: Runtime,
{
    hash: Hasher<Rt>,
    eq: Comparator<Rt>,
}

impl<Rt> Keying<Rt>
where
    Rt: Runtime,
{
    fn of<K, E>(rt: &Rt, hash: HashOf<K, E, Rt>, eq: EqOf<K, E, Rt>) -> Self
    where
        K: Var<kind::Type>,
        E: Var<kind::Effect>,
    {
        Self {
            hash: Closure::new(rt, hash.into_value()),
            eq: Closure::new(rt, eq.into_value()),
        }
    }
}

/// An iterator over references into a borrowed container, read by
/// position. `iterator::lent_iter` is this function over a container that
/// is `Var<kind::Type>` in Rust; a declared extension type carries that
/// bound only where its own parameters are `TyArg`, which a handler's
/// parameters are not, so the map reads its entries through this one.
fn lent_parts<C, T, E, I, Rt>(
    container: Ref<C, Shared, Rt>,
    at: impl Fn(&C, usize) -> Option<&T> + Send + Sync + 'static,
) -> Iter<Ref<T, Shared, Rt>, E, I, Rt>
where
    C: Send + Sync + 'static,
    T: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    let mut index = 0;
    Iter::generate(move |rt| {
        let part = container.try_map(rt, |c| at(c, index));
        index += 1;
        part
    })
}

struct Entry<Rt>
where
    Rt: Runtime,
{
    hash: i64,
    binding: Binding<Rt>,
}

/// A key and the value it names.
struct Binding<Rt>
where
    Rt: Runtime,
{
    key: Owned<Rt>,
    value: Owned<Rt>,
}

/// Where a key belongs: its hash, and the entry already holding it.
struct Probe {
    hash: i64,
    at: Option<usize>,
}

/// The position of an entry `Table::push` has just added: it is in
/// `entries` and not yet in `positions`, so it is not a candidate for its
/// own key. Only `seek_pushed`, `seek_pushed_now` and `settle` take one,
/// and `settle` consumes it.
struct Pushed(usize);

/// What became of a binding the table took in: the position its key
/// occupies, and the binding turned away where the key was already there.
struct Placed<Rt>
where
    Rt: Runtime,
{
    at: usize,
    turned_away: Option<Binding<Rt>>,
}

/// The storage a `Map` and a `Set` are both laid out as: entries in
/// insertion order, and the positions each hash occupies.
pub struct Table<Rt>
where
    Rt: Runtime,
{
    entries: Vec<Entry<Rt>>,
    positions: FxHashMap<i64, Vec<usize>>,
    keying: Keying<Rt>,
}

/// An entry's key or value at the element type the handler declares. The
/// storage holds the runtime's own value, and `TransparentOver<Rt>` is the
/// promise that a `T` is that value's layout — the cast `acvus-extern`'s
/// `reference` module makes to read a container's elements in place.
fn at_element<T, Rt>(owned: &Owned<Rt>) -> &T
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    // SAFETY: `T: TransparentOver<Rt>` and `Owned<Rt>` is `repr(transparent)`
    // over `Rt::Value`, so the two are one layout.
    unsafe { &*(owned as *const Owned<Rt>).cast::<T>() }
}

fn at_element_mut<T, Rt>(owned: &mut Owned<Rt>) -> &mut T
where
    T: TransparentOver<Rt>,
    Rt: Runtime,
{
    // SAFETY: as `at_element`, with the caller's exclusive loan.
    unsafe { &mut *(owned as *mut Owned<Rt>).cast::<T>() }
}

/// A reference value naming a held value where it lies.
fn lend<Rt>(rt: &Rt, held: &Owned<Rt>) -> Rt::Value
where
    Rt: Runtime,
{
    // SAFETY: the storage is live for the whole call, and a closure the
    // reference is handed to keeps it no longer than the call (RFC-0018).
    unsafe { rt.reference(held) }
}

fn binding<K, V, Rt>(rt: &Rt, key: K, value: V) -> Binding<Rt>
where
    K: OneValue<Rt>,
    V: OneValue<Rt>,
    Rt: Runtime,
{
    Binding {
        key: Owned::<Rt>::from_value(key.erase(rt)),
        value: Owned::<Rt>::from_value(value.erase(rt)),
    }
}

/// A set's binding: the key, and the unit the runtime writes for a value
/// nothing reads.
fn keyed<Rt>(rt: &Rt, key: Rt::Value) -> Binding<Rt>
where
    Rt: Runtime,
{
    Binding {
        key: Owned::<Rt>::from_value(key),
        value: Owned::<Rt>::from_value(rt.undef()),
    }
}

impl<Rt> Table<Rt>
where
    Rt: Runtime,
{
    fn new(keying: Keying<Rt>, capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            positions: FxHashMap::default(),
            keying,
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.positions.clear();
    }

    /// The entries whose key hashes to `hash`. A hash no entry has is a
    /// hash with no candidates, so the empty list is the answer.
    fn candidates(&self, hash: i64) -> Vec<usize> {
        self.positions.get(&hash).cloned().unwrap_or_default()
    }

    /// Where the key `probe` — a `&K` reference value — belongs.
    fn seek_now(&self, ctx: &mut Ctx<'_, Rt>, probe: Rt::Value) -> Probe {
        let rt = ctx.rt;
        let hash = self.keying.hash.call_now(ctx, (Ref::new(probe),));
        for at in self.candidates(hash) {
            let held = lend(rt, &self.entries[at].binding.key);
            let same = (Ref::new(probe), Ref::new(held));
            if self.keying.eq.call_now(ctx, same) {
                return Probe { hash, at: Some(at) };
            }
        }
        Probe { hash, at: None }
    }

    async fn seek(&self, ctx: &mut Ctx<'_, Rt>, probe: Rt::Value) -> Probe {
        let rt = ctx.rt;
        let hash = self.keying.hash.call(ctx, (Ref::new(probe),)).await;
        for at in self.candidates(hash) {
            let held = lend(rt, &self.entries[at].binding.key);
            let same = (Ref::new(probe), Ref::new(held));
            if self.keying.eq.call(ctx, same).await {
                return Probe { hash, at: Some(at) };
            }
        }
        Probe { hash, at: None }
    }

    fn push(&mut self, binding: Binding<Rt>) -> Pushed {
        let at = self.entries.len();
        self.entries.push(Entry { hash: 0, binding });
        Pushed(at)
    }

    fn seek_pushed_now(&self, ctx: &mut Ctx<'_, Rt>, fresh: &Pushed) -> Probe {
        let rt = ctx.rt;
        self.seek_now(ctx, lend(rt, &self.entries[fresh.0].binding.key))
    }

    async fn seek_pushed(&self, ctx: &mut Ctx<'_, Rt>, fresh: &Pushed) -> Probe {
        let rt = ctx.rt;
        self.seek(ctx, lend(rt, &self.entries[fresh.0].binding.key))
            .await
    }

    fn settle(&mut self, probe: Probe, fresh: Pushed) -> Placed<Rt> {
        let Some(at) = probe.at else {
            self.entries[fresh.0].hash = probe.hash;
            self.positions.entry(probe.hash).or_default().push(fresh.0);
            return Placed {
                at: fresh.0,
                turned_away: None,
            };
        };
        let turned_away = self.entries.pop().expect("the entry `push` just added");
        Placed {
            at,
            turned_away: Some(turned_away.binding),
        }
    }

    /// Rust's `insert`: the new value takes the key's entry, which keeps
    /// the position it was first inserted at, and the value it displaced is
    /// the answer.
    fn put_now(&mut self, ctx: &mut Ctx<'_, Rt>, binding: Binding<Rt>) -> Option<Owned<Rt>> {
        let fresh = self.push(binding);
        let probe = self.seek_pushed_now(ctx, &fresh);
        self.displace(probe, fresh)
    }

    async fn put(&mut self, ctx: &mut Ctx<'_, Rt>, binding: Binding<Rt>) -> Option<Owned<Rt>> {
        let fresh = self.push(binding);
        let probe = self.seek_pushed(ctx, &fresh).await;
        self.displace(probe, fresh)
    }

    fn displace(&mut self, probe: Probe, fresh: Pushed) -> Option<Owned<Rt>> {
        let placed = self.settle(probe, fresh);
        let new = placed.turned_away?.value;
        Some(std::mem::replace(
            &mut self.entries[placed.at].binding.value,
            new,
        ))
    }

    /// Rust's `entry(k).or_insert(v)` and `HashSet::insert`: a key already
    /// there keeps the entry it has, and the binding offered is turned
    /// away.
    fn occupy_now(&mut self, ctx: &mut Ctx<'_, Rt>, binding: Binding<Rt>) -> Placed<Rt> {
        let fresh = self.push(binding);
        let probe = self.seek_pushed_now(ctx, &fresh);
        self.settle(probe, fresh)
    }

    async fn occupy(&mut self, ctx: &mut Ctx<'_, Rt>, binding: Binding<Rt>) -> Placed<Rt> {
        let fresh = self.push(binding);
        let probe = self.seek_pushed(ctx, &fresh).await;
        self.settle(probe, fresh)
    }

    /// As `IndexMap::shift_remove`: the entries after it move down, so what
    /// is left keeps the order it was inserted in.
    fn take_out(&mut self, at: usize) -> Entry<Rt> {
        let entry = self.entries.remove(at);
        self.reindex();
        entry
    }

    fn reindex(&mut self) {
        self.positions.clear();
        for (at, entry) in self.entries.iter().enumerate() {
            self.positions.entry(entry.hash).or_default().push(at);
        }
    }

    fn drain_entries(&mut self) -> Vec<Entry<Rt>> {
        self.positions.clear();
        std::mem::take(&mut self.entries)
    }

    fn refill(&mut self, entries: Vec<Entry<Rt>>) {
        self.entries = entries;
        self.reindex();
    }
}

/// A count or a capacity wider than the address space is the overflow
/// `Vec::with_capacity` itself reports.
fn as_capacity(n: u64) -> usize {
    let Ok(n) = usize::try_from(n) else {
        panic!("capacity overflow")
    };
    n
}

/// Any type is a key or a value.
fn unbounded<T>() -> TyVarBound {
    TyVarBound::Any
}

/// No slot of a map or a set specializes: the storage holds the runtime's
/// own values whatever the key and value types are.
fn uniform_slot<T>() -> bool {
    false
}

/// The crossing of a declared extension type stored as itself, at the one
/// runtime its own parameter names.
///
/// `#[derive(ExternType)]` stores a type as its payload, and `Ref::with` —
/// the reader behind `get`, `keys` and `values` — reads a reference's
/// storage as the declared Rust type, so a `Ref` into a derived map derefs
/// at the payload's type and trips the runtime's type check. `Deque` is
/// stored as itself and has no such gap; `cross_as_stored!` writes that for
/// a Rust type naming no runtime, and a map names one, so the impls are
/// here. The second runtime parameter `cross_as_stored!` would add is what
/// this does not have: it would let a value of one runtime erase into
/// another.
macro_rules! stored_extern_type {
    ($t:ident<$($k:ident),+>, name: $name:literal) => {
        impl<$($k,)+ E, Rt> Var<kind::Type> for $t<$($k,)+ E, Rt>
        where
            $($k: Var<kind::Type>,)+
            E: Var<kind::Effect>,
            Rt: Runtime,
        {
        }

        impl<$($k,)+ E, Rt> TyArg for $t<$($k,)+ E, Rt>
        where
            $($k: TyArg + Var<kind::Type>,)+
            E: Term<kind::Effect> + Var<kind::Effect>,
            Rt: Runtime,
        {
            fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
                PolyTy::UserDefined {
                    id: QualifiedRef::root(i.intern($name)),
                    type_args: vec![$(TypeArg::uniform($k::poly_ty(i, vars))),+],
                    effect_args: vec![<E as Term<kind::Effect>>::poly(vars)],
                    identity_args: vec![],
                }
            }
        }

        impl<$($k,)+ E, Rt> ExternTypeDecl for $t<$($k,)+ E, Rt>
        where
            $($k: Var<kind::Type>,)+
            E: Var<kind::Effect>,
            Rt: Runtime,
        {
            fn type_decl(i: &Interner) -> UserDefinedDecl {
                UserDefinedDecl {
                    qref: QualifiedRef::root(i.intern($name)),
                    type_params: vec![$(unbounded::<$k>()),+],
                    effect_params: 1,
                    identity_params: 0,
                    specializable: vec![$(uniform_slot::<$k>()),+],
                }
            }
        }

        impl<$($k,)+ E, Rt> Stored<Rt> for $t<$($k,)+ E, Rt>
        where
            $($k: Var<kind::Type>,)+
            E: Var<kind::Effect>,
            Rt: Runtime,
        {
        }

        impl<$($k,)+ E, Rt> Borrowable<Rt> for $t<$($k,)+ E, Rt>
        where
            $($k: Var<kind::Type>,)+
            E: Var<kind::Effect>,
            Rt: Runtime,
        {
        }

        impl<$($k,)+ E, Rt> BorrowableSpecialized<Rt> for $t<$($k,)+ E, Rt>
        where
            $($k: Var<kind::Type>,)+
            E: Var<kind::Effect>,
            Rt: Runtime,
        {
        }

        impl<$($k,)+ E, Rt> Cross<Rt> for $t<$($k,)+ E, Rt>
        where
            $($k: Var<kind::Type>,)+
            E: Var<kind::Effect>,
            Rt: Runtime,
        {
            type Form = One;
            type ReturnForm = One;

            unsafe fn from_run(rt: &Rt, run: &[Rt::Value]) -> Self {
                // SAFETY: the caller's contract, at one value.
                unsafe { <Self as OneValue<Rt>>::from_run(rt, run) }
            }

            fn into_run(self, rt: &Rt, out: &mut [Rt::Value]) {
                <Self as OneValue<Rt>>::into_run(self, rt, out)
            }
        }

        impl<$($k,)+ E, Rt> OneValue<Rt> for $t<$($k,)+ E, Rt>
        where
            $($k: Var<kind::Type>,)+
            E: Var<kind::Effect>,
            Rt: Runtime,
        {
            acvus_extern::whole_box!($t<$($k,)+ E, Rt>, Rt);
        }

        impl<$($k,)+ E, Rt> OneValue<Rt, Specialized> for $t<$($k,)+ E, Rt>
        where
            $($k: Var<kind::Type>,)+
            E: Var<kind::Effect>,
            Rt: Runtime,
        {
            acvus_extern::whole_box!($t<$($k,)+ E, Rt>, Rt);
        }

        unsafe impl<$($k,)+ E, Rt> FromValue<Rt> for $t<$($k,)+ E, Rt>
        where
            $($k: Var<kind::Type>,)+
            E: Var<kind::Effect>,
            Rt: Runtime,
        {
            unsafe fn from_value(rt: &Rt, value: Rt::Value) -> Self {
                acvus_extern::debug_assert_erased_from!(rt, &value, $t<$($k,)+ E, Rt>);
                // SAFETY: the trait's contract — a table crosses as itself
                // (`whole_box!` above), so the value was erased from this
                // table type at the site the checker matched.
                unsafe { rt.materialize::<$t<$($k,)+ E, Rt>>(value) }
            }
        }

        borrowed_as_self!(
            $t<$($k,)+ E, Rt>,
            $($k: Var<kind::Type>,)+ E: Var<kind::Effect>, Rt: Runtime
        );
    };
}

// -- The map ------------------------------------------------------------

pub struct HashMap<K, V, E, Rt>(Table<Rt>, PhantomData<(K, V, E)>)
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime;

stored_extern_type!(HashMap<K, V>, name: "HashMap");

fn new_map<K, V, E, Rt>(
    rt: &Rt,
    hash: HashOf<K, E, Rt>,
    eq: EqOf<K, E, Rt>,
    capacity: usize,
) -> HashMap<K, V, E, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    HashMap(Table::new(Keying::of(rt, hash, eq), capacity), PhantomData)
}

#[extern_fn(effect = pure)]
fn hash_map<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    hash: HashOf<K, E, Rt>,
    eq: EqOf<K, E, Rt>,
) -> HashMap<K, V, E, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    new_map(rt, hash, eq, 0)
}

#[extern_fn(effect = pure)]
fn with_capacity<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    n: u64,
    hash: HashOf<K, E, Rt>,
    eq: EqOf<K, E, Rt>,
) -> HashMap<K, V, E, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    new_map(rt, hash, eq, as_capacity(n))
}

#[extern_fn(effect = pure)]
fn len<K, V, E, Rt>(m: &HashMap<K, V, E, Rt>) -> u64
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    m.0.len() as u64
}

#[extern_fn(effect = pure)]
fn is_empty<K, V, E, Rt>(m: &HashMap<K, V, E, Rt>) -> bool
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    m.0.is_empty()
}

#[extern_fn(effect = pure)]
fn clear<K, V, E, Rt>(m: &mut HashMap<K, V, E, Rt>)
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    m.0.clear();
}

#[extern_fn(effect = pure)]
fn keys<K, V, E, I, Rt>(
    m: Ref<HashMap<K, V, E, Rt>, Shared, Rt>,
) -> Iter<Ref<K, Shared, Rt>, E, I, Rt>
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    lent_parts(m, |m, at| {
        m.0.entries.get(at).map(|e| at_element(&e.binding.key))
    })
}

#[extern_fn(effect = pure)]
fn values<K, V, E, I, Rt>(
    m: Ref<HashMap<K, V, E, Rt>, Shared, Rt>,
) -> Iter<Ref<V, Shared, Rt>, E, I, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    lent_parts(m, |m, at| {
        m.0.entries.get(at).map(|e| at_element(&e.binding.value))
    })
}

// There is no `values_mut`. Declared as
// `Iter<Ref<V, Mut, Rt>, E, I, Rt>` it compiles, and a script that stores
// through an element of it is refused with "cannot store through &i64: not
// a `&mut`": an iterator's element type argument does not carry the
// exclusive loan. `get_mut` is the exclusive loan the boundary does carry,
// one key at a time.

fn drained<K, V, T, E, I, Rt>(
    m: HashMap<K, V, E, Rt>,
    part: impl Fn(Entry<Rt>) -> Owned<Rt> + Send + 'static,
) -> Iter<T, E, I, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    T: Var<kind::Type> + OneValue<Rt> + FromValue<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    let mut entries = m.0.entries.into_iter();
    Iter::generate(move |rt| {
        entries
            .next()
            // SAFETY: the element contract at this module's head.
            .map(|entry| unsafe { T::from_value(rt, part(entry).into_value()) })
    })
}

#[extern_fn(effect = pure)]
fn into_keys<K, V, E, I, Rt>(m: HashMap<K, V, E, Rt>) -> Iter<K, E, I, Rt>
where
    K: Var<kind::Type> + OneValue<Rt> + FromValue<Rt>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    drained(m, |entry| entry.binding.key)
}

#[extern_fn(effect = pure)]
fn into_values<K, V, E, I, Rt>(m: HashMap<K, V, E, Rt>) -> Iter<V, E, I, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + OneValue<Rt> + FromValue<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    drained(m, |entry| entry.binding.value)
}

fn insert_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    key: K,
    value: V,
) -> Option<V>
where
    K: Var<kind::Type> + OneValue<Rt>,
    V: Var<kind::Type> + OneValue<Rt> + FromValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let old = m.0.put_now(ctx, binding(rt, key, value))?;
    // SAFETY: the element contract at this module's head.
    Some(unsafe { V::from_value(rt, old.into_value()) })
}

#[extern_fn(effect = E, sync = insert_now)]
async fn insert<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    key: K,
    value: V,
) -> Option<V>
where
    K: Var<kind::Type> + OneValue<Rt>,
    V: Var<kind::Type> + OneValue<Rt> + FromValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let old = m.0.put(ctx, binding(rt, key, value)).await?;
    // SAFETY: the element contract at this module's head.
    Some(unsafe { V::from_value(rt, old.into_value()) })
}

fn get_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: Ref<HashMap<K, V, E, Rt>, Shared, Rt>,
    key: Ref<K, Shared, Rt>,
) -> Option<Ref<V, Shared, Rt>>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let at = m.with(rt, |m| &m.0).seek_now(ctx, key.into_value()).at?;
    Some(m.map(rt, |m| at_element(&m.0.entries[at].binding.value)))
}

#[extern_fn(effect = E, sync = get_now)]
async fn get<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: Ref<HashMap<K, V, E, Rt>, Shared, Rt>,
    key: Ref<K, Shared, Rt>,
) -> Option<Ref<V, Shared, Rt>>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let at = m.with(rt, |m| &m.0).seek(ctx, key.into_value()).await.at?;
    Some(m.map(rt, |m| at_element(&m.0.entries[at].binding.value)))
}

fn get_mut_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: Ref<HashMap<K, V, E, Rt>, Mut, Rt>,
    key: Ref<K, Shared, Rt>,
) -> Option<Ref<V, Mut, Rt>>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let at = m.with(rt, |m| &m.0).seek_now(ctx, key.into_value()).at?;
    Some(m.map(rt, |m| at_element_mut(&mut m.0.entries[at].binding.value)))
}

#[extern_fn(effect = E, sync = get_mut_now)]
async fn get_mut<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: Ref<HashMap<K, V, E, Rt>, Mut, Rt>,
    key: Ref<K, Shared, Rt>,
) -> Option<Ref<V, Mut, Rt>>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let at = m.with(rt, |m| &m.0).seek(ctx, key.into_value()).await.at?;
    Some(m.map(rt, |m| at_element_mut(&mut m.0.entries[at].binding.value)))
}

fn contains_key_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &HashMap<K, V, E, Rt>,
    key: Ref<K, Shared, Rt>,
) -> bool
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    m.0.seek_now(ctx, key.into_value()).at.is_some()
}

#[extern_fn(effect = E, sync = contains_key_now)]
async fn contains_key<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &HashMap<K, V, E, Rt>,
    key: Ref<K, Shared, Rt>,
) -> bool
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    m.0.seek(ctx, key.into_value()).await.at.is_some()
}

fn remove_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    key: Ref<K, Shared, Rt>,
) -> Option<V>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + FromValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let at = m.0.seek_now(ctx, key.into_value()).at?;
    // SAFETY: the element contract at this module's head.
    Some(unsafe { V::from_value(rt, m.0.take_out(at).binding.value.into_value()) })
}

#[extern_fn(effect = E, sync = remove_now)]
async fn remove<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    key: Ref<K, Shared, Rt>,
) -> Option<V>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + FromValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let at = m.0.seek(ctx, key.into_value()).await.at?;
    // SAFETY: the element contract at this module's head.
    Some(unsafe { V::from_value(rt, m.0.take_out(at).binding.value.into_value()) })
}

fn or_insert_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: Ref<HashMap<K, V, E, Rt>, Mut, Rt>,
    key: K,
    value: V,
) -> Ref<V, Mut, Rt>
where
    K: Var<kind::Type> + OneValue<Rt>,
    V: Var<kind::Type> + OneValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let one = binding(rt, key, value);
    let at = m.with(rt, |m| m.0.occupy_now(ctx, one)).at;
    m.map(rt, |m| at_element_mut(&mut m.0.entries[at].binding.value))
}

#[extern_fn(effect = E, sync = or_insert_now)]
async fn or_insert<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: Ref<HashMap<K, V, E, Rt>, Mut, Rt>,
    key: K,
    value: V,
) -> Ref<V, Mut, Rt>
where
    K: Var<kind::Type> + OneValue<Rt>,
    V: Var<kind::Type> + OneValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let one = binding(rt, key, value);
    let at = m.with(rt, |m| m.0.occupy(ctx, one)).await.at;
    m.map(rt, |m| at_element_mut(&mut m.0.entries[at].binding.value))
}

fn extend_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    other: HashMap<K, V, E, Rt>,
) where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    for entry in other.0.entries {
        drop(m.0.put_now(ctx, entry.binding));
    }
}

#[extern_fn(effect = E, sync = extend_now)]
async fn extend<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    other: HashMap<K, V, E, Rt>,
) where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    for entry in other.0.entries {
        drop(m.0.put(ctx, entry.binding).await);
    }
}

type KeepOf<K, V, E, Rt> = Closure<(Ref<K, Shared, Rt>, Ref<V, Shared, Rt>), bool, E, Rt>;

fn retain_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    keep: KeepOf<K, V, E, Rt>,
) where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut kept = Vec::with_capacity(m.0.len());
    for entry in m.0.drain_entries() {
        let key = Ref::new(lend(rt, &entry.binding.key));
        let value = Ref::new(lend(rt, &entry.binding.value));
        if keep.call_now(ctx, (key, value)) {
            kept.push(entry);
        }
    }
    m.0.refill(kept);
}

#[extern_fn(effect = E, sync = retain_now)]
async fn retain<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    keep: KeepOf<K, V, E, Rt>,
) where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut kept = Vec::with_capacity(m.0.len());
    for entry in m.0.drain_entries() {
        let key = Ref::new(lend(rt, &entry.binding.key));
        let value = Ref::new(lend(rt, &entry.binding.value));
        if keep.call(ctx, (key, value)).await {
            kept.push(entry);
        }
    }
    m.0.refill(kept);
}

pub fn map_registry<Rt>() -> Registry<Rt>
where
    Rt: Runtime,
{
    extern_registry! {
        ns: "map",
        types: [HashMap<_, _, _, Rt>],
        fns: [
            hash_map, with_capacity, len, is_empty, clear,
            keys, values, into_keys, into_values,
            insert, get, get_mut, contains_key, remove, or_insert, extend, retain,
        ],
    }
}

// -- The set ------------------------------------------------------------

pub struct HashSet<K, E, Rt>(Table<Rt>, PhantomData<(K, E)>)
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime;

stored_extern_type!(HashSet<K>, name: "HashSet");

impl<K, E, Rt> HashSet<K, E, Rt>
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn table(&self) -> &Table<Rt> {
        &self.0
    }

    fn table_mut(&mut self) -> &mut Table<Rt> {
        &mut self.0
    }
}

#[extern_fn(effect = pure)]
fn hash_set<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    hash: HashOf<K, E, Rt>,
    eq: EqOf<K, E, Rt>,
) -> HashSet<K, E, Rt>
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    HashSet(Table::new(Keying::of(rt, hash, eq), 0), PhantomData)
}

#[extern_fn(name = "len", effect = pure)]
fn set_len<K, E, Rt>(s: &HashSet<K, E, Rt>) -> u64
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    s.table().len() as u64
}

#[extern_fn(name = "is_empty", effect = pure)]
fn set_is_empty<K, E, Rt>(s: &HashSet<K, E, Rt>) -> bool
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    s.table().is_empty()
}

#[extern_fn(name = "clear", effect = pure)]
fn set_clear<K, E, Rt>(s: &mut HashSet<K, E, Rt>)
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    s.table_mut().clear();
}

#[extern_fn(instance_of = sig::as_iter, effect = pure)]
fn as_iter_set<K, E, I, Rt>(
    s: Ref<HashSet<K, E, Rt>, Shared, Rt>,
) -> Iter<Ref<K, Shared, Rt>, E, I, Rt>
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    lent_parts(s, |s, at| {
        s.table()
            .entries
            .get(at)
            .map(|e| at_element(&e.binding.key))
    })
}

#[extern_fn(instance_of = sig::into_iter, effect = pure)]
fn into_iter_set<K, E, I, Rt>(s: HashSet<K, E, Rt>) -> Iter<K, E, I, Rt>
where
    K: Var<kind::Type> + OneValue<Rt> + FromValue<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    let mut entries = s.0.entries.into_iter();
    Iter::generate(move |rt| {
        entries
            .next()
            // SAFETY: the element contract at this module's head.
            .map(|entry| unsafe { K::from_value(rt, entry.binding.key.into_value()) })
    })
}

/// Rust's `HashSet::insert` answers whether the set gained the key, and a
/// key already there is left as it was.
fn set_insert_now<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, s: &mut HashSet<K, E, Rt>, key: K) -> bool
where
    K: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let one = keyed(rt, key.erase(rt));
    s.table_mut().occupy_now(ctx, one).turned_away.is_none()
}

#[extern_fn(name = "insert", effect = E, sync = set_insert_now)]
async fn set_insert<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, s: &mut HashSet<K, E, Rt>, key: K) -> bool
where
    K: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let one = keyed(rt, key.erase(rt));
    s.table_mut().occupy(ctx, one).await.turned_away.is_none()
}

fn contains_now<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    s: &HashSet<K, E, Rt>,
    key: Ref<K, Shared, Rt>,
) -> bool
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    s.table().seek_now(ctx, key.into_value()).at.is_some()
}

#[extern_fn(effect = E, sync = contains_now)]
async fn contains<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    s: &HashSet<K, E, Rt>,
    key: Ref<K, Shared, Rt>,
) -> bool
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    s.table().seek(ctx, key.into_value()).await.at.is_some()
}

/// Rust's `HashSet::remove` answers whether the key was there.
fn set_remove_now<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    s: &mut HashSet<K, E, Rt>,
    key: Ref<K, Shared, Rt>,
) -> bool
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let table = s.table_mut();
    let Some(at) = table.seek_now(ctx, key.into_value()).at else {
        return false;
    };
    drop(table.take_out(at));
    true
}

#[extern_fn(name = "remove", effect = E, sync = set_remove_now)]
async fn set_remove<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    s: &mut HashSet<K, E, Rt>,
    key: Ref<K, Shared, Rt>,
) -> bool
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let table = s.table_mut();
    let Some(at) = table.seek(ctx, key.into_value()).await.at else {
        return false;
    };
    drop(table.take_out(at));
    true
}

fn set_extend_now<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    s: &mut HashSet<K, E, Rt>,
    other: HashSet<K, E, Rt>,
) where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    for entry in other.0.entries {
        s.table_mut().occupy_now(ctx, entry.binding);
    }
}

#[extern_fn(name = "extend", effect = E, sync = set_extend_now)]
async fn set_extend<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    s: &mut HashSet<K, E, Rt>,
    other: HashSet<K, E, Rt>,
) where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    for entry in other.0.entries {
        s.table_mut().occupy(ctx, entry.binding).await;
    }
}

/// Whether `b` holds the key: `b`'s own hasher and comparator decide, as
/// they do for every lookup in `b`.
fn keeps_now<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, b: &HashSet<K, E, Rt>, key: &Owned<Rt>) -> bool
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    b.table().seek_now(ctx, lend(rt, key)).at.is_some()
}

async fn keeps<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, b: &HashSet<K, E, Rt>, key: &Owned<Rt>) -> bool
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    b.table().seek(ctx, lend(rt, key)).await.at.is_some()
}

/// Rust's `intersection` and `difference` borrow both sets and yield
/// references, which needs a clone of a key to build a set out of;
/// `Runtime` offers no clone of a value, so these two consume both sets and
/// return the one that is left. The hasher and comparator kept are the
/// receiver's.
fn intersection_now<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    mut a: HashSet<K, E, Rt>,
    b: HashSet<K, E, Rt>,
) -> HashSet<K, E, Rt>
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut kept = Vec::new();
    for entry in a.table_mut().drain_entries() {
        if keeps_now(ctx, &b, &entry.binding.key) {
            kept.push(entry);
        }
    }
    a.table_mut().refill(kept);
    a
}

#[extern_fn(effect = E, sync = intersection_now)]
async fn intersection<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    mut a: HashSet<K, E, Rt>,
    b: HashSet<K, E, Rt>,
) -> HashSet<K, E, Rt>
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut kept = Vec::new();
    for entry in a.table_mut().drain_entries() {
        if keeps(ctx, &b, &entry.binding.key).await {
            kept.push(entry);
        }
    }
    a.table_mut().refill(kept);
    a
}

fn difference_now<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    mut a: HashSet<K, E, Rt>,
    b: HashSet<K, E, Rt>,
) -> HashSet<K, E, Rt>
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut kept = Vec::new();
    for entry in a.table_mut().drain_entries() {
        if !keeps_now(ctx, &b, &entry.binding.key) {
            kept.push(entry);
        }
    }
    a.table_mut().refill(kept);
    a
}

#[extern_fn(effect = E, sync = difference_now)]
async fn difference<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    mut a: HashSet<K, E, Rt>,
    b: HashSet<K, E, Rt>,
) -> HashSet<K, E, Rt>
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut kept = Vec::new();
    for entry in a.table_mut().drain_entries() {
        if !keeps(ctx, &b, &entry.binding.key).await {
            kept.push(entry);
        }
    }
    a.table_mut().refill(kept);
    a
}

fn is_subset_now<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    a: &HashSet<K, E, Rt>,
    b: &HashSet<K, E, Rt>,
) -> bool
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    for entry in &a.table().entries {
        if !keeps_now(ctx, b, &entry.binding.key) {
            return false;
        }
    }
    true
}

#[extern_fn(effect = E, sync = is_subset_now)]
async fn is_subset<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    a: &HashSet<K, E, Rt>,
    b: &HashSet<K, E, Rt>,
) -> bool
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    for entry in &a.table().entries {
        if !keeps(ctx, b, &entry.binding.key).await {
            return false;
        }
    }
    true
}

fn from_iter_now<K, E, I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    mut it: Iter<K, E, I, Rt>,
    hash: HashOf<K, E, Rt>,
    eq: EqOf<K, E, Rt>,
) -> HashSet<K, E, Rt>
where
    K: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut s = HashSet(Table::new(Keying::of(rt, hash, eq), 0), PhantomData);
    crate::iter::drain_now!(it, ctx, |value| {
        let one = keyed(rt, value);
        s.table_mut().occupy_now(ctx, one);
    });
    s
}

#[extern_fn(effect = E, sync = from_iter_now)]
async fn from_iter<K, E, I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    mut it: Iter<K, E, I, Rt>,
    hash: HashOf<K, E, Rt>,
    eq: EqOf<K, E, Rt>,
) -> HashSet<K, E, Rt>
where
    K: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    let rt = ctx.rt;
    let mut s = HashSet(Table::new(Keying::of(rt, hash, eq), 0), PhantomData);
    crate::iter::drain!(it, ctx, |value| {
        let one = keyed(rt, value);
        s.table_mut().occupy(ctx, one).await;
    });
    s
}

pub fn set_registry<Rt>() -> Registry<Rt>
where
    Rt: Runtime,
{
    extern_registry! {
        ns: "set",
        types: [HashSet<_, _, Rt>],
        fns: [
            hash_set, set_len, set_is_empty, set_clear,
            as_iter_set, into_iter_set,
            set_insert, contains, set_remove, set_extend,
            intersection, difference, is_subset, from_iter,
        ],
    }
}
