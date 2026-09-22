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
//! **The element contract.** The table holds its keys at `K` and its values
//! at `V` — the handler's own type variables — and the two closures at the
//! `K` and `E` the map's type carries (RFC-0068 D3). A handler takes
//! `m: &HashMap<K, V, E, Rt>` or a `HashSet` as a declared parameter, so the
//! checker unified those variables with the key and value types of the table
//! the argument names, and Rust's own checker carries that decision from
//! there: nothing below reads a runtime value back at a type, because
//! nothing below holds a runtime value.

use std::marker::PhantomData;
use std::ops::DerefMut;

use acvus_extern::{Instance, Later};
use acvus_extern::{
    Borrowable, BorrowableSpecialized, Closure, ClosureFn, Cross, Ctx, ExternType, ExternTypeDecl,
    FxHashMap, Interner, One, OneValue, PassedByValue, PolyTy, PolyVars, QualifiedRef, Ref,
    Registry, Runtime, Shared, Specialized, Stored, Term, TransparentOver, TyArg, TyVarBound,
    TypeArg, UserDefinedDecl, Var, borrowed_as_self, extern_fn, extern_registry, kind,
};

use crate::iter::{Items, Refs, sig};

/// The hash and the comparator as the declaration sees them, and as the
/// table keeps them: at the key type and the effect the map's own type
/// carries. A closure is stored at the types it was declared with and is
/// not re-spelled (RFC-0068 D1).
type HashOf<K, E, Rt> = Closure<(Ref<K, Shared, Rt>,), i64, E, Rt>;
type EqOf<K, E, Rt> = Closure<(Ref<K, Shared, Rt>, Ref<K, Shared, Rt>), bool, E, Rt>;

/// A key's hash and equality as the table keeps them.
///
/// Concrete, and not `Box<dyn>`. `ClosureFn::call_now` takes the context as
/// `&mut Ctx<'_, Rt>`, a pointer to the one the machine owns; behind a
/// vtable that pointer would be to `Op::run`'s own slot, lent to a callee
/// LLVM must assume keeps it, and its sibling-call rule then refuses the
/// tail jump every operation owes its successor (RFC-0052, enforced by
/// `acvus-interpreter-test/benches/asm_probe.rs`).
struct Keying<K, E, Rt>
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    hash: HashOf<K, E, Rt>,
    eq: EqOf<K, E, Rt>,
}

struct Entry<K, V> {
    hash: i64,
    binding: Binding<K, V>,
}

/// A key and the value it names, each at the type the declaration gave it.
struct Binding<K, V> {
    key: K,
    value: V,
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
struct Placed<K, V> {
    at: usize,
    turned_away: Option<Binding<K, V>>,
}

/// The storage a `Map` and a `Set` are both laid out as: entries in
/// insertion order, and the positions each hash occupies. A set's value
/// slot is `()`: nothing reads it, and a runtime value written there would
/// be a value the table owns for no reader.
pub struct Table<K, V, E, Rt>
where
    K: Var<kind::Type>,
    V: Send + Sync + 'static,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    entries: Vec<Entry<K, V>>,
    positions: FxHashMap<i64, Vec<usize>>,
    keying: Keying<K, E, Rt>,
}

impl<K, V, E, Rt> Table<K, V, E, Rt>
where
    K: Var<kind::Type>,
    V: Send + Sync + 'static,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    fn new(keying: Keying<K, E, Rt>, capacity: usize) -> Self {
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

    fn push(&mut self, binding: Binding<K, V>) -> Pushed {
        let at = self.entries.len();
        self.entries.push(Entry { hash: 0, binding });
        Pushed(at)
    }

    fn settle(&mut self, probe: Probe, fresh: Pushed) -> Placed<K, V> {
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

    fn displace(&mut self, probe: Probe, fresh: Pushed) -> Option<V> {
        let placed = self.settle(probe, fresh);
        let new = placed.turned_away?.value;
        Some(std::mem::replace(
            &mut self.entries[placed.at].binding.value,
            new,
        ))
    }

    /// As `IndexMap::shift_remove`: the entries after it move down, so what
    /// is left keeps the order it was inserted in.
    fn take_out(&mut self, at: usize) -> Entry<K, V> {
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

    fn drain_entries(&mut self) -> Vec<Entry<K, V>> {
        self.positions.clear();
        std::mem::take(&mut self.entries)
    }

    fn refill(&mut self, entries: Vec<Entry<K, V>>) {
        self.entries = entries;
        self.reindex();
    }
}

/// Everything a lookup reaches: the key crosses into the two closures as
/// Rust's `&K`, which is the passed form of the `&K` their declaration
/// names (`Passed`, RFC-0018).
impl<K, V, E, Rt> Table<K, V, E, Rt>
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    V: Send + Sync + 'static,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    /// Where the key `probe` belongs.
    fn seek_now(&self, ctx: &mut Ctx<'_, Rt>, probe: &K) -> Probe {
        let hash = self.keying.hash.call_now(ctx, (probe,));
        for at in self.candidates(hash) {
            let same = (probe, &self.entries[at].binding.key);
            if self.keying.eq.call_now(ctx, same) {
                return Probe { hash, at: Some(at) };
            }
        }
        Probe { hash, at: None }
    }

    async fn seek(&self, ctx: &mut Ctx<'_, Rt>, probe: &K) -> Probe {
        let hash = self.keying.hash.call(ctx, (probe,)).await;
        for at in self.candidates(hash) {
            let same = (probe, &self.entries[at].binding.key);
            if self.keying.eq.call(ctx, same).await {
                return Probe { hash, at: Some(at) };
            }
        }
        Probe { hash, at: None }
    }

    fn seek_pushed_now(&self, ctx: &mut Ctx<'_, Rt>, fresh: &Pushed) -> Probe {
        self.seek_now(ctx, &self.entries[fresh.0].binding.key)
    }

    async fn seek_pushed(&self, ctx: &mut Ctx<'_, Rt>, fresh: &Pushed) -> Probe {
        self.seek(ctx, &self.entries[fresh.0].binding.key).await
    }

    /// Rust's `insert`: the new value takes the key's entry, which keeps
    /// the position it was first inserted at, and the value it displaced is
    /// the answer.
    fn put_now(&mut self, ctx: &mut Ctx<'_, Rt>, binding: Binding<K, V>) -> Option<V> {
        let fresh = self.push(binding);
        let probe = self.seek_pushed_now(ctx, &fresh);
        self.displace(probe, fresh)
    }

    async fn put(&mut self, ctx: &mut Ctx<'_, Rt>, binding: Binding<K, V>) -> Option<V> {
        let fresh = self.push(binding);
        let probe = self.seek_pushed(ctx, &fresh).await;
        self.displace(probe, fresh)
    }

    /// Rust's `entry(k).or_insert(v)` and `HashSet::insert`: a key already
    /// there keeps the entry it has, and the binding offered is turned
    /// away.
    fn occupy_now(&mut self, ctx: &mut Ctx<'_, Rt>, binding: Binding<K, V>) -> Placed<K, V> {
        let fresh = self.push(binding);
        let probe = self.seek_pushed_now(ctx, &fresh);
        self.settle(probe, fresh)
    }

    async fn occupy(&mut self, ctx: &mut Ctx<'_, Rt>, binding: Binding<K, V>) -> Placed<K, V> {
        let fresh = self.push(binding);
        let probe = self.seek_pushed(ctx, &fresh).await;
        self.settle(probe, fresh)
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

        borrowed_as_self!(
            $t<$($k,)+ E, Rt>,
            $($k: Var<kind::Type>,)+ E: Var<kind::Effect>, Rt: Runtime
        );
    };
}

// -- The map ------------------------------------------------------------

pub struct HashMap<K, V, E, Rt>(Table<K, V, E, Rt>, PhantomData<(K, V, E)>)
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime;

stored_extern_type!(HashMap<K, V>, name: "HashMap");

fn new_map<K, V, E, Rt>(
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
    HashMap(Table::new(Keying { hash, eq }, capacity), PhantomData)
}

#[extern_fn(effect = pure)]
fn hash_map<K, V, E, Rt>(hash: HashOf<K, E, Rt>, eq: EqOf<K, E, Rt>) -> HashMap<K, V, E, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    new_map(hash, eq, 0)
}

#[extern_fn(effect = pure)]
fn with_capacity<K, V, E, Rt>(
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
    new_map(hash, eq, as_capacity(n))
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

/// The map a borrowed reading names, and the position it has reached.
///
/// `iter::Refs` is this over a container whose elements are its own one
/// type. A map has two element types and a stage carries one `iter::next`,
/// so reading the keys and reading the values are two stages, each with a
/// body of its own.
pub struct KeysBody<K, V, E, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    map: Ref<HashMap<K, V, E, Rt>, Shared, Rt>,
    at: usize,
}

/// The borrowed source over a map's keys.
#[derive(ExternType)]
#[extern_type(name = "Keys")]
#[repr(transparent)]
pub struct Keys<K, V, E, I, Rt>(KeysBody<K, V, E, Rt>, PhantomData<I>)
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime;

impl<K, V, E, I, Rt> Keys<K, V, E, I, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    /// The key at this step's position, and the step.
    fn step<'a>(&'a mut self, ctx: &Ctx<'_, Rt>) -> Option<&'a K> {
        let index = self.0.at;
        self.0.at += 1;
        self.0
            .map
            .with(ctx.rt, |m| m.0.entries.get(index).map(|e| &e.binding.key))
    }
}

/// As `KeysBody`, over the values.
pub struct ValuesBody<K, V, E, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    map: Ref<HashMap<K, V, E, Rt>, Shared, Rt>,
    at: usize,
}

/// The borrowed source over a map's values.
#[derive(ExternType)]
#[extern_type(name = "Values")]
#[repr(transparent)]
pub struct Values<K, V, E, I, Rt>(ValuesBody<K, V, E, Rt>, PhantomData<I>)
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime;

impl<K, V, E, I, Rt> Values<K, V, E, I, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    /// The value at this step's position, and the step.
    fn step<'a>(&'a mut self, ctx: &Ctx<'_, Rt>) -> Option<&'a V> {
        let index = self.0.at;
        self.0.at += 1;
        self.0
            .map
            .with(ctx.rt, |m| m.0.entries.get(index).map(|e| &e.binding.value))
    }
}

#[extern_fn(effect = pure)]
fn keys<K, V, E, I, Rt>(m: Ref<HashMap<K, V, E, Rt>, Shared, Rt>) -> Keys<K, V, E, I, Rt>
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Keys(KeysBody { map: m, at: 0 }, PhantomData)
}

#[extern_fn(instance_of = sig::next, effect = pure)]
fn next_keys<'a, K, V, E, I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &'a mut Keys<K, V, E, I, Rt>,
) -> Option<&'a K>
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    it.step(ctx)
}

#[extern_fn(effect = pure)]
fn values<K, V, E, I, Rt>(m: Ref<HashMap<K, V, E, Rt>, Shared, Rt>) -> Values<K, V, E, I, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Values(ValuesBody { map: m, at: 0 }, PhantomData)
}

#[extern_fn(instance_of = sig::next, effect = pure)]
fn next_values<'a, K, V, E, I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &'a mut Values<K, V, E, I, Rt>,
) -> Option<&'a V>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    it.step(ctx)
}

// There is no `values_mut`. A `Values` whose `iter::next` answered
// `Option<&mut V>` compiles, and a script that stores through an element of
// it is refused with "cannot store through &i64: not a `&mut`": a stage's
// element type argument does not carry the exclusive loan. `get_mut` is the
// exclusive loan the boundary does carry, one key at a time.

#[extern_fn(effect = pure)]
fn into_keys<K, V, E, I, Rt>(m: HashMap<K, V, E, Rt>) -> Items<K, I, Rt>
where
    K: Var<kind::Type> + Stored<Rt> + Cross<Rt>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(m.0.entries.into_iter().map(|e| e.binding.key).collect())
}

#[extern_fn(effect = pure)]
fn into_values<K, V, E, I, Rt>(m: HashMap<K, V, E, Rt>) -> Items<V, I, Rt>
where
    K: Var<kind::Type>,
    V: Var<kind::Type> + Stored<Rt> + Cross<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(m.0.entries.into_iter().map(|e| e.binding.value).collect())
}

fn insert_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    key: K,
    value: V,
) -> Option<V>
where
    K: Var<kind::Type> + OneValue<Rt> + TransparentOver<Rt>,
    V: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    m.0.put_now(ctx, Binding { key, value })
}

#[extern_fn(effect = E, sync = insert_now)]
async fn insert<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    key: K,
    value: V,
) -> Option<V>
where
    K: Var<kind::Type> + OneValue<Rt> + TransparentOver<Rt>,
    V: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    m.0.put(ctx, Binding { key, value }).await
}

/// The result is Rust's borrow of the map the caller lent (RFC-0047 §3,
/// RFC-0068 D4), so the declaration runs at `Task::Sync`: there is no
/// awaited form of a result that names the frame the call laid its
/// arguments on.
#[extern_fn(effect = E)]
fn get<'m, K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &'m HashMap<K, V, E, Rt>,
    key: &K,
) -> Option<&'m V>
where
    K: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let at = m.0.seek_now(ctx, key).at?;
    Some(&m.0.entries[at].binding.value)
}

/// As `get`'s, with the exclusive loan the boundary carries one key at a
/// time.
#[extern_fn(effect = E)]
fn get_mut<'m, K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &'m mut HashMap<K, V, E, Rt>,
    key: &K,
) -> Option<&'m mut V>
where
    K: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let at = m.0.seek_now(ctx, key).at?;
    Some(&mut m.0.entries[at].binding.value)
}

fn contains_key_now<K, V, E, Rt>(ctx: &mut Ctx<'_, Rt>, m: &HashMap<K, V, E, Rt>, key: &K) -> bool
where
    K: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    m.0.seek_now(ctx, key).at.is_some()
}

#[extern_fn(effect = E, sync = contains_key_now)]
async fn contains_key<K, V, E, Rt>(ctx: &mut Ctx<'_, Rt>, m: &HashMap<K, V, E, Rt>, key: &K) -> bool
where
    K: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt>,
    V: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    m.0.seek(ctx, key).await.at.is_some()
}

fn remove_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    key: &K,
) -> Option<V>
where
    K: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt>,
    V: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let at = m.0.seek_now(ctx, key).at?;
    Some(m.0.take_out(at).binding.value)
}

#[extern_fn(effect = E, sync = remove_now)]
async fn remove<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    key: &K,
) -> Option<V>
where
    K: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt>,
    V: Var<kind::Type> + OneValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let at = m.0.seek(ctx, key).await.at?;
    Some(m.0.take_out(at).binding.value)
}

/// As `get_mut`'s: the result is a borrow of the map, so the declaration
/// runs at `Task::Sync`.
#[extern_fn(effect = E)]
fn or_insert<'m, K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &'m mut HashMap<K, V, E, Rt>,
    key: K,
    value: V,
) -> &'m mut V
where
    K: Var<kind::Type> + OneValue<Rt> + TransparentOver<Rt>,
    V: Var<kind::Type> + OneValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let at = m.0.occupy_now(ctx, Binding { key, value }).at;
    &mut m.0.entries[at].binding.value
}

fn extend_now<K, V, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    m: &mut HashMap<K, V, E, Rt>,
    other: HashMap<K, V, E, Rt>,
) where
    K: Var<kind::Type> + TransparentOver<Rt>,
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
    K: Var<kind::Type> + TransparentOver<Rt>,
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
    K: Var<kind::Type> + TransparentOver<Rt>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut kept = Vec::with_capacity(m.0.len());
    for entry in m.0.drain_entries() {
        if keep.call_now(ctx, (&entry.binding.key, &entry.binding.value)) {
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
    K: Var<kind::Type> + TransparentOver<Rt>,
    V: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut kept = Vec::with_capacity(m.0.len());
    for entry in m.0.drain_entries() {
        if keep
            .call(ctx, (&entry.binding.key, &entry.binding.value))
            .await
        {
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
        types: [HashMap<_, _, _, Rt>, Keys<_, _, _, _, Rt>, Values<_, _, _, _, Rt>],
        fns: [
            hash_map, with_capacity, len, is_empty, clear,
            keys, next_keys, values, next_values, into_keys, into_values,
            insert, get, get_mut, contains_key, remove, or_insert, extend, retain,
        ],
    }
}

// -- The set ------------------------------------------------------------

/// A set's value slot is `()`: the table's shape is the map's, and the only
/// thing a set does not have is a value to hold.
type SetTable<K, E, Rt> = Table<K, (), E, Rt>;

pub struct HashSet<K, E, Rt>(SetTable<K, E, Rt>, PhantomData<(K, E)>)
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
    fn table(&self) -> &SetTable<K, E, Rt> {
        &self.0
    }

    fn table_mut(&mut self) -> &mut SetTable<K, E, Rt> {
        &mut self.0
    }
}

/// A set's binding: the key, and the value slot nothing reads.
fn keyed<K>(key: K) -> Binding<K, ()> {
    Binding { key, value: () }
}

#[extern_fn(effect = pure)]
fn hash_set<K, E, Rt>(hash: HashOf<K, E, Rt>, eq: EqOf<K, E, Rt>) -> HashSet<K, E, Rt>
where
    K: Var<kind::Type>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    HashSet(Table::new(Keying { hash, eq }, 0), PhantomData)
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

/// A set has one element type, so its borrowed source is `iter::Refs` over
/// the set itself; only the `iter::next` that reads a set by position is
/// declared here.
#[extern_fn(instance_of = sig::as_iter, effect = pure)]
fn as_iter_set<K, E, I, Rt>(s: Ref<HashSet<K, E, Rt>, Shared, Rt>) -> Refs<HashSet<K, E, Rt>, I, Rt>
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Refs::of(s)
}

#[extern_fn(instance_of = sig::next, effect = pure)]
fn next_refs_set<'a, K, E, I, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: &'a mut Refs<HashSet<K, E, Rt>, I, Rt>,
) -> Option<&'a K>
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    it.step(ctx, |s, at| {
        s.table().entries.get(at).map(|e| &e.binding.key)
    })
}

#[extern_fn(instance_of = sig::into_iter, effect = pure)]
fn into_iter_set<K, E, I, Rt>(s: HashSet<K, E, Rt>) -> Items<K, I, Rt>
where
    K: Var<kind::Type> + Stored<Rt> + Cross<Rt>,
    E: Var<kind::Effect>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(s.0.entries.into_iter().map(|e| e.binding.key).collect())
}

/// Rust's `HashSet::insert` answers whether the set gained the key, and a
/// key already there is left as it was.
fn set_insert_now<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, s: &mut HashSet<K, E, Rt>, key: K) -> bool
where
    K: Var<kind::Type> + OneValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    s.table_mut()
        .occupy_now(ctx, keyed(key))
        .turned_away
        .is_none()
}

#[extern_fn(name = "insert", effect = E, sync = set_insert_now)]
async fn set_insert<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, s: &mut HashSet<K, E, Rt>, key: K) -> bool
where
    K: Var<kind::Type> + OneValue<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    s.table_mut()
        .occupy(ctx, keyed(key))
        .await
        .turned_away
        .is_none()
}

fn contains_now<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, s: &HashSet<K, E, Rt>, key: &K) -> bool
where
    K: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    s.table().seek_now(ctx, key).at.is_some()
}

#[extern_fn(effect = E, sync = contains_now)]
async fn contains<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, s: &HashSet<K, E, Rt>, key: &K) -> bool
where
    K: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    s.table().seek(ctx, key).await.at.is_some()
}

/// Rust's `HashSet::remove` answers whether the key was there.
fn set_remove_now<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, s: &mut HashSet<K, E, Rt>, key: &K) -> bool
where
    K: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let table = s.table_mut();
    let Some(at) = table.seek_now(ctx, key).at else {
        return false;
    };
    drop(table.take_out(at));
    true
}

#[extern_fn(name = "remove", effect = E, sync = set_remove_now)]
async fn set_remove<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, s: &mut HashSet<K, E, Rt>, key: &K) -> bool
where
    K: Var<kind::Type> + Borrowable<Rt> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let table = s.table_mut();
    let Some(at) = table.seek(ctx, key).await.at else {
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
    K: Var<kind::Type> + TransparentOver<Rt>,
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
    K: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    for entry in other.0.entries {
        s.table_mut().occupy(ctx, entry.binding).await;
    }
}

fn union_now<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    mut a: HashSet<K, E, Rt>,
    b: HashSet<K, E, Rt>,
) -> HashSet<K, E, Rt>
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    for entry in b.0.entries {
        a.table_mut().occupy_now(ctx, entry.binding);
    }
    a
}

#[extern_fn(effect = E, sync = union_now)]
async fn union<K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    mut a: HashSet<K, E, Rt>,
    b: HashSet<K, E, Rt>,
) -> HashSet<K, E, Rt>
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    for entry in b.0.entries {
        a.table_mut().occupy(ctx, entry.binding).await;
    }
    a
}

/// Whether `b` holds the key: `b`'s own hasher and comparator decide, as
/// they do for every lookup in `b`.
fn keeps_now<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, b: &HashSet<K, E, Rt>, key: &K) -> bool
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    b.table().seek_now(ctx, key).at.is_some()
}

async fn keeps<K, E, Rt>(ctx: &mut Ctx<'_, Rt>, b: &HashSet<K, E, Rt>, key: &K) -> bool
where
    K: Var<kind::Type> + TransparentOver<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    b.table().seek(ctx, key).await.at.is_some()
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
    K: Var<kind::Type> + TransparentOver<Rt>,
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
    K: Var<kind::Type> + TransparentOver<Rt>,
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
    K: Var<kind::Type> + TransparentOver<Rt>,
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
    K: Var<kind::Type> + TransparentOver<Rt>,
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
    K: Var<kind::Type> + TransparentOver<Rt>,
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
    K: Var<kind::Type> + TransparentOver<Rt>,
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

fn from_iter_now<It, K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: It,
    hash: HashOf<K, E, Rt>,
    eq: EqOf<K, E, Rt>,
    next: Instance<sig::next<It, K, E, Rt>, It, Rt, Later>,
) -> HashSet<K, E, Rt>
where
    It: Var<kind::Type> + DerefMut<Target = Rt::Value>,
    K: Var<kind::Type> + Stored<Rt> + Cross<Rt> + TransparentOver<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut s = HashSet(Table::new(Keying { hash, eq }, 0), PhantomData);
    while let Some(key) = next.call(ctx, &mut it, ()) {
        s.table_mut().occupy_now(ctx, keyed(key));
    }
    s
}

#[extern_fn(effect = E, sync = from_iter_now)]
async fn from_iter<It, K, E, Rt>(
    ctx: &mut Ctx<'_, Rt>,
    it: It,
    hash: HashOf<K, E, Rt>,
    eq: EqOf<K, E, Rt>,
    next: Instance<sig::next<It, K, E, Rt>, It, Rt, Later>,
) -> HashSet<K, E, Rt>
where
    It: Var<kind::Type> + DerefMut<Target = Rt::Value>,
    K: Var<kind::Type> + Stored<Rt> + Cross<Rt> + TransparentOver<Rt> + PassedByValue<Rt>,
    E: Var<kind::Effect>,
    Rt: Runtime,
{
    let mut it = it;
    let mut s = HashSet(Table::new(Keying { hash, eq }, 0), PhantomData);
    while let Some(key) = next.call_await(ctx, &mut it, ()).await {
        s.table_mut().occupy(ctx, keyed(key)).await;
    }
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
            as_iter_set, next_refs_set, into_iter_set,
            set_insert, contains, set_remove, set_extend,
            union, intersection, difference, is_subset, from_iter,
        ],
    }
}
