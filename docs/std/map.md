# `HashMap<K, V>` and `HashSet<K>`

A map holds what a key names. The key's hash and equality are passed at
construction as two closures, the map stores them, and every operation that
has to find a key calls them — the `BuildHasher`-and-comparator form Rust
offers. The map's effect is the join of the two closures', so a lookup
carries whatever they carry, and over `core::hash` and `==` that is `Pure`.

A script names the two like this:

```
let counts = hash_map(|k| -> hash(k), |a, b| -> a == b);
```

The comparator is written over the two references. `*a == *b` moves a
`String` out of the reference the comparator was lent and the checker
refuses it; `a == b` compares what the two references name, at every key
type. `eq(a, b)` does not work: an instance of `core::eq` is chosen where
the lambda is written, and the key type is not known until the first
`insert`, so the choice falls on the wrong instance.

The constructors are `hash_map` and `hash_set` and the types are `HashMap`
and `HashSet`, which are Rust's own names. A constructor named `map` would
join the overload set of `iter::map`, and the two are then ambiguous at a
call whose lambda parameter type is still open — `examples/grades` stops
compiling.

A function that walks a map or a set answers a pipeline, and a pipeline's
type is the stage itself: `Keys<K, V>` and `Values<K, V>` borrow the map and
yield `&K` and `&V`, `Refs<HashSet<K>>` borrows the set and yields `&K`, and
`Items<K>` owns the keys a consumed map or set gave up. A function that
*takes* a pipeline takes the stage's type variable, written `I` below; any
stage whose element fits stands there.

## `HashMap<K, V>` — namespace `map`

| name | signature | Rust twin | difference |
| --- | --- | --- | --- |
| `hash_map` | `Fn(Fn(&K) -> i64, Fn(&K, &K) -> Bool) -> HashMap<K, V>` | `HashMap::with_hasher` | the comparator is passed too; Rust takes `Eq` from the key |
| `with_capacity` | `Fn(u64, Fn(&K) -> i64, Fn(&K, &K) -> Bool) -> HashMap<K, V>` | `HashMap::with_capacity_and_hasher` | a capacity past the address space traps with `capacity overflow`, as `Vec::with_capacity` does |
| `len` | `Fn(&HashMap<K, V>) -> u64` | `HashMap::len` | none |
| `is_empty` | `Fn(&HashMap<K, V>) -> Bool` | `HashMap::is_empty` | none |
| `clear` | `Fn(&mut HashMap<K, V>) -> ()` | `HashMap::clear` | none |
| `insert` | `Fn(&mut HashMap<K, V>, K, V) -> Option<V>` | `HashMap::insert` | a replaced key keeps the position it was first inserted at |
| `get` | `Fn(&HashMap<K, V>, &K) -> Option<&V>` | `HashMap::get` | the probe is a `&K`; there is no `Borrow<Q>` |
| `get_mut` | `Fn(&mut HashMap<K, V>, &K) -> Option<&mut V>` | `HashMap::get_mut` | as `get`; what the call binds is a `&mut V`, and the one spelling refused binds through a pattern — see "What the boundary refuses" below |
| `contains_key` | `Fn(&HashMap<K, V>, &K) -> Bool` | `HashMap::contains_key` | the probe is a `&K` |
| `remove` | `Fn(&mut HashMap<K, V>, &K) -> Option<V>` | `HashMap::remove` | the entries after it move down, as `IndexMap::shift_remove` does, so the order of what is left is the order it was |
| `or_insert` | `Fn(&mut HashMap<K, V>, K, V) -> &mut V` | `HashMap::entry(k).or_insert(v)` | one call rather than an `Entry` value; the default is evaluated whether or not it is used |
| `extend` | `Fn(&mut HashMap<K, V>, HashMap<K, V>) -> ()` | `HashMap::extend` | the argument is another map, consumed; a sequence of pairs does not cross |
| `retain` | `Fn(&mut HashMap<K, V>, Fn(&K, &V) -> Bool) -> ()` | `HashMap::retain` | the closure is lent `&V`, not `&mut V` |
| `keys` | `Fn(&HashMap<K, V>) -> Keys<K, V>` | `HashMap::keys` | insertion order; `Keys` is a borrowing source whose element is a `&K` |
| `values` | `Fn(&HashMap<K, V>) -> Values<K, V>` | `HashMap::values` | insertion order; `Values` is a borrowing source whose element is a `&V` |
| `into_keys` | `Fn(HashMap<K, V>) -> Items<K>` | `HashMap::into_keys` | insertion order; consumes the map |
| `into_values` | `Fn(HashMap<K, V>) -> Items<V>` | `HashMap::into_values` | insertion order; consumes the map |

## `HashSet<K>` — namespace `set`

| name | signature | Rust twin | difference |
| --- | --- | --- | --- |
| `hash_set` | `Fn(Fn(&K) -> i64, Fn(&K, &K) -> Bool) -> HashSet<K>` | `HashSet::with_hasher` | the comparator is passed too |
| `len` | `Fn(&HashSet<K>) -> u64` | `HashSet::len` | none |
| `is_empty` | `Fn(&HashSet<K>) -> Bool` | `HashSet::is_empty` | none |
| `clear` | `Fn(&mut HashSet<K>) -> ()` | `HashSet::clear` | none |
| `insert` | `Fn(&mut HashSet<K>, K) -> Bool` | `HashSet::insert` | none; the answer is whether the set gained the key, and a key already there is left as it was |
| `contains` | `Fn(&HashSet<K>, &K) -> Bool` | `HashSet::contains` | the probe is a `&K` |
| `remove` | `Fn(&mut HashSet<K>, &K) -> Bool` | `HashSet::remove` | the keys after it move down, so the order of what is left is the order it was |
| `extend` | `Fn(&mut HashSet<K>, HashSet<K>) -> ()` | `HashSet::extend` | the argument is another set, consumed |
| `union` | `Fn(HashSet<K>, HashSet<K>) -> HashSet<K>` | `HashSet::union` | consumes both and answers a set, as `intersection`; the hasher, comparator and order kept are the first set's. Its body writes the second set's keys into the first in place rather than calling `set::extend`: the earlier `union` lent a handler local `&mut a` to that un-inlined handler, and LLVM's sibling-call rule refuses a tail call out of any function an alloca's address escapes, so every `Op::run` reaching it landed with a call |
| `intersection` | `Fn(HashSet<K>, HashSet<K>) -> HashSet<K>` | `HashSet::intersection` | consumes both and answers a set; Rust borrows both and yields references, which needs a clone of a key to build a set from, and the runtime offers none. The hasher, comparator and order kept are the first set's; the second set's own hasher and comparator decide each membership |
| `difference` | `Fn(HashSet<K>, HashSet<K>) -> HashSet<K>` | `HashSet::difference` | consumes both and answers a set, as `intersection` |
| `is_subset` | `Fn(&HashSet<K>, &HashSet<K>) -> Bool` | `HashSet::is_subset` | the second set's own hasher and comparator decide each membership |
| `from_iter` | `Fn(I, Fn(&K) -> i64, Fn(&K, &K) -> Bool) -> HashSet<K>` | `HashSet::from_iter` | `I` is any pipeline whose element is a `K`; the hasher and comparator are passed; a repeat keeps the first key |
| `as_iter` | `Fn(&HashSet<K>) -> Refs<HashSet<K>>` | `HashSet::iter` | insertion order; the name is the language's shared source signature, and the element is a `&K` |
| `into_iter` | `Fn(HashSet<K>) -> Items<K>` | `HashSet::into_iter` | insertion order; consumes the set |

## Waiting on RFC-0067

| name | signature | Rust twin | difference |
| --- | --- | --- | --- |
| `hash_map` / `hash_set`, no arguments | `Fn() -> HashMap<K, V>` where `K: Instance<hash<K>> + Instance<eq<K>>` | `HashMap::new` | not built. The key's own `core::hash` and `core::eq` instances stand where the two closures stand now, the map stores no closure, and a map built this way is journaled — see below |

## Iteration order

Rust leaves `HashMap`'s order unspecified. This one iterates in insertion
order, and that is a decision: the corpus is run twice and the two runs are
compared (`differential`), so an order that varied between runs would be a
failing corpus rather than a licence. The storage is a `Vec` of entries in
insertion order plus an index from hash to the positions holding it, which
is `indexmap`'s layout.

`insert` over a key already present replaces the value and leaves the entry
where it was. `remove` and `retain` drop entries and leave the order of what
is left.

## A map is not held by a space

`Deque` declares `ExternTypeDecl::space` and commits with `--commit`. A map
does not, and that is a decision. `Journaled::decode_state` builds the whole
value back from canonical bytes; a closure has no canonical bytes, which is
`acvus-interpreter`'s layout answering that a value of a function type is not
held by a space; and a map that lost its hasher on reload would answer every
lookup wrongly. The map a space can hold is the one whose key operations
come from the key's instances at each call site, which stores no closure. It
arrives with RFC-0067.

## What the boundary refuses

**There is no `iter` or `entries` over `(K, V)`.** No tuple crosses the
extern boundary — a tuple has `TyArg` and no `Cross` — so a stage cannot
carry a pair, and for the same reason there is no `HashMap::from_iter` over
pairs. `keys` and `values` walk the same insertion order, so the *n*-th of
one belongs with the *n*-th of the other.

**There is no `values_mut`.** A `Values` whose `iter::next` answered
`Option<&mut V>` compiles, and a script that stores through an element of it
is refused with `cannot store through &i64: not a &mut`: a stage's element
type argument does not carry the exclusive loan. `get_mut` is the exclusive
loan the boundary does carry, one key at a time.

**What `get_mut` binds is a mutable reference; a pattern's binding is not.**
Both of these write 99 at both `Opt`s, whether the `Option` is named first
or the two calls are one expression:

```
let o = get_mut(&mut m, &q); let v = o.unwrap(); *v = 99; let r = 1; *get(&m, &r).unwrap()
let v = get_mut(&mut m, &q).unwrap(); *v = 99; let r = 1; *get(&m, &r).unwrap()
```

and `let v = get_mut(&mut m, &q).unwrap(); *v` reads the value through the
same binding. `unwrap` is declared over `Option` and over `Result`, so a
call of it is a `Signature` decision that waits on its argument's head
(`docs/solver.md` R2); in the one-expression form that head is another
call's open decision, and a binding settles the decisions open before it, so
`v` is a `&mut i64` where the store and the read are checked.

The spelling still refused binds through a pattern, which lends a shared
reference:

```
if let Some(v) = get_mut(&mut m, &q) { *v = 99; }
```

It is refused at both `Opt`s with "cannot store through `v`, of type &_: not
a `&mut`; bind it with `&mut`".
`acvus-interpreter-test/tests/map.rs`'s
`a_chained_unwrap_of_get_mut_is_a_mutable_reference_where_it_is_bound` pins
all four programs.

**A `get_mut` whose key is read again refuses at `Opt::Full`.** A result
that is a reference borrows the call, not a named parameter, so the `&mut V`
`get_mut` answers holds the probe's loan as well as the map's. This is not
admitted, because `q` is borrowed a second time while the first loan is
live:

```
let o = get_mut(&mut m, &q); let v = o.unwrap(); *v = 99; contains_key(&m, &q)
```

That program compiles at `Opt::None` and aborts the compiler at `Opt::Full`
with a `BorrowConflict` the optimizer raised; the refusal that belongs in
front of it is not written yet.

**A reference into the map is a loan on the map.** An `insert` while a `get`
result is live is refused, with "`m` is written here while a reference to it
is live" (RFC-0064).
