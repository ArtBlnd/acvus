# Transparency-constrained type inference

The types a body writes are inferred from their uses, and transparency decides where a type is decided by the values it meets and where a value is coerced to a type already given. Each meeting slot is decided once, from all of its values; each coercion is the one normal form its two types allow, made at one site.

`Small` in the examples is an extern-defined type with a declared cast to `i64`.

## 1. Definitions

A **type** is a finite tree: a constructor with parts and members. The **constructor** is a type's outermost former: what the type is apart from its parts and members. For example, `i32`, `i64`, `f32`, `f64`, `bool`, an extern-defined `N`, `&` or `&mut` with its target's constructor, `Tuple(n)`, `Array(n)`, `Option`, `Result`, `Object`, `Enum(N)`, `Struct(N)`, `Fn` and `!`. The **parts** are the types inside it. The **members** are an object's field names and an enum's variant tags. The six **transparent** constructors are `Object`, `Enum(N)`, `Tuple(n)`, `Array(n)`, `Option` and `Result`. A declared cast goes from an extern-defined type, or a reference to one, to a type of the same mode; the casts form a tree.

```
(Fact)        A fact is a property a type carries beside its constructor, parts and members: an
              order with a join and a top, its unknown. Facts meet by join, and where the join is
              not defined they collapse to the top. A fact is decided once, from its sources.
```

| Fact | Carried by | Order; top | Sources |
|---|---|---|---|
| Effect | `Fn`; an extern-defined type | pure below opaque; opaque | A function's body; for an extern-defined type, the functions its values hold |
| Flows | `Fn` | listed flows below every flow; every flow | The function's body |
| Calls | `Fn` | many below once; once | After inference, `(Strongest)` (§14) |
| Holds | `Fn` | nothing below shared below exclusive; exclusive | After inference, its captures |
| Values | `Fn` | sets of lambdas below any; any | The lambdas written |
| Mark | a whole concrete extern-defined type | marked below unmarked; unmarked | After inference, the extern fn that makes its value, and each use that takes it unmarked: a slot it enters, fixed slots included, a declaration's type variable standing for it, a type it meets, a cast from it |
| Declared facts | a declaration | what is stated below nothing stated; nothing stated | The declaration |

A function that becomes a value carries its facts in its type, and a call performs the callee's effect in the body it is in.

A **slot** is where a value goes, and a **site** is where one value enters one slot. The **values of a slot** are fixed by the program's text: a binding's initializer, assignments and binding pattern; a parameter's argument at each call or instance; a result's tails, `return`s and `?` exits.

## 2. Transparency

```
(Writer)      Every type has one writer: the body whose text makes its values, a lambda belonging
              to the body it is written in; the impl blocks of a struct, for its fields' types; a
              declaration; the host. A value of a transparent constructor that a body receives
              from another writer, a call's result or a value a declaration or the host gives,
              has there the type that body writes, by the names it carries: the value is the one
              given.
(Transparent) A slot is a meeting slot at a site inside its type's writer: its type is decided
              there, once, from every value it is given. At any other site it is a fixed slot: its
              type is given, and a value is coerced to it.
(Visible)     A struct's fields are named only inside its impl blocks.
```

A function's result where its caller takes it, a place path and `as` are no sites. An operator is a call, and its operands are its arguments.

## 3. Variance

A part of a transparent constructor is **covariant**, and every other part is **invariant**. Structure is met (§4) and coerced (§6) only at a covariant part; an invariant part is equal, its facts included, since a fact relaxed through a `&mut` would let the callee store what the caller then reads at the stricter fact.

| Constructor | Position | Variance | Step |
|---|---|---|---|
| `Object` | each field's type | covariant | `(Lift)` |
| `Object` | its fields | narrowed at a fixed slot | `(Narrow)` |
| `Enum(N)` | each payload | covariant | `(Lift)` |
| `Enum(N)` | its variants | widened at a fixed slot | `(Widen)` |
| `Tuple(n)`, `Array(n)`, `Option`, `Result` | each part | covariant | `(Lift)` |
| `Result` | its error part, where `(ErrMeet)` merges | wrapped | `(Wrap)` |
| `&`, `&mut` | the target | invariant | |
| `&mut` | the mode | weakened to `&` | `(Reborrow)` |
| an extern-defined `N` | each argument | invariant | |
| `Struct(N)` | each parameter | invariant | |
| a slice | the element | invariant | |
| `Fn` | each parameter, the result | invariant | |
| any | its own facts | at or below the slot's, checked where decided | `(Facts)` |
| any | an identity it carries | hidden where every occurrence of it is covariant | `(Pack)` |
| `!` | | below every type | `(Bot)` |

| Coercion | Outcome |
|---|---|
| `Option<Small>` to `Option<i64>`; `[Small; 3]` to `[i64; 3]` | `(Lift)` with the cast |
| `&mut S` to `&S` | `(Reborrow)` |
| `&(Small, X)` to `&(i64, X)`; `&mut Small` to `&mut i64`; `Vec<Small>` to `Vec<i64>`; `&[Small]` to `&[i64]`; with only `Small → i64` declared | none: a reference's target, an extern-defined type's argument and a slice's element are invariant |
| `&mut Fn[many]` to `&mut Fn[once]`; `Fn(Fn[many]) -> R` to `Fn(Fn[once]) -> R` | none: facts at an invariant part are equal |

## 4. Exactness

### 4.1 Uses

```
(Exact)       A value enters a meeting slot exactly where its constructor is the slot's: the
              two are one type there, with no coercion. Members are no part of exactness: an
              object's fields and an enum's variants grow by union in the one type, as (Grow)
              allows (§5). Each covariant part of the one type is a slot of its own, whose values
              are the values' parts there, judged by this definition in turn; each invariant
              part is equal in all of them. A value enters a fixed slot by a coercion (§6).
```

| Case | Outcome | Waits |
|---|---|---|
| A meeting slot given one value | exact | no |
| A part of a construction | exact with what is written there | no |
| A meeting slot given several values | exact where a value's constructor is the slot's, coerced where it differs (`(Meet)`, §8) | for the constructors |
| A fixed slot | coerced; the coercion is empty where the types are one | |
| A value of `!` | enters any slot, decides nothing | no |

A type that would be one of its own parts is no type, and is a conflict.

| Meeting | Outcome |
|---|---|
| `Object` with `Object`; `Enum(N)` with `Enum(N)` | One constructor; members by union; each part a meeting slot. |
| `Tuple(n)`, `Array(n)`, `Option`, `Result` | One constructor; each part a meeting slot. |
| `Vec<?a>` with `Vec<i64>` | The argument is invariant and equal: `?a` is `i64`. |
| `Vec<Small>` with `Vec<i64>` | Refused: one constructor, whose argument is invariant and equal in every value. |

| Program | Outcome |
|---|---|
| `let a = 0; let b = a; let c = b;` | One type. |
| `g(c)` with `g<T>(T)` | `T` is `c`'s type. |
| `if c { {x: 1} } else { {y: s} }` | One object, `{x, y}`. |
| `if c { (s, 1) } else { (t, 2) }`, `t: i64` | One `Tuple(2)`; part 0 is a meeting slot given `Small` and `i64`: `i64`, and `(s, 1)` is coerced by `(Lift)` where it entered. |

### 4.2 Construction

```
(Construct)   A construction of a transparent constructor gives its value's constructor and
              members by its syntax: (e1, e2) a tuple, {a: e} an object, Some(e) and E::A(e) a
              variant, [e1, e2] an array. Each of its parts is a slot whose value is what is
              written there.
```

### 4.3 Projection

```
(Project)     A projection names exactly one part of its operand's type: p.y a field, p.0 a
              part, p[i] the element, *r the target, and a call f(ā) one at each parameter and
              one at the result. Being one to one, it is exact and is solved in Exact. Which
              implementation gives the part, an index through as_slice or the declaration a call
              reaches, is a decision by type (§8).
(Read)        Reading a place gives a value of the place's type, whatever path reaches it.
(Lend)        &ₘ e, e a place of type &ₙ T, is &ₘ T: a lend of a reference lends its target, since
              no type is a reference to a reference.
```

A store `p.y = v` is the projection `p.y` and the site where `v` enters the field.

### 4.4 Test

```
(Test)        A pattern, in a let, an if let, a match or a ?, is a projection at each part it
              names: a field pattern names the field, and a variant pattern its variant's payload
              and the variant itself, which (Grow) adds where it allows.
(Bind)        A pattern that binds through references in its scrutinee's type binds by reference,
              at the least mutability of the references it passes and of the bound value's own
              reference, if it is one.
```

## 5. Structured type variables

```
(Kind)        A type variable may carry a kind: what is known of its type before its
              constructor. A literal gives its kind, the integer kind W(n) or the float kind
              (§10); a projection gives the structure it names (§4.3). Kinds meet by conjunction:
              the variable has what each of them states. A value that gives the variable a
              constructor must have the kind's structure, and its parts are one type with the
              kind's. A slot the text gives no value, whose kind is a projection's structure that
              exactly one constructor has, is that constructor, in Exact. A literal's kind takes
              its default at convergence (§9, §10). Any other variable with a kind whose
              constructor no value gives is refused as ambiguous.
(Grow)        A projection adds the member it names, a field or a variant, to its operand's type
              where the body writes that type; an object's field only where no reference, index
              or captured name lies on the projection's path. The members of Option, Result and
              Error are the language's (Reserved).
```

A value's field has a value only on the paths that write it; the initialization check refuses a use of a value whole on a path where one of its fields has none.

| Program | Outcome |
|---|---|
| `let a; a.x = 0; f(a);` | The projection gives `a` the kind `{x: ?1}`; the text gives `a` no value, and an object is the one constructor that has `x`: `a: {x: i32}`. |
| `let x; let v = &mut x.y; *v = 0;` | The projection `x.y` adds `y`; the store through `v` enters it: `x: {y: i32}`, as `x.y = 0` gives. |
| `let o = {x: 1}; let p = o; p.y = 2;` | One type, `{x, y}`; reading `o.y` is refused, `y` being unwritten on every path to it. |
| `r.a = v` with `r: &mut T`; `xs[0].a = v`; a captured `c.a = v` | `a` must already be a field: no field is added. |
| `*r = E::B` with `r: &mut E{A}`, `E` written by the body | `E::B` meets in `*r`, members by union: `E` gains `B`. |
| `let o = mk(); o.b = 2;`, `mk` giving `{a}` | `o` has the script's type, which grows: `o: {a, b}`; `mk`'s type stays `{a}`. |
| `let e = get_e(); e = E::B; take_e(e);`, declared `get_e() -> E{A}` and `take_e(E{A})` | `e: E{A, B}` in the script; at `take_e`'s fixed slot `(Widen)` needs `E{A}` to hold `B`, so the call is refused. |

| Program | Outcome |
|---|---|
| `match e { E::B => … }`, `E` written by the body | `E` gains `B`. |
| A test, through a reference, of a variant that the referent's type, which another writer wrote, lacks | Refused: that type does not grow. |
| `match r { Err(Error::Timeout(t)) => … }`, `r: Result<T, Error { IoError(IoError), ParseError(ParseError) }>` | Refused: `Error`'s variants are the language's, here `IoError` and `ParseError`. |

## 6. Coercion

### 6.1 Generators

A **coercion** from `T` to `U` makes, at one site, a value of `U` from one of `T`. It is the one normal form of §6.2 from `T` to `U`, built from these steps, and `T ≤ U` says it exists:

```
(Cast)      a declared cast S → T, via f                    S ≤ T ⇝ f
(Reborrow)                                                  &mut X ≤ &X ⇝ reborrow
(Lift)      F transparent,  Tᵢ ≤ Uᵢ ⇝ cᵢ                    F(T̄) ≤ F(Ū) ⇝ F(c̄)
(Narrow)                                                    {f̄, ḡ} ≤ {f̄} ⇝ project
(Widen)                                                     E{Ā} ≤ E{Ā, B̄} ⇝ inject
(Facts)     T and U one but for their own facts, each of
            T's at or below U's (Fact)                      T ≤ U ⇝ empty
(Pack)      ι an identity T carries, each occurrence of it
            at a covariant position (§3)                  T ≤ ∃ι. T ⇝ empty
(Bot)                                                       ! ≤ U ⇝ absurd
(Wrap)      at an error part (ErrMeet) merges, N the name of
            E's constructor                                 E ≤ Error{…, N(E), …} ⇝ inject N
```

`(Reborrow)` makes a shared reference through a `&mut` at the site; the `&mut` keeps its type, and how long the reborrow lives is borrowck's.

### 6.2 Stages

```
Exact  ≺  Reborrow  ≺  Cast  ≺  Shape
```

Shape is `(Lift)`, `(Narrow)`, `(Widen)` and `(Wrap)`. `!` reaches any type alone. `(Facts)` and `(Pack)` are no stage: they change no value, and each fact is checked where it is decided.

- **Normal form.** A coercion is `Reborrow? ; Cast^k ; Shape?`, each part by its own normal form. Shape follows Cast because a cast goes from an extern-defined type or a reference to one, and every step of Shape gives a transparent type; Reborrow precedes Cast so that only shared access is asked. `&mut S → &T` therefore needs the shared path `&S → … → &T`.
- **Coherence.** Toward a given type, each stage has at most one step: the modes decide Reborrow, the tree decides Cast, the structure decides Shape. The normal form is unique.
- **Rank.** A coercion ranks by its highest stage, then the number of cast steps in it, its parts' included, then the lower stages it takes, compared in that order.

### 6.3 Sites

```
(Fixed)     fixed slot U,  e : T,  T ≤ U ⇝ c                       c(e)
(Convert)   fixed slot U,  e : T,  no T ≤ U,
            exactly one declared conversion with an instance from T
            to U, g                                                 g(e)
(Recv)      receiver, parameter P: the first of  T ≤ P,  &T ≤ P,  &mut T ≤ P
(Reach)     at each argument in order, keep the candidates of least rank, a receiver taking
            (Recv)'s order first; the result, ranked as an argument is toward the type it is
            taken at, decides last; exactly one remains
```

In a reach, `!` matches every candidate and decides nothing, a literal's variable matches as `(Lit)` says, and no conversion is taken. A value is coerced once, at its site, after its slot is decided, and keeps its own type; under `(Lift)` each of its parts enters the slot's part, a site of its own.

A transparent constructor answers a generic function from its parts' answers, as the language defines that function there; a reach that takes such an answer reads its parts' types.

| Program | Outcome |
|---|---|
| `(a, b) == (c, d)`, `a, c: i64`, `b, d: bool` | `eq` at `Tuple(2)`, answered from `eq` at `i64` and at `bool`. |
| `(s, 1) == (t, 2)`, `Small` with no `eq` | Refused: a part has no answer. |

## 7. Identity

```
(Identity)    An extern-defined type may carry an identity, a parameter of its type. A type
              variable is chosen by each use; an identity is chosen by the execution that makes
              it, a call, a call of a lambda or an iteration of a loop, and is that identity only
              within it. The types of what outlives an execution, its body's result, a slot
              outside it and the types such a slot holds, carry none of its identities: where a
              value carrying one reaches them, or meets a value of another identity, its type
              there is the packed ∃ι. T where (Pack) gives it. Each read of a place of a packed
              type opens it to a fresh identity, distinct from every other. A declaration writes
              its result's identity as a parameter's, which the call passes on, or as one it
              makes. Two identities are compared only where their types are related.
```

| Program | Outcome |
|---|---|
| `let a; loop { a = has_identity(); }`, `has_identity() -> F@new` | Each iteration's identity outlives no iteration: `a: ∃ι. F@ι`, and each read of `a` opens a fresh identity. |
| `let x = if c { open() } else { open() };`, `open() -> F@new` | Two calls, two identities, meeting: `x: ∃ι. F@ι`. `let y = x; y.at(y.index_of())` is admitted; `x.at(x.index_of())` reads `x` twice and is refused. |
| `let mut v = Vec::new(); for k in ks { v.push(open()); } v[0].at(v[1].index_of());` | `T` is one type with `v`'s element, which outlives each iteration: `v: Vec<∃ι. F@ι>`. Each index reads and opens its own identity, and `at` is refused. |
| `if c { &mut x1 } else { &mut x2 }`, `x1: F@ι1`, `x2: F@ι2` | Refused: the identities sit under an invariant target, which `(Pack)` does not reach. |
| `let i = f.index_of(k);`, `index_of(&F@p) -> Idx@p` | The declaration passes `f`'s identity on; none is made. |
| `let a = take_id(); a.push(other_id());`, `a: X<Y@0>`, `other_id()` giving `Z@1`, `push(self: &mut X<T>, v: U)` | `Z@1` enters `U`, which no rule relates to `Y@0`: the two identities never meet, and the call is admitted. |
| The same with `push(self: &mut X<T>, v: T)` | `T` is `Y@0`, fixed by the receiver's invariant target, and `Z@1` would have to enter it: the call is refused. |

## 8. Join

A value depends on a slot when its type is computed from the slot's own type.

```
(Meet)        A meeting slot's constructor is decided once, from the values whose types do not
              depend on its own, when those are settled: the one constructor where they have one,
              and otherwise their join, the nearest common ancestor on the tree that all of them
              reach as one type, and &mut X ⊔ &X = &X. A value whose type depends on the slot is
              exact with it or coerced up into it, and is refused where it would raise it. Where
              no join exists, the slot is refused, naming every value.
```

| Meeting | Outcome |
|---|---|
| `Enum(N)` with `Enum(M)`; `Object` with `Enum`; one tag with a payload on one side only | Refused. |
| `&mut Small` with `&mut i64`, only `Small → i64` declared | Refused: no join; `Small → i64` gives no cast between references. |
| Facts | Joined, a join not defined collapsing to the top (`(Fact)`). |
| Function types | Constraints by union. |

| Program | Outcome |
|---|---|
| `let mut xs = Vec::new(); for v in items { xs = xs.push(v); }` | `Vec` from the independent value; `push(self: Vec<T>, v: T)` reached at it; the dependent `Vec<V>` exact. |
| `let mut x = 0i64; while c { x = x + 1; }` | `i64`; `x + 1` exact. |
| `let mut x = s; while c { x = widen(x); }`, `widen(Small) -> i64` | `Small`; the dependent `i64` would raise it: refused. |
| `let mut acc = None; for v in vs { acc = Some(match acc { None => v, Some(a) => a + v }); }` | The payload is a slot; `v` is independent of it and decides it. |
| `let mut acc = None; for v in vs { if let Some(a) = acc { acc = Some(a + 1); } }` | The payload's one value depends on it, and nothing independent decides it: Unresolved, refused. |

## 9. Solving

```
loop { Exact → Instance → Join } until the loop converges
at convergence: the defaults, then the loop again
converged with no default left: refuse what remains Unresolved or conflicting, at once
```

| Phase | Settles |
|---|---|
| Exact | Every exact relation and every projection, as soon as it can be told (§4), and each decision by type, a reach, a `for`, a `?`, an index's implementation and a lend, once what it reads is settled. |
| Instance | Each instance's constraints, at its types. |
| Join | Each meeting slot that is not all exact (§4, §8), and each value at its site (§6.3), in a fixed order. |

Each phase reads only what is settled, settles only what is determined, never revises, never refuses and leaves the rest Unresolved. Instance and Join read each other's results, `l(if p { s } else { 1i64 })` and `g(l(a))`, so the phases repeat.

A decision or a slot **waits** while an input it reads is not settled; it settles once that input does, in a later phase or round, with no default. The loop **converges** when a round settles nothing, its state the same as the round's before. Only convergence applies the defaults, and only to what no value and no waiting decision can still decide; a declaration's bound on a variable counts as such a decision. A type variable with no kind is `!`, which decides nothing for what reads it; a literal's kind takes its default (`(LitDefault)`); a function's effect is opaque, and its flows every flow. Each default is one value whatever else is defaulted, so the defaults of one convergence apply together. A decision whose inputs are all settled and that has more than one answer is refused as ambiguous.

## 10. Literals

An unsuffixed integer literal `n` is a type variable of the integer kind `W(n)` (`(Kind)`), the integer constructors that hold `n`; an unsuffixed float literal is one of the float kind, `f32` and `f64`. A suffixed literal is its suffix, `"s"` is `&str`, and `b"…"` is `&[u8; k]`.

```
(Lit)        A literal's variable ν is decided once, by every slot it enters or meets in whose
             constructor is an integer one, for an integer literal, or a float one, for a float
             literal, fixed slots included; that constructor must be one and lie in ν's kind,
             or ν is refused, naming every slot. In a meeting ν gives no constructor and takes
             the slot's: an integer literal meets only integer constructors, and a float
             literal only float ones. ν meeting ν′, both integer or both float, is one variable
             over the intersection of their kinds. In a reach ν matches every candidate whose
             constructor there is an integer one, for an integer literal, or a float one, for a
             float literal, and decides nothing.
(LitDefault) An undecided integer variable is i32, and a float variable f64; where the default
             does not hold the literal's value, ν is refused.
```

| Program | Outcome |
|---|---|
| `let x = 1; f(x);`, `f(u8)` | `x: u8` |
| `let x = 1; f(x); g(x);`, `f(u8)`, `g(i64)` | Refused, naming both calls. |
| `x + 1`, `x: u8` | `add(u8, u8)`; `1: u8` |
| `1 + 2` | The default: `i32`. |
| `if c { 2u8 } else { 1 }` | `u8` |
| `if c { s } else { 1 }`, `s: Small` | Refused: an integer literal meets only integer constructors. |
| `if c { s } else { 1i64 }` | `i64`: a suffixed literal is its type, joined on the tree. |
| `if c { 1.5 } else { 2u8 }` | Refused: a float literal meets only float constructors. |
| `let y = 2.0; h(y);`, `h(f32)` | `y: f32` |
| `let z = 3_000_000_000;`, nothing deciding it | `i32` does not hold it: refused. |
| `None`, `[]` | Their parts are given no value and stay `!`; they enter `Option<i64>` and `[i64; 0]` by `(Lift)` and `(Bot)`. |

## 11. `?`

```
(Try)        e? waits for e's constructor, Option or Result. Its value is the Some or Ok payload,
             a part of e's type. Its exit, None : Option<!> or Err(x) : Result<!, E>, enters the
             innermost body's result as return does: a lambda's or a fn's, else the unit's. In
             a template, outside every lambda, ? is refused: the unit's result is the String it
             builds.
(ErrMeet)    At a meeting slot, a Result's error part given several constructors is one
             Enum(Error): each value not of Enum(Error) is coerced at its site by (Wrap), and all
             then meet in it as (Meet) says. Two constructors of one name are refused.
(Reserved)   Error, Option and Result name the language's enums, and self the receiver. Error's
             variants are made only by (ErrMeet), and a pattern may name them.
```

| Program | Outcome |
|---|---|
| `a()?; b()?;`, both erring with `IoError` | Error part `IoError`. |
| `a()?` with `IoError`, `b()?` with `ParseError` | `Error { IoError(IoError), ParseError(ParseError) }` |
| A caller, `f()?; g()?;`, `g` erring with `Timeout` | `Error { IoError(IoError), ParseError(ParseError), Timeout(Timeout) }` |
| `opt?` and `res?` in one body | Refused: `Option` and `Result` do not meet. |
| `Err(s)` and `Err(1i64)` | `Error { Small(Small), i64(i64) }` |

## 12. Lambdas and units

```
(Lam)        |p̄| e : Fn(P̄ → R; φ), each Pᵢ its parameter's slot, R the slot of its results, and
             φ its facts (Fact), its Values fact this lambda.
(Let)        A let-bound lambda is solved first, alone, without defaults. A variable still free
             that the enclosing scope does not hold is quantified, with its kind. A decision that
             reads a quantified variable is a constraint of the scheme; any other is made once.
(Use)        Each use is an instance: fresh quantified variables, and each constraint made there
             as the caller's own, its sites at the use. A use that would reach its own lambda
             through the calls of the values it is given is refused, so instances nest finitely.
(Expected)   A lambda written where a fixed slot of a function type takes it takes that type's
             parameters and result; its results are coerced at their sites.
(LamJoin)    Function values meeting take their constraints by union; a use instantiates each
             lambda their Values fact names.
(Unit)       A script or a template is a unit. It is solved alone, its inputs its parameters, and
             what stays free is quantified as (Let) quantifies: that is its interface. A call of
             it instantiates the interface as a declaration's is instantiated. A call that would
             reach its own unit is refused.
```

| Program | Outcome |
|---|---|
| `let id = \|x\| x; id(1u8); id("s")` | `∀a. a → a`; two instances. |
| `let f = \|x\| x.len(); f(v); f(s)` | `len` is a constraint, reached at each instance. |
| `let h = \|\| c.len()` | Made once. |
| `xs.map(\|x\| x + 1)`, `map` taking `Fn(T) -> U` | `x: T` |
| acvum `fn apply(f, x) { f(x) }`; `apply(\|y\| y + 1, 2u8)` | The constraint calls the lambda with `u8` at the use: `y: u8`. |
| acvum `fn set_a(r, v) { r.a = v; }`; `set_a(&mut o, 1i64)` with `o.a` given `s` | The store is the caller's: `o.a` is `i64`, `s` coerced. |
| acvum `fn with_z(o) { o.z = 1; o }` given the script's `{a}` | The script's type grows to `{a, z}`; a value of it that never went through `with_z` is refused where it is used whole. |

## 13. Modules

An `acvum` file holds modules: a `struct` with its `impl` blocks, or a free `fn`.

```
(Struct)     struct N makes Struct(N): nominal, its members its declared fields; a field not
             declared is refused. A value is given its fields by N { … } and field by field
             within one body.
(Method)     self, &self and &mut self are Struct(N), &Struct(N) and &mut Struct(N) from the
             start. e.m(ā) reaches m by e's constructor, its receiver by (Recv).
(Group)      The fns of modules that call one another are solved and generalized together, as
             (Let) solves, with the field types of their structs. A struct carries its fields'
             types as a tuple carries its parts': each type variable a field's type still holds,
             and each identity it carries, is a parameter of the struct, Struct(N)<ᾱ>. A value
             moved into a field keeps its identity. A use inside the group is monomorphic, one
             outside an instance. A free fn is a let-bound lambda with no environment that may
             recur within its group.
(Loans)      Each value holds its own loans, a value of Struct(N) at each of its fields.
(Scope)      A refusal falls on the smallest of these that it leaves untyped: a use, where it
             reads an instance's types or the loans of one call; a method, where its body fails
             apart from any use, each call of it then refused at that cause; the struct, where
             it leaves a field's type untyped, every use of it then refused at that cause.
```

| Program | Outcome |
|---|---|
| `fn set(&mut self, v) { self.a = v; }` | `v` enters `Point.a`: `∀α β. (&mut Point<α, β>, α) → ()`. |
| `fn new(a, b) -> Self { Point { a, b } }`; outside, `Point::new(1, 2).sum()` | `Point<α, β>`; the call instantiates `Point<i32, i32>`, and `sum` reaches `add(i32, i32)` there. |
| `self.count = 0` in a method | `count` holds the integer kind and is a parameter: `Counter<ν>`; a use decides it, or its default. |
| `fn new(a) -> Self { let p = Point { a }; p.b = 0; p }` | Whole at the return. |
| `fn bad(a) -> Self { Point { a } }` | Refused: `b` unwritten at the return. |
| Outside, `p.add(q)`; `p.a` | `&p` by `(Recv)`; `p.a` refused. |
| `let v = View::new(&x); x = 2; use(v);` | `x = 2` refused: `v` holds a loan on `x` at `.r`. |
| `self.a = "s"` and `self.a = 1` in two methods | Every use of `Point` refused, naming both stores. |
| `struct Holder { f }`, `fn new() -> Self { Holder { f: open() } }`; `let a = Holder::new(); let b = Holder::new(); b.get(a.idx())` | `Holder<ι>`, as `(open(),)` is `(F@ι,)`. `new`'s result outlives its execution, so it is `∃ι. Holder<ι>`, and each call is read and opened: `a` and `b` hold two identities, and the call is refused. |

## 14. Access

After the loop, each use of a place is classified on its instance's decided types, by what it asks of the place: one of three strengths, `shared < exclusive < take`. A lend with `&` is shared, a write or a lend with `&mut` exclusive, and a read as `(Refine)` says; a take moves the value. Whether the place's path permits what a use asks, a move or a `&mut` through a `&`, is borrowck's.

A type **copies** where it is `!`, a number, `bool`, `char`, `()`, a `&`, or a transparent constructor whose parts all copy.

```
(Demand)     A place is used by where it stands: in a value's position it is read; under &ₘ,
             an autoref (Recv), a by-reference iteration or a binding by reference (Bind), it
             is lent at m; on the left of = it is written.
(Refine)     A read takes its strength from the place's type: a type that copies is read
             shared, a &mut is reborrowed exclusive, any other is taken. A call of a function
             value reads it by its facts: once takes it, one holding an exclusive capture is
             exclusive, any other is shared.
(Path)       A use of a place is a use of each place on its path at the same strength; past a
             deref it is a reborrow of the reference, exclusive where the use is exclusive or
             take, shared where it is shared.
(Strongest)  What is decided for a place is decided by its strongest use: a capture's mode,
             & where that use is shared and &mut otherwise; whether a move lambda is once,
             once where it takes a captured name; an index's view, as_slice_mut where the
             element is used exclusive or taken.
```

Each instance is classified alone, and a lambda's capture modes and once/many are the join over its instances, `&` below `&mut` and many below once. A lambda is classified after the lambdas it calls, which `(Use)` keeps finite.

| Program | Outcome |
|---|---|
| `let n = x; let m = x;`, `x: i64` | Two shared reads. |
| `bump(xs[0])`, `xs: [&mut i64; 2]` | `xs[0]` is a `&mut`, reborrowed exclusive, so the view is `as_slice_mut`. |
| `*r = 5` | A write; past the deref, `r` is reborrowed exclusive. |
| `p.a.b = 1` | A write of `p.a.b`, and so an exclusive use of `p.a` and `p`. |
| `\|\| c.push(1)`, `push(&mut self, …)` | An autoref at `&mut`, so `c` is captured `&mut`. |
| `move \|\| consume(v)`, `v: Vec<i64>` | `v` is taken, so the lambda is once. |

## 15. Properties

- **Locality.** A fixed slot's type is given by another writer, so each body is closed. Most relations are exact without waiting. A constructor only rises on a finite tree and members only grow, so every write is monotone; a decision is made once, on settled inputs. The join is associative and commutative, so the order values are met in decides nothing. The work is polynomial in the number of slots and decisions; instances can make that number exponential in the program's text, each `let` that uses the one before it twice doubling it. Which instances share code is devirtualization's.
- **Refusal.** A meeting slot is decided from all of its values, so its refusal names all of them. Refusals are made after the loop, in a fixed order, and a mismatch at a fixed slot or in an instance's constraint is reported at the use. A dependent value is blamed before an independent one (§8).

How precisely a flow between two ends of different structures names their parts is borrowck's.

## 16. Prior work

This system was derived anew from its own requirements. Rust, Swift, C# and Coq, in which its author worked at length, influenced it (§16.1); the research in §16.2 established corresponding mechanisms independently. Each entry states what the language feature or the work is and which part of this note corresponds to it.

### 16.1 Languages

**Rust**

- Matsakis, Klock (2014). *The Rust language*. HILT. https://doi.org/10.1145/2663171.2663188 — Ownership, borrowing and traits. Here: borrowck after lowering (§14), and generic functions answered by impls (§6.3).
- Weiss, Gierczak, Patterson, Ahmed (2019). *Oxide: the essence of Rust*. arXiv. https://doi.org/10.48550/arXiv.1903.00982 — A formal model of Rust's places, loans and reborrows. Here: `(Reborrow)`, `(Lend)` and access by place (§6.1, §4.3, §14).
- Rust RFC 401, *Coercions*. https://rust-lang.github.io/rfcs/0401-coercions.html — Coercions made only at coercion sites. Here: fixed slots as the sites of coercion (§2, §6.3).
- The Rust Reference, *Type coercions*. https://doc.rust-lang.org/reference/type-coercions.html — The coercion sites and coercions such as `&mut T` to `&T`. Here: `(Reborrow)` and `(Fixed)` (§6).
- The Rustonomicon, *Subtyping and Variance*. https://doc.rust-lang.org/nomicon/subtyping.html — `&mut T` invariant in `T`. Here: a reference's target invariant (§3).
- The Rust Reference, *Method call expressions*. https://doc.rust-lang.org/reference/expressions/method-call-expr.html — The receiver taken as written, then by `&` and `&mut`. Here: `(Recv)` (§6.3).
- The Rust Reference, *Closure types*. https://doc.rust-lang.org/reference/types/closure.html — Capture modes inferred from how each captured place is used; `Fn`, `FnMut` and `FnOnce`. Here: `(Strongest)` and the calls and holds facts (§1, §14).
- The Rust Reference, *Operator expressions*, the question mark operator. https://doc.rust-lang.org/reference/expressions/operator-expr.html — `?` returns early with the error, converted to the function's error type. Here: `(Try)` and `(Wrap)` (§11).
- Rust RFC 212, *Restore integer fallback*. https://rust-lang.github.io/rfcs/0212-restore-int-fallback.html — An unconstrained integer literal falls back to `i32`. Here: `(LitDefault)` (§10).

**Swift**

- The Swift compiler, *Type Checker Design and Implementation*. https://github.com/swiftlang/swift/blob/main/docs/TypeChecker.md — Constraint-based type checking in which a literal is typed through a literal protocol and bound to its default literal type where nothing else decides it, and competing solutions are compared by a score. Here: literal kinds and their defaults (§10), and the rank by which exactly one candidate remains (§6.2–§6.3).
- Swift Evolution SE-0176, *Enforce Exclusive Access to Memory*. https://github.com/swiftlang/swift-evolution/blob/main/proposals/0176-enforce-exclusive-access-to-memory.md — Accesses classified as reads or modifications, a modification excluding any overlapping access. Here: access strengths (§14).

**C#**

- Kennedy, Syme (2001). *Design and implementation of generics for the .NET Common Language Runtime*. PLDI. https://doi.org/10.1145/378795.378797 — Generic code instantiated at its type arguments, shared where their representations agree. Here: instances, and which of them share code (§12, §15).
- Emir, Kennedy, Russo, Yu (2006). *Variance and generalized constraints for C# generics*. ECOOP. https://doi.org/10.1007/11785477_18 — Covariant and contravariant type parameters. Here: variance by part (§3).
- C# language specification, *Conversions*. https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/language-specification/conversions — Implicit and user-defined conversions, and an anonymous function converted to the delegate type its context gives. Here: `(Convert)` and `(Expected)` (§6.3, §12).
- C#, *Anonymous types*. https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/types/anonymous-types — Objects written by their members, `new { a = 1 }`, typed by them. Here: object constructions (§4.2).

**Coq (Rocq)**

- Saïbi (1997). *Typing algorithm in type theory with inheritance*. POPL. https://doi.org/10.1145/263699.263742 — Coq's implicit coercions: coercions declared between classes, forming a graph under a coherence condition. Here: declared casts forming a tree, and the unique normal form (§1, §6).
- The Rocq reference manual, *Implicit coercions*. https://rocq-prover.org/doc/master/refman/addendum/implicit-coercions.html — The same mechanism as documented. Here: the same.

### 16.2 Corresponding research

**Polymorphism and generalization**

- Milner (1978). *A theory of type polymorphism in programming*. Journal of Computer and System Sciences. https://doi.org/10.1016/0022-0000(78)90014-4 — Type inference for ML with let-polymorphism. Here: `(Let)` and `(Use)` (§12), a let-bound lambda generalized and each use an instance.
- Damas, Milner (1982). *Principal type-schemes for functional programs*. POPL. https://doi.org/10.1145/582153.582176 — Principal type schemes for let-polymorphism. Here: the same rules.
- Mycroft (1984). *Polymorphic type schemes and recursive definitions*. LNCS. https://doi.org/10.1007/3-540-12925-1_41 — Polymorphic recursion: a recursive definition used at instances of its own scheme. Here: `(Group)` (§13), which uses a group's fns monomorphically inside it.
- Henglein (1993). *Type inference with polymorphic recursion*. TOPLAS. https://doi.org/10.1145/169701.169692 — Type inference with polymorphic recursion is undecidable. Here: the ground for `(Group)`'s monomorphic use inside a group.
- Mairson (1990). *Deciding ML typability is complete for deterministic exponential time*. POPL. https://doi.org/10.1145/96709.96748 — Let-polymorphism makes typability exponential in the program's text. Here: §15's statement on instances.

**Kinds, records and variants**

- Ohori (1995). *A polymorphic record calculus and its compilation*. TOPLAS. https://doi.org/10.1145/218570.218572 — Kinded type variables: a variable carries the record structure known of it before its type is known. Here: `(Kind)` and a projection's kind (§5).
- Rémy (1989). *Type checking records and variants in a natural extension of ML*. POPL. https://doi.org/10.1145/75277.75284 — Extensible records and variants typed by row variables. Here: members that grow by union (§4.1, §5).
- Wand (1991). *Type inference for record concatenation and multiple inheritance*. Information and Computation. https://doi.org/10.1016/0890-5401(91)90050-c — Type inference for records extended and combined. Here: the same.
- Castagna, Petrucciani, Nguyễn (2016). *Set-theoretic types for polymorphic variants*. ICFP. https://doi.org/10.1145/2951913.2951928 — A type system for OCaml's polymorphic variants, whose variant sets grow with their uses and tests. Here: variants added by values and tests (`(Grow)`, `(Test)`, §4.4, §5).
- Anderson, Giannini, Drossopoulou (2005). *Towards type inference for JavaScript*. ECOOP, LNCS. https://doi.org/10.1007/11531142_19 — Type inference for objects whose fields are added by assignment, with a field read only where it is defined. Here: an object's fields added by stores (`(Grow)`) and the initialization check (§5).

**Subtyping, joins and coercions**

- Reynolds (1980). *Using category theory to design implicit conversions and generic operators*. LNCS. https://doi.org/10.1007/3-540-10250-7_24 — Implicit conversions as arrows that compose, and the coherence of their choice. Here: the coercion generators and the unique normal form (§6.1–§6.2).
- Breazu-Tannen, Coquand, Gunter, Scedrov (1991). *Inheritance as implicit coercion*. Information and Computation. https://doi.org/10.1016/0890-5401(91)90055-7 — Subtyping read as coercions inserted into the program, with coherence. Here: a coercion made at its site (§6.3).
- Luo (1999). *Coercive subtyping*. Journal of Logic and Computation. https://doi.org/10.1093/logcom/9.1.105 — Subtyping given by declared coercions under coherence conditions. Here: `(Cast)`, declared casts forming a tree (§1, §6.1).
- Henglein (1994). *Dynamic typing: syntax and proof theory*. Science of Computer Programming. https://doi.org/10.1016/0167-6423(94)00004-2 — Coercions with a normal form and a minimality order among completions. Here: the normal form and its rank (§6.2).
- Swamy, Hicks, Bierman (2009). *A theory of typed coercions and its applications*. ICFP. https://doi.org/10.1145/1596550.1596598 — Coercion insertion at sites, with ambiguity and coherence treated explicitly. Here: `(Fixed)`, `(Convert)` and `(Reach)` (§6.3).
- Pottier (1996). *Simplifying subtyping constraints*. ICFP. https://doi.org/10.1145/232627.232642 — Inference with subtyping constraints and their simplification. Here: a meeting slot decided from all of its values (§8).
- Dolan, Mycroft (2017). *Polymorphism, subtyping, and type inference in MLsub*. POPL. https://doi.org/10.1145/3009837.3009882 — Principal type inference with subtyping, types forming a lattice with joins. Here: a meeting slot's constructor as the join of its values (`(Meet)`, §8).
- Parreaux (2020). *The simple essence of algebraic subtyping*. ICFP. https://doi.org/10.1145/3409006 — A simple algorithm for the same. Here: the same.

**Expected types and constraint solving**

- Pierce, Turner (2000). *Local type inference*. TOPLAS. https://doi.org/10.1145/345099.345100 — Expected types propagated to unannotated lambdas. Here: `(Expected)` (§12).
- Dunfield, Krishnaswami (2021). *Bidirectional typing*. ACM Computing Surveys. https://doi.org/10.1145/3450952 — A survey of checking against an expected type and synthesizing one. Here: the same.
- Vytiniotis, Peyton Jones, Schrijvers, Sulzmann (2011). *OutsideIn(X): modular type inference with local assumptions*. Journal of Functional Programming. https://doi.org/10.1017/S0956796811000098 — Constraints generated first and solved afterwards, those not yet solvable kept until more is known. Here: the solving loop and its waiting decisions (§9).

**Effects and usage**

- Lucassen, Gifford (1988). *Polymorphic effect systems*. POPL. https://doi.org/10.1145/73560.73564 — Types that carry the effects of computations. Here: the effect fact (§1).
- Talpin, Jouvelot (1992). *Polymorphic type, region and effect inference*. Journal of Functional Programming. https://doi.org/10.1017/S0956796800000393 — Inference of effects with types. Here: the same.
- Turner, Wadler, Mossin (1995). *Once upon a type*. FPCA. https://doi.org/10.1145/224164.224168 — Inference of whether a value is used at most once. Here: the calls fact, once or many (§1, §14).

**Identities and representation**

- Mitchell, Plotkin (1988). *Abstract types have existential type*. TOPLAS. https://doi.org/10.1145/44501.45065 — Existential types, each opening giving a fresh abstract type. Here: an identity fresh at each execution (`(Identity)`, §7).
- Launchbury, Peyton Jones (1994). *Lazy functional state threads*. PLDI. https://doi.org/10.1145/178243.178246 — A rank-2 type that brands a computation so that its references stay in it. Here: the same.
- Kiselyov, Shan (2007). *Lightweight static capabilities*. Electronic Notes in Theoretical Computer Science. https://doi.org/10.1016/j.entcs.2006.10.039 — Values branded by a type-level name, so that an index is checked once and used only with its own array. Here: an identity passed from a value to what is derived from it, as `index_of` and `at` (§7).
- Yanovski, Dang, Jung, Dreyer (2021). *GhostCell: separating permissions from data in Rust*. ICFP. https://doi.org/10.1145/3473597 — Branded tokens in Rust through invariant lifetimes. Here: identities at the extern boundary and at invariant parts (§7, §3).
- Leroy (1992). *Unboxed objects and polymorphic typing*. POPL. https://doi.org/10.1145/143165.143205 — Specialized representations where the types are known, a uniform one where polymorphic code needs it. Here: the mark, a specialized representation kept only where every use allows it (§1).

**Without a DOI**

- The Haskell 2010 Report, chapter 4, §4.3.4 *Ambiguous Types, and Defaults for Overloaded Numeric Operations* and §4.5.1 *Dependency Analysis*. https://www.haskell.org/onlinereport/haskell2010/haskellch4.html — Defaulting of ambiguous numeric types; binding groups typed together. Here: defaults at convergence (§9) and `(Group)` (§13).
