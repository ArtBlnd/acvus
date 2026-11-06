# Transparency-constraint type system

This document defines how the types of a program are decided. The system is as simple as it can be and deterministic, refuses a program at what registered the constraints a rule fails on, and has no type declarations.

The rules are the obligations (Obligations), each stated once under its identifier. Every other statement of this document is a definition an obligation uses, a consequence of obligations, or an example. An implementation implements the obligations and nothing besides: a consequence implemented as a rule of its own is a rule added. Each obligation is witnessed by at least one case whose outcome changes where the obligation is left out; the examples name the obligations they witness.

## Definitions

### Compilation graph

A program is one compilation graph. Its compilation nodes are units, extern fns and contexts, and its edges are uses: a body, the text of a free fn, a method or a lambda, uses what it names.

| Compilation node | Sort | Use |
|---|---|---|
| Unit | module, script, template | calls its free fns and methods |
| Extern fn | a declaration of an external function | calls it |
| Context | a declared `@x` | reads it, takes from it, lends it and commits into it (context §3) |

| Unit | Free fns | Struct |
|---|---|---|
| Module | zero or more | at most one, the module itself, whose impl blocks hold its methods |
| Script | one, whose body is the script's text | none |
| Template | one, whose body is the template's text | none |

A template is a script whose body is the `String` formatted from its text, and whose result type is fixed to `String`. `{{ e }}` is a use of `core::display`. A script's or a template's input `$x` is a parameter of its function (programs §4).

A lambda lies inside another body, its enclosing body. Free fns and methods that reach one another by calls form one binding group, and a lambda is a binding group of its own, lying inside the binding group of its enclosing body.

### Transparency

A body sees the target of a use at one of three levels.

| Target of the use | What the body sees | Level |
|---|---|---|
| itself | all of its own text | Transparent |
| another body: a free fn, a method or a lambda | its text, separately at each use | Semi-transparent |
| an extern fn, a context | its declaration | Opaque |

A module's struct is constructed, and has its fields named, only inside that module's impl blocks. A context's type is given by the host, and is called its context type. A context type is external and has no hole, it holds neither a reference nor a function, and a brand type in it is that context's own, never made one with another (context §3).

### Types

A type is a finite term: one constructor, its members and its parts, and each part is again a type.

| Constructor | Notation | Parts | Members |
|---|---|---|---|
| Reference | `&T`, `&mut T` | the target | |
| Tuple | `(T₁, …, Tₙ)` | each component | |
| Array | `[T; n]` | the element | |
| Slice | `[T]` | the element | |
| Option | `Option⟨T⟩` | the payload | `Some`, `None` |
| Result | `Result⟨T, E⟩` | the ok, the error | `Ok`, `Err` |
| Error | `Error{V̄(T̄)}`, `Error⟨E⟩` | each payload; `E` | its variants; none |
| Function type | `Fn(P̄ → R)` | each parameter, the result | |
| Function value type | `λ̄` | none | its closures |
| Object | `{f̄: T̄}` | each field's type | its fields |
| Enum | `E{V̄(T̄)}` | each payload | its variants |
| Struct | `Self⟨T̄⟩` | one type parameter for each field, in the order the struct declares them | |
| Extern-defined type | `N⟨T̄, ι⟩` | each argument, and the brand type | |
| Scalar | the integers of 8 to 64 bits, `usize`, `f32`, `f64`, `bool`, `char` | none | |
| String | `String` | none | |
| str | `str` | none | |
| Unit type | `()` | none | |
| Bottom | `!` | none | |

`usize` is an integer type of its own, whose width the host gives. It is not any integer of that width: the two are different types, and no coercion relates them.

A function type is written only by a declaration. A function value is what a lambda expression gives (O29), and its type is a function value type: its parameter and result types are no parts of it, as each use of it makes them anew (O26).

A reference has a mutability: `&` is shared, and `&mut` is exclusive. A reference's lifetime is not written in the text, and is decided at each use as the least one.

Two types are equal when their constructors, members and parts are all equal. A constructor is what remains of the notation without its members and parts: a reference's mutability, an enum's name, a struct's module, an extern-defined type's name, and a tuple's or an array's length are in it.

The specialization mark of an extern-defined type (types §8) is not handled by this type system.

### Equivalence classes

A type variable is one type, represented by an equivalence class (a class).

| Record | What it holds |
|---|---|
| lower bounds | what enters the class: what the text writes (Lower bounds), what meets in it (Meetings), and the classes O2 and O14 make its lower bounds |
| upper bounds | what the class goes to: the types given it, the classes it is a lower bound of, and the upper bounds of each class whose class node its head is in (Upper bounds) |
| structural constraints | the parts and members its projections require the type to have, before its head is decided |
| parts | the structural components of the class, each connected to a class of its own, named as the constructor names it: the target, a component, a field, the element, a payload, an argument |
| domain | the domain registered in it by a literal with no suffix (Domains) |

| Decision | What it holds |
|---|---|
| head | the type's constructor and its members (Types) |

The class node of a type is what a change goes by: for a reference, its mutability and its target's constructor; for any other type, its constructor. As no reference has a reference as its target (O23), a class node never reaches deeper than that.

### Domains

A class's domain is the set of types the class may be. A literal with no suffix registers a domain in its class: the types of its family that hold its value, among the integer types for `10` and among the float types for `1.5`.

| Domain | Its types | Fallback type |
|---|---|---|
| every type | every type | `!` |
| the integer types | the integers of 8 to 64 bits, and `usize` | `i32` |
| the float types | `f32`, `f64` | `f64` |

### Constraints

A lower bound and an upper bound are each a type. Each is read as its constructor, its members and its parts:

| Read | What it holds |
|---|---|
| constructor | the constructor of the type, with what it holds: a reference's mutability, a tuple's or an array's length, an enum's name, a struct's module, an extern-defined type's name, which `Error` it is, a function type's number of parameters, which scalar |
| members | an object's fields, an enum's or an `Error`'s variants, a function value type's closures. An `Option`'s and a `Result`'s members are given by the constructor |
| parts | each part, a class of its own, read in the same order |

Every type registered as a constructor, members or parts, and every domain registered (Domains), is kept, so that a refusal names what registered it (O22).

### Lower bounds

| Read | Relation to the class's head (O6) |
|---|---|
| constructor, members | is joined into the head (Joins), and reaches it at its own place by the reborrow where its mutability is exclusive and the head's shared; the reborrow is recorded with it |
| parts | each part is one type with the corresponding part of the class, and what that records is told by the state of the corresponding part (O9) |

| What the text writes | constructor | members | parts |
|---|---|---|---|
| a literal of one type: `1u32`, `true`, `'a'`, `()` | that type's | — | — |
| a lend `&ₘ p` (O23) | reference, with its mutability | — | the target: the class of the place's type with the references above it taken off |
| a constructor application: `(a, b)`, `[a, b]`, `{f: a}`, `V(a)`, `Some(a)`, `None`, `Ok(a)`, `Err(e)` | its constructor | its members | each argument's class |
| a lambda (O29) | function value type | its closure | — |
| `return e` (Meetings) | `!` | — | — |
| `e?` (O25) | as O25 states | as O25 states | as O25 states |

### Upper bounds

| Where | constructor, members | parts |
|---|---|---|
| at a boundary (O2) | the class reaches it by exactly one change: one stage of coercion, then one chain of conversions (O11) | each part below its class node is one type with the corresponding part of the type the change goes to, and what that records is told by the state of the corresponding part (O9). Through a conversion, its signature gives the correspondence (O12) |
| at a part (O9) | equal: no change is made at a part | the same, at each part |

| What gives it | Where |
|---|---|
| a parameter type a declaration writes | at a boundary (O2) |
| the common part's parameter, until the use is decided | at a boundary (O14) |
| a context type, at a commit into its context | at a boundary (O2) |
| the external part corresponding to a part, at a change | at a part (O9) |
| `usize`, of an index's class | at the index's class (Structural constraints) |

These are the types given a class. A class's head is in the class node of each class it is a lower bound of, whose head is the join of its lower bounds' heads (Joins), and of each reference whose target it is. A class's upper bounds are the types given it, the classes it is a lower bound of, and the upper bounds of each class whose class node its head is in.

Each upper bound is reached through a class node: the class's own, for a type given it and for a class it is a lower bound of, and that class's, for an upper bound of a class whose class node its head is in. What of that class node is decided is kept, and the class's head fills in what is not.

An upper bound gives the class heads for what is not decided of the class node it is reached through: a type, or a class with a head, gives what its class node has there; a class with lower bounds and no head yet gives the join of the heads they have given; and a class of the common part of a use not yet decided gives what each candidate writes in its place, every head where a candidate writes a hole (O13). The use not yet decided of whose common part a class is a class gives it the same. A head goes to an upper bound where the class node the upper bound is reached through, so filled in, goes to it: to one given at a boundary, by a change, the empty change included (O11), and, for a class of the common part of a use not yet decided, to one of the heads the candidates write there; to one given at a part, by the empty change; and to a class it is a lower bound of, where that class gives no head or its join with the head that class gives is that head (Joins).

The head a class's upper bounds give it is the one head, of those it is given and in its domain, that goes to every one of its upper bounds. Upper bounds check the head a class's lower bounds decide, and decide the head of a class whose dimension is upper (O17, O19).

### Structural constraints

A projection records a structural constraint on its operand's class, names that part's class `τ`, and records the upper bound the table gives.

| Projection | Example | Structural constraint | Upper bound |
|---|---|---|---|
| Deref | `*r` | a reference to `τ` | none |
| Field | `p.f` | has field `f: τ` | none |
| Component | `t.0` | has component `i: τ` | none |
| Elem | `a[i]` | indexes `τ` | `usize`, of the class of `i` |
| Payload | `x` in `if let Some(x) = o` | has variant `V` with payload `τ` | none |
| Call | `f(ā)` | called with the arguments `ā`, giving the result `τ` | none |

### Closures and instances

A closure is a lambda together with the classes its captures are where its expression lies (Landin 1964). A closure enters a class when a lower bound of the class has it among its members.

An instance of a body at a use is that body's text recorded anew at the use (O26).

### Meetings

Values meet where the text puts them in one place: an expression assigned to a place, or bound to it by `let`, meets in the place's class; the branches of an `if` or a `match` meet in its class; and a body's tail, its `return`s and its `?` exits meet in the body's result class.

### Changes

A change goes from one type to another by their class nodes and members, made of coercions and conversions; the change made of none is the empty change. A coercion is a change the language gives:

| Coercion | What changes |
|---|---|
| `&mut T → &T` | the mutability |
| `! → U` | the class node |
| an object to one with only some of its fields | the members |
| an enum, or an `Error`, to one with more variants | the members |
| `&[T; n] → &[T]`, `&mut [T; n] → &mut [T]` | the target's constructor |
| a function value type to one with more closures | the members |
| a function value type to a function type | the class node; the function type's parameters and result are unified with the parameter and result classes of each closure's instance (O30) |

A conversion is a declared function `S → T`. An auto-conversion is a chain of conversions, `S → U → T` from the conversions `S → U` and `U → T`.

### Joins

The heads a class's lower bounds give are joined, and the join is the class's head (O6). `!` has no values: it is the least head, and its join with any head is that head. The head classes are the references, the objects, the function value types, the slices, the `Option`s, the `Result`s, the tuples of each length, the arrays of each length, the enums of each name, the `Error`s with variants, the `Error⟨E⟩`s, the function types of each number of parameters, the structs of each module, the extern-defined types of each name, each scalar, `String`, `str` and `()`. Two heads of one head class join as that head class, with the shared mutability where either is a shared reference, and with the union of their members; two heads of two head classes are a type mismatch (O22).

### Common part

The common part of a use's candidates is their least general generalization, by anti-unification (Plotkin 1970, Reynolds 1970). It keeps each constructor all candidates share, puts a class made anew wherever they differ, and one class wherever they differ by the same types. The rest of a candidate is what its signature writes beyond the common part. Each candidate is an instance of it.

| Candidates | Common part |
|---|---|
| `pick({x: i32}) -> {x: i32}`, `pick(u8) -> u8` | `pick(?a) -> ?a` |
| `push(&mut Vec⟨T⟩, T)`, `push(&mut Vec⟨u8⟩, u8)` | `push(&mut Vec⟨?a⟩, ?a)` |
| `f(T, T)`, `f(u8, u16)` | `f(?a, ?b)` |
| `new() -> Vec⟨T⟩` alone | `new() -> Vec⟨?t⟩` |

### Adjustments

An adjustment is an operation the language applies where the text gives one:

- the auto-deref, by which a projection other than a Deref applies to its operand's target;
- the adjustment of a method call's receiver: the first of the receiver as written, its `&` and its `&mut` that reaches a candidate, each lend taken as far as the place allows.

### Brand types

A brand type is a part of an extern-defined type, the name of a storage (ownership §6). Only an extern-defined type has a brand type among its parts; a part of any type may be branded. A scope is the text of a body or a lambda, or one iteration of a loop, and scopes nest. A class belongs to the scope in whose text it is made, and two classes unified into one belong to the outer of their scopes.

## Obligations

### Where types are decided

**O1.** What meets within one transparency is a lower bound of the class it meets in (Meetings).

**O2.** Where an argument of a use is passed to a parameter of the signature or the common part made anew at that use (O13), which is parameter passing, and where an expression is committed into a context (context §3), what is recorded is told by the parameter's type, or the context type:

| That type | What is recorded |
|---|---|
| a hole | the argument's class is a lower bound of that class |
| any other type | a class made anew there, the boundary's class, whose lower bound is the class of the argument or of the expression committed, and whose upper bound is that type |

**O3.** A class's head is decided along one of three dimensions (O17): lower, as the join of what enters it (O6); upper, as the head its upper bounds give it (O19); and domain, as its domain's fallback type (O20).

### Classes

**O4.** Solving only adds to a class's records, and makes each of its decisions once.

**O5.** No class is a part of itself, or a part of one of its parts, at any depth (the occurs check, Robinson 1965).

**O6.** Classes that enter one another, at any depth, are one class (cycle elimination, Fähndrich et al. 1998). The head of a class that a lower bound enters is the join of the heads its lower bounds give (Joins).
**O7.** A structural constraint gives the part it names a class, decides no head, and holds of the head decided, which has every part it names.

**O8.** A class's domain is the intersection of every domain registered in it, and every type where none is. A domain holds at least one type, and holds the head decided for its class and the type fallback gives it.

### Changes

**O9.** A change changes a type's class node and members, and the type of no part below its class node: each such part is one type with the corresponding part of the type the change goes to, and what that records is told by the state of the corresponding part:

| The corresponding part (O13) | What is recorded |
|---|---|
| free | the two classes are unified |
| external | the corresponding part is an upper bound of that part |

**O10.** The one coercion made within a class is the reborrow, where a lower bound reaches its class's head (Lower bounds). Every other coercion, and each auto-conversion, is made at a boundary alone, and an auto-conversion once, after the class node of the boundary's class is decided.

**O11.** No coercion follows a boundary's conversions: the change at a boundary is one stage of coercion followed by one chain of conversions. Exactly one such change goes from the class node of the boundary's class to the upper bound's class node.

**O12.** The parameter of a conversion writes its class node with distinct holes as the parts below the class node. The signatures of the chosen conversions are made anew at the boundary (O13), and their functions are called at the boundary.

### Uses

**O13.** At each use, the signature its candidate writes, or the common part of its candidates, is made anew as classes with maximal sharing: one class for each distinct type it writes, at any depth, so that one type is one class wherever the signature writes it. The class of a hole, and each class the common part puts where the candidates differ, is free; every other class is external, with the head the signature writes. At each use of a context, its context type is made anew in the same way, every class of it external, with the head the host gives; the class of the place `@x` is the one it makes.

**O14.** A use records the common part of its candidates before it is decided, and until then the upper bound of each argument's boundary is the common part of the candidates' parameters. Once the use is decided, the candidate it is decided as is recorded as at a use with that candidate alone (O13, O2): each class of the candidate is unified with the class the common part has in its place, and each boundary with the boundary the argument already has; where the candidate writes a hole as that parameter, the argument's boundary class is a lower bound of the hole's class, as at O2, and no change is made at that boundary.

**O15.** A use is decided as the one candidate that what is decided of its arguments and its result reaches: each argument by a change to the candidate's parameter, and the result by equality to the candidate's result type.

**O16.** A use's result is unified with the class of the result type its candidate writes (O13). No change is made there.

### Solving

**O17.** A class's dimension is lower where a lower bound enters it; upper where none does and its upper bounds, or the use not yet decided of whose common part it is a class, give it a head (Upper bounds); and domain otherwise. Only its dimension decides its head; where that dimension gives no one head, the class waits.

**O18.** Solving records every constraint the text gives as soon as it is given, on no condition of what has been decided, and settles: it decides the head of each class whose dimension is lower once each of its lower bounds has given its head and each use not yet decided of whose common part it is a class is decided (O6, O14), makes the change at each boundary whose two class nodes are decided (O11), takes the adjustment of each method call's receiver once the receiver's class node is decided (Adjustments), and decides each use with one candidate left (O15), until nothing changes: the fixpoint. The outcome, and what every refusal names, do not depend on the order in which settling takes its steps.

**O19.** At the fixpoint, each class with no head whose dimension is upper takes the head its upper bounds give it (Upper bounds), where there is one, all at once. Where a class takes one, solving returns to settling.

**O20.** Where O19 gives no class a head, each class with no head whose dimension is domain takes the fallback type of the least domain in Domains that contains its domain, all at once. Where a class takes one, solving returns to settling.

**O21.** Solving ends where neither O19 nor O20 gives a class a head. A class with no head, and a use not decided, when solving ends are refused (O22).

**O22.** A program is refused wherever a rule fails, at what registered the constraints it fails on (Constraints).

### References

**O23.** No type has a reference directly as its target; the parts inside a target are not subject to this, the argument of `&N⟨&U⟩` for example. A lend `&ₘ p` gives a reference whose target is the place's type with the references above it taken off, and whose mutability is exclusive where `m` and every reference taken off are exclusive, and shared otherwise.

**O24.** Each lower bound records the coercion it takes (Lower bounds), and each projection and each method call's receiver records the adjustment it takes (Adjustments).

### Error

**O25.** `Error` is a keyword, the language's enum, and its variants are made only by `?`. What `e?` puts into the result's error is told by the error type `E` it exits with:

| `E` | What enters the result's error |
|---|---|
| a type of a name `N`: an enum, a struct, an extern-defined type | an `Error` with the one variant `N(E)` |
| an `Error` | that `Error`, whose variants are united with the result's, side by side |
| a type with no name, `i32` for example | `Error⟨E⟩`, which has no variant |

`Error⟨E⟩` and an `Error` with variants have different constructors, and the variants of one `Error` have different names.

### Other bodies

**O26.** At recording, each use of another body records an instance of that body at the use: the classes of the body's own text are made anew, and every other class its text names is that same class. The instance's parameter and result classes are unified with the use's arguments and result.

**O27.** A use that lies in the binding group of the body it reaches, or in an instance of that body, records no instance, and takes the classes of that body's text, or of the instance it lies in.

### Brand types

**O28.** A type is branded by at most one brand type, and two different brand types are never made one. A brand type in the result of a declaration or a body that no parameter's type is branded by is made anew, as a free class, at each use, in the scope the use lies in. A brand type does not leave the scope it is made in: a program is refused where a class outside that scope would be branded by it (the skolem escape check, Läufer & Odersky 1994).

### Function values

**O29.** A lambda expression gives its class a lower bound: a function value type whose one member is the closure of that lambda (Lower bounds).

**O30.** A use of a function value is a Call constraint on a class, or an external function type given a class as an upper bound (Upper bounds). For each closure that enters the class and each use on it, the instance of O26 is recorded, whose captures are the classes the closure holds, and whose parameter and result classes are unified with the Call's arguments and result, or with the function type's parameters and result. The head decided for a class has, at each use on it, an instance of each of its members.

**O31.** The text of a free fn, a method or a lambda is recorded once more, with every class it names made anew, and only settled (O18).

## Consequences

Each statement below follows from the obligations it names, and is no rule of its own.

- **Upper bounds decide only a class that no lower bound enters** (O9, O17, O19). A part that a lower bound enters must satisfy its upper bound, the corresponding external part, and a part that none enters takes the head its upper bounds give at the fixpoint. So `take(Some(&mut x))` with `take(Option⟨&T⟩)` is refused at `take`'s argument: the payload's head is `&mut`, from `&mut x`, and no change alters a part.
- **No coercion follows a conversion** (O11). An argument of type `&mut Vec⟨u8⟩` passed to a parameter of type `&[u8]` goes by `&mut → &` and then `as_slice`; `as_slice_mut` ends at `&mut [u8]`, and no coercion follows it.
- **Within a class, the heads joined are of one head class** (O6, O10, Joins). Apart from `!`, which every join absorbs, the join is of heads of one head class, and conversions are made at a boundary alone, so lower bounds of two head classes in one class are refused there (O22).
- **The reborrow is the coercion a lower bound takes** (O6, O10). In `if c { &mut x } else { &y }`, the join of the two heads is the shared reference, and the lower bound `&mut x` reaches it at its own place by the reborrow, which it records.
- **A value that meets in a place keeps its own class** (O1, O10). It is a lower bound of the place's class, so the place's class is the join of what meets there, and the value reaches that join at its own place, by the reborrow where it is an exclusive reference and the join is shared; its own class is unchanged.
- **A branch that does not complete leaves its `if` the other branch's head** (O1, O6, Joins). `return e` gives `!`, the least head, and the join of `!` with a head is that head.
- **A class of the common part of a use not yet decided joins what enters it only once the use is decided** (O14, O18). The decision unifies it with the class its candidate writes, and, where that candidate writes a hole as a parameter, makes the argument's boundary class a lower bound of it, so what enters it before the decision is not all that will enter it. So where a candidate writes one hole at a parameter and at the result, what enters the result's class does not decide the use (O15).
- **A class that no lower bound enters takes the head its upper bounds give before any fallback** (O17, O19, O20). The class of a literal passed to a boundary with `u8` has that `u8` among its upper bounds, so its dimension is upper and it takes `u8` at the fixpoint; fallback gives `i32` only to a literal's class that no upper bound gives a head.
- **The heads a class may take are the heads the program gives** (O6, O19, O20, Upper bounds). A join is of heads lower bounds give, the head of a class's upper bounds is one they give, and a fallback type is one the Domains table gives; so a class takes from its upper bounds only a head one of them gives, and never a type that only a conversion goes from, or `!` where none of them gives it.
- **`!` holds no structural constraint** (O7, O20, O22). Fallback gives `!` to a class whose domain is every type; `!` has no part, so no structural constraint holds of it, and a class with a structural constraint that falls back to `!` is refused. There are no higher-kinded types: in `Option⟨?k⟩`, `?k` may be `!`, and `Option` may not.
- **The signature's sharing is the type's equality** (O13). A class is one type and equality is structural, so one type is one class wherever a signature writes it: in `x(T) -> T` the parameter and the result are the one class `T`, and in `pop(&mut Vec⟨T⟩) -> Option⟨T⟩` the parameter and the result are two classes sharing the part `T`. A hole is one class wherever it is written, and the common part's one class for one difference is the same sharing.
- **Settling is order-independent** (O4, O6, O8, O17, O18, O26, O30). A class whose dimension is lower takes its head only once each of its lower bounds has given its head; a domain only narrows; candidates only decrease; a class node is decided once; and what upper bounds give and fallback are taken only at the fixpoint, for every class at once (O19, O20). Settling is a monotone operator on the records, and every fair order of its steps reaches the same fixpoint (chaotic iteration, Cousot and Cousot 1977). This is what makes the last sentence of O18 attainable.
- **Instances are made at recording** (O18, O26). An instance unifies its parameters with the use's arguments, and its text may add a lower bound to an argument's class, as `|x| x = &b` does. Were an instance recorded after a decision, that class's head could be decided before the lower bound is added, and the outcome would depend on the order. So every instance is recorded at recording.
- **A lambda's captures are the enclosing body's classes** (O26, O30). Every instance of a closure names the same classes for its captures, as the substitution of a let-bound term at each use keeps its free variables bound to the enclosing binders (Pierce, TAPL §22.7). A capture written by a lambda is decided with every instance's writes, and a lambda waits for no capture.
- **A closure that only a decision lets enter a class is refused at its uses** (O14, O22, O30). A closure entering through the rest of a decided candidate has no instance recorded, so the head decided does not satisfy O30.
- **A lambda's own text is checked once** (O31). Its refusals that depend on no use and no capture are found once. Its parameters and captures are left to its instances: the recording makes them anew, so it writes into no class of the enclosing body, and it is only settled (O18), so what decides them there is only what enters them.
- **Polymorphic recursion is not admitted** (O27). A recursion that changes its parameters' types (Mycroft 1984) unifies them in the body's own types, and a lambda reached within its own instance takes that instance's classes, so recording ends.
- **A context type never changes** (O2, O4, O6, O9, O13, O22). Every class of it has the head the host gives, and a head is decided once (O4). What is committed into a context reaches its context type by the change at a boundary (O2, O11), a lower bound of one of its classes is joined into that class's head (O6), and no change alters a part below a class node (O9), so a lend of a context reaches no other target. Anything else is refused (O22).

## Examples

Below, `c` is a `bool`, and a function whose signature is in a comment is an extern declaration.

### 1. A literal and a `u32` in one `if` (O1, O6, O8, O19)

```
let a = 1u32;
let b = if c { a } else { 10 };
```

Outcome: `a`, `b` and `10` are `u32`. The class of `a` and the class of `10` meet in the `if`'s class, which enters `b`'s class. The class of `10` has no lower bound, and its domain is the integer types that hold `10`; its upper bound, the `if`'s class, gives it `u32`, the join of what has entered it, so at the fixpoint it takes `u32`, which lies in that domain. The `if`'s class and `b`'s are then `u32`.

### 2. An `if` with information on one side (O1, O6)

```
let a = if c { Some(1u8) } else { None };
```

Outcome: `a` is `Option⟨u8⟩`. The lower bounds `Some(1u8)` and `None` both have the constructor `Option`, and their payloads are unified with the class's payload. The payload's one lower bound is `u8`.

### 3. A reborrow within a class (O6, O10, O24)

```
let r = if c { &mut x } else { &y };
```

Outcome: `r` is a `&`, and the lower bound `&mut x` records the reborrow.

### 4. A part a boundary decides (O2, O9, O13, O18, O19)

```
let a = Vec::new();      // new() -> Vec⟨T⟩
fill_all(&mut a);        // fill_all(&mut Vec⟨u8⟩)
```

Outcome: `a` is `Vec⟨u8⟩`.

1. `new`'s `T` is `?0`, and `a`'s class is `Vec⟨?0⟩`.
2. The boundary's class node is `(&mut, Vec)`, and so is its upper bound's. Settling makes the empty change, and the part `?0` has the upper bound `u8`.
3. No lower bound enters `?0`, and its upper bound gives it `u8`, so its dimension is upper, and at the fixpoint it takes `u8`.

### 5. A conversion through a hole (O11, O12, O19)

```
let v = Vec::new();      // new() -> Vec⟨T⟩
sum(&v);                 // sum(&[u8]); a conversion as_slice: &Vec⟨T⟩ → &[T]
```

Outcome: `v` is `Vec⟨u8⟩`.

1. `v`'s class is `Vec⟨?0⟩`. The boundary's class node is `(&, Vec)`, and its upper bound's `(&, [])`.
2. One change goes between the two class nodes: no coercion, then `as_slice`. Its `T` is made anew as `?t`.
3. On the parameter's side `?0` and `?t` are one type and are unified; on the result's side `&[?t]` corresponds to `&[u8]`, so `?t` has the upper bound `u8`, which gives it `u8` at the fixpoint.

### 6. A boundary does not change what enters a part (O2, O9, O17, O22)

```
let mut buf = Vec::new();   // new() -> Vec⟨T⟩
push(&mut buf, 1u16);       // push(&mut Vec⟨T⟩, T)
consume(buf);               // consume(Vec⟨u8⟩)
```

Outcome: refused at `consume`'s argument.

1. `buf` is `Vec⟨?t⟩`, and `push`'s `T` is `?T`.
2. At the first argument, the two class nodes are `(&mut, Vec)`, so `?t` and `?T` are unified. The second argument is passed to a parameter whose type is `T`, the free class `?T`, so `u16` is a lower bound of `?T`, and `?T` is `u16`.
3. At `consume`, the class nodes are `Vec` on both sides, and the part `?t`, which what enters it decided as `u16`, does not satisfy its upper bound `u8`.

### 7. Upper bounds reached through a reference (O3, O19)

```
let a;
g(&a);                   // g(&[u8])
h(&a);                   // h(&Vec⟨u8⟩); a conversion as_slice: &Vec⟨T⟩ → &[T]
```

Outcome: `a` is `Vec⟨u8⟩`.

1. No lower bound enters `a`'s class, so neither boundary's class node is decided, and settling reaches its fixpoint.
2. `a`'s class is the target of both boundaries' references, so their upper bounds, `&[u8]` and `&Vec⟨u8⟩`, are its upper bounds, and its dimension is upper. What of the two class nodes is decided is `&`, and for the target's constructor the upper bounds give `[]` and `Vec`. From `(&, Vec)` a change goes to both; from `(&, [])` none goes to `&Vec⟨u8⟩`. So the head they give `a`'s class is `Vec`, and it takes `Vec`.
3. Solving returns to settling, and the two boundaries proceed as in 4 and 5.

### 8. Upper bounds that give no one head (O19, O21)

```
let a;
g(&a);                   // g(&[u8])
k(&a);                   // k(&str)
```

Outcome: refused at `a`, with the two boundaries. For the target's constructor the upper bounds give `[]` and `str`, and no change goes between the two, so they give it no one head, and `a`'s class has no head when solving ends.

### 9. A literal passed to a boundary (O8, O17, O19)

```
let x = 0;
take(x);                 // take(u8)
```

Outcome: `x` is `u8`. The class of `0` has no lower bound, and its domain is the integer types that hold `0`. It enters `x`'s class, which enters the boundary's class, whose upper bound is `u8`; so its upper bounds give it `u8`, and its dimension is upper. At the fixpoint it takes `u8`, which lies in that domain, and `x`'s class and the boundary's class are then `u8`, as joins.

### 10. A structural constraint with nothing that gives a head (O7, O20, O22)

```
let a;
a.x = 0;
```

Outcome: refused at `a.x`. The structural constraint "has field `x`" gives no head, and `a`'s class has no lower bound and no upper bound, so its dimension is domain. Its domain is every type, so fallback gives it `!`, of which the structural constraint does not hold.

### 11. A payload nothing enters (O20)

```
let x = None;
```

Outcome: `x` is `Option⟨!⟩`. The payload's class has no lower bound and no upper bound, so its dimension is domain, and its domain is every type, so it falls back to `!`.

### 12. A part whose head what enters it decides, at a boundary (O9, O17, O22)

```
take(Some(&mut x));      // take(Option⟨&T⟩)
```

Outcome: refused at `take`'s argument. The payload's head is `&mut`, from `&mut x`; the change at the boundary between the two `Option` class nodes is empty, and alters no part, so the payload does not satisfy its upper bound `&T`.

### 13. A coercion before a conversion (O11)

```
let mut v = bytes();     // bytes() -> Vec⟨u8⟩
sum(&mut v);             // sum(&[u8]); conversions as_slice: &Vec⟨T⟩ → &[T], as_slice_mut: &mut Vec⟨T⟩ → &mut [T]
```

Outcome: accepted, by `&mut → &` and then `as_slice`, the one change from `(&mut, Vec)` to `(&, [])`.

### 14. A capture its lambda writes (O26, O29, O30)

```
let mut last = None;
let remember = |x| last = Some(x);
remember(1u8);
```

Outcome: `last` is `Option⟨u8⟩`.

1. The closure of the lambda enters the class of `remember`, and the call is a Call constraint on that class.
2. At recording, the call records an instance: `x` is made anew as `x'` and unified with the class of `1u8`, and `last` is the class of `last` itself.
3. The instance's `last = Some(x')` is a meeting at the call, so `Some(x')` is a lower bound of `last`'s class, with `None`. The class's constructor is `Option`, and its payload has the lower bound `u8`.

### 15. One capture, two types (O6, O22, O26, O30)

```
let mut last = None;
let remember = |x| last = Some(x);
remember(1u8);
remember(1u16);
```

Outcome: refused. The two instances put `u8` and `u16` into the one payload of `last`, and no coercion relates them.

### 16. A lambda never used (O20, O31)

```
let mut last = None;
let remember = |x| last = Some(x);
```

Outcome: `last` is `Option⟨!⟩`. No instance is recorded, and the text's own recording makes every class it names anew and is only settled. The payload of `last` has no lower bound and no upper bound, so it falls back to `!`.

### 17. A closure that only a decision lets enter (O13, O14, O22, O30)

```
choose(|x| x, 1u8)(2u8); // choose(F, u8) -> F, choose(u8, F) -> F
```

Outcome: refused at the call of the result. The common part is `choose(?a, ?b) -> ?c`, with three classes, so at recording no closure enters `?c` and the call records no instance. The use of `choose` is decided as the first candidate, whose rest unifies `?c` with the class the closure enters, and the head decided for `?c` then has a member with no instance at the call.

### 18. A literal passed to overloaded candidates (O14, O19)

```
let y = pick(0);         // pick(i32) -> i32, pick(String) -> String
```

Outcome: `y` is `i32`.

1. The common part is `pick(?a) -> ?a`, and `?a`, the use's result, enters `y`'s class. The argument's boundary class has the lower bound `X`, the class of `0`, and the upper bound `?a`.
2. Nothing enters `X` or `?a`. The use, not yet decided, gives `?a` the heads the candidates write in its place, `i32` and `String`, and `?a` gives them to `X` through the boundary's class node. Settling reaches its fixpoint.
3. Of `i32` and `String`, `i32` alone lies in `X`'s domain, the integer types that hold `0`, so `X` takes `i32`. `?a` is given two, and waits.
4. Solving returns to settling: the boundary's class is `i32`, the argument reaches `pick(i32)` alone, the use is decided, and `?a` and `y` are `i32`.

### 19. Two candidates in one domain (O17, O21)

```
let y = pick(0);         // pick(i32) -> i32, pick(u8) -> u8
```

Outcome: refused at `0`, at `y` and at the use. `?a` gives `i32` and `u8`, both in the integer types, and no change goes between the two, so the upper bounds of the class of `0` give it no one head. Its dimension is upper, and only that dimension decides it (O17), so fallback, which would give `i32`, does not reach it. When solving ends, the class of `0`, the boundary's class, `?a` and `y`'s class have no head, and the use is not decided.

### 20. A head that another class's upper bounds settle (O19, O21)

```
let y;
take(y);                 // take(u8)
pick2(0, y);             // pick2(i32, i32), pick2(u8, u8)
```

Outcome: `y` and `0` are `u8`.

1. The common part is `pick2(?a, ?a)`, and settling reaches its fixpoint with no head decided.
2. `y`'s class has the upper bounds `u8`, from `take`, and `?a`, which gives `i32` and `u8`; `u8` alone goes to both, so `y`'s class takes `u8`. At the same time the class of `0` has `i32` and `u8` from `?a`, both in its domain, and waits.
3. Settling decides the use as `pick2(u8, u8)`, the one candidate `y` reaches, and `?a` is `u8`.
4. At the next fixpoint the class of `0` takes `u8`. Were a class refused where its upper bounds give it no one head at one fixpoint, this program would be refused at step 2.

### 21. A literal passed to a hole with a typed argument (O6, O19)

```
f(0, 1u8);               // f(T, T)
```

Outcome: `T` and `0` are `u8`. `T`'s lower bounds are the classes of `0` and of `1u8`, and its head waits for both. The class of `1u8` gives `u8`. The class of `0` has no lower bound, and its upper bound `T` gives it `u8`, the join of what has entered `T`, which lies in its domain; so at the fixpoint it takes `u8`, and `T`'s head is then `u8`.

### 22. Literals passed to a hole alone (O17, O20)

```
f(0, 1);                 // f(T, T)
```

Outcome: `T`, `0` and `1` are `i32`. The classes of `0` and `1` have no lower bound, and their upper bound `T` gives them no head, as no head has entered it; so their dimension is domain, and they fall back to `i32`. `T`'s head is then their join, `i32`.

### 23. Fallback after what upper bounds give (O20)

```
let a;
if let Some(v) = a {}
take(a);                 // take(Option⟨u8⟩)
```

Outcome: `v` is `u8`.

1. `a`'s class has the structural constraint "has variant `Some` with payload `τ`", where `τ` is `v`'s class, and no lower bound. It enters the boundary's class, whose upper bound `Option⟨u8⟩` gives it `Option`, so its dimension is upper. `τ` has no lower bound and no upper bound yet.
2. At the fixpoint `a`'s class takes `Option`. Solving returns to settling before any fallback: the boundary's class, as its join, is `Option`, the change is empty, and the payload, which is `τ`, has the upper bound `u8`.
3. At the next fixpoint `τ` takes `u8`. Were fallback taken with what upper bounds give, `τ` would be `!`, and refused at its upper bound.

### 24. A branch that does not complete (O1, O6)

```
let a = if c { return 0u8 } else { 1u8 };
```

Outcome: `a` is `u8`. `return 0u8` gives the `if`'s class the lower bound `!`, and `1u8` gives it `u8`; their join is `u8`, as `!` is the least head (Joins). Were `!` a head class of its own, the two would be a type mismatch.

### 25. A value that meets in a place keeps its own class (O1, O10)

```
let mut x = 1u8;
let y = 2u8;
let p = &mut x;
let r = if c { p } else { &y };
touch(p);                // touch(&mut u8)
```

Outcome: accepted; `r` is `&u8`, and `p` is `&mut u8`. `p`'s class meets in the `if`'s class with `&y`, so the `if`'s class is their join, the shared reference, and `p`'s class reaches it at its own place by the reborrow. `p`'s class itself stays `&mut u8`, so `touch(p)` goes by the empty change. Were what meets unified into one class, `p` would be `&u8`, and refused at `touch`.
