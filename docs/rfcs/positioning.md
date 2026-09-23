# Positioning

acvus is a statically typed scripting language. It has no standard library of
its own and does nothing by itself; a host supplies every capability through
ExternFn. It exists to build DSLs on. This document decides what the compiler
is, where the outside world enters, who executes a program, and what the
language declines to own.

## RFC-0001: acvus is an enrichment pipeline; information is lost at the execution boundary alone

Status: Accepted

1. Parsing discards surface syntax and nothing else.
2. Every pass after parsing records in the MIR a property that was implicit in
   the source. No pass removes a property another pass could still use.
3. Information is lost at one place, the execution boundary, and validation
   runs on the MIR as it stands immediately before it.
4. There are no lowering dialects between the MIR and execution.
5. Values carry no runtime type tag; the types the MIR accumulates make
   erasure at the boundary sound.
6. A new feature is admitted only if it loses no information.

Rust is the implementation language and acvus the composition language: an
ExternFn's signature describes its Rust body's semantics, so a composition is
optimized without reading the body.

**Why.** With one cliff, a wrong result is either a wrong MIR or a wrong
materialization of it. A lowering chain loses information at every stage, and
a wrong result then names no stage.

**Rejected.**
- A chain of lowering dialects — loss at every stage; a wrong result cannot be
  attributed without reconstructing what each stage discarded.

## RFC-0002: ExternFn is the only door to the outside world, and everything inside it is sound

Status: Accepted

1. The MIR is parametric over the front end (the AST) and the runtime
   context; replacing those two hosts another DSL. Behaviour bound to one DSL
   belongs in its front end or runtime context, not in the MIR.
2. ExternFn is the only door to the outside world. An extension that enters
   is a first-class citizen of the SSA; nothing enters halfway, and everything
   inside the door is required to be sound.
3. acvus has no C FFI.
4. acvus has no JIT: executable memory is never allocated.
5. Declarations sit on the ExternFn side: its author writes them, the script
   author writes none, and complexity may be placed on the ExternFn author.

**Why.** The stack is scripts on ExternFn, ExternFn on Rust, the interpreter
on Rust's memory safety; with no C FFI the bottom stays Rust, so acvus adds
ordering on a memory-safe base instead of establishing memory safety itself.
Without executable memory, generated code cannot become code injection.
**Cost.** The JIT refusal is a decision not to build, not a property the IR
enforces: the MIR is typed SSA over words and pointers, and a fork that wants
a native backend finds nothing in the way.

**Rejected.**
- C FFI — native calls move scheduling out of the interpreter and put a
  non-Rust layer below acvus.
- JIT — a code-injection surface, and it ends the interpreter's control of
  every execution (RFC-0003).

## RFC-0003: The interpreter mediates every execution

Status: Accepted

Nothing is lowered to native code. The specializations a JIT would perform at
run time are performed ahead of time from facts already in the MIR; the
machine that runs them is RFC-0052's.

The interpreter owns scheduling, so it can run a program under an adversarial
schedule: a verification mode reorders operations the program declared
order-irrelevant and reports a difference in result as a wrong declaration.

**Why.** Verifying a declared ordering freedom needs an executor that chooses
the order; native code hands scheduling to the OS and the hardware. A JIT
would also pay complexity, executable memory and warm-up to recover type and
scheduling information acvus never lost.

## RFC-0004: The language has no map; a map is an extension type

Status: Accepted

The type system has no map type and the grammar no map index or field syntax.
A map is an extension type a registry declares, entering through ExternFn like
any other capability.

**Why.** A map is an Object that has lost its field count and names; the
language keeps Object, and a host that needs dynamic keys supplies the
structure.
