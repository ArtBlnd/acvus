# Machine

How the interpreter runs a prepared body: what preparation decides once, the
synchronous machine and its one async driver, failure as a panic, who owns a
runtime value and when it is released, the operation and frame
representation, and the shape of a closure value.

## RFC-0044: A body is prepared once into a `Code`, and the machine that runs it is synchronous

Status: Accepted

1. **A `MirBody` is not what the interpreter runs.** At module load each body,
   `main` and every closure, is prepared once into a `Code`: operations whose
   every static fact — operand slots, arithmetic type, literal word, branch
   target, extern handler — is resolved then and never looked up again. The
   module's `Prepared` owns every `Code` (RFC-0069), and nothing about a body
   is rebuilt per call. What an operation is, is RFC-0052's.

   **The machine is synchronous.** It leaves its loop only to return or to hand
   a future up. The one `async fn` is the driver around it: it awaits the
   pending future, stores the result in the slot the operation named, and
   re-enters at the next operation. No future is boxed for a body that never
   suspends.

   **A call into a body that cannot suspend runs to its result inside the
   calling operation.** Whether it can is the callee's task (RFC-0046), which
   `prepare` reads statically (RFC-0052 rule 4).
   At the extern boundary the same fact is `Runtime::call_is_sync`, asked once
   when a closure argument is materialized for a handler, not per element;
   `Runtime::call_now` is the call that follows a true answer.

   **A failure at run time is a Rust `panic!`.** A division by zero, a
   `MIN / -1`, an index out of range, a `Diverge` reached, a broken extern
   contract — each panics where it happens, with Rust's message for the same
   operation. There is no failure channel: no operation or boundary tests
   whether the one below it failed. A host that survives a failing script
   catches the panic; a spawned task's panic is resumed on the awaiting run's
   thread, so `Eval` sees it as its own. `Runtime` has no `Error` type, no
   `trap`, and no `empty`/`is_empty`. What a panic owes the frame is
   RFC-0048 rule 8.

   **A panic message names the operation, not the source position.** Naming
   the position needs one store per operation for the unwinder to read, about
   a tenth of every loop iteration.

   **Asynchrony is a mechanism and purity is a semantics.** Whether a lazy
   pipeline can suspend is its type's task (RFC-0046), fixed when it is
   built; `iter::sum` is `Pure` for a pure pipeline and still awaits its
   element closures.

   **Specialization at prepare time is the interpreter's instance selection**
   (RFC-0020, RFC-0040): `BinOp::Add` at `Float` prepares to the `f64`
   addition, at `Int(I64)` to the `i64` one; a `Take` through a word to a
   word load; a `Const` to a word constant with the bits inline; an extern
   call to its handler's arity form with the handler in hand. `val_types` is
   not read at run time: every question it answered is a preparation-time
   choice. The preparation is one
   exhaustive `match` over `InstKind` and the operand types, so an instruction
   kind added to the IR fails to compile until it is prepared.

2. **Slots are selected, not taken from the `ValueId`.** One assignment runs
   over a body before any operation is prepared. Liveness is a backward
   dataflow over the body's jumps; two values share a slot unless one is live
   where the other is written. On that relation:

   - A jump argument and the block parameter it feeds become one class where
     they do not interfere, so the move on that edge drops.
   - A call site whose handler takes a window gets a contiguous run of `arity`
     registers; an argument whose last use is that call is allocated into its
     window slot, and one live past the call is copied there.
   - Everything else takes the lowest slot free over its live range.

   A storage a place names directly keeps one slot for the whole body, because
   a reference holds a pointer into its register.

3. **A synchronous handler takes its arguments by value, up to three.** A
   `Value` is a scalar pair, so up to three arguments cross in registers and
   constrain no register run. Four or more arguments, every asynchronous
   handler and every spawn take a window: the handler takes each argument out
   of the register it was lent — an asynchronous one before it builds its
   `'static` future. The handler ABI is RFC-0059's.

   **A definition does not drop what it overwrites; an assignment does.**
   `prepare` tells the two apart, never the run. A definition is one store:
   the slot selector shares a slot only between non-interfering values, and
   drop insertion (RFC-0018) has already placed the `Drop` where a move-only
   value's life ended. An assignment — an `Assign`,
   a `FieldSet`, a `Commit` — writes a storage that may hold a live value and
   releases it (RFC-0048 rule 4).

4. **A `while` whose head and body transfer no control is one operation.**
   `prepare` recognizes the shape the lowering gives a `while` (its test a
   `JumpIf`, RFC-0063) when the head label is named only by the entry and the
   back edge, the body label only by the test, no outside jump names a label
   inside, and every operation in the head and body is straight-line.
   Recognition runs inner-first, so a nested region already prepared is
   straight-line. A `for` is its own terminator (RFC-0057). The region's form
   is RFC-0052 rule 3.

5. **An arithmetic chain is one operation, and a body that is one chain has no
   frame.** A maximal run of arithmetic over registers of one inline numeric
   type, each intermediate used once by the next, is one operation,
   optionally ending in one comparison. A numeric literal read only by
   arithmetic or a comparison becomes a register the frame's entry fills, so
   a leaf is always a register. A chain is at most three nodes; a longer run
   is split from the root into pieces joined through registers.
   The operator alphabet is `{Add, Mul}`. A closure body that is
   `params -> return` or `params -> one chain -> return` prepares to
   `Code::Expr`, evaluated with no frame, machine or operation loop; a body
   with more than `ExprChain::MAX_OPERANDS` operands, or with captures, keeps
   its frame.

6. **An `if`/`else` whose arms transfer no control is one operation.** It is
   read from the `Diamond` terminator (RFC-0063). A recognized diamond is
   straight-line, so a `while` whose head or body holds a branch is still one
   loop operation.

7. **A run of extern calls and its deref is one operation.** A maximal run of
   synchronous by-value extern calls in one block, each result used once by
   the next call, each call taking at most two runtime values, optionally
   closed by one `Take` through the last result, is one operation whose
   intermediates are Rust locals. A call carrying an RFC-0007 order edge, a
   window-form or asynchronous call, and a result read twice stay unfused; a
   one-word literal argument is an entry constant and does not break the
   run. The operation is specialized on the
   run's shape — call count and tail as const parameters — because a shape
   walked at run time costs what the dispatch it saves costs.

**Why.** An interpreter that walks the IR recovers at every execution what the
checker and the lowering already fixed; a `Code` makes the per-call clones,
map lookups and type matches inexpressible. A synchronous core keeps
`.await` and boxing off every body that never suspends.
A panic costs nothing on the path that does not fail, where a channel costs a
test per operation and a wider return per call.

**Cost.** A preparation pass and a slot assignment per body at module load. A
storage keeps its slot for the whole body, so a body that takes many
addresses reuses few registers; reading `analysis::loans` would narrow this
and is not built. A `while` whose head or body can suspend prepares as the
blocks it was.

**Rejected.**
- A failure channel — `Result<Value, RuntimeError>` returns, a `Runtime::trap`
  side call, a thread-local flag — every loop with an extern call pays it,
  and `wasm32` refuses the thread-local.
- Choosing the synchronous call path from the callee's effect — a pure callee
  may await (`iter::sum`).
- Deciding ready-or-later per element (`Pull { Ready, Later }`) — it belongs
  at construction.
- One growable contiguous register stack — pushing a callee's frame can
  reallocate it under the pointers references hold (RFC-0052 rule 7).
- Registers as `u64` — an extern reads a lent value through `&Value`, so every
  lend would materialize one.
- Words packed as three `usize` — an operation's inline capacity would differ
  by platform.
- An owned or lent argument buffer at the handler boundary — each costs more
  than the allocation it removes; the arguments are already in registers
  (rule 2).
- A free list of register files, or a thread-local one for extern callbacks —
  the thread-local access dominates the call.
- A chain as a postfix stack of micro-operations, one dispatch for a run of
  chain pieces, `{Add, Sub, Mul}`, four-node trees, a leaf that is an operand
  or a constant — each measured slower or bought no time for its instance
  text.
- Keeping the IR walker beside the machine — two semantics under one suite
  say nothing about either.

## RFC-0048: Ownership is the machine's — a value copies, a register is written once

Status: Accepted

**A `Value` is `Copy` and has no `Drop`.** Releasing a `Large` is an explicit
act of the machine — a drop instruction or the frame's exit — or of a Rust
holder that took ownership. Everything else copies.

1. **Two types, one bit pattern, one trait.** The extern crate defines
   `trait Release: Copy { fn release(self); }` and `Runtime::Value: Release`.
   `Owned<R>` is `Erased<R, Never>`, `#[repr(transparent)]` over
   `ManuallyDrop<R::Value>`, whose `Drop` calls `release`: the one owning
   type, defined once for every runtime, held wherever Rust owns a runtime
   value — a container's elements, an `Erased`, a closure carrier's value, an
   iterator stage's function. The ABI — handler signatures, `Ref`,
   `Elements`, the window — is `R::Value`. `Owned::from_value` is the
   identity and `into_value` is `ManuallyDrop::take`; each takes the
   runtime's `Holding` (RFC-0068 rule 1), so the glue and the runtime do both
   and a handler does neither. The interpreter's `Value` implements
   `Release`: a `Large` drops through its header, a word does nothing.
2. **The header keeps one thing the machine cannot know: the drop.** A `Large`
   erased from an extension type carries `drop_slot::<T>` in its header,
   because a Rust holder releases values whose language type erased their Rust
   type, and no instruction stands at that release. A vtable is a constant of
   the type it describes, so erasing a `Large` takes no lock and hashes no
   `TypeId`.
3. **A frame is aligned cells with mark words.** A frame is a run of
   `#[repr(C, align(64))]` cells of `MaybeUninit<Value>` slots, and bit `i` of
   a mark word is "register `i` owns a `Large`". The mark words sit in the
   slots past the frame's registers. Nothing runs per slot at frame end but
   the sweep. The cell width, the frame bound and the mark-word count are
   RFC-0050 rule 2.
4. **A register is written exactly once per definition.** A define of a
   move-only type (`define::<LARGE>`, which `prepare` chooses) writes the slot
   and sets its bit; a define of a word writes the slot. `assign` (RFC-0045)
   releases the old value if its bit is set, writes, and sets or clears the
   bit. Nothing writes an empty value after a take, and there is no
   `Kind::Empty`; `Undef` stays, as an SSA definition.
5. **A take is static and batched.** `prepare` emits which operand slots each
   operation consumes, and the operation reads them and clears their bits
   with one constant mask it carries (`take_mask`); a copied word touches
   neither slot nor mask. The double-take check is `debug_assert!(marked & mask ==
   mask)`. Tuple helpers are `#[inline(always)]`, since a `(Value, Value)`
   crosses a call boundary through memory.
6. **Release is an instruction or the sweep.** The `Drop` instruction drop
   insertion places (RFC-0018) takes the slot, calls `release` and clears the
   bit. The frame's exit releases the slots whose bits are set. A `Value` the
   machine holds outside a register is released where the machine lets go of
   it.
7. **A Rust holder holds `Owned<R>`.** Every type in `acvus-extern` and
   `acvus-ext` that stores a runtime value it owns holds `Owned<R>`, and Rust's
   `Drop` releases it. `Ref`, `RefMut` and `Elements` borrow and hold
   `R::Value`. The compiler enumerates the holders: an `R::Value` is not stored
   where an `Owned` is expected without `from_value`, nor an `Owned` passed
   where the ABI wants an `R::Value` without `into_value` — both at the glue,
   never in a body.
8. **A panic is an exit, not a release.** A run-time failure is a panic
   (RFC-0044); no cleanup runs and none is owed. A host that catches a
   script's panic and lives sweeps the frame's mark words at the catch. A
   trap is not ordered with effects: which effects a run issued before it
   is not stated, and which trap a run reports is (the least in iteration
   and stage, RFC-0089 rule 5). A trap stays where control puts it: an
   operation that can trap moves only to where it runs on exactly the
   paths it ran on, and a pass removes one whose value nothing reads only
   where `analysis::raise` shows it cannot trap, the one predicate every
   pass that removes an instruction asks. An integer `/` or `%` and a
   checked index cannot trap where the interval domain (RFC-0047 rule 7)
   clears them. A call of an extern cannot trap where the instance it
   names is declared `total` (RFC-0082 rule 9). A call of a local function
   cannot trap where none of the callee's instructions can, decided
   callees first over the call graph, so a call inside a cycle of it can
   trap; a call through a function value can trap. An overflow is removed
   as RFC-0037 rule 3 says.

**Why.** A droppable `Value` needs an address wherever it may drop, which puts
unwind landing pads in handlers and keeps values in memory; it forces a second
store after every move so the frame's `Drop` does not drop twice, and a scan of
every slot at every return. `prepare` already knows at every operand whether it
is consumed or copied, and drop insertion places every release as an
instruction, so the machine owns every fact `Drop` re-derives.

**Cost.** Two names for one bit pattern and a conversion at every store
boundary, free at run time. `mem::drop(v)` on a `Copy` value is a no-op that
reads like a release, which is why the method is `release`. One bit test per
`Large` define or assign, one mask clear per batched take, one bit iteration
per frame exit. An unused call of an extern not declared `total`, of a local
function that may trap, or through a function value still runs, as does an
unused checked index the interval domain cannot clear; each local function a
call names is put through SSA once more to decide whether it can trap.

**Rejected.**
- Changing `Value`'s layout to make the second store cheap — it fixes a store
  that should not exist.
- `&[Released]` for the arity-N window — the glue would know how this
  interpreter tracks ownership; a `Copy` bound is all extern glue may know.
- `Value: Drop` kept, with `ManuallyDrop` only in the register file — keeps the
  landing pads and the address requirement in every handler.
- `Value: Copy` with a manual `release` in each holder's `Drop` — the compiler
  cannot enumerate the holders, so each can forget; `Owned` makes the omission
  unwritable.
- The drop function in the operation instead of the header — no instruction
  stands at the release a Rust holder makes under an erased type.
- Lazy drop (a define releases a still-marked slot) and drop fusion
  (`drop_mask`) — not refused; later stages, each to be measured alone.
- Removing a Pure call whose value nothing reads, whatever its callee does —
  drops the callee's trap: an unused call of a local function that divides
  by zero ran past it, and so did an unused `a[5]` on a two-element array.
- Whether a function can trap as an axis of its effect type — the interval
  proofs that clear a division or an index are MIR facts, computed after the
  checker has closed the function's type, so the type could only say that a
  body holding any division or index may trap.

## RFC-0052: An operation is a struct, and the machine calls it once

Status: Accepted

1. **An operation is a struct that implements one trait, and it holds its
   successor.**

   ```rust
   pub type Exit = u64;
   pub trait Op: Send + Sync {
       fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit;
   }
   ```

   A decision `prepare` made is a type or a field, boxed once; a `run` body
   holds no `match`, no `let … else` and no kind test on a fact its own type
   carries.

   A straight-line run is a **chain**: each operation holds the next and ends
   by tail-calling it, so the run is a line of `jmp *`. An operation with no
   successor ends the chain and hands back its `Exit`: at a joint, the
   `BlockId` the machine enters next; inside a region, the word the part
   computed last. There is no `Terminator` trait and no `Block`.

2. **The stream holds the joints.** `Body::heads: Box<[Box<dyn Op>]>` is indexed
   by `BlockId`, one chain head per joint — a place where control genuinely
   chooses. The machine's loop runs only there:

   ```rust
   loop { at = heads[at].run(self, 0); if at >= SENTINEL { return at } }
   ```

   `RETURN`, `SUSPEND` and the verdicts of rule 3 sit above every block id,
   so one compare per joint leaves the loop.

3. **A region is an operation, and its parts are chains.** A recognized branch
   or loop chooses nothing `prepare` did not already know, so it is an
   operation of the chain it sits in: `Loop` (a `while`, RFC-0044 rule 4),
   `For` (RFC-0057), `Diamond` (RFC-0063), and a region form per `Switch`
   whose arms all rejoin (RFC-0051). Each part heads its own chain, ended by
   `Yield`, which hands the region the part's last word, so no `BlockId`
   leaves a region. A region's parts hold no terminator,
   structurally: a region is admitted only where `prepare::straight_run`
   reached the shape's own edges through straight-line instructions.

   **A part's word is a verdict where the part can exit.** A chain ends in
   `Yield`, or in `Fall`, `Break`, `Continue` or `Return`, which hand back
   `FALL`, `LEAVE`, `AGAIN` and `RETURN`; a `break`, `continue`, `?` or
   `return` under a test is an `Escape` whose escaping arm ends in a
   verdict. A region is monomorphized over whether it can escape, so a loop
   with no way out holds no verdict compare; the two endings never share a
   chain, so a computed word never reads as a verdict.

   **A terminator owns blocks and ids, never a move.** A jump's parallel move
   is `Mov` operations in the block the edge leaves from; a two-way edge
   carrying moves takes a block of its own, and a diamond's join moves end
   each arm. `Fused` is an operation, and so is a chain: `Chain1/2/3` carry
   the operand places as type parameters and the plan as a field, with their
   kernels inline.

4. **No operation that may suspend is fused.** The callee's task is in the
   type the checker settled (RFC-0046), and `prepare` reads it off the callee.
   A call whose task is `Sync` is an operation (`CallDirect`,
   `CallIndirect`) that runs the callee to its value through the window,
   reaching it by index because a caller's `prepare` may run before the
   callee's body exists. A call above `Sync` is a terminator of its own block
   that hands the driver a future. A region or `Fused` holds only the first,
   so a fused operation never re-enters. An assert where a frame is entered
   synchronously catches a disagreement between the checker's effect and the
   body `prepare` produced.

5. **A `Value` is read and written through `&mut Machine`.** The register file
   is the frame of cells and mark words of RFC-0048 rule 3. An operation holds a byte
   displacement, `Off`, multiplied once by `prepare`; `Slot`, the
   language-level index, is a different type and stays inside `prepare`. A
   define is a store, plus one mark bit when the operation's type says `Large`
   (RFC-0050 rule 2). Slot access is unchecked in release, on `prepare`'s
   proof that every slot is below `frame_len`. A slot's kind is static, so
   `prepare` writes the kind byte of every word-typed slot once, when the
   frame is made, and a word-typed operation stores and reads the word only;
   only an option-typed slot, whose kind a run changes (RFC-0039), is written
   whole.

   **A word with one use rides in the argument register.** A word-typed SSA
   value whose one use is the next operation of the same chain is never
   written to the frame: the producer returns it in `r0` and the consumer
   reads it there. Where a word is, is a type, `Place` — `Slot` at an `Off`,
   or `R0`. Two operands never both ride. A word does not ride across a
   joint: a `Loop` reads its condition after its head chain has returned,
   while a `Diamond`'s condition rides.

6. **An extern call holds its handler.** No `Arc<dyn Fn>` and no handler enum:
   `prepare` reads the declaration's arity form once (RFC-0044 rule 3) and
   picks the operation's type, so the synchronous arity-1 call makes no
   compare before the call. The handler is handed the runtime, never the
   machine; its ABI is RFC-0059's.

7. **A frame is made where the machine already is.** A synchronous callee's
   frame is the window above the caller's `frame_len` in the same store,
   marked by its own mark words and released by one sweep: a call makes one
   capacity compare and steps past its own cells. A callee that does not fit,
   or a chain deeper than the store's cells, roots a `Store` of its own.

   **A call's arguments are the callee's first registers.** The caller writes
   each argument straight into the register the callee reads, so a call holds
   an `arity`, not an argument array. An extern's closure call takes the same
   window: a handler is lent the window above the calling frame as
   `Runtime::Frame<'_>`, and the closure's arguments cross straight into its
   run (RFC-0059).

   **The caller owns the frame.** A synchronous closure call is handed
   `Runtime::Frame<'a>`, a borrow of cells someone else owns, and cannot own,
   make or free it. No site in `acvus-ext` makes a frame: a handler passes the
   one it receives down, and a stage takes it as a parameter of `next`. The
   handle is one word, the `FrameState` the frame below owns; its cells are a
   raw slice, because `Frame<'a>` has one lifetime parameter and `&mut` is
   invariant.

   A `Cell` holds only registers: a mark word between cells would break the
   displacement an `Off` is. An unbound `Store` is an empty `Vec`.
   `Store::bind(body)` is the only way to reach a frame, and skips laying the
   body's slot kinds and entry constants when the frame already carries them:
   a word register's kind byte survives every word store (rule 5) and an
   entry constant's register has no writer. The `Vec` does not grow while a
   chain runs, because every frame below the growth point borrows from it.

   A call whose future outlives the frame owns its arguments, because its
   future is `'static`: its glue owns a `Store` and lends a window out of it
   (`Runtime::Rooted`).

**Why.** A fixed op record with a payload table and a returned control enum
made one dispatch and then four more, each re-deciding what `prepare` knew. A
chain of tail calls walks a straight run in three instructions and one taken
branch per operation, where a slice of fat pointers takes seven and two.

**Cost.**
- One vtable-slot load per operation that a function pointer in the stream
  would not pay; it has no dependency on the previous result.
- The tail call is a guarantee only a probe holds: stable Rust has no
  `become`. `acvus-interpreter-test/benches/asm_probe.rs` asserts that every
  `run` in the release binary ends in `jmp *`, but for a list, closed both
  ways, of families a stack address forces to `call` + `ret`.
- The `Place` axis multiplies every family's instances, all emitted whether
  a program reaches them or not.

**Rejected.**
- A `call`/`ret` per operation through a slice of fat pointers — no loop
  shape removes its cost while the stream is a slice.
- Unrolling the dispatch loop — a straight-line group is a superinstruction,
  the recognizers' job.
- A chain kernel reached through a pointer rather than inlined.
- `OpFn` + payload index, or `OpFn` + `Box<dyn Data>` — the pairing of
  function and data is a convention the type system cannot see.
- Variable-length inline records — a wrong jump offset is memory corruption,
  and facts are read through casts.
- `Flow` as a returned word — a compare per operation, not per joint.
- A window passed by value to a handler — three words, past the two SysV
  passes in registers, and its escaping address kills the tail call.
- Aligning every function entry to a cache line — it trades rows against each
  other, and costs text.
- A cold hint on the machine's rare paths — LLVM already lays them after the
  tail jump.

## RFC-0074: A diamond of two pure arms is a select

Status: Accepted

Where a `Diamond`'s arms are pure word arithmetic, the recognizer produces an
operation that evaluates the work and picks the word with no arm call:

```rust
pub struct Select<T: Num, C: Place, D: Place, const R: u8, const COMPUTES_ON_TRUE: bool>
    { cond: C::At, plan: Plan, passed: Off, dst: D::At, next: Box<dyn Op> }
```

1. **One node against a pass-through arm.** `plan` is a one-node chain plan
   evaluated inline; `passed` is the register the other arm hands the join.
   One arm's block holds exactly one `BinOp`, the other arm is the test's own
   edge into the join carrying the incoming word, the join has one word
   parameter owning no `Large`, and the computed value's one use is the join
   edge. `prepare::select_shape` holds the list; nothing is tested at run time.
2. **Every other diamond stays a `Diamond`.** Refused:
   - `/` and `%` — the node runs on both paths, and their traps (RFC-0037
     rule 2) would end a run on the path the program does not take
     (RFC-0048 rule 8). Their trap is no flag of a wrapped result: the
     machine's division faults on a divisor of zero and on `MIN / -1`
     before any flag exists. A flag form would test the divisor first and
     divide by a substituted one; that is a compare and a substitution
     added to every such select, for a shape no measured body has, and it
     is not built. A float `/` and `%` are refused too.
   - An arm of more than one operation — its intermediate would reach a
     register on the path not taken, and `assign_slots`, which runs first, may
     have given that register to a value live outside the arm.
   - Both arms computing — `assign_slots` coalesced both arms' words into the
     join's one register.
   - A join of more than one parameter — there is no one word to select.
   - Anything that is not a `BinOp`: a call, a `Large` define, a store, a path
     write, an indexed read.
   - Operands outside a chain's types (the integer widths and `f64`).

   The program's `+`, `-` and `*` are admitted. They trap where they
   overflow (RFC-0037 rule 3), and the select runs the node as its
   overflowing form: the wrapped value and a flag, which traps nothing.
   The run then ends with the operation's trap text exactly where the flag
   is set and the computing side is the one the test chose; the wrapped
   value reaches no register. That is every run, and only the runs, on
   which the arm's operation trapped, at the same place, since nothing
   stands between the test and the arm's one operation. A `+`, `-` or `*` a
   pass wrote wraps and has no flag to test.

**Why.** One node is the cost rule: a second needs its shape in the type or a
`match` inside `run`, which gives back the branch the select removes.

**Rejected.**
- Refusing the program's `+`, `-` and `*` — the arm becomes a `Diamond`,
  which costs a branch and two arm dispatches in the body that has the most
  selects (`accum`'s `branch while` ran 78 % slower), while the flag keeps
  the trap exact.
- Speculating a raising arm behind a divisor proof — buys one shape and puts a
  numeric proof in the recognizer.
- A select of arbitrary arm width — two arm chains is a loss against a
  `Diamond`'s one, and inlining the width squares the chain family's
  instances.

## RFC-0069: A closure is a code word beside its captures

Status: Accepted

1. **The run is the caller's.** A closure value holds neither the interpreter
   nor the page. A call takes them from where it is made: the `Machine` that
   runs `CallIndirect`, or the `Ctx` a handler was handed. A closure does not
   outlive the run that made it (RFC-0014), and a closure a `Space` holds is
   re-entered through that space's runtime.

2. **The code is a word.** What a closure runs is a `Code` the module's
   `Prepared` owns. A closure names it by address, one word fixed by `prepare`
   where `MakeClosure` is built; nothing counts references to it. The head of a
   `Code` is the function that enters it, chosen once where the `Code` is made,
   so a call is two loads and an indirect call: no `dyn`, and no match on the
   shape at the call. This is RFC-0067's instance word for a body the
   interpreter prepared.

3. **A closure of no captures is its code word.** Its `Value` is inline — the
   kind says closure, the word is the code. Making it allocates nothing,
   copying it is a register copy, and no sweep releases it.

4. **Captures are one block.** A capturing closure is one allocation: the
   code word, the capture count, and the captures behind them. A call binds
   each capture into the callee's frame as a reference (RFC-0018) to the
   capture where it lies; reading it in place is RFC-0073.

5. **One synchronous entry, chosen where the `Code` is made.** The script's
   `CallIndirect` and a handler's closure call reach the same function — the
   `Code`'s head — handing it the window whose first `arity` registers hold the
   arguments and the run it is called in. The head is chosen as narrowly as the
   body allows: a chain body's head is the chain's own evaluator,
   monomorphized on operand type, operators and cast leaves; a body that
   returns an argument has a head that reads that argument. One indirect call
   per closure call and none inside it. The closure value crosses the entry by
   value, in two registers, because an address handed to an indirect callee
   costs the calling operation its tail call. `start` stays a second entry for
   a framed body that may suspend: it fills a frame that exists before the
   future does and hands back that frame, so it cannot share the synchronous
   signature; a chain body's `start` lays a window of its own and calls the
   head. A closure call and an extern call stay different operations to
   `prepare`.

**Why.** The run is already in every caller's hand, and the module owns the
code for as long as the run lasts; what is left of a closure is its captures.
A value that also carries the run and a counted entry costs two allocations
and several atomic operations to make, copy and release, for a closure of no
captures as for one of ten.

**Cost.** The capture block's layout is written once, one `unsafe` site in the
interpreter. A `Code` must not move or die while a word names it — the
`Prepared` outlives every frame of its run; the obligation is stated where the
word is minted.

**Rejected.**
- Keeping the counted fields and making the call cheaper — the value's shape
  is what costs a capture-free lambda two allocations, and every copy and
  release pays it.
- One box for every closure — a closure of no captures has nothing to box.

## RFC-0073: A capture is read in place

Status: Proposed

1. A call puts the capture block's address in the callee's frame once, and a
   capture read is a load at an offset `prepare` fixed. No reference is written
   per capture per call.
2. Loans analysis still reads a capture as a reference (RFC-0018); only the
   machine's read changes.

**Why.** Binding writes one reference per capture on every call (RFC-0069 rule 4),
a cost a closure called in a loop pays per iteration for values that do not
move.
**Cost.** The capture read becomes an operation of its own, reading from the
block address rather than from a register.
