# undefined behaviour

## 1. Any-target lowering

The language can be lowered to any target. What every target cannot keep at no cost is not promised: it is undefined, or left to the implementation.

An extreme target: an assembler whose contexts are registers (`@rax`, `@rbx`), whose extern fns are opaque, and whose syscalls are async. The language lowers to it. A context is a typed window whatever backs it, a register included. An extern fn is only its declaration. An async syscall is an effect that consumes an order token and yields the next.

With `@rax` and `@rbx` declared as `u64` contexts and `sys::getpid`, `sys::write` declared as effectful extern fns:

```acvus
@rax = sys::getpid();
anyorder {
    sys::write(1, "a");
    sys::write(2, "b");
}
@rbx = @rax + 1;
@rbx
```

An illustrative lowering (no such backend exists; the point is that nothing in the program stops one):

```asm
; contexts are registers: @rax is rax, @rbx is rbx
    call    sys_getpid          ; an opaque extern: only its declaration is known
                                ; its result lands in rax, which is @rax
    ; anyorder: both writes consume the entry token
    submit  write, 1, "a"   -> t1
    submit  write, 2, "b"   -> t2
    wait    t1, t2              ; the merge: the chain resumes after both
    lea     rbx, [rax + 1]      ; no overflow check: overflow is undefined
    ret
```

## 2. Premises

A run that breaks a premise the language takes as given has no meaning.

- Integer arithmetic stays within its type: overflow is undefined.
- Every declaration the program relies on is true: every declared fact of an extern fn, and every `anyorder` the program writes.

## 3. Implementation limits

Call depth, stack, registers and frames are the implementation's. A backend that cannot run an admitted program has reached its own limit; the program is not refused. A trap an implementation raises at such a limit belongs to the implementation, not to the language.
