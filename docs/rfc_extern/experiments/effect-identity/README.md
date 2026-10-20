# Probes: effects and identities through declarations

The evidence for how a declaration's effect and identity reach the compiler
(compiler: extern fn §2, §6, §8; types §3). `probes.rs` builds declarations
by hand, as a registry gives them, checks small programs with typeck, and
prints each outcome beside its expectation. An effect is observed through a
pure slot: `apply` takes a pure function, so a lambda whose work the compiler
finds opaque is refused there. The fixture is the probe's means, not a
decided primitive.

| Probe | What it checks | Expected |
|---|---|---|
| e01 | opaque work in the pure slot | refused |
| e02 | work carried in a result's type (`seal`), then run by a declaration that runs its variable (`run`) | refused |
| e03 | the same with pure work | admitted |
| e04 | two functions of one effect variable, one opaque, run | refused |
| e05 | the same, both pure | admitted |
| e06 | two functions of one variable, mixed, carried, then run | refused |
| e07 | two functions of one variable, mixed, outside a slot | admitted |
| c01 | a declaration consuming its variable as its own effect, running nothing | verify refused |
| c02 | the same declaration, own effect pure, running its variable | verify admitted |
| c03 | a variable neither consumed nor carried | verify refused |
| r01 | a generic function called directly, its impl opaque | refused |
| r02 | the same impl reached through a declaration's requirement | refused |
| r03 | the requiring declaration running the requirement's effect variable | refused |
| r04 | the requiring declaration outside a slot | admitted |
| r05 | the requiring declaration's own function type carrying the requirement's effect variable, in the slot | refused |
| r06 | the same, running that variable | refused |
| r07 | the same, outside a slot | admitted |
| i01 | a parameter's identity given to the result, met with the parameter | admitted |
| i02 | two made identities met | refused |
| i03 | identities carried by a type's arguments, each taken back as its own | admitted |
| i04 | the same, taken back swapped | refused |
| i05 | the other argument, taken back as its own | admitted |

Run `docs/rfc_extern/experiments/effect-identity/run.sh`. It builds against
the repository's compiler; build outputs go to `target/` here, which git
ignores.

Result, 2026-10-06, with stable rustc 1.97.1, against the compiler at
`cac021bb`: eighteen as expected, four not.
- e02, e04, e06 are `ClosureExceedsSlot`: one effect variable bounds every function that writes it, carried or run.
- c01 and c03 are `EffectUnrecorded`: the compiler records consumption by the variables a declaration runs, never by its own effect.
- i02 and i04 are `NoConversion`: two identities never meet, and an identity a type argument carries stays with that argument.
- **r02 is admitted:** an opaque impl reached through a requirement leaves no effect in the requiring call, which a pure slot then admits.
- **r03 is refused by verify, `RunsNoEffectVariable`:** a requirement's effect variable is no variable of the declaration's own signature, so the declaration cannot run it.
- **r05 to r07 are `Mismatch` of effects, Pure against Opaque, even outside a slot:** with its own function type carrying the requirement's effect variable, the declaration verifies, but at each call the compiler takes that variable as pure, since no parameter gives it, and the opaque impl the requirement reaches does not fit.
