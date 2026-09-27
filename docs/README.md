# docs

Three sets of RFCs, and nothing else:

- `rfc_compiler`: the language, as the compiler sees it. It means something with no runtime at all.
- `rfc_extern`: the extern side, and the runtime contract between extern fns and whatever runs them.
- `rfc_interpreter`: this interpreter.

## The rule

**The lists are fixed.** No document is added to a set, and none is split from one. Each document keeps its form: a concept, stated in a few abstract sentences, in the language's own terms. When a concept changes, the whole set is rewritten; nothing is patched in place.

What this rules out, and what to do instead:

- **Adding a document for a new feature.**
  Don't: add `rfc_compiler/iterators.md` because iterators arrived.
  Do: find the concept the feature is an instance of (a `for` over an iterator is control flow deciding by type) and let that section's rule cover it. If no concept covers it, the concept set is wrong: rewrite the set.

- **Listing cases where a rule applies.**
  Don't: "`i64` copies. `(i64, bool)` copies. `[u8; 4]` copies. `&T` copies. `Option<i64>` ..."
  Do: "A value copies when its type is a primitive, a shared reference, or a structural type whose every part copies."

- **Describing how the compiler proceeds.**
  Don't: "The solver collects candidates, tries them in declaration order, and keeps the first that unifies."
  Do: "A call reaches the declaration its arguments' types decide. The outcome does not depend on the order the program is checked in."

- **Naming the implementation.**
  Don't: "`InstKind::Merge` joins the `Order` values in `AnyorderScope::acc` (lower.rs:760)."
  Do: "The block yields a merge of every token its calls yielded, and the chain resumes from that merge."

- **Stating one fact in two places.**
  Don't: repeat the capture rules in ownership and in functions.
  Do: state them once, in functions §2, and write "(functions §2)" where another document needs them.

- **Turning an implementation limit into a language rule.**
  Don't: "A reference is not a component of an array."
  Do: nothing in the language. A backend that cannot run an admitted program has reached its own limit (undefined behaviour §3).

- **Growing a section with exceptions.**
  Don't: add a bullet per surprising case under equality.
  Do: state the one real exception in the rule itself: "Equality means what it means in Rust, with one exception: `NaN == NaN` is true."

- **Explaining standard concepts.**
  Don't: "`!` is a type with no values; it can stand in for any type because ..."
  Do: "`!` is the bottom type, and every type constructor is invariant."

- **Turning a direction into numbered rules.**
  Don't: split what refusals aim for into rules 1 to 9.
  Do: write the aim as prose, as refusals and compilation are written.

- **Recording who decided, and when.**
  Don't: "(owner, 2026-09-27) The checker is always right."
  Do: "The checker is right." The history is in version control.
