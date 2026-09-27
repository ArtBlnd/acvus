# host

The host embeds the interpreter. It declares the types of its contexts and of each entry's inputs and result, passes the inputs into a run, and receives the result. Its declared types are extern-defined (types §1).

## 1. The page

The host owns the page behind every context. Within a run, a context lives in a slot of the body that names it:

- The body fetches each context it names from the page when it is entered.
- Before a call whose effect may read or write a context, the body commits that context to the page, and it fetches it again after the call. A call whose effect declares the contexts it touches brackets only those; an opaque one brackets every context the body holds.
- When the body returns, it commits every context it holds.

What the page does with a commit, and what a fetch returns, is the host's (context §3).
