# host

The host embeds the interpreter. It declares the types of its contexts and of each entry's inputs and result, passes the inputs into a run, and receives the result. Its declared types are extern-defined (types §1).

## 1. The page

The host owns the page behind every context. Where a body loads and commits a context is the compiler's (compiler: context §3); what the page does with a commit, and what a load returns, is the host's.
