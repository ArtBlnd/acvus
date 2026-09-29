# host

The host embeds the interpreter. It declares the types of its contexts and of the inputs and result of each function a run may start at, passes the inputs into a run, and receives the result. Its declared types are extern-defined (types §1).

## 1. The page

The host owns the page behind every context. Where a body takes, lends and commits a context is the compiler's (compiler: context §3); what the page does with them is the host's.
