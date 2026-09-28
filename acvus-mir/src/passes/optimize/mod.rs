//! Transforms (passes-structure §1, §4). Each reads facts through the
//! analysis manager and changes IR only through `ir::cfg::Rewriter`. Each
//! states the analysis-domain pairs it reads, the form it requires and keeps,
//! its standard algorithm and cost, why it keeps the sequential meaning
//! (compilation §1), and that it keeps extern-defined type identity
//! (compilation §3). None is written yet.
