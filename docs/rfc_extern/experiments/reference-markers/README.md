# Experiment: a reference held uniformly

The evidence for references §4, a reference held uniformly, and for the
layout exception of variables §4. `model.rs` is a model runtime of
sixteen-byte `Value`s with `Uniform`, `Ref`, `Mut`, the `Shared` and
`Exclusive` markers and `TypeVar` as references and variables state them; its storage counts
owners instead of freeing, so a wrong count is seen without undefined
behaviour.

| Part | What it checks | Run |
|---|---|---|
| `behavior.rs` | release, the borrowed marker, the held forms, and a negative control | `cargo test --manifest-path docs/rfc_extern/experiments/reference-markers/Cargo.toml` |
| the same, under Miri | that the model has no undefined behaviour | `cargo +nightly miri test --manifest-path docs/rfc_extern/experiments/reference-markers/Cargo.toml` |
| `probes/` | what rustc admits and refuses, each probe's second line stating the expected outcome | `docs/rfc_extern/experiments/reference-markers/run_probes.sh` |
| `layout.rs` | an author-defined container's layout under each element view, against the generic view | `docs/rfc_extern/experiments/reference-markers/run_layout.sh` |

Every build output goes to `target/` here, which git ignores. The layout run
needs a nightly toolchain; std is not rebuilt with `-Zrandomize-layout`, so
only the containers defined in `layout.rs` are randomized.
