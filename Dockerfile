# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM rust:1.86-slim AS builder

WORKDIR /app

# System deps needed by lalrpop / linking
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
  && rm -rf /var/lib/apt/lists/*

# Copy manifests first for layer-cached dependency fetch
COPY Cargo.toml Cargo.lock ./
COPY acvus-ast/Cargo.toml           acvus-ast/Cargo.toml
COPY acvus-mir/Cargo.toml           acvus-mir/Cargo.toml
COPY acvus-mir-cli/Cargo.toml       acvus-mir-cli/Cargo.toml
COPY acvus-mir-test/Cargo.toml      acvus-mir-test/Cargo.toml
COPY acvus-runtime/Cargo.toml       acvus-runtime/Cargo.toml
COPY acvus-playground/Cargo.toml    acvus-playground/Cargo.toml

# Stub source files so `cargo fetch` / dep compilation succeeds
RUN mkdir -p acvus-ast/src acvus-mir/src acvus-mir-cli/src \
             acvus-mir-test/tests acvus-runtime/src acvus-playground/src && \
    echo 'fn main(){}' > acvus-ast/src/lib.rs && \
    echo 'fn main(){}' > acvus-mir/src/lib.rs && \
    echo 'fn main(){}' > acvus-mir-cli/src/main.rs && \
    echo ''             > acvus-mir-test/tests/e2e.rs && \
    echo 'fn main(){}' > acvus-runtime/src/lib.rs && \
    echo 'fn main(){}' > acvus-playground/src/main.rs && \
    touch acvus-playground/src/index.html

# Pre-build deps (cached unless Cargo.toml / Cargo.lock change)
RUN cargo build --release -p acvus-playground 2>&1 | tail -5 || true
RUN rm -rf acvus-ast/src acvus-mir/src acvus-mir-cli/src \
           acvus-mir-test/tests acvus-runtime/src acvus-playground/src

# Copy real source
COPY acvus-ast         acvus-ast
COPY acvus-mir         acvus-mir
COPY acvus-runtime     acvus-runtime
COPY acvus-playground  acvus-playground

# Build the playground binary (release)
RUN cargo build --release -p acvus-playground

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/acvus-playground /usr/local/bin/acvus-playground

EXPOSE 3000

ENTRYPOINT ["acvus-playground"]
