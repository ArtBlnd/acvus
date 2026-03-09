# Build context: workspace root (parent of pomollu-frontend)
#   docker build -f pomollu-frontend/Dockerfile .

# ── Stage 1: WASM build ──────────────────────────────────────────────────────
FROM rust:1.93-slim AS wasm-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config curl \
  && rm -rf /var/lib/apt/lists/*

RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

WORKDIR /app

# Copy manifests for layer caching
COPY Cargo.toml Cargo.lock ./
COPY acvus-ast/Cargo.toml             acvus-ast/Cargo.toml
COPY acvus-mir/Cargo.toml             acvus-mir/Cargo.toml
COPY acvus-mir-pass/Cargo.toml        acvus-mir-pass/Cargo.toml
COPY acvus-mir-cli/Cargo.toml         acvus-mir-cli/Cargo.toml
COPY acvus-interpreter/Cargo.toml     acvus-interpreter/Cargo.toml
COPY acvus-interpreter-test/Cargo.toml acvus-interpreter-test/Cargo.toml
COPY acvus-mir-test/Cargo.toml        acvus-mir-test/Cargo.toml
COPY acvus-orchestration/Cargo.toml   acvus-orchestration/Cargo.toml
COPY acvus-chat/Cargo.toml            acvus-chat/Cargo.toml
COPY acvus-chat-cli/Cargo.toml        acvus-chat-cli/Cargo.toml
COPY acvus-ext/Cargo.toml             acvus-ext/Cargo.toml
COPY acvus-utils/Cargo.toml       acvus-utils/Cargo.toml
COPY acvus-playground/Cargo.toml      acvus-playground/Cargo.toml
COPY pomollu-engine/Cargo.toml        pomollu-engine/Cargo.toml

# Stub sources so cargo can resolve the workspace
RUN mkdir -p acvus-ast/src acvus-mir/src acvus-mir-pass/src acvus-mir-cli/src \
             acvus-interpreter/src acvus-interpreter-test/src \
             acvus-interpreter-test/tests \
             acvus-mir-test/tests \
             acvus-orchestration/src acvus-chat/src acvus-chat-cli/src \
             acvus-ext/src acvus-utils/src \
             acvus-playground/src pomollu-engine/src && \
    echo 'fn main(){}' > acvus-ast/src/lib.rs && \
    echo 'fn main(){}' > acvus-mir/src/lib.rs && \
    echo 'fn main(){}' > acvus-mir-pass/src/lib.rs && \
    echo 'fn main(){}' > acvus-mir-cli/src/lib.rs && \
    echo 'fn main(){}' > acvus-interpreter/src/lib.rs && \
    echo 'fn main(){}' > acvus-interpreter-test/src/lib.rs && \
    echo 'fn main(){}' > acvus-interpreter-test/tests/fixtures.rs && \
    echo ''            > acvus-mir-test/tests/e2e.rs && \
    echo 'fn main(){}' > acvus-orchestration/src/lib.rs && \
    echo 'fn main(){}' > acvus-chat/src/lib.rs && \
    echo 'fn main(){}' > acvus-chat-cli/src/main.rs && \
    echo 'fn main(){}' > acvus-ext/src/lib.rs && \
    echo 'fn main(){}' > acvus-utils/src/lib.rs && \
    echo 'fn main(){}' > acvus-playground/src/main.rs && \
    touch acvus-playground/src/index.html && \
    echo 'fn main(){}' > pomollu-engine/src/lib.rs

RUN cargo fetch 2>&1 | tail -3 || true

# Remove stubs, copy real sources (only what pomollu-engine needs)
RUN rm -rf acvus-ast/src acvus-mir/src acvus-mir-pass/src \
           acvus-interpreter/src acvus-orchestration/src \
           acvus-chat/src acvus-ext/src acvus-utils/src \
           pomollu-engine/src

COPY acvus-ast         acvus-ast
COPY acvus-mir         acvus-mir
COPY acvus-mir-pass    acvus-mir-pass
COPY acvus-interpreter acvus-interpreter
COPY acvus-orchestration acvus-orchestration
COPY acvus-chat        acvus-chat
COPY acvus-ext         acvus-ext
COPY acvus-utils   acvus-utils
COPY pomollu-engine    pomollu-engine

RUN wasm-pack build --release --target bundler pomollu-engine

# ── Stage 2: Frontend build ──────────────────────────────────────────────────
FROM node:22-slim AS frontend-builder

WORKDIR /app

COPY pomollu-frontend/package.json pomollu-frontend/package-lock.json ./
RUN npm ci

COPY pomollu-frontend/ ./

# Copy WASM artifacts from stage 1
COPY --from=wasm-builder /app/pomollu-engine/pkg/pomollu_engine.js           src/lib/wasm/
COPY --from=wasm-builder /app/pomollu-engine/pkg/pomollu_engine.d.ts         src/lib/wasm/
COPY --from=wasm-builder /app/pomollu-engine/pkg/pomollu_engine_bg.js        src/lib/wasm/
COPY --from=wasm-builder /app/pomollu-engine/pkg/pomollu_engine_bg.wasm      src/lib/wasm/
COPY --from=wasm-builder /app/pomollu-engine/pkg/pomollu_engine_bg.wasm.d.ts src/lib/wasm/
COPY --from=wasm-builder /app/pomollu-engine/pkg/package.json                src/lib/wasm/

RUN npm run build

# ── Stage 3: Serve ───────────────────────────────────────────────────────────
FROM nginx:alpine

COPY --from=frontend-builder /app/build /usr/share/nginx/html

# SPA fallback
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
