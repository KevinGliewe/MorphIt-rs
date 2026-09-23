# MorphIt HTTP server (crates/morphit-server) with the web UI and examples.
#
#   docker build -t morphit-server .                                # CPU image
#   docker build --target runtime-gpu -t morphit-server:gpu .       # NVIDIA GPU image
#
# Stage 1 compiles the server; the runtime stages hold only the binary and
# web/. The CPU image has no Vulkan loader, so `--device auto` packs on the
# CPU. The GPU image adds the Vulkan loader and the NVIDIA ICD; run it with
# `docker run --gpus all`. Results are identical on either device.
#
# Behind a TLS-intercepting proxy, pass its CA chain (PEM) as a build secret:
#   docker build --secret id=ca_certs,src=corp-ca.pem -t morphit-server .

FROM rust:1-bookworm AS builder
WORKDIR /src
# rust-toolchain.toml is left out on purpose: the image's stable toolchain is used.
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    --mount=type=secret,id=ca_certs,required=false \
    if [ -s /run/secrets/ca_certs ]; then \
        cat /etc/ssl/certs/ca-certificates.crt /run/secrets/ca_certs > /tmp/ca.pem; \
        export CARGO_HTTP_CAINFO=/tmp/ca.pem SSL_CERT_FILE=/tmp/ca.pem; \
    fi \
 && cargo build --release --locked -p morphit-server \
 && cp target/release/morphit-server /usr/local/bin/morphit-server


FROM debian:bookworm-slim AS runtime-cpu
RUN useradd --create-home --shell /usr/sbin/nologin morphit
COPY --from=builder /usr/local/bin/morphit-server /usr/local/bin/morphit-server
COPY web/ /app/web/
ENV MORPHIT_WEB_DIR=/app/web \
    MORPHIT_BIND=0.0.0.0:8000 \
    MORPHIT_DEVICE=auto
USER morphit
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["morphit-server", "--healthcheck"]
CMD ["morphit-server"]


# No CUDA needed: the server computes through Vulkan (wgpu). The NVIDIA
# container toolkit injects the driver into this plain image.
FROM ubuntu:24.04 AS runtime-gpu
RUN --mount=type=secret,id=ca_certs,required=false \
    APT_CA=""; \
    if [ -s /run/secrets/ca_certs ]; then \
        cat /etc/ssl/certs/ca-certificates.crt /run/secrets/ca_certs > /tmp/ca.pem; \
        APT_CA="-o Acquire::https::CaInfo=/tmp/ca.pem"; \
    fi \
 && apt-get $APT_CA update \
 && apt-get $APT_CA install -y --no-install-recommends libvulkan1 \
 && rm -rf /var/lib/apt/lists/* /tmp/ca.pem \
 && useradd --create-home --shell /usr/sbin/nologin morphit
# Vulkan driver manifest for the NVIDIA libraries the container toolkit
# mounts in (graphics capability).
COPY docker/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
COPY --from=builder /usr/local/bin/morphit-server /usr/local/bin/morphit-server
COPY web/ /app/web/
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    MORPHIT_WEB_DIR=/app/web \
    MORPHIT_BIND=0.0.0.0:8000 \
    MORPHIT_DEVICE=auto
USER morphit
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["morphit-server", "--healthcheck"]
CMD ["morphit-server"]


# Default target: the CPU image.
FROM runtime-cpu
