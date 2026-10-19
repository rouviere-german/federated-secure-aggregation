# syntax=docker/dockerfile:1
#
# Multi-stage, non-root image. Base images are pinned by digest (resolved 2026-09-13); Dependabot's
# docker ecosystem proposes digest bumps. python:3.12-slim is multi-arch, so the index digest works
# on amd64 and arm64.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
# Drop the vendored test suites of the four scientific wheels (~99 MB of the 388 MB venv).
# Nothing in the coordinator or a silo imports them, and the public test helpers live outside
# these directories (numpy.testing, pandas._testing, scipy._lib), so what the application can
# import is unchanged. Compiled bytecode is deliberately *kept*: the runtime filesystem is
# read-only, so a .pyc that is missing at start-up can never be written.
RUN set -eux; \
    site=/opt/venv/lib/python3.12/site-packages; \
    for pkg in numpy scipy pandas sklearn; do \
        find "$site/$pkg" -type d -name tests -prune -exec rm -rf {} +; \
    done

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="federated-secure-aggregation" \
      org.opencontainers.image.licenses="MIT"
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000
# The coordinator's only writable path: the privacy budget ledger lives here. It is created in
# the image and owned by the app user so that a named volume mounted over it inherits that
# ownership; the rest of the filesystem stays read-only. A budget that reset on restart would
# make the claimed epsilon a statement about uptime rather than about the data.
RUN install -d -o 10001 -g 10001 -m 0750 /var/lib/fedsecagg
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-m", "federated_secure_aggregation.core.healthcheck"]
ENTRYPOINT ["python", "-m", "federated_secure_aggregation"]
CMD ["serve"]
