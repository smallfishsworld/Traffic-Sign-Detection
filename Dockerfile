# syntax=docker/dockerfile:1.7
# Inference service image: OpenVINO + FastAPI, without PyTorch. The model is mounted at run time
# (see docker-compose.yml). Build:  docker build -t tsd-serve:0.1.0 .
# Behind a TLS-intercepting proxy, pass its CA bundle as a build secret (never stored in a
# layer):  docker build --secret id=ca_bundle,src=/path/to/ca.pem ...

# Pinned by digest (the image the recorded build used); bump deliberately, e.g. with Renovate.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

FROM ${PYTHON_IMAGE} AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy UV_COMPILE_BYTECODE=0 UV_PYTHON_DOWNLOADS=never \
    UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=2
WORKDIR /app
RUN --mount=type=secret,id=ca_bundle,required=false \
    if [ -f /run/secrets/ca_bundle ]; then export PIP_CERT=/run/secrets/ca_bundle; fi; \
    pip install --no-compile uv==0.11.32
# Dependencies first, from the lock file alone (bind mounts, so a README or source edit does not
# invalidate this layer; the uv download cache persists across builds): openvino + serve extras.
RUN --mount=type=secret,id=ca_bundle,required=false \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    if [ -f /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --locked --no-dev --no-install-project --extra openvino --extra serve
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=secret,id=ca_bundle,required=false \
    --mount=type=cache,target=/root/.cache/uv \
    if [ -f /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --locked --no-dev --no-editable --extra openvino --extra serve

FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --create-home --uid 10001 tsd
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TSD_HOST=0.0.0.0 TSD_PORT=8000 \
    TSD_MODEL_PATH=/models/yolo11n_openvino_int8/yolo11n.xml
WORKDIR /app
USER tsd
EXPOSE 8000
# Probe the port the service actually listens on (TSD_PORT can be overridden at run time).
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('TSD_PORT', '8000'), timeout=2)"
CMD ["tsd", "serve"]
