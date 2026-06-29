# Reproducible NMJ analysis image (linux/arm64 + linux/amd64 via multi-arch base digest).
# Python deps: requirements.lock (universal, hash-verified).
# Regenerate lock: scripts/lock_requirements.sh
FROM python@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf AS builder

ARG CMAKE_VERSION=3.31.6-2
ARG BUILD_ESSENTIAL_VERSION=12.12
ARG GIT_VERSION=1:2.47.3-0+deb13u1
ARG UV_VERSION=0.9.7

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake=${CMAKE_VERSION} \
        build-essential=${BUILD_ESSENTIAL_VERSION} \
        git=${GIT_VERSION} \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock .
RUN pip install --no-cache-dir uv==${UV_VERSION} \
    && uv pip install --system --no-cache --require-hashes -r requirements.lock

FROM python@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf AS runtime

ARG LIBGOMP_VERSION=14.2.0-19

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1=${LIBGOMP_VERSION} \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Code is mounted at runtime via docker-compose
# Default CMD is overridden by docker-compose; batch uses 8503, single-image uses 8504.
CMD ["streamlit", "run", "BTX.py", "--server.port=8504", "--server.address=0.0.0.0"]
