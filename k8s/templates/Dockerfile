FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG USER={{lastname}}
ARG UID={{uuid}}

RUN adduser ${USER} --uid ${UID} --home /localdir/ --disabled-password --gecos "" --no-create-home

SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update \
  && apt-get install -y \
  ninja-build \
  git \
  curl \
  libfl-dev \
  && rm -rf /var/lib/apt/lists/*
# https://askubuntu.com/questions/1465653/usr-bin-ld-cannot-find-ll-no-such-file-or-directory/1465654#1465654

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /localdir/

# Create venv with Python 3.11 (uv downloads Python automatically)
RUN uv venv --python 3.11 /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

RUN chown -R ${USER}:${USER} /localdir/

COPY . .

# Install dependencies
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV UV_INDEX_STRATEGY=unsafe-best-match
ENV UV_LINK_MODE=copy

# Use lockfile for reproducible installs
# Pre-install build dependencies needed for --no-build-isolation
# flash-attn requires torch to be present during its build
RUN uv pip install --no-cache packaging wheel setuptools psutil ninja \
    "torch==$(grep -E '^torch==' requirements.lock | cut -d'=' -f3)"
RUN uv pip sync --no-cache --no-build-isolation requirements.lock
RUN uv pip freeze

USER ${USER}
WORKDIR /localdir/
