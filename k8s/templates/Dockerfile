FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG USER={{lastname}}
ARG UID={{uuid}}

# 1. Create the home directory first
RUN mkdir -p /localdir/

# 2. Delete the default 'ubuntu' user (UID 1000)
RUN userdel -r ubuntu || true

# 3. Create your custom user
RUN adduser ${USER} --uid ${UID} --home /localdir/ --disabled-password --gecos "" --no-create-home

SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# 4. Install Packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libfl-dev \
    libibverbs1 \
    libnl-3-dev \
    libnl-route-3-dev \
    libudev-dev \
    ninja-build \
    pkg-config \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 5. Fix Python Symlinks
# I removed the trailing '\' at the end of this block
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# 6. Install Pip
# Now this runs as a fresh command
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /localdir/
RUN chown -R ${USER}:${USER} /localdir/

COPY . .
RUN python -m pip install packaging wheel
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch
RUN python -m pip install --no-build-isolation -r requirements.txt

USER ${USER}
WORKDIR /localdir/