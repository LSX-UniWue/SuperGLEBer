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
# This flag is necessary on Ubuntu 24.04 to allow pip to install globally
# without creating a virtual environment (PEP 668 override).
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# 4. Install Packages
# Added python3-pip here
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
    python3-pip \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 5. Fix Python Symlinks
# Ensure python and python3 point to python3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# (Removed the manual get-pip.py installation step)

WORKDIR /localdir/
RUN chown -R ${USER}:${USER} /localdir/

COPY . .

# 6. Install Python packages
RUN python -m pip install packaging wheel
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 torch \
    && rm -rf /root/.cache/pip
RUN python -m pip install --no-build-isolation -r requirements.txt

USER ${USER}
WORKDIR /localdir/