FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG USER={{lastname}}
ARG UID={{uuid}}

# 1. Create the home directory first
RUN mkdir -p /localdir/

# 2. Delete the default 'ubuntu' user (UID 1000) to avoid conflict
RUN userdel -r ubuntu || true

# 3. Create your custom user
RUN adduser ${USER} --uid ${UID} --home /localdir/ --disabled-password --gecos "" --no-create-home

SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# 4. Merged Package List
# Note: python3-distutils removed (deprecated in Py3.12)
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
    && rm -rf /var/lib/apt/lists/* \


RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /localdir/
RUN chown -R ${USER}:${USER} /localdir/

COPY . .
RUN python -m pip install packaging wheel
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch
RUN python -m pip install --no-build-isolation -r requirements.txt

#RUN python -m pip install flash-attn==2.5.7 --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
#RUN python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex/

USER ${USER}
WORKDIR /localdir/
