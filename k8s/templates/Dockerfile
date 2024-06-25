FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG USER={{lastname}}
ARG UID={{uuid}}

RUN adduser ${USER} --uid ${UID} --home /localdir/ --disabled-password --gecos "" --no-create-home

SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update \
  && apt-get update \
  && apt-get install -y \
  python3-dev \
  python3-venv \
  python3-distutils \
  ninja-build \
  git \
  curl \
  && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /localdir/
RUN chown -R ${USER}:${USER} /localdir/

COPY . .
# we need these two as installing them together with the rest of the requirements will fail
RUN python -m pip install Cython==0.29.36 numpy==1.25.2
RUN python -m pip install --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu117 -r requirements.txt
#RUN python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex/

USER ${USER}
WORKDIR /localdir/
