FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG USER={{lastname}}
ARG UID={{uuid}}

RUN adduser ${USER} --uid ${UID} --home /localdir/ --disabled-password --gecos "" --no-create-home

SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update \
  && apt-get install -y software-properties-common \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y \
  python3.11 \
  python3.11-dev \
  python3.11-venv \
  ninja-build \
  git \
  curl \
  libfl-dev \
  && rm -rf /var/lib/apt/lists/*
# https://askubuntu.com/questions/1465653/usr-bin-ld-cannot-find-ll-no-such-file-or-directory/1465654#1465654

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /localdir/
RUN chown -R ${USER}:${USER} /localdir/

COPY . .
RUN python -m pip install packaging wheel psutil ninja
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128
RUN python -m pip install --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

USER ${USER}
WORKDIR /localdir/
