FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
#FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ARG USER={{lastname}}
ARG UID={{uuid}}

RUN adduser ${USER} --uid ${UID} --home /localdir/ --disabled-password --gecos "" --no-create-home

SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update \
  && apt-get install -y \
  python3-dev \
  python3-venv \
  python3-distutils \
  ninja-build \
  git \
  curl \
  libfl-dev \
  && rm -rf /var/lib/apt/lists/*
# https://askubuntu.com/questions/1465653/usr-bin-ld-cannot-find-ll-no-such-file-or-directory/1465654#1465654

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /localdir/
RUN chown -R ${USER}:${USER} /localdir/

COPY . .
RUN python -m pip install packaging wheel
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.3.1+cu121
RUN python -m pip install --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

#RUN python -m pip install flash-attn==2.5.7 --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
#RUN python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex/

USER ${USER}
WORKDIR /localdir/
