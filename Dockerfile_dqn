FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_ROOT /usr/local/cuda/bin

ENV http_proxy=135.245.192.7:8000
ENV https_proxy=135.245.192.7:8000
ENV HTTP_PROXY=135.245.192.7:8000
ENV HTTPS_PROXY=135.245.192.7:8000
ENV LANG C.UTF-8

COPY proxy.conf /etc/apt/apt.conf.d

RUN apt-get update && \
    apt-get install -y \
    nano \
    curl \
    git \
    wget \
    xz-utils \
    build-essential \
    libsqlite3-dev \
    libreadline-dev \
    libssl-dev \
    openssl \
    libfreetype6-dev \
    libxft-dev \
    liblapack-dev \
    libopenblas-dev \
    libbz2-dev \
    software-properties-common \
    nvidia-cuda-toolkit \
    graphviz

EXPOSE 9998
EXPOSE 6006
EXPOSE 6007
EXPOSE 6008
EXPOSE 6009
EXPOSE 6010
EXPOSE 6011
EXPOSE 6012
EXPOSE 6013
EXPOSE 6014
EXPOSE 6015

RUN apt-get install -y \
    python3 \
    python3-pip \
    python3.7

ENTRYPOINT bash
