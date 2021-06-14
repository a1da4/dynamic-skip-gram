#FROM python:3.6-slim
FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y tzdata && \
    apt-get install -y \
    git \
    vim \
    libsndfile-dev \
    apt-utils \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    python3-tk && \
    apt clean autoclean && \
    apt autoremove -y

RUN ln -fns /usr/bin/python3.6 /usr/bin/python && \
    ln -fns /usr/bin/python3.6 /usr/bin/python3 && \
    ln -fns /usr/bin/pip3 /usr/bin/pip

ENV TZ Asia/Tokyo
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:jp
ENV LC_ALL ja_JP.UTF-8

RUN pip install --upgrade pip && \
    pip install --upgrade setuptools

RUN python3 -m pip install numpy scipy matplotlib

WORKDIR /work
COPY src/ /work/src/
