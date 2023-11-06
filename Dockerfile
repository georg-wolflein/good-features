FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install Python 3.9
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.9 python3-pip python3.9-distutils git && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# OpenGL is needed for OpenCV
RUN apt update && \
    apt install -y libgl1-mesa-glx vim && \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

VOLUME /app
VOLUME /data
VOLUME /metadata

ENV PYTHONPATH "/app:${PYTHONPATH}"

ENV DATA_FOLDER /data
ENV HYDRA_FULL_ERROR 1

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN mkdir -p /app
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser
WORKDIR /app

RUN git config --global --add safe.directory /app

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD sleep 365d