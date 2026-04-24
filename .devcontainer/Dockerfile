FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash \
    ca-certificates \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    gdb \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    libfaiss-dev \
    python3-faiss \
    tree \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python || true \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip || true

RUN pip3 install --no-cache-dir \
    tensorflow-datasets \
    tensorflow \
    numpy \
    pandas \
    matplotlib

RUN useradd -m -s /bin/bash -u 904 vetlean \
    && echo 'vetlean ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER vetlean

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>