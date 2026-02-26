FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# system dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-numpy \
        python3-pip \
        vim \
        gnupg \
        lsb-release \
        software-properties-common \
        libeigen3-dev \
        libxinerama-dev \
        libglfw3-dev \
        libxcursor-dev \
        libxi-dev \
        libxrandr-dev \
        libxxf86vm-dev \
        x11-apps \
        libx11-dev \
        libxext-dev \
        libxrender-dev \
        libxfixes-dev \
        xvfb \
        && rm -rf /var/lib/apt/lists/*

# python aliases
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
        && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# PyTorch
RUN pip3 install --no-cache-dir \
        torch \
        torchvision \
        torchaudio \
        numpy

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

# set working directory
WORKDIR /workspace

# auto source python environment
RUN echo "[ -f /workspace/.venv/bin/activate ] && source /workspace/.venv/bin/activate" >> ~/.bashrc

# when container starts
CMD ["/bin/bash"]
