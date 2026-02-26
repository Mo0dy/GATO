FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# system dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        curl \
        ca-certificates \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
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

RUN pip3 install --no-cache-dir cmake

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# set working directory
WORKDIR /workspace

# create venv and install project dependencies via uv
# (pyproject.toml is copied first for Docker layer caching)
COPY pyproject.toml ./
RUN uv venv .venv --python python${PYTHON_VERSION} \
        && uv pip install --python .venv/bin/python torch torchvision torchaudio \
        && uv pip install --python .venv/bin/python -e ".[dev]"

ENV LD_LIBRARY_PATH=/workspace/.venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}

# auto-activate venv on shell entry
RUN echo '[ -f /workspace/.venv/bin/activate ] && source /workspace/.venv/bin/activate' >> ~/.bashrc

# when container starts
CMD ["/bin/bash"]
