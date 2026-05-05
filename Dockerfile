FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV UV_PROJECT_ENVIRONMENT=/opt/gato-venv

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

RUN pip3 install --no-cache-dir cmake==3.24.0

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
        && ln -sf /root/.local/bin/uv /usr/local/bin/uv

ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:/root/.local/bin:${PATH}"

# install Python dependencies into an image-backed environment
WORKDIR /tmp/gato-deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --group dev --no-install-project

ENV LD_LIBRARY_PATH=${UV_PROJECT_ENVIRONMENT}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}

# set working directory
WORKDIR /workspace

# auto-activate venv on shell entry
RUN echo '[ -f /opt/gato-venv/bin/activate ] && source /opt/gato-venv/bin/activate' >> ~/.bashrc
RUN echo 'export PYTHONPATH=/workspace/python${PYTHONPATH:+:$PYTHONPATH}' >> ~/.bashrc

# when container starts
CMD ["/bin/bash"]
