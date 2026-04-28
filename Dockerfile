# start with NVIDIA CUDA base image and ROS Humble
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS cuda
# TODO: is ROS still needed?
FROM ros:humble-ros-base


# CUDA
COPY --from=cuda /usr/local/cuda /usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV UV_PROJECT_ENVIRONMENT=/opt/gato-venv

# system dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-numpy \
        python3-pip \
        vim \
        gnupg \
        lsb-release \
        software-properties-common \
        ros-humble-urdfdom \
        ros-humble-hpp-fcl \
        ros-humble-urdfdom-headers \
        python3-colcon-common-extensions \
        python3-rosdep \
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

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN ln -sf /root/.local/bin/uv /usr/local/bin/uv

ENV PATH=${UV_PROJECT_ENVIRONMENT}/bin:/root/.local/bin:${PATH}

# install Python dependencies into an image-backed environment
WORKDIR /tmp/gato-deps
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --group dev

ENV LD_LIBRARY_PATH=${UV_PROJECT_ENVIRONMENT}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}

# set working directory
WORKDIR /workspace

# Do not auto-source ROS in the default shell: ROS injects its own PYTHONPATH
# and can mix ROS eigenpy with the venv pinocchio, causing
# `undefined symbol: EIGENPY_ARRAY_APIPyArray_RUNTIME_VERSION` on import.
# Source /opt/ros/humble/setup.bash manually only in shells that need ROS tools.
#RUN echo "[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash" >> ~/.bashrc

# auto source python environment
RUN echo "[ -f /opt/gato-venv/bin/activate ] && source /opt/gato-venv/bin/activate" >> ~/.bashrc
RUN echo "export PYTHONPATH=/workspace/python\${PYTHONPATH:+:\$PYTHONPATH}" >> ~/.bashrc

# Do not inherit ROS's /ros_entrypoint.sh from the base image: it sources ROS
# before bash starts, which pollutes the default shell environment and can mix
# ROS eigenpy with the venv pinocchio.
ENTRYPOINT []

# when container starts
CMD ["/bin/bash"]
