# GATO
> GPU-Accelerated Trajectory Optimization

Numerical experiments and the open-source solver from  ["GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control"](https://arxiv.org/abs/2510.07625)

## Installation

```sh
git clone https://github.com/A2R-Lab/GATO.git
cd GATO
```

### Requirements
- Docker
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Docker is used for containerization and is strongly advised.

### Setup

```sh
./tools/install.sh
```

This will:

- initialize git submodules
- rebuild the Docker image
- recreate the Docker container from that image
- build the project inside the container

To enter the container later:

```sh
./tools/docker.sh
```

To force a fresh image rebuild:

```sh
./tools/docker.sh --rebuild-image
```

### Build Options

You can control which Python extension modules are built by selecting plant models and horizon lengths at CMake configure time:

```sh
mkdir -p build && cd build
cmake -DPLANT="indy7;iiwa14" -DKNOTS="8;32;128" ..
cmake --build . --parallel
```

- `PLANT`: semicolon-separated list of plant targets (`indy7`, `iiwa14`).
- `KNOTS`: semicolon-separated list of horizon lengths.

Built Python modules are written to `python/bsqp/` as `bsqpN{N}_{plant}.so`.

### Reference Environment

- Ubuntu 22.04
- CUDA 12.9
- CMake 3.24+
- C++17
- gcc 11.4.0
- Python 3.10.12
- Docker 28.1.0

## Usage

See [batch_sqp.cu](examples/bsqp.cu) for a minimal example of a batched trajectory optimization solve in C++/CUDA. Example Jupyter notebooks and python benchmarks using GATO for MPC are in [examples/](examples/).

The container shell automatically picks up the image-backed Python environment
and exports `PYTHONPATH=/workspace/python`.

Run the Python benchmark example inside the container with:

```sh
python examples/benchmark_fig8.py
```

Run the C++ example with:

``` sh
./build/bsqp
```

## Related

- The open-source [MPCGPU solver](https://github.com/A2R-Lab/MPCGPU)
- [GRiD](https://github.com/A2R-Lab/GRiD), a GPU-accelerated library for computing rigid body dynamics with analytical gradients

## Cite

```bibtex
@misc{du2025gatogpuacceleratedbatchedtrajectory,
      title={GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control},
      author={Alexander Du and Emre Adabag and Gabriel Bravo and Brian Plancher},
      year={2025},
      eprint={2510.07625},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.07625},
}
```
