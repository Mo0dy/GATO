#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/common.sh"

BUILD_DIR="${GATO_BUILD_DIR:-build}"
FRESH=0
TARGET=""
NATIVE_CUDA_ARCH=0
declare -a CMAKE_CONFIGURE_ARGS=()
declare -a CMAKE_BUILD_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./tools/build.sh [options]

Incrementally configure and build GATO with CMake. By default the existing
build directory is reused so unchanged targets are not rebuilt.

Options:
  --fresh                         Delete the build directory before configure.
  --target <name>                 Build one CMake target.
  --plant <list>                  Configure PLANT, e.g. "indy7;iiwa14".
  --knots <list>                  Configure KNOTS, e.g. "8;32;128".
  --cuda-architectures <list>     Configure CMAKE_CUDA_ARCHITECTURES.
  --native-cuda-arch              Build only for the first visible GPU arch.
  -h, --help                      Show this help.

Environment:
  GATO_BUILD_DIR                  Build directory, default: build.
  GATO_SKIP_CUDA_COMPATIBILITY_TEST=1
                                  Skip post-build CUDA artifact validation.
EOF
}

require_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        printf "\n${RED}${BOLD}${CROSS} %s requires a value.${RESET}\n" "${option}"
        exit 1
    fi
}

detect_native_cuda_architecture() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf "\n${RED}${BOLD}${CROSS} --native-cuda-arch requires nvidia-smi on PATH.${RESET}\n"
        exit 1
    fi

    local compute_capability
    if ! compute_capability="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
            | tr -d ' ' \
            | sed '/^$/d' \
            | head -n 1
    )"; then
        printf "\n${RED}${BOLD}${CROSS} --native-cuda-arch could not query GPU compute capability.${RESET}\n"
        exit 1
    fi

    if [[ -z "${compute_capability}" ]]; then
        printf "\n${RED}${BOLD}${CROSS} --native-cuda-arch could not detect a visible GPU compute capability.${RESET}\n"
        exit 1
    fi

    printf "%s-real" "${compute_capability/./}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh)
            FRESH=1
            shift
            ;;
        --target)
            require_value "$1" "${2:-}"
            TARGET="$2"
            shift 2
            ;;
        --plant)
            require_value "$1" "${2:-}"
            CMAKE_CONFIGURE_ARGS+=("-DPLANT=$2")
            shift 2
            ;;
        --knots)
            require_value "$1" "${2:-}"
            CMAKE_CONFIGURE_ARGS+=("-DKNOTS=$2")
            shift 2
            ;;
        --cuda-architectures)
            require_value "$1" "${2:-}"
            CMAKE_CONFIGURE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=$2")
            shift 2
            ;;
        --native-cuda-arch)
            NATIVE_CUDA_ARCH=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf "\n${RED}${BOLD}${CROSS} Unknown option: %s${RESET}\n" "$1"
            usage
            exit 1
            ;;
    esac
done

cd "${REPO_ROOT}"

if [[ "${NATIVE_CUDA_ARCH}" -eq 1 ]]; then
    native_architecture="$(detect_native_cuda_architecture)"
    CMAKE_CONFIGURE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${native_architecture}")
    printf "${YELLOW}${BOLD}${GEAR} Configuring native CUDA architecture: %s${RESET}\n" "${native_architecture}"
fi

if [[ "${FRESH}" -eq 1 ]]; then
    printf "${YELLOW}${BOLD}${GEAR} Removing build directory: %s${RESET}\n" "${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

printf "${YELLOW}${BOLD}${GEAR} Configuring project with CMake in %s...${RESET}\n" "${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}" "${CMAKE_CONFIGURE_ARGS[@]}"

if [[ -n "${TARGET}" ]]; then
    CMAKE_BUILD_ARGS+=(--target "${TARGET}")
    printf "${YELLOW}${BOLD}${GEAR} Building target: %s${RESET}\n" "${TARGET}"
else
    printf "${YELLOW}${BOLD}${GEAR} Building project...${RESET}\n"
fi

cmake --build "${BUILD_DIR}" --parallel "${CMAKE_BUILD_ARGS[@]}"

if [[ -n "${TARGET}" ]]; then
    printf "${YELLOW}${BOLD}${GEAR} Skipping CUDA compatibility validation for focused target build.${RESET}\n"
elif [[ "${GATO_SKIP_CUDA_COMPATIBILITY_TEST:-0}" == "1" ]]; then
    printf "${YELLOW}${BOLD}${GEAR} Skipping CUDA compatibility validation because GATO_SKIP_CUDA_COMPATIBILITY_TEST=1.${RESET}\n"
else
    printf "${YELLOW}${BOLD}${GEAR} Validating CUDA compatibility for built artifacts...${RESET}\n"
    ./tools/test_cuda_compatibility.sh --strict --build-dir "${BUILD_DIR}"
fi

printf "${GREEN}${BOLD}${CHECK} Build complete.${RESET}\n"
