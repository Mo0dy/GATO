#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"

STRICT=0
BUILD_DIR="${GATO_BUILD_DIR:-build}"

require_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        printf "\n${RED}${BOLD}${CROSS} %s requires a value.${RESET}\n" "${option}"
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --strict)
            STRICT=1
            shift
            ;;
        --build-dir)
            require_value "$1" "${2:-}"
            BUILD_DIR="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./tools/test_cuda_compatibility.sh [--strict] [--build-dir <dir>]

Checks whether the built CUDA artifacts in this repository contain compatible
cubin or PTX code for the currently visible GPU(s). By default the script
warns and exits successfully when required tools or GPUs are unavailable.
With --strict it fails instead.

Options:
  --build-dir <dir>  CMake build directory, default: GATO_BUILD_DIR or build.
EOF
            exit 0
            ;;
        *)
            printf "\n${RED}${BOLD}${CROSS} Unknown option: %s${RESET}\n" "$1"
            exit 1
            ;;
    esac
done

warn_or_fail() {
    local message="$1"
    if [[ "${STRICT}" -eq 1 ]]; then
        printf "\n${RED}${BOLD}${CROSS} %s${RESET}\n" "${message}"
        exit 1
    fi
    printf "\n${YELLOW}${BOLD}${GEAR} %s${RESET}\n" "${message}"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn_or_fail "Skipping CUDA compatibility validation because nvidia-smi is unavailable."
    exit 0
fi

if ! command -v cuobjdump >/dev/null 2>&1; then
    warn_or_fail "Skipping CUDA compatibility validation because cuobjdump is unavailable."
    exit 0
fi

mapfile -t GPU_CAPABILITIES < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sed '/^$/d')
if [[ "${#GPU_CAPABILITIES[@]}" -eq 0 ]]; then
    warn_or_fail "Skipping CUDA compatibility validation because no visible GPUs were reported."
    exit 0
fi

declare -a ARTIFACTS=()
if [[ -x "${BUILD_DIR}/bsqp" ]]; then
    ARTIFACTS+=("${BUILD_DIR}/bsqp")
fi
while IFS= read -r artifact; do
    ARTIFACTS+=("${artifact}")
done < <(find python/bsqp -maxdepth 1 -type f -name '*.so' | sort 2>/dev/null || true)

if [[ "${#ARTIFACTS[@]}" -eq 0 ]]; then
    warn_or_fail "Skipping CUDA compatibility validation because no built CUDA artifacts were found."
    exit 0
fi

cc_to_code() {
    local cc="$1"
    local major="${cc%%.*}"
    local minor="${cc##*.}"
    printf "%d" "$((10 * major + minor))"
}

extract_arch_codes() {
    local mode="$1"
    local artifact="$2"
    cuobjdump "${mode}" "${artifact}" 2>/dev/null \
        | sed -n 's/.*sm_\([0-9][0-9]*\).*/\1/p' \
        | sort -u
}

real_compatible() {
    local gpu_code="$1"
    local cubin_code="$2"
    local gpu_major="$((gpu_code / 10))"
    local cubin_major="$((cubin_code / 10))"
    local gpu_minor="$((gpu_code % 10))"
    local cubin_minor="$((cubin_code % 10))"
    [[ "${gpu_major}" -eq "${cubin_major}" && "${gpu_minor}" -ge "${cubin_minor}" ]]
}

ptx_compatible() {
    local gpu_code="$1"
    local ptx_code="$2"
    [[ "${gpu_code}" -ge "${ptx_code}" ]]
}

printf "\n${CYAN}${BOLD}--------------------------------------------------${RESET}\n"
printf "${BOLD}${GREEN}${GEAR} Validating CUDA artifact compatibility.${RESET}\n"
printf "${BOLD}${GREEN}${EYE} Visible GPU compute capabilities:${RESET} %s\n" "${GPU_CAPABILITIES[*]}"

for artifact in "${ARTIFACTS[@]}"; do
    mapfile -t ELF_CODES < <(extract_arch_codes --list-elf "${artifact}")
    mapfile -t PTX_CODES < <(extract_arch_codes --list-ptx "${artifact}")

    if [[ "${#ELF_CODES[@]}" -eq 0 && "${#PTX_CODES[@]}" -eq 0 ]]; then
        printf "\n${RED}${BOLD}${CROSS} %s${RESET}\n" "No embedded cubin or PTX code found in ${artifact}."
        exit 1
    fi

    printf "\n${BOLD}${GREEN}${ARROW} Checking ${artifact}${RESET}\n"
    if [[ "${#ELF_CODES[@]}" -gt 0 ]]; then
        printf "  cubins: %s\n" "${ELF_CODES[*]}"
    else
        printf "  cubins: none\n"
    fi
    if [[ "${#PTX_CODES[@]}" -gt 0 ]]; then
        printf "  ptx:    %s\n" "${PTX_CODES[*]}"
    else
        printf "  ptx:    none\n"
    fi

    for gpu_cc in "${GPU_CAPABILITIES[@]}"; do
        gpu_code="$(cc_to_code "${gpu_cc}")"
        compatible=0

        for elf_code in "${ELF_CODES[@]}"; do
            if real_compatible "${gpu_code}" "${elf_code}"; then
                compatible=1
                break
            fi
        done

        if [[ "${compatible}" -eq 0 ]]; then
            for ptx_code in "${PTX_CODES[@]}"; do
                if ptx_compatible "${gpu_code}" "${ptx_code}"; then
                    compatible=1
                    break
                fi
            done
        fi

        if [[ "${compatible}" -eq 0 ]]; then
            printf "\n${RED}${BOLD}${CROSS} Artifact %s is not compatible with visible GPU compute capability %s.${RESET}\n" "${artifact}" "${gpu_cc}"
            printf "Rebuild with -DCMAKE_CUDA_ARCHITECTURES=<your-sm> or a portable multi-arch list.\n"
            exit 1
        fi

        printf "  ${CHECK} compatible with GPU compute capability %s\n" "${gpu_cc}"
    done
done

printf "\n${GREEN}${BOLD}${CHECK} CUDA compatibility validation passed.${RESET}\n"
printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n\n"
