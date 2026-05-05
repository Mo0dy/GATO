#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"

CONTAINER_NAME="gato-container"

git submodule update --init --recursive

printf "\n${CYAN}${BOLD}--------------------------------------------------${RESET}\n"
printf "${BOLD}${GREEN}${GEAR} Bootstrapping Docker-first GATO setup.${RESET}\n"

if ! command -v docker >/dev/null 2>&1; then
    printf "\n${RED}${BOLD}${CROSS} Docker is required but was not found on PATH.${RESET}\n"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    printf "\n${RED}${BOLD}${CROSS} Docker is installed but the daemon is not reachable.${RESET}\n"
    exit 1
fi

printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n"
printf "${BOLD}${GREEN}${GEAR} Rebuilding image and ensuring container is running.${RESET}\n"
"$(dirname "$0")/docker.sh" --rebuild-image --no-attach

printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n"
printf "${BOLD}${GREEN}${GEAR} Building project inside the container.${RESET}\n"
docker exec "${CONTAINER_NAME}" bash -lc "cd /workspace && ./tools/build.sh"

printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n"
printf "${GREEN}${BOLD}${CHECK} Setup complete.${RESET}\n"
printf "${BOLD}${GREEN}${ARROW} Enter the container:${RESET} ./tools/docker.sh\n"
printf "${BOLD}${GREEN}${ARROW} Run a Python example:${RESET} python examples/benchmark_fig8.py\n"
printf "${BOLD}${GREEN}${ARROW} Run the C++ example:${RESET} ./build/bsqp\n"
printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n\n"
