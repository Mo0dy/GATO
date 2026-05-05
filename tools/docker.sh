#!/usr/bin/env bash
set -Eeuo pipefail

# Source common definitions
source "$(dirname "$0")/common.sh"

IMAGE_NAME="gato"
CONTAINER_NAME="gato-container"
REBUILD_IMAGE=0
ATTACH_SHELL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild-image)
            REBUILD_IMAGE=1
            shift
            ;;
        --no-attach)
            ATTACH_SHELL=0
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./tools/docker.sh [--rebuild-image] [--no-attach]

  --rebuild-image  Rebuild the Docker image and recreate the container.
  --no-attach      Ensure the container is running, but do not open a shell.
EOF
            exit 0
            ;;
        *)
            printf "\n${RED}${BOLD}${CROSS} Unknown option: %s${RESET}\n" "$1"
            exit 1
            ;;
    esac
done

printf "\n${CYAN}${BOLD}--------------------------------------------------${RESET}\n"
printf "${BOLD}${GREEN}${GEAR} Setting up docker container.${RESET}\n"

if [[ "${REBUILD_IMAGE}" -eq 1 ]]; then
    printf "${YELLOW}${BOLD}${GEAR} Rebuilding image '${IMAGE_NAME}'.${RESET}\n\n"
    if docker ps -q -f name=^/${CONTAINER_NAME}$ | grep -q .; then
        docker stop "${CONTAINER_NAME}" >/dev/null
    fi
    if docker ps -aq -f name=^/${CONTAINER_NAME}$ | grep -q .; then
        docker rm -f "${CONTAINER_NAME}" >/dev/null
    fi
    docker build -t "${IMAGE_NAME}" .
    printf "\n${GREEN}${BOLD}${CHECK} Image '${IMAGE_NAME}' rebuilt successfully.${RESET}\n"
elif ! docker image inspect ${IMAGE_NAME} >/dev/null 2>&1; then
    printf "${YELLOW}${BOLD}${GEAR} Image '${IMAGE_NAME}' not found. Building...${RESET}\n\n"
    docker build -t ${IMAGE_NAME} .
    printf "\n${GREEN}${BOLD}${CHECK} Image '${IMAGE_NAME}' built successfully.${RESET}\n"
else
    printf "${GREEN}${BOLD}${CHECK} Image '${IMAGE_NAME}' found.${RESET}\n"
fi

# Prepare for GUI forwarding if possible, but do not fail if X11 is unavailable.
export DISPLAY=${DISPLAY:-:0} # Default to :0 if not set
if command -v xhost >/dev/null 2>&1; then
    if ! xhost +local:docker >/dev/null 2>&1; then
        printf "${YELLOW}${BOLD}${GEAR} Skipping optional xhost setup; X11 forwarding is unavailable.${RESET}\n"
    fi
fi

EXIT_CODE=0

# Check if container exists and is running
if docker ps -q -f name=^/${CONTAINER_NAME}$ | grep -q .; then
    if [[ "${ATTACH_SHELL}" -eq 1 ]]; then
        printf "${BOLD}${GREEN}${ARROW} Attaching to running container ${YELLOW}${CONTAINER_NAME}${RESET}...\n\n"
        docker exec -it "${CONTAINER_NAME}" /bin/bash
        EXIT_CODE=$?
    else
        printf "${GREEN}${BOLD}${CHECK} Container '${CONTAINER_NAME}' is already running.${RESET}\n"
    fi
# Check if container exists but is stopped
elif docker ps -aq -f status=exited -f name=^/${CONTAINER_NAME}$ | grep -q .; then
    if [[ "${ATTACH_SHELL}" -eq 1 ]]; then
        printf "${YELLOW}${BOLD}${GEAR} Starting stopped container ${YELLOW}${CONTAINER_NAME}${RESET} and attaching...\n\n"
        docker start "${CONTAINER_NAME}" >/dev/null
        docker exec -it "${CONTAINER_NAME}" /bin/bash
        EXIT_CODE=$?
    else
        printf "${YELLOW}${BOLD}${GEAR} Starting stopped container ${YELLOW}${CONTAINER_NAME}${RESET}...\n\n"
        docker start "${CONTAINER_NAME}" >/dev/null
    fi
# Container does not exist, run a new one
else
    if [[ "${ATTACH_SHELL}" -eq 1 ]]; then
        printf "${BOLD}${GREEN}${ARROW} Running new container (name: ${YELLOW}${CONTAINER_NAME}${YELLOW})...${RESET}\n\n"
        docker run -it \
            --gpus all \
            --network=host \
            -e DISPLAY="${DISPLAY}" \
            -v "$(pwd)":/workspace:Z \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            --name "${CONTAINER_NAME}" \
            "${IMAGE_NAME}"
        EXIT_CODE=$?
    else
        printf "${BOLD}${GREEN}${ARROW} Creating new container (name: ${YELLOW}${CONTAINER_NAME}${YELLOW})...${RESET}\n\n"
        docker run -d -it \
            --gpus all \
            --network=host \
            -e DISPLAY="${DISPLAY}" \
            -v "$(pwd)":/workspace:Z \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            --name "${CONTAINER_NAME}" \
            "${IMAGE_NAME}" >/dev/null
    fi
fi

if [[ "${ATTACH_SHELL}" -eq 0 ]]; then
    printf "\n${GREEN}${BOLD}${CHECK} Container ${YELLOW}${CONTAINER_NAME}${RESET} is ready.${RESET}\n"
elif [ "$EXIT_CODE" -eq 0 ]; then
    printf "\n${GREEN}${BOLD}${CHECK} Exited container session for ${YELLOW}${CONTAINER_NAME}${RESET} successfully.${RESET}\n"
else
    printf "\n${RED}${BOLD}${CROSS} Command failed or container exited with error (Code: $EXIT_CODE).${RESET}\n"
fi
printf "${CYAN}${BOLD}--------------------------------------------------${RESET}\n\n"
