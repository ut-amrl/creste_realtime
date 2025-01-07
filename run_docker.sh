#!/bin/bash

USER=$(whoami)

# Function to display usage
display_usage() {
    echo "Usage: $0 -n <container_name> -i <image_name> -m <src1:dst1> <src2:dst2> ..."
    echo "  -n: Name of the container to run"
    echo "  -i: Docker image to use"
    echo "  -m: Mounts in the format src:dst (space-separated for multiple mounts)"
}

DEFAULT_IMAGE_NAME="amrl-infra-ros1"
# Default directory mounts
default_mounts=(
    "./:/home/creste_realtime"
    "/robodata/arthurz/Research/amrl_infrastructure/amrl_msgs:/home/amrl_msgs"
    "/robodata/jackal_logs/arthurz/lsmap_bags:/home/lsmap_bags"
)

# Parse command-line arguments
while getopts "n:i:m:" opt; do
    case $opt in
        n) CONTAINER_NAME="$OPTARG" ;;
        i) IMAGE_NAME="$OPTARG" ;;
        m) MOUNTS+=("$OPTARG") ;;
        *) display_usage; exit 1 ;;
    esac
done

# Assign default container name if not provided
if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME=$DEFAULT_IMAGE_NAME
fi

# Check if required arguments are provided
if [ -z "$CONTAINER_NAME" ]; then
    echo "Error: Container name not provided"
    display_usage
    exit 1
fi

# Construct the Docker run command
DOCKER_CMD="docker run --rm --gpus=all --net=host -it --name $CONTAINER_NAME"

# Add default mounts
for DEFAULT_MOUNT in "${default_mounts[@]}"; do
    SRC="$(echo $DEFAULT_MOUNT | cut -d: -f1)"
    DST="$(echo $DEFAULT_MOUNT | cut -d: -f2)"
    DOCKER_CMD+=" -v $SRC:$DST"
done

# Add user-specified mounts
if [ ! -z "$MOUNTS" ]; then
    for MOUNT in "${MOUNTS[@]}"; do
        SRC="$(echo $MOUNT | cut -d: -f1)"
        DST="$(echo $MOUNT | cut -d: -f2)"
        DOCKER_CMD+=" -v $SRC:$DST"
    done
fi

DOCKER_CMD+=" $IMAGE_NAME"

# Print and execute the command
echo "Running: $DOCKER_CMD"
eval $DOCKER_CMD
