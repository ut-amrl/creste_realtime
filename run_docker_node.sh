#!/bin/bash

USER=$(whoami)

# Function to display usage
display_usage() {
    echo "Usage: $0 -n <container_name> -i <image_name> -m <src1:dst1> <src2:dst2> ..."
    echo "  -n: Name of the container to run"
    echo "  -i: Docker image to use"
    echo "  -m: Mounts in the format src:dst (space-separated for multiple mounts)"
}

DEFAULT_IMAGE_NAME="creste_realtime_ros2"
DEFAULT_CONTAINER_NAME="creste_realtime_container"

FRODO_WS_DIR="$HOME/frodo_autonomy"
TARGET_CRESTE_DIR="/creste_ws/src"

# Default directory mounts
default_mounts=(
    "$FRODO_WS_DIR/src/creste_realtime:$TARGET_CRESTE_DIR/creste_realtime"
    "$FRODO_WS_DIR/src/amrl_msgs:$TARGET_CRESTE_DIR/amrl_msgs"
)

RUN_SCRIPT="$TARGET_CRESTE_DIR/creste_realtime/scripts/run_inference.sh"
MOUNTS=()

# Parse command-line arguments
while getopts "n:i:m:" opt; do
    case $opt in
        n) CONTAINER_NAME="$OPTARG" ;;
        i) IMAGE_NAME="$OPTARG" ;;
        m) MOUNTS+=("$OPTARG") ;;
        *) display_usage; exit 1 ;;
    esac
done

# Assign default image name if not provided
if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME=$DEFAULT_IMAGE_NAME
    echo "Image name not provided. Using default: $IMAGE_NAME"
fi

# Check if required arguments are provided
if [ -z "$CONTAINER_NAME" ]; then
    CONTAINER_NAME=$DEFAULT_CONTAINER_NAME
    echo "Container name not provided. Using default: $CONTAINER_NAME"
fi

# Kill existing container with the same name (if it exists)
existing_container=$(docker ps -a -q --filter "name=^/${CONTAINER_NAME}$")
if [ -n "$existing_container" ]; then
    echo "Container '$CONTAINER_NAME' exists. Killing it..."
    docker rm -f "$CONTAINER_NAME"
fi

# Construct the Docker run command
DOCKER_CMD="docker run --rm --gpus=all --net=host --ipc=host --pid=host -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -it --name $CONTAINER_NAME"

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

# Append the image name and the startup command
DOCKER_CMD+=" $IMAGE_NAME bash -c '$RUN_SCRIPT'"

# Print and execute the command
echo "Running: $DOCKER_CMD"
eval $DOCKER_CMD
