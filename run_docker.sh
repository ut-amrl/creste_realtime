#!/bin/bash
USER=$(whoami)

export DISPLAY=${DISPLAY:-:1}
xhost +local:docker

docker run -it --rm \
    --gpus all \
    --net=host \
    --ipc=host \
    --pid=host \
    -e ROS_DOMAIN_ID=42 \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e DISPLAY=$DISPLAY \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v $(pwd):/lift-splat-map-realtime \
    -v $(pwd)/../../gridmap_ws:/gridmap_ws \
    --user $(id -u):$(id -g) \
    lsmap_realtime

        # -v $(pwd)/data/os1:/point_clouds \
    # -v $(pwd)/data/cam0:/images \
    # -v $(pwd)/data/calibrations:/calibrations \