#!/bin/bash
USER=$(whoami)

export DISPLAY=localhost:10.0  # Ensure this matches the DISPLAY variable used on the server
xhost +local:podman

docker run -it --rm \
    --gpus all \
    --net=host \
    --security-opt label=disable \
    --env="DISPLAY" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v /robodata/arthurz/Research/lift-splat-map/data/coda/3d_comp/os1:/point_clouds \
    -v /robodata/arthurz/Research/lift-splat-map/data/coda/downsampled_2/2d_rect/cam0:/images \
    -v /robodata/arthurz/Datasets/CODa_v2/calibrations:/calibrations \
    -v $(pwd):/lift-splat-map-realtime \
    -v $(pwd)/../../gridmap_ws:/gridmap_ws \
    lsmap_realtime