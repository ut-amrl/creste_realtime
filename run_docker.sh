#!/bin/bash
USER=$(whoami)

docker run -it --net=host \
    --gpus all \
    -v /robodata/arthurz/Research/lift-splat-map/data/coda/3d_comp/os1:/point_clouds \
    -v /robodata/arthurz/Research/lift-splat-map/data/coda/downsampled_2/2d_rect/cam0:/images \
    -v /robodata/arthurz/Datasets/CODa_v2/calibrations:/calibrations \
    -v $(pwd):/lift-splat-map-realtime \
    -v $(pwd)/../../gridmap_ws:/gridmap_ws \
    lsmap_realtime