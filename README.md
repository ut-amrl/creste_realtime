# creste-realtime

The easiest way to get started with this repository is to build and run using the provided Dockerfile.

## ROS Dependencies

1. [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
2. [AMRL ROS Messages](https://github.com/ut-amrl/amrl_msgs)

You do not need to install ROS Noetic if working in the docker container. We have tested this Docker container on both Ubuntu 22.04 and Ubuntu 20.04 systems.

## Docker Container Installation
```bash
./build_docker.sh
```

## Docker Container Usage

Make sure to modify the `run_docker.sh` script to mount the correct `amrl_msgs` directory. The directory is
by default set to `/robodata/arthurz/Research/amrl_infrastructure/amrl_msgs:/home/amrl_msg`. You must change the
first path to the location of your `amrl_msgs` directory.
```bash
./run_docker.sh
```

## Running the Code

In the docker container, run the following commands to start the `creste-realtime` package. Make sure to set the
`CONFIG_PATH` and `WEIGHTS_PATH` to the correct path to the configuration file and model weights path file respectively.
```bash
./scripts/run_inference.sh --config_path="$CONFIG_PATH" --weights_path="$WEIGHTS_PATH"
```
