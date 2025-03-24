#!/bin/bash

# Default paths for config and weights
CONFIG_PATH="./config/creste.yaml"
WEIGHTS_PATH=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config_path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --weights_path)
      WEIGHTS_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# Check if ros2 is installed (i.e. if the "ros2" command is available)
if command -v ros2 > /dev/null 2>&1; then
  ROS_VERSION=2
else
  ROS_VERSION=1
fi

./bin/creste_node --config_path "$CONFIG_PATH" --weights_path "$WEIGHTS_PATH"

