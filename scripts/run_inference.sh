#!/bin/bash

# Default paths for config and weights
CONFIG_PATH="./config/creste.yaml"
WEIGHTS_PATH="./traversability_model_trace_distill128_cfs.pt"

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
      # Skip unknown options, Error on unknown arguments
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# Debug Messages
echo "Running inference with weights path: $WEIGHTS_PATH"

# Launch the ROS node with specified flags
rosrun creste_realtime creste_node --config_path="$CONFIG_PATH" --weights_path="$WEIGHTS_PATH"
