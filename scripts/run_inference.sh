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
      # Error on unknown arguments (or skip them, depending on your preference)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

if [[ -n "$WEIGHTS_PATH" ]]; then
  # If a weights_path was specified
  echo "Running inference with weights path: $WEIGHTS_PATH"
  rosrun creste_realtime creste_node --config_path="$CONFIG_PATH" --weights_path="$WEIGHTS_PATH"
else
  # If no weights_path is given, omit it entirely
  echo "Running inference with default config only (no weights path)."
  rosrun creste_realtime creste_node --config_path="$CONFIG_PATH"
fi
