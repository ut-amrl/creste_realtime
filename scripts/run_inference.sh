#!/bin/bash

SESSION="create_inference"
CRESTE_WS="/creste_ws"

tmux has-session -t "$SESSION" 2>/dev/null
if [ $? != 0 ]; then
    tmux new-session -d -s "$SESSION"
fi

tmux send-keys -t "$SESSION" "cd $CRESTE_WS" C-m

# Default paths for config and weights
CONFIG_PATH="$CRESTE_WS/src/creste_realtime/config/frodo/creste_mono.yaml"
WEIGHTS_PATH="$CRESTE_WS/src/creste_realtime/frodo_ckpts/traversability_model_trace_frozendinov2_cfs.pt"

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

# Check if CRESTE_DIR exists, error if not
if [ ! -d "$CRESTE_WS" ]; then
  echo "Error: CRESTE_DIR ($CRESTE_WS) does not exist."
  exit 1
fi
# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: Config file ($CONFIG_PATH) does not exist."
  exit 1
fi
# Check if weights file exists
if [ ! -f "$WEIGHTS_PATH" ]; then
  echo "Error: Weights file ($WEIGHTS_PATH) does not exist."
  exit 1
fi

tmux send-keys -t "$SESSION" "colcon build --packages-select amrl_msgs creste_realtime" C-m
tmux send-keys -t "$SESSION" "source install/setup.bash" C-m
tmux send-keys -t "$SESSION" "ros2 run creste_realtime creste_node --config_path \"$CONFIG_PATH\" --weights_path \"$WEIGHTS_PATH\"" C-m

tmux send-keys -t "$SESSION" "sleep 1" C-m

# 3) now set use_sim_time to true on the running node
# tmux send-keys -t "$SESSION" \
#   "ros2 param set /creste_node use_sim_time true" C-m

# Optionally attach to the tmux session so you can view the output.
tmux attach -t "$SESSION"
