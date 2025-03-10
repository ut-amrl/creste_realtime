#!/bin/bash

# Default service name
DEFAULT_SERVICE_NAME="/navigation/deep_cost_map_service"

# Local frame
LOCAL_FRAME="base_link"

# Use the provided service name or the default if none is provided
SERVICE_NAME="${1:-$DEFAULT_SERVICE_NAME}"

# Call the ROS service with the header only
echo "Sending goal to frame $LOCAL_FRAME using service $SERVICE_NAME..."
rosservice call "$SERVICE_NAME" "header:
  frame_id: '$LOCAL_FRAME'"
