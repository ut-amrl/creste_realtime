#!/bin/bash

# Default service name
DEFAULT_SERVICE_NAME="/navigation/deep_cost_map_service"

# Local frame to include in the header
LOCAL_FRAME="base_link"

# Use the provided service name or the default if none is provided
SERVICE_NAME="${1:-$DEFAULT_SERVICE_NAME}"

echo "Sending goal to frame $LOCAL_FRAME using service $SERVICE_NAME..."

# Call the ROS 2 service. The service type is amrl_msgs/srv/CostmapSrv.
ros2 service call "$SERVICE_NAME" "amrl_msgs/srv/CostmapSrv" "{header: {frame_id: '$LOCAL_FRAME'}}"
