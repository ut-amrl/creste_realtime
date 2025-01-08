#!/bin/bash

# Service name and local frame
SERVICE_NAME="/navigation/carrot_planner"
LOCAL_FRAME="base_link"

# Goal parameters (5 meters ahead in local coordinates)
GOAL_X=10.0  # 5 meters forward
GOAL_Y=0.0  # No lateral movement
GOAL_Z=0.0  # Same height
QUAT_X=0.0  # Orientation quaternion (facing forward)
QUAT_Y=0.0
QUAT_Z=0.0
QUAT_W=1.0

# Call the ROS service with the local frame goal
echo "Sending goal to ($GOAL_X, $GOAL_Y, $GOAL_Z) in frame $LOCAL_FRAME..."
rosservice call "$SERVICE_NAME" "carrot:
  header:
    frame_id: '$LOCAL_FRAME'
  pose:
    position:
      x: $GOAL_X
      y: $GOAL_Y
      z: $GOAL_Z
    orientation:
      x: $QUAT_X
      y: $QUAT_Y
      z: $QUAT_Z
      w: $QUAT_W"
