#!/bin/bash

# Get cwd
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Define variables
REMOTE_USER="arthurz"
REMOTE_HOST="robolidar.csres.utexas.edu"
REMOTE_DIRS=("/robodata/arthurz/Research/lift-splat-map/data/coda/3d_comp/os1" "/robodata/arthurz/Research/lift-splat-map/data/coda/2d_rect/cam0" "/robodata/arthurz/Datasets/CODa_v2/calibrations")
LOCAL_MOUNT_BASE="$DIR/data"

# Store mount points
MOUNT_POINTS=()

# Function to unmount directories
function unmount_all {
    echo "Unmounting directories..."
    for LOCAL_MOUNT_POINT in "${MOUNT_POINTS[@]}"; do
        if mountpoint -q "$LOCAL_MOUNT_POINT"; then
            fusermount -u "$LOCAL_MOUNT_POINT"
            echo "Unmounted $LOCAL_MOUNT_POINT."
        fi
    done
}

# Trap SIGINT (Ctrl+C) to trigger unmounting before exit
trap unmount_all EXIT

# Check if SSHFS is installed
if ! command -v sshfs &> /dev/null; then
    echo "SSHFS is not installed. Please install it using 'sudo apt-get install sshfs'."
    exit 1
fi

# Create local mount points and mount remote directories
for REMOTE_DIR in "${REMOTE_DIRS[@]}"; do
    # Extract the directory name to use as the local mount point
    DIR_NAME=$(basename "$REMOTE_DIR")
    LOCAL_MOUNT_POINT="$LOCAL_MOUNT_BASE/$DIR_NAME"

    # Create the local mount directory if it doesn't exist
    if [ ! -d "$LOCAL_MOUNT_POINT" ]; then
        mkdir -p "$LOCAL_MOUNT_POINT"
    fi

    # Mount the remote directory
    echo "Mounting $REMOTE_DIR to $LOCAL_MOUNT_POINT..."
    sshfs -o uid=$(id -u),gid=$(id -g),allow_other "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_MOUNT_POINT"

    # Check if the mount was successful
    if mountpoint -q "$LOCAL_MOUNT_POINT"; then
        echo "Successfully mounted $REMOTE_DIR to $LOCAL_MOUNT_POINT."
        MOUNT_POINTS+=("$LOCAL_MOUNT_POINT")
    else
        echo "Failed to mount $REMOTE_DIR. Please check your SSH connection and paths."
    fi
done

# Keep the script running until interrupted
echo "Press Ctrl+C to unmount and exit."
while true; do
    sleep 1
done