# Use the official ROS2 Humble base image
FROM ros:humble-ros-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary tools and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rclcpp \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-std-msgs \
    ros-humble-pcl-conversions \
    libpcl-dev \
    software-properties-common \
    tmux

# # Add NVIDIA package repositories
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
#     && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
#     && wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.65.01-1_amd64.deb \
#     && dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.65.01-1_amd64.deb \
#     && apt-key add /var/cuda-repo-ubuntu2004-11-7-local/7fa2af80.pub \
#     && apt-get update \
#     && apt-get install -y cuda

# # Install NVIDIA cuDNN
# RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb \
#     && dpkg -i nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb \
#     && apt-get update \
#     && apt-get install -y libcudnn8 libcudnn8-dev

# # Install NVIDIA NCCL
# RUN apt-get install -y libnccl2 libnccl-dev

# # Install NVIDIA TensorRT
# RUN apt-get install -y tensorrt libnvinfer-dev libnvinfer-plugin-dev

# Install TorchScript (PyTorch C++ API)
RUN pip3 install torch torchvision torchaudio

# Add the user workspace
RUN mkdir -p /workspace/src
WORKDIR /workspace

# Copy the package to the workspace
COPY . /workspace/src/lsmap_realtime

# Build the workspace
RUN . /opt/ros/humble/setup.sh \
    && colcon build

# Source the setup file
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && bash"]
