# Use the NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary tools and libraries
RUN apt-get update && apt-get install -y \
    lsb-release \
    gnupg2 \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    libopencv-dev \
    python3-pip \
    software-properties-common \
    tmux \
    unzip

# Add the ROS 2 apt repository
RUN apt-get update && apt-get install -y curl gnupg lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Install ROS 2 Humble
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-rclcpp \
    ros-humble-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-std-msgs \
    ros-humble-pcl-conversions \
    python3-colcon-common-extensions

# Install PCL library
RUN apt-get update && apt-get install -y libpcl-dev

# Install TorchScript (PyTorch C++ API)
RUN pip3 install torch torchvision torchaudio

# Download and install libtorch for cuda 12.1
RUN curl -LO https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip -d /usr/local && \
    rm libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip

# Set up environment
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Add the user workspace
RUN mkdir -p /workspace/src
WORKDIR /workspace

# Copy the package to the workspace
COPY . /workspace/src/lsmap_realtime

RUN /bin/bash -c "source /opt/ros/humble/setup.sh"

# Source the setup file
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && exec bash"]