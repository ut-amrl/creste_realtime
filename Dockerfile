# Start from the NVIDIA CUDA 12.1.0 / cuDNN 8 devel image on Ubuntu 22.04
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# -----------------------------------------------------------
# 1) Basic dependencies and tools
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    lsb-release \
    gnupg2 \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    libopencv-dev \
    libgtest-dev \
    libgoogle-glog-dev \
    python3-pip \
    software-properties-common \
    tmux \
    vim \
    unzip \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------
# 2) Add ROS2 Humble apt repository for Ubuntu 22.04 and install
# -----------------------------------------------------------
# Official instructions from:
#  https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Binary.html
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    && rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Add the ROS2 apt repository
RUN add-apt-repository universe && apt-get update

#  - Add official ROS2 GPG key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -

#  - Add apt sources for ROS2 (Humbl):
#    We use $(. /etc/os-release; echo $UBUNTU_CODENAME) to get "jammy"
RUN sh -c 'echo "deb [arch=amd64] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release;echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2-humble.list'

# Install ROS2 Humble desktop and some commonly used packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-pcl-conversions \
    python3-argcomplete \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------
# 3) Environment setup for ROS2 + GPU
# -----------------------------------------------------------
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# We set a default RMW. You can change it if you prefer Cyclone.
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# We'll set AMENT_PREFIX_PATH in the shell profile
# If you build new custom stuff, you might also want to source your colcon workspace
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "export AMENT_PREFIX_PATH=/opt/ros/humble:\$AMENT_PREFIX_PATH" >> ~/.bashrc

# -----------------------------------------------------------
# 4) Install PCL, Torch, and other dependencies
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends libpcl-dev && \
    rm -rf /var/lib/apt/lists/*

# PyTorch + TorchScript via pip (adjust version if needed):
RUN pip3 install --no-cache-dir torch torchvision torchaudio

# Download and install LibTorch (C++ distribution) for CUDA 12.1
# Adjust version if there's a newer build you prefer
RUN curl -LO https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip -d /usr/local && \
    rm libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip

ENV Torch_DIR=/opt/libtorch
ENV LD_LIBRARY_PATH="/opt/libtorch/lib:$LD_LIBRARY_PATH"

# Install Eigen3
RUN apt-get update && apt-get install -y --no-install-recommends libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------
# 5) Additional environment additions
# -----------------------------------------------------------
# Make sure /usr/local/lib is in LD_LIBRARY_PATH
RUN echo "export LD_LIBRARY_PATH=/usr/local/libtorch/lib:/usr/local/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
# RUN echo "export AMENT_PREFIX_PATH=\$AMENT_PREFIX_PATH:/creste_ws/src/amrl_msgs" >> ~/.bashrc

WORKDIR /creste_ws

# Example: if you want to define a custom CMD that always sources ROS2
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && exec bash"]
