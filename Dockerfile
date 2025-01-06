# Use the NVIDIA CUDA base image with Ubuntu 20.04 for ROS Noetic compatibility
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

#-----------------------------------------------------------
# 1) Basic dependencies and tools
#-----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    vim \
    unzip && \
    rm -rf /var/lib/apt/lists/*

#-----------------------------------------------------------
# 2) Setup ROS 1 Noetic apt repository and install
#    (For Ubuntu 20.04)
#-----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    lsb-release && \
    # Add ROS Noetic sources
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros1-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update

# Install ROS Noetic (minimal + needed packages)
RUN apt-get install -y --no-install-recommends \
    ros-noetic-desktop \
    ros-noetic-cv-bridge \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-pcl-conversions \
    # ros-noetic-grid-map-core \
    # ros-noetic-grid-map-ros \
    # ros-noetic-grid-map-cv \
    # ros-noetic-grid-map-msgs \
    # for rviz, you could also add ros-noetic-rviz if desired
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep if needed:
RUN apt-get update && apt-get install -y python3-rosdep && \
    rosdep init && \
    rosdep update

#-----------------------------------------------------------
# 3) Environment setup
#-----------------------------------------------------------
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

#-----------------------------------------------------------
# 4) Install PCL, Torch, and other dependencies
#-----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends libpcl-dev && \
    rm -rf /var/lib/apt/lists/*

# PyTorch + TorchScript via pip (you may adjust the version if needed)
RUN pip3 install --no-cache-dir torch torchvision torchaudio

# Download and install libtorch for CUDA 12.1
RUN curl -LO https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip -d /usr/local && \
    rm libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip

# Install Eigen3
RUN apt-get update && apt-get install -y --no-install-recommends libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

#-----------------------------------------------------------
# 5) Source ROS in .bashrc
#-----------------------------------------------------------
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "export ROS_PACKAGE_PATH=/home/creste_realtime/home/amrl_msgs:\$ROS_PACKAGE_PATH" >> ~/.bashrc

# Note: In ROS 1, you typically use catkin_make or catkin tools
# We won't build here by default, because you may want to build interactively.

CMD ["/bin/bash", "-c", "source /opt/ros/noetic/setup.bash && exec bash"]
