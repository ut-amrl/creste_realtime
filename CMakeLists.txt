cmake_minimum_required(VERSION 3.10)
project(creste_realtime)

set(VTK_DIR "/usr/local/vtk-9.3.0/lib/cmake/vtk-9.3" CACHE PATH "" FORCE)
list(PREPEND CMAKE_PREFIX_PATH "${VTK_DIR}")

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install" CACHE PATH "Installation Directory" FORCE)
endif()

# Enable the new IN_LIST operator usage for CMake
if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()

message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

# --------------------------------------------------------------------
# Determine ROS version from environment variable
# --------------------------------------------------------------------
if(DEFINED ENV{ROS_VERSION})
  if("$ENV{ROS_VERSION}" STREQUAL "1")
    add_definitions(-DROS1)
    set(USE_ROS1 TRUE)
    message(STATUS "Detected ROS_VERSION=1 -> Using ROS1 (rosbuild)")
  elseif("$ENV{ROS_VERSION}" STREQUAL "2")
    add_definitions(-DROS2)
    set(USE_ROS1 FALSE)
    message(STATUS "Detected ROS_VERSION=2 -> Using ROS2 (ament_cmake)")
  else()
    message(FATAL_ERROR "Unknown ROS_VERSION: $ENV{ROS_VERSION}. Must be '1' or '2'.")
  endif()
else()
  message(FATAL_ERROR "ROS_VERSION environment variable is not defined")
endif()

# --------------------------------------------------------------------
# Compiler flags
# --------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wno-unknown-pragmas -Werror")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(STATUS "Additional Flags for Release mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2 -DNDEBUG")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message(STATUS "Additional Flags for Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

# --------------------------------------------------------------------
# ROS1 vs. ROS2 Setup
# --------------------------------------------------------------------

if(USE_ROS1)
  # ------------------------------------------
  # ROS1 (rosbuild)
  # ------------------------------------------
  if(NOT DEFINED ENV{ROS_ROOT})
    message(FATAL_ERROR "ROS_ROOT is not set; cannot build for ROS1.")
  endif()
  include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
  rosbuild_init()

  # Optionally control static/shared libs with these lines:
  # set(ROS_BUILD_STATIC_LIBS true)
  # set(ROS_BUILD_SHARED_LIBS false)

else()
  # ------------------------------------------
  # ROS2 (ament_cmake)
  # ------------------------------------------
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(amrl_msgs REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(pcl_conversions REQUIRED)
  message(STATUS "Building for ROS2 using ament_cmake")
endif()

# --------------------------------------------------------------------
# Common Third-Party Dependencies (Non-ROS)
# --------------------------------------------------------------------
# e.g., Torch, OpenCV, PCL, etc.
list(APPEND CMAKE_PREFIX_PATH "/usr/local/libtorch")  # Adjust if needed

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters octree)

find_package(VTK REQUIRED CONFIG)
include(${VTK_USE_FILE})  # If needed

# We'll gather them in a variable
set(non_ros_libs
  glog
  gflags
  yaml-cpp
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${VTK_LIBRARIES}  # if needed
)

# Include directories
include_directories(
  include
  src/shared
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
)

add_subdirectory(src/shared)

# Link directories
link_directories(${PCL_LIBRARY_DIRS})

# --------------------------------------------------------------------
# (Optional) If you have custom messages or services
# --------------------------------------------------------------------
if(USE_ROS1)
  rosbuild_genmsg()
  rosbuild_gensrv()
else()
  # For ROS2, you might do something like:
  # rosidl_generate_interfaces(${PROJECT_NAME}
  #   "msg/MyMsg.msg"
  #   "srv/MySrv.srv"
  #   DEPENDENCIES std_msgs
  # )
endif()

# --------------------------------------------------------------------
# Building
# --------------------------------------------------------------------
if(USE_ROS1)
  # ------------------------------------------
  # Build with rosbuild
  # ------------------------------------------
  # Example: add a shared library
  # rosbuild_add_library(shared_lib src/shared/foo.cc)
  # target_link_libraries(shared_lib ${non_ros_libs})

  # Executable #1: creste_node
  rosbuild_add_executable(creste_node
    src/creste_node_main.cc
    src/creste_node.cc
    src/utils.cc
    src/visualization.cc
  )
  # Link with ros libs + third-party libs
  target_link_libraries(creste_node
    roslib roscpp  # typical ROS1 libs
    ${non_ros_libs}
    pcl_conversions
  )

else()
  # ------------------------------------------
  # Build with ament_cmake (ROS2)
  # ------------------------------------------
  add_executable(creste_node
    src/creste_node_main.cc
    src/creste_node.cc
    src/utils.cc
    src/visualization.cc
  )
  target_link_libraries(creste_node
    ${non_ros_libs}
    rclcpp::rclcpp
  )
  ament_target_dependencies(creste_node
    rclcpp
    sensor_msgs
    geometry_msgs
    amrl_msgs
    cv_bridge
    tf2
    tf2_ros
    pcl_conversions
  )

endif()

# --------------------------------------------------------------------
# Output paths
# --------------------------------------------------------------------
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Print some helpful info
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs: ${TORCH_LIBRARIES}")

# --------------------------------------------------------------------
# Installation / ament_package
# --------------------------------------------------------------------
if(USE_ROS1)
  # For rosbuild, there's typically no 'install' or 'ament_package'.
  message(STATUS "No ROS1 install step by default. Add custom logic if needed.")
else()
  # ROS2: ament
  install(TARGETS creste_node # publisher_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  ament_package()
endif()
