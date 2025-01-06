# Minimum version for older ROS distributions:
# (rosbuild often used 2.6 as a baseline)
cmake_minimum_required(VERSION 2.6)
project(creste_realtime)

message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

# Compiler flags
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wno-unknown-pragmas -Werror")

# Handle Debug/Release flags
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  message(STATUS "Additional Flags for Release mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2 -DNDEBUG")
elseif(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  message(STATUS "Additional Flags for Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

#-----------------------------------------------------------------------------
# rosbuild initialization
#-----------------------------------------------------------------------------
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
rosbuild_init()

# Optionally, control whether you build shared/static libs
# set(ROS_BUILD_STATIC_LIBS true)
# set(ROS_BUILD_SHARED_LIBS false)

#-----------------------------------------------------------------------------
# Find Libtorch, OpenCV, PCL, etc.
# (Same as your original approach, but without ament)
#-----------------------------------------------------------------------------
# Adjust prefix path for Libtorch if needed
list(APPEND CMAKE_PREFIX_PATH "/usr/local/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(grid_map_core REQUIRED)
# find_package(grid_map_msgs REQUIRED)
# find_package(grid_map_ros REQUIRED)
# find_package(grid_map_cv REQUIRED)
find_package(yaml-cpp REQUIRED)

# If you need GridMap libs (grid_map_core, grid_map_ros, etc.), 
# you can locate them as normal ROS packages:
# rosbuild_find_ros_package(grid_map_core)
# rosbuild_find_ros_package(grid_map_ros)
# rosbuild_find_ros_package(grid_map_cv)
# rosbuild_find_ros_package(grid_map_msgs)

# If your code uses sensor_msgs, std_msgs, pcl_conversions, etc.
# you typically reference them as “rosbuild depends” in your manifest (.xml).
# 
# For linking, we’ll gather them into a variable.
# You can also add roslib, roscpp, etc. as needed.
set(ros_libs
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  pcl_conversions
  # grid_map_core
  # grid_map_ros
  # grid_map_cv
  # grid_map_msgs
)

#-----------------------------------------------------------------------------
# Include directories
#-----------------------------------------------------------------------------
# The rosbuild macros automatically include some ROS-related paths,
# but we can add extra includes if needed:
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  # ${grid_map_core_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS} 
)

link_directories(${PCL_LIBRARY_DIRS})

#-----------------------------------------------------------------------------
# Generate messages/services if your package defines them
#-----------------------------------------------------------------------------
rosbuild_genmsg()
rosbuild_gensrv()

#-----------------------------------------------------------------------------
# Build executables
# In rosbuild, we typically use the macro: rosbuild_add_executable
#-----------------------------------------------------------------------------
rosbuild_add_executable(creste_node
  src/lsmap_node_main.cc
  src/lsmap_node.cc
)

# Link libraries for lsmap_node
target_link_libraries(creste_node
  ${ros_libs}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  ${TORCH_LIBRARIES}
  # Add Eigen or CUDA if you link them explicitly, e.g.
  # ${EIGEN3_LIBRARIES}
  # ${CUDA_LIBRARIES}
)

#-----------------------------------------------------------------------------
# Another executable (publisher_node), if you have it:
#-----------------------------------------------------------------------------
rosbuild_add_executable(publisher_node
  src/publisher_node_main.cc
)

target_link_libraries(publisher_node
  ${ros_libs}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  # etc.
)

#-----------------------------------------------------------------------------
# If you produce shared libraries or run gtests, you’d do similarly:
#-----------------------------------------------------------------------------
# add_library(my_library src/my_library.cc)
# target_link_libraries(my_library ${ros_libs} ${TORCH_LIBRARIES} ...)
#
# rosbuild_add_gtest(test_something tests/test_something.cpp)
# target_link_libraries(test_something my_library)

#-----------------------------------------------------------------------------
# Output paths, if desired:
#-----------------------------------------------------------------------------
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH    ${PROJECT_SOURCE_DIR}/lib)

# If you want to confirm the final flags/paths:
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "LibTorch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "OpenCV includes: ${OpenCV_INCLUDE_DIRS}")

# Done!
