PROJECT(lsmap_realtime LANGUAGES CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

MESSAGE(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

SET(CMAKE_CUDA_FLAGS "-std=c++17")
SET(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wno-unknown-pragmas -Werror")
SET(CMAKE CXX_STANDARD_REQUIRED True)

IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
  MESSAGE(STATUS "Additional Flags for Release mode")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -O2 -DNDEBUG")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  MESSAGE(STATUS "Additional Flags for Debug mode")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")
ENDIF()

# Set the path to libtorch
list(APPEND CMAKE_PREFIX_PATH "/usr/local/libtorch")

message(STATUS "Compiling with ROS2 COLCON")
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(PCL 1.10 REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Link directories
link_directories(${PCL_LIBRARY_DIRS})

# Add executables
add_executable(lsmap_node
  src/lsmap_node_main.cc
  src/lsmap_node.cc
)

add_executable(publisher_node
  src/publisher_node_main.cc
)

# Link libraries
ament_target_dependencies(lsmap_node
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  grid_map_msgs
  grid_map_ros
)

ament_target_dependencies(publisher_node
  rclcpp
  geometry_msgs
  sensor_msgs
  cv_bridge
  OpenCV
  Eigen3
)

target_link_libraries(lsmap_node
  ${PCL_LIBRARIES}
  ${TORCH_LIBRARIES}
)

target_link_libraries(publisher_node
  ${PCL_LIBRARIES}
)

# Install targets
install(TARGETS lsmap_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS publisher_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install include directories
install(DIRECTORY src/
  DESTINATION src/${PROJECT_NAME}
)

ament_target_dependencies(lsmap_node
  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
  PCL
  OpenCV
)

ament_target_dependencies(publisher_node
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  pcl_conversions
  PCL
  tf2_ros
  OpenCV
)

# Install launch files, if any
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()