#ifndef UTILS_H
#define UTILS_H

#include <torch/torch.h>

#include <Eigen/Dense>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------------
// Conditional includes for ROS1 vs ROS2, plus type aliases for
// logging/publisher
// -----------------------------------------------------------------------------
#ifdef ROS1
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

// For logging macros (LOG_INFO, etc.), you can define them in the .cc or here:
#define LOG_INFO(...) ROS_INFO(__VA_ARGS__)
#define LOG_WARN(...) ROS_WARN(__VA_ARGS__)
#define LOG_ERROR(...) ROS_ERROR(__VA_ARGS__)
#define GET_TIME ros::Time::now()
inline int64_t to_nanoseconds(const ros::Time& stamp) {
  return static_cast<int64_t>(stamp.sec) * 1000000000LL +
         static_cast<int64_t>(stamp.nsec);
}

#else
#include <cv_bridge/cv_bridge.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

// Minimal logging macros as an example
#include <iostream>
#define LOG_INFO(...) \
  RCLCPP_INFO(rclcpp::get_logger("creste_node"), __VA_ARGS__)
#define LOG_WARN(...) \
  RCLCPP_WARN(rclcpp::get_logger("creste_node"), __VA_ARGS__)
#define LOG_ERROR(...) \
  RCLCPP_ERROR(rclcpp::get_logger("creste_node"), __VA_ARGS__)
#define GET_TIME rclcpp::Clock().now()
#include <builtin_interfaces/msg/time.hpp>
inline int64_t to_nanoseconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<int64_t>(stamp.sec) * 1000000000LL +
         static_cast<int64_t>(stamp.nanosec);
}

#endif
// -----------------------------------------------------------------------------

namespace creste {

// -----------------------------------------------------------------------------
// Structures
// -----------------------------------------------------------------------------
struct CalibInfo {
  int rows;
  int cols;
  std::vector<float> data;
};

struct PlannerParams {
  float max_v;
  float max_w;
  float max_dv;
  float max_dw;
  int partitions;
  float dt;
  int max_iters;
  float max_time;
  float goal_tolerance;
  float cost_weight;
};

struct MapParams {
  float height;      // cells
  float width;       // cells
  float resolution;  // meters/cell
  float origin_x;    // cells
  float origin_y;    // cells
  int map_height;    // pixels
  int map_width;     // pixels
};

struct Pose2D {
  float x;
  float y;
  float theta;

  Pose2D() : x(0.0f), y(0.0f), theta(0.0f) {}
  Pose2D(float x, float y, float theta) : x(x), y(y), theta(theta) {}
};

// -----------------------------------------------------------------------------
// Function Declarations
// -----------------------------------------------------------------------------
torch::Tensor UpsampleDepthImage(int target_height, int target_width,
                                 const torch::Tensor& depth_image);

void TensorToVec2D(const torch::Tensor& tensor,
                   std::vector<std::vector<float>>& vec);

cv::Mat TensorToMat(const torch::Tensor& tensor);

inline torch::Tensor createTrapezoidalFovMask(int H, int W,
                                              float fovTopAngle = 110,
                                              float fovBottomAngle = 110,
                                              float near = 5, float far = 200) {
  // Implementation is the same as before
  torch::Tensor mask_tensor = torch::zeros({H, W}, torch::kBool);

  float centerX = W / 2.0;
  float centerY = H / 2.0;

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float dx = x - centerX;
      float dy = centerY - y;
      float distance = std::sqrt(dx * dx + dy * dy);
      float angle = std::atan2(dx, dy) * 180.0 / M_PI;
      if (angle < -180.0) angle += 360.0;

      float angularSpread;
      if (distance <= near) {
        angularSpread = fovTopAngle / 2.0;
      } else if (distance >= far) {
        angularSpread = fovBottomAngle / 2.0;
      } else {
        float t = (distance - near) / (far - near);
        angularSpread =
            (1 - t) * (fovTopAngle / 2.0) + t * (fovBottomAngle / 2.0);
      }

      if (distance >= near && distance <= far &&
          std::abs(angle) <= angularSpread) {
        mask_tensor[y][x] = true;
      }
    }
  }
  return mask_tensor;
}

std::vector<float> linspace(float start, float end, int num);

torch::Tensor computePCA(const torch::Tensor& features);

void saveElevationImage(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key);

// Publishing function for "traversability" images
void PublishTraversability(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key,
#if ROS1
    Publisher& traversability_pub
#else
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr traversability_pub
#endif
);

// Publishing function for "depth" images
void PublishCompletedDepth(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key,
#ifdef ROS1
    Publisher& depth_pub
#else
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub
#endif
);
}  // namespace creste

#endif  // UTILS_H