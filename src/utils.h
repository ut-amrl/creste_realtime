#ifndef UTILS_H
#define UTILS_H
#include <ros/ros.h>
#include <torch/torch.h>  // <-- or #include <ATen/ATen.h>

#include <Eigen/Dense>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace creste {

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

torch::Tensor UpsampleDepthImage(int target_height, int target_width,
                                 const torch::Tensor& depth_image);

void TensorToVec2D(const torch::Tensor& tensor,
                   std::vector<std::vector<float>>& vec);

cv::Mat TensorToMat(const torch::Tensor& tensor);

inline torch::Tensor createTrapezoidalFovMask(int H, int W,
                                              float fovTopAngle = 80,
                                              float fovBottomAngle = 80,
                                              float near = 0, float far = 200) {
  // Initialize the mask
  torch::Tensor mask_tensor = torch::zeros({H, W}, torch::kBool);  // False

  // Center coordinates
  float centerX = W / 2.0;
  float centerY = H / 2.0;

  // Loop through each pixel in the grid
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      // Calculate distance and angle
      float dx = x - centerX;
      float dy = centerY - y;
      float distance = std::sqrt(dx * dx + dy * dy);
      float angle = std::atan2(dx, dy) * 180.0 / M_PI;

      // Adjust angles to be in the range [-180, 180]
      if (angle < -180.0) {
        angle += 360.0;
      }

      // Determine angular spread
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

      // Create the mask
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

// void saveSemanticImage(
//     const std::unordered_map<std::string, torch::Tensor>& output,
//     const std::string& key);

void PublishTraversability(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key, ros::Publisher& depth_pub);

void PublishCompletedDepth(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key, ros::Publisher& depth_pub);

}  // namespace creste

#endif  // UTILS_H