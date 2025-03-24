#include "utils.h"

#include <stdio.h>

#ifdef ROS1
using Publisher = ros::Publisher;
using ImageMsg = sensor_msgs::Image;
using Header = std_msgs::Header;
#else
using Publisher = rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr;
using ImageMsg = sensor_msgs::msg::Image;
using Header = std_msgs::msg::Header;
#endif
namespace creste {

torch::Tensor UpsampleDepthImage(int target_height, int target_width,
                                 const torch::Tensor& depth_image) {
  // Check input shape [1, H, W]
  if (depth_image.dim() != 3 || depth_image.size(0) != 1) {
    throw std::invalid_argument(
        "UpsampleDepthImage: depth_image must be [1,H,W].");
  }
  std::vector<int64_t> size = {target_height, target_width};

  auto upsampled = torch::nn::functional::interpolate(
      depth_image.unsqueeze(0),
      torch::nn::functional::InterpolateFuncOptions().size(size).mode(
          torch::kNearest));

  return upsampled.squeeze(0);
}

void TensorToVec2D(const torch::Tensor& tensor,
                   std::vector<std::vector<float>>& vec) {
  torch::Tensor tensor_cpu = tensor.to(torch::kCPU).squeeze();
  int rows = tensor_cpu.size(0);
  int cols = tensor_cpu.size(1);

  vec.resize(rows, std::vector<float>(cols));
  auto accessor = tensor_cpu.accessor<float, 2>();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      vec[i][j] = accessor[i][j];
    }
  }
}

cv::Mat TensorToMat(const torch::Tensor& tensor) {
  torch::Tensor tensor_cpu = tensor.to(torch::kCPU).squeeze();
  int rows = tensor_cpu.size(0);
  int cols = tensor_cpu.size(1);

  if (tensor_cpu.dim() != 2) {
    throw std::invalid_argument("TensorToMat: Tensor must be 2D");
  }

  // Create a CV_32FC1 matrix
  cv::Mat mat(rows, cols, CV_32FC1);
  auto accessor = tensor_cpu.accessor<float, 2>();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      mat.at<float>(i, j) = accessor[i][j];
    }
  }

  // scale to [0,255]
  double min_val, max_val;
  cv::minMaxLoc(mat, &min_val, &max_val);

  cv::Mat scaled_8u;
  if (min_val == max_val) {
    mat.convertTo(scaled_8u, CV_8UC1, 1.0, 0.0);
  } else {
    mat.convertTo(scaled_8u, CV_8UC1, 255.0 / (max_val - min_val),
                  -255.0 * min_val / (max_val - min_val));
  }

  return scaled_8u;
}

std::vector<float> linspace(float start, float end, int num) {
  std::vector<float> result;
  if (num == 1) {
    result.push_back(start);
    return result;
  }
  float step = (end - start) / (num - 1);
  for (int i = 0; i < num; ++i) {
    result.push_back(start + i * step);
  }
  return result;
}

torch::Tensor computePCA(const torch::Tensor& features) {
  // shape [B, F, H, W] => reorder to [F, B*H*W]
  auto sizes = features.sizes();
  int64_t B = sizes[0], F = sizes[1], H = sizes[2], W = sizes[3];
  std::cout << "PCA: B=" << B << ", F=" << F << ", H=" << H << ", W=" << W
            << std::endl;

  auto X = features.permute({1, 0, 2, 3}).reshape({F, B * H * W});
  auto mean = X.mean(1, /*keepdim=*/true);
  X = X - mean;  // center

  auto cov = torch::mm(X, X.t()) / static_cast<float>(H * W - 1);
  auto eigen = torch::linalg_eigh(cov);
  auto eigenvalues = std::get<0>(eigen);
  auto eigenvectors = std::get<1>(eigen);

  // top 3
  auto top_eigenvectors = eigenvectors.index(
      {torch::indexing::Slice(), torch::indexing::Slice(F - 3, F)});
  auto projected = torch::mm(top_eigenvectors.t(), X);
  auto reduced = projected.view({3, B, H, W}).permute({1, 0, 2, 3});
  return reduced;
}

void saveElevationImage(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key) {
  if (output.count(key) <= 0) return;
  auto elevation_tensor = output.at(key).to(torch::kCPU).to(torch::kFloat32);
  if (elevation_tensor.dim() < 4) {
    LOG_WARN("Elevation shape <4 dims, skipping.");
    return;
  }
  int b = elevation_tensor.size(0);
  int c = elevation_tensor.size(1);
  int h = elevation_tensor.size(2);
  int w = elevation_tensor.size(3);
  if (b != 1 || c < 1) {
    LOG_WARN("Elevation shape mismatch. Skipping.");
    return;
  }

  // We'll just take channel 0
  auto first_channel = elevation_tensor.index({0, 0});
  // Turn to cv::Mat
  const float* data_ptr = first_channel.data_ptr<float>();
  cv::Mat mat(h, w, CV_32FC1, (void*)data_ptr);

  double min_val, max_val;
  cv::minMaxLoc(mat, &min_val, &max_val);

  cv::Mat scaled;
  if (min_val == max_val) {
    mat.convertTo(scaled, CV_8UC1, 1.0, 0.0);
  } else {
    mat.convertTo(scaled, CV_8UC1, 255.0 / (max_val - min_val),
                  -255.0 * min_val / (max_val - min_val));
  }
  cv::Mat color_map;
  cv::applyColorMap(scaled, color_map, cv::COLORMAP_JET);

  std::string filename = key + ".png";
  if (!cv::imwrite(filename, color_map)) {
    std::cout << "Failed to save " << filename;
  } else {
    std::cout << "Saved elevation image to " << filename;
  }
}

// -----------------------------------------------------------------------------
// The next two functions publish an image to a given Publisher
// -----------------------------------------------------------------------------
void PublishTraversability(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key, Publisher traversability_pub) {
  if (output.count(key) == 0) {
    LOG_WARN("Key %s not found, skipping PublishTraversability.", key.c_str());
    return;
  }
  auto trav_tensor = output.at(key).to(torch::kCPU).to(torch::kFloat32);
  if (trav_tensor.dim() < 4) {
    LOG_WARN("Traversability dim < 4, skipping.");
    return;
  }
  int b = trav_tensor.size(0);
  int c = trav_tensor.size(1);
  int h = trav_tensor.size(2);
  int w = trav_tensor.size(3);
  if (b != 1 || c != 1) {
    LOG_WARN("Traversability shape mismatch: %d,%d,%d,%d", b, c, h, w);
    return;
  }

  const float* ptr = trav_tensor.data_ptr<float>();
  cv::Mat trav_mat(h, w, CV_32FC1, (void*)ptr);

  double minv, maxv;
  cv::minMaxLoc(trav_mat, &minv, &maxv);

  cv::Mat scaled_8u;
  if (minv == maxv) {
    trav_mat.convertTo(scaled_8u, CV_8UC1, 1.0, 0.0);
  } else {
    trav_mat.convertTo(scaled_8u, CV_8UC1, 255.0 / (maxv - minv),
                       -255.0 * minv / (maxv - minv));
  }

  // Publish as mono8
  auto msg = cv_bridge::CvImage(Header(), "mono8", scaled_8u).toImageMsg();
  msg->header.stamp = GET_TIME;
  msg->header.frame_id = "bev";
#ifdef ROS1
  traversability_pub.publish(msg);
#else
  traversability_pub->publish(*msg);
#endif
  LOG_INFO("Published traversability image (ROS2).");
}

void PublishCompletedDepth(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key, Publisher depth_pub
  ) {
  if (!output.count(key)) {
    LOG_WARN("Key %s not found. Skipping PublishCompletedDepth.", key.c_str());
    return;
  }
  auto depth_tensor = output.at(key).to(torch::kCPU).to(torch::kFloat32);
  if (depth_tensor.dim() < 3) {
    LOG_WARN("Depth tensor has dim=%ld", depth_tensor.dim());
    return;
  }
  int b = depth_tensor.size(0);
  int h = depth_tensor.size(1);
  int w = depth_tensor.size(2);
  if (b != 1) {
    LOG_WARN("Depth shaep mismatch [b=%d]", b);
    return;
  }

  const float* data = depth_tensor.data_ptr<float>();
  cv::Mat depth_mat(h, w, CV_32FC1, (void*)data);

  double minv, maxv;
  cv::minMaxLoc(depth_mat, &minv, &maxv);

  cv::Mat depth_8u;
  if (minv == maxv) {
    depth_mat.convertTo(depth_8u, CV_8UC1, 1.0, 0.0);
  } else {
    depth_mat.convertTo(depth_8u, CV_8UC1, 255.0 / (maxv - minv),
                        -255.0 * minv / (maxv - minv));
  }

  auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", depth_8u)
                 .toImageMsg();
  msg->header.stamp = GET_TIME;
  msg->header.frame_id = "left_optical";

#ifdef ROS1
  depth_pub.publish(*msg);
#else
  depth_pub->publish(*msg);
#endif
  LOG_INFO("Published normalized depth image (ROS2).");
}

}  // namespace creste
