#include "utils.h"

#include <cv_bridge/cv_bridge.h>

#include "sensor_msgs/Image.h"

using namespace torch::indexing;

namespace creste {

torch::Tensor UpsampleDepthImage(int target_height, int target_width, const torch::Tensor& depth_image) {
  // Check input tensor dimensions
  if (depth_image.dim() != 3 || depth_image.size(0) != 1) {
    throw std::invalid_argument("Input depth_image must be of shape [1, H, W].");
  }

  // Define the target output size
  std::vector<int64_t> size = {target_height, target_width};

  // Use nearest neighbor interpolation to upsample the depth image
  auto upsampled = torch::nn::functional::interpolate(
      depth_image.unsqueeze(0),
      torch::nn::functional::InterpolateFuncOptions()
          .size(size)
          .mode(torch::kNearest));

  return upsampled.squeeze(0);
}

void TensorToVec2D(const torch::Tensor& tensor,
                   std::vector<std::vector<float>>& vec) {
  // Ensure the tensor is on the CPU
  torch::Tensor tensor_cpu =
      tensor.to(torch::kCPU).squeeze();  // Remove the batch dimension

  // Get the number of rows and columns
  int rows = tensor_cpu.size(0);
  int cols = tensor_cpu.size(1);

  // Resize the vector
  vec.resize(rows, std::vector<float>(cols));

  // Copy the data
  auto accessor = tensor_cpu.accessor<float, 2>();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      vec[i][j] = accessor[i][j];
    }
  }
}

cv::Mat TensorToMat(const torch::Tensor& tensor) {
  // Ensure the tensor is on the CPU
  torch::Tensor tensor_cpu = tensor.to(torch::kCPU).squeeze();  // Remove the batch dimension if present

  // Get the number of rows and columns (assuming single-channel image)
  int rows = tensor_cpu.size(0);
  int cols = tensor_cpu.size(1);

  // Ensure the tensor has the correct shape (single channel, 2D tensor)
  if (tensor_cpu.dim() != 2) {
    throw std::invalid_argument("Tensor must be a 2D tensor.");
  }

  // Create a single-channel 1 byte int matrix
  auto mat = cv::Mat(rows, cols, CV_32FC1);  // CV_32FC1 corresponds to 32-bit float, single-channel

  // Copy the data from the tensor to the OpenCV mat
  auto accessor = tensor_cpu.accessor<float, 2>();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      mat.at<float>(i, j) = accessor[i][j];
    }
  }

  // Scale the data into [0, 255] range for visualization
  double min_val, max_val;
  cv::minMaxLoc(mat, &min_val, &max_val);

  cv::Mat scaled_8u;
  // We scale so that anything at min_val -> 0, max_val -> 255
  // If min_val == max_val, convertTo(...) will produce a uniform image.
  if (min_val == max_val) {
    // entire image is constant
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

// Utility to compute PCA reduction from [B,F,H,W] to [1,3,H,W]
torch::Tensor computePCA(const torch::Tensor& features) {
    // Ensure the input is on CPU and of type float for PCA computation.
    // auto features_cpu = features.to(torch::kFloat32).cpu();

    // features_cpu shape: [1, F, H, W]
    auto sizes = features.sizes();
    int64_t B = sizes[0], F = sizes[1], H = sizes[2], W = sizes[3];
    printf("B: %ld, F: %ld, H: %ld, W: %ld\n", B, F, H, W);

    // Reshape to [F, N] where N = H*W
    // auto X = features_cpu.permute(1, 0, 2, 3).view({F, B * H * W});
    auto X = features.permute({1, 0, 2, 3}).reshape({F, B * H * W});

    // Center the data (subtract the mean for each feature channel)
    auto mean = X.mean(1, /*keepdim=*/true);
    X = X - mean;

    // Compute covariance matrix of shape [F, F]
    auto cov = torch::mm(X, X.t()) / static_cast<float>(H * W - 1);

    // Compute eigenvalues and eigenvectors of the covariance matrix
    // Note: torch::linalg_eigh returns eigenvalues in ascending order
    auto eigen = torch::linalg_eigh(cov);
    auto eigenvalues = std::get<0>(eigen);
    auto eigenvectors = std::get<1>(eigen);

    // Select the top 3 eigenvectors corresponding to the largest eigenvalues
    // Since eigenvalues are in ascending order, the top eigenvectors are at the end
    auto top_eigenvectors = eigenvectors.index({Slice(), Slice(F - 3, F)}); // shape [F,3]

    // Project the original centered data onto the top eigenvectors
    auto projected = torch::mm(top_eigenvectors.t(), X); // shape [3, H*W]

    // Reshape result back to [3, B, H, W] -> [B, 3, H, W]
    auto reduced = projected.view({3, B, H, W}).permute({1, 0, 2, 3});
    
    return reduced;
}

// void CresteNode::SaveDepthImage(const cv::Mat& depthMatrix,
//                                const std::string& filename) {
//   if (depthMatrix.empty()) {
//     ROS_ERROR("Invalid input matrix for save_depth_image.");
//     return;
//   }

//   ROS_INFO("Saving depth image of size [%d x %d]", depthMatrix.rows,
//            depthMatrix.cols);
//   cv::Mat normDepthImage = cv::Mat::zeros(depthMatrix.size(), CV_8UC1);
//   double min_depth, max_depth;
//   cv::minMaxLoc(depthMatrix, &min_depth, &max_depth);
//   for (int i = 0; i < depthMatrix.rows; ++i) {
//     for (int j = 0; j < depthMatrix.cols; ++j) {
//       float depth = depthMatrix.at<float>(i, j);
//       uchar normDepth = static_cast<uchar>(255 * (depth - min_depth) /
//                                            (max_depth - min_depth));
//       normDepthImage.at<uchar>(i, j) = normDepth;
//     }
//   }

//   if (!cv::imwrite(filename, normDepthImage)) {
//     ROS_ERROR("Failed to save depth image to %s", filename.c_str());
//   } else {
//     ROS_INFO("Depth image saved to %s", filename.c_str());
//   }
// }

void saveElevationImage(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key) {
  if (output.count(key) <= 0) return;

  const auto& elevation_tensor =
      output.at("elevation");  // shape [1, 2, H, W] now

  // Move to CPU float
  auto elevation_cpu = elevation_tensor.to(torch::kCPU).to(torch::kFloat32);

  if (elevation_cpu.dim() < 4) {
    ROS_WARN(
        "Elevation tensor has unexpected shape. Skipping elevation "
        "colormap.");
  } else {
    int batch = elevation_cpu.size(0);  // should be 1
    int chan = elevation_cpu.size(1);   // should be 2
    int h = elevation_cpu.size(2);
    int w = elevation_cpu.size(3);

    // Expect [1, 2, H, W]
    if (batch != 1 || chan != 2) {
      ROS_WARN(
          "Elevation shape is [%d, %d, %d, %d]. Expected [1,2,H,W]. "
          "Adjust code or skip.",
          batch, chan, h, w);
    } else {
      // Slice out the first channel -> shape [1,1,H,W]
      auto first_channel =
          elevation_cpu.narrow(/*dim=*/1, /*start=*/0, /*length=*/1);

      // Access raw float data from that slice
      const float* elev_data = first_channel.data_ptr<float>();

      // Convert to CV_32FC1
      cv::Mat elevation_mat(h, w, CV_32FC1, (void*)elev_data);

      // Find min/max
      double min_val, max_val;
      cv::minMaxLoc(elevation_mat, &min_val, &max_val);

      // Scale to [0,255]
      cv::Mat scaled;
      elevation_mat.convertTo(scaled, CV_8UC1, 255.0 / (max_val - min_val),
                              -255.0 * min_val / (max_val - min_val));

      // Apply a colormap (e.g. JET)
      cv::Mat color_map;
      cv::applyColorMap(scaled, color_map, cv::COLORMAP_JET);

      // Save to disk
      std::string elev_filename = key + ".png";
      if (!cv::imwrite(elev_filename, color_map)) {
        ROS_ERROR("Failed to save %s", elev_filename.c_str());
      } else {
        ROS_INFO("Saved elevation colormap to %s", elev_filename.c_str());
      }
    }
  }
}

// void saveSemanticImage(
//     const std::unordered_map<std::string, torch::Tensor>& output,
//     const std::string& key) {
//   if (output.count(key) <= 0) return;

//   // 2) PCA decomposition of a feature map
//   // ------------------------------------------------
//   // Example: "traversability" or any other feature map
//   // (Assume it's shaped [1, C, H, W]. We do PCA -> top 3 dims -> visualize.)
//   const auto& feature_tensor = output.at(key);
//   // shape e.g. [1, C, H, W]

//   // We’ll reuse computePCA(...) to get top 3 principal components
//   int components = 3;
//   auto [pca_result, pca_matrix] = computePCA(feature_tensor, components);
//   // pca_result is shape [N, 3], where N = H*W in the typical flatten
//   // approach Reshape to [1, 3, H, W]

//   // You might need the original height & width from the feature_tensor
//   int batch = feature_tensor.size(0);
//   // int featC = feature_tensor.size(1);
//   int featH = feature_tensor.size(2);
//   int featW = feature_tensor.size(3);
//   // pca_result: shape [N, 3], N = featH*featW if batch=1

//   if (batch != 1) {
//     ROS_WARN("Feature map has batch=%d, skipping PCA image generation", batch);
//     return;
//   }

//   // reshape [N,3] -> [1, 3, H, W]
//   pca_result = pca_result.reshape({1, 3, featH, featW});

//   // Scale each channel to [0,1] (for color)
//   auto minv = pca_result.min().item<float>();
//   auto maxv = pca_result.max().item<float>();
//   auto pca_scaled = (pca_result - minv) / (maxv - minv);  // now in [0,1]

//   // Move to CPU float
//   pca_scaled = pca_scaled.to(at::kCPU).to(torch::kFloat32);

//   // Convert to CV_8UC3 (assuming the shape is [1, 3, H, W])
//   const float* pca_data = pca_scaled.data_ptr<float>();
//   // We'll build an interleaved CV_32FC3 first, then convert to CV_8UC3
//   cv::Mat pca_mat_32f(featH, featW, CV_32FC3);
//   // We need to reorder the channels because we have [3, H, W].
//   // We'll copy each pixel from pca_data => pca_mat_32f.
//   for (int y = 0; y < featH; ++y) {
//     for (int x = 0; x < featW; ++x) {
//       int idx = (y * featW + x);
//       float c0 = pca_data[0 * featH * featW + idx];  // channel 0
//       float c1 = pca_data[1 * featH * featW + idx];  // channel 1
//       float c2 = pca_data[2 * featH * featW + idx];  // channel 2
//       pca_mat_32f.at<cv::Vec3f>(y, x) = cv::Vec3f(c0, c1, c2);
//     }
//   }

//   // Convert [0,1] float -> [0,255] 8-bit
//   cv::Mat pca_mat_8u;
//   pca_mat_32f.convertTo(pca_mat_8u, CV_8UC3, 255.0);

//   // Save as an RGB image
//   std::string pca_filename = key + ".png";
//   if (!cv::imwrite(pca_filename, pca_mat_8u)) {
//     ROS_ERROR("Failed to save %s", pca_filename.c_str());
//   } else {
//     ROS_INFO("Saved PCA-based feature image to %s", pca_filename.c_str());
//   }
// }

void PublishTraversability(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key,
    ros::Publisher& traversability_pub)  // <-- new parameter for publishing
{
  // 1) Check existence of the specified key in 'output'
  if (output.count(key) == 0) {
    ROS_WARN("Key '%s' not found in output map. Skipping.", key.c_str());
    return;
  }

  // 2) Retrieve the tensor (assumed shape [1,1,H,W])
  const auto& traversability_tensor = output.at(key);

  // 3) Move to CPU and ensure float32
  auto trav_cpu = traversability_tensor.to(torch::kCPU).to(torch::kFloat32);

  // 4) Check dimensions
  if (trav_cpu.dim() < 4) {
    ROS_WARN("Traversability tensor has unexpected shape (dim=%ld). Skipping.",
             trav_cpu.dim());
    return;
  }

  int batch = trav_cpu.size(0);  // should be 1
  int chan = trav_cpu.size(1);   // should be 1
  int h = trav_cpu.size(2);
  int w = trav_cpu.size(3);

  if (batch != 1 || chan != 1) {
    ROS_WARN(
        "Traversability shape is [%d, %d, %d, %d], expected [1,1,H,W]. "
        "Skipping.",
        batch, chan, h, w);
    return;
  }

  // 5) Convert the single-channel data into a CV_32FC1 Mat
  const float* trav_data = trav_cpu.data_ptr<float>();
  cv::Mat trav_mat(h, w, CV_32FC1, (void*)trav_data);

  // 6) Scale the data into [0, 255] range for visualization
  double min_val, max_val;
  cv::minMaxLoc(trav_mat, &min_val, &max_val);

  cv::Mat scaled_8u;
  // We scale so that anything at min_val -> 0, max_val -> 255
  // If min_val == max_val, convertTo(...) will produce a uniform image.
  if (min_val == max_val) {
    // entire image is constant
    trav_mat.convertTo(scaled_8u, CV_8UC1, 1.0, 0.0);
  } else {
    trav_mat.convertTo(scaled_8u, CV_8UC1, 255.0 / (max_val - min_val),
                       -255.0 * min_val / (max_val - min_val));
  }

  // 7) Save to disk as grayscale
  // std::string out_filename = key + ".png";
  // if (!cv::imwrite(out_filename, scaled_8u)) {
  //   ROS_ERROR("Failed to save %s", out_filename.c_str());
  // } else {
  //   ROS_INFO("Saved traversability grayscale image to %s",
  //            out_filename.c_str());
  // }

  // 8) Publish the same image over ROS
  sensor_msgs::ImagePtr trav_msg =
      cv_bridge::CvImage(std_msgs::Header(), "mono8", scaled_8u).toImageMsg();
  trav_msg->header.stamp = ros::Time::now();  // or set an appropriate stamp
  trav_msg->header.frame_id = "bev";
  traversability_pub.publish(trav_msg);

  ROS_INFO("Published traversability image (mono8) on topic '%s'",
           traversability_pub.getTopic().c_str());
}

void PublishCompletedDepth(
    const std::unordered_map<std::string, torch::Tensor>& output,
    const std::string& key, ros::Publisher& depth_pub) {
  // 1) Check if the specified key exists
  if (output.count(key) == 0) {
    ROS_WARN("Key '%s' not found in output map. Skipping depth save.",
             key.c_str());
    return;
  }

  // 2) Retrieve the depth tensor
  const auto& depth_tensor = output.at(key);

  // 3) Move to CPU float
  auto depth_cpu = depth_tensor.to(torch::kCPU).to(torch::kFloat32);

  // 4) Verify shape (expecting [1,H,W], for example)
  if (depth_cpu.dim() < 3) {
    ROS_WARN("Depth tensor has dim=%ld, expected 3. Skipping.",
             depth_cpu.dim());
    return;
  }
  int batch = depth_cpu.size(0);
  int h = depth_cpu.size(1);
  int w = depth_cpu.size(2);
  if (batch != 1) {
    ROS_WARN("Depth shape is [%d, %d, %d], expected [1,H,W]. Skipping.", batch,
             h, w);
    return;
  }

  // 5) Create an OpenCV Mat (CV_32FC1) pointing to the tensor data
  const float* depth_data = depth_cpu.data_ptr<float>();
  cv::Mat depth_mat(h, w, CV_32FC1, (void*)depth_data);

  // 6) Find min/max for normalization
  double min_val, max_val;
  cv::minMaxLoc(depth_mat, &min_val, &max_val);

  // 7) Convert to [0,255] range (8-bit) for saving/publishing
  cv::Mat depth_8u;
  if (max_val == min_val) {
    // Avoid divide-by-zero if the entire image is constant
    depth_mat.convertTo(depth_8u, CV_8UC1, 1.0, 0.0);
  } else {
    depth_mat.convertTo(depth_8u, CV_8UC1, 255.0 / (max_val - min_val),
                        -255.0 * min_val / (max_val - min_val));
  }

  // 8) Save to disk
  // std::string filename = "completed_depth.png";
  // if (!cv::imwrite(filename, depth_8u)) {
  //   ROS_ERROR("Failed to save %s", filename.c_str());
  // } else {
  //   ROS_INFO("Saved normalized depth image to %s", filename.c_str());
  // }

  // 9) Publish the same image via depth_pub
  sensor_msgs::ImagePtr depth_msg =
      cv_bridge::CvImage(std_msgs::Header(), "mono8", depth_8u).toImageMsg();
  depth_msg->header.stamp = ros::Time::now();   // or use a relevant timestamp
  depth_msg->header.frame_id = "left_optical";  // adapt as needed
  depth_pub.publish(depth_msg);

  ROS_INFO("Published normalized depth image on topic '%s'",
           depth_pub.getTopic().c_str());
}

}  // namespace creste