#include "lsmap_node.h"

namespace lsmap {

void LSMapNode::pointcloud_callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message");
  cloud_queue_.push(msg);

  // Convert ROS2 PointCloud2 message to PCL point cloud
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  pcl::fromROSMsg(*msg, pcl_cloud);

  // Process the PCL point cloud (example: just log the size)
  RCLCPP_INFO(this->get_logger(), "PointCloud size: %lu", pcl_cloud.size());
}

void LSMapNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  RCLCPP_INFO(this->get_logger(), "Received Image message");
  image_queue_.push(msg);

  // // Convert ROS2 Image to OpenCV image
  // cv_bridge::CvImagePtr cv_ptr;
  // try {
  //     cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
  // } catch (cv_bridge::Exception& e) {
  //     RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  //     return;
  // }

  // // Process the OpenCV image
  // cv::Mat image = cv_ptr->image;

  // // For example, convert to grayscale
  // cv::Mat gray_image;
  // cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

  // // Convert OpenCV image back to ROS2 Image and publish
  // auto output_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8",
  // gray_image).toImageMsg(); image_publisher_->publish(*output_msg);
}

void LSMapNode::p2p_callback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received pixel to point message");
  pixel_to_point_ = *msg;
}

void LSMapNode::camera_info_callback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received CameraInfo message");
  camera_info_ = *msg;
}

void LSMapNode::save_depth_image(const cv::Mat& depthMatrix,
                                 const std::string& filename) {
  if (depthMatrix.empty()) {
    std::cerr << "Invalid input matrix." << std::endl;
    return;
  }

  std::cout << "depth image size: " << depthMatrix.size() << std::endl;
  cv::Mat normDepthImage = cv::Mat::zeros(depthMatrix.size(), CV_8UC1);
  double min_depth, max_depth;
  cv::minMaxLoc(depthMatrix, &min_depth, &max_depth);
  for (int i = 0; i < depthMatrix.rows; ++i) {
    for (int j = 0; j < depthMatrix.cols; ++j) {
      float depth = depthMatrix.at<float>(i, j);
      uchar normDepth = static_cast<uchar>(255 * (depth - min_depth) /
                                           (max_depth - min_depth));

      normDepthImage.at<uchar>(i, j) = normDepth;
    }
  }

  if (!cv::imwrite(filename, normDepthImage)) {
    std::cerr << "Failed to save depth image." << std::endl;
  } else {
    std::cout << "Depth image saved to " << filename << std::endl;
  }
}

std::tuple<torch::Tensor, torch::Tensor> LSMapNode::projection(
    sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,
    sensor_msgs::msg::Image::SharedPtr image_msg) {
  // Convert ROS2 PointCloud2 message to PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

  // Convert ROS2 Image to OpenCV image
  cv::Mat bgr_image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;
  Eigen::Matrix<float, 3, 4, Eigen::RowMajor> pt2pixel;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      pt2pixel(i, j) = camera_info_.p[i * 4 + j];
    }
  }
  // Convert bgr to rgb
  cv::Mat rgb_image;
  cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

  // Save RGB iamge
  // cv::imwrite("rgb_image.png", rgb_image);
  rgb_image.convertTo(rgb_image, CV_32FC3, 1.0 / 255.0);
  cv::Mat depth_image(rgb_image.rows, rgb_image.cols, CV_32FC1,
                      std::numeric_limits<float>::min());

  size_t num_points = pcl_cloud->points.size();
  Eigen::MatrixXf points(4, num_points);
  for (size_t i = 0; i < num_points; ++i) {
    points(0, i) = pcl_cloud->points[i].x;
    points(1, i) = pcl_cloud->points[i].y;
    points(2, i) = pcl_cloud->points[i].z;
    points(3, i) = 1;
  }

  // Project point cloud to image space
  Eigen::MatrixXf pixels = pt2pixel * points;
  for (size_t i = 0; i < num_points; ++i) {
    float u = pixels(0, i) / pixels(2, i);
    float v = pixels(1, i) / pixels(2, i);
    float z = pixels(2, i);

    if (u >= 0 && u < rgb_image.cols && v >= 0 && v < rgb_image.rows &&
        z > 0.0) {
      size_t pixel_x = static_cast<size_t>(u);
      size_t pixel_y = static_cast<size_t>(v);
      depth_image.at<float>(pixel_y, pixel_x) =
          std::max(depth_image.at<float>(pixel_y, pixel_x), z);
    }
  }
  depth_image.setTo(0.0, depth_image == std::numeric_limits<float>::min());
  depth_image *= 1000.0;  // Convert depth to mm

  // Downsample image by 2 for faster inference
  cv::resize(rgb_image, rgb_image, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
  cv::resize(depth_image, depth_image, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);

  // // Print max and min depth values
  // double min_depth, max_depth;
  // cv::minMaxLoc(depth_image, &min_depth, &max_depth);
  // RCLCPP_INFO(this->get_logger(), "Min depth: %lf, Max depth: %lf",
  // min_depth,
  //             max_depth);

  // save_depth_image(depth_image, "depth_image.png");
  // Save raw depth image
  // cv::imwrite("depth_image.png", depth_image);
  // cv::Mat depth_image_uint16 = cv::Mat(depth_image.size(), CV_16UC1);
  // depth_image.convertTo(depth_image_uint16, CV_16UC1);
  // cv::imwrite("depth_image.png", depth_image_uint16);

  // Create RGBD input image for jit trace
  cv::Mat rgbd_image;
  std::vector<cv::Mat> channels;
  cv::split(rgb_image, channels);
  channels.push_back(depth_image);
  cv::merge(channels, rgbd_image);
  torch::Tensor rgbd_tensor = torch::from_blob(
      rgbd_image.data, {rgbd_image.rows, rgbd_image.cols, 4}, torch::kFloat32);
  rgbd_tensor = rgbd_tensor.permute({2, 0, 1});  // [H, W, C] -> [C, H, W]
  rgbd_tensor = rgbd_tensor.unsqueeze(0).unsqueeze(
      0);  // [C, H, W] -> [B, #cams, C, H, W]
  rgbd_tensor = rgbd_tensor.to(torch::kCUDA);

  // Create point2pixel matrix for inference
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> pixel2point;
  // pixel2point << -0.0002, -0.0013, 1.0949, 0.0180,
  //                  -0.0104, 0.0002, 0.7840, 0.0795,
  //                  -0.0002, -0.0103, 0.5617, -0.1033,
  //                  -0.0000, -0.0000, 0.0000, 1.0000;
  // pixel2point << -0.00021055,     -0.00137477,      1.09806240, 0.01806181,
  //           -0.01046783,      0.00003336,      0.80283189,      0.07945612,
  //           -0.00000575,     -0.01037925,      0.55110741,     -0.10325236,
  //           -0.00000000,      0.00000000,     -0.00000000,      1.00000000;
  // print out pixel2point
  // tensor([[[[    -0.0002,     -0.0014,      1.0980,      0.0181], // gt seq 1
  //         [    -0.0104,     -0.0000,      0.8046,      0.0794],
  //         [     0.0000,     -0.0104,      0.5460,     -0.1033],
  //         [    -0.0000,      0.0000,     -0.0000,      1.0000]]]])

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      pixel2point(i, j) = pixel_to_point_.data[i * 4 + j];
      // RCLCPP_INFO(this->get_logger(), "pixel2point[%ld][%ld]: %0.6f", i, j,
      //             pixel2point(i, j));
    }
  }

  torch::Tensor pixel2point_tensor =
      torch::from_blob(pixel2point.data(), {4, 4}, torch::kFloat32).clone();
  pixel2point_tensor = pixel2point_tensor.unsqueeze(0).unsqueeze(
      0);  // [4, 4] -> [B, #cams, 4, 4]
  pixel2point_tensor = pixel2point_tensor.to(torch::kCUDA);

  // Create tuple for model input
  auto inputs = std::make_tuple(rgbd_tensor, pixel2point_tensor);
  return inputs;
}

void LSMapNode::tensorToGridMap(
    const std::unordered_map<std::string, torch::Tensor>& output,
    grid_map::GridMap& map) {
  // Iterate through all map tensors
  bool is_map_set = false;
  int grid_height = 0;
  int grid_width = 0;
  for (const auto& item : output) {
    // Extract key as a string and value as a tensor
    std::string field = item.first;
    torch::Tensor tensor = item.second;

    if (!map.exists(field))  // Skip if the field does not exist in the map
      continue;

    if (!is_map_set) {
      // Get tensor dimensions
      float grid_res = 0.1;
      float map_height = tensor.size(2) * grid_res;
      float map_width = tensor.size(3) * grid_res;
      map.setGeometry(grid_map::Length(map_width, map_height), grid_res,
                      grid_map::Position(0, 0));
      grid_height = map.getSize().y();
      grid_width = map.getSize().x();
      is_map_set = true;
    }

    torch::Tensor tensor_cpu = tensor.to(torch::kCPU).to(torch::kFloat32);
    const float* tensor_data = tensor_cpu.data_ptr<float>();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_height; ++i) {
      for (int j = 0; j < grid_width; ++j) {
        float grid_val = tensor_data[i * grid_width + j];
        if (!this->fov_mask_[i][j]) {
          grid_val = 0.0;
        }

        map.at(field, grid_map::Index(i, j)) = grid_val;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor> LSMapNode::computePCA(
    const at::Tensor& input_tensor, int components) {
  // Ensure the tensor is on CPU and of type float
  at::Tensor cpu_tensor = input_tensor.to(at::kCPU).to(torch::kFloat32);

  // print tensor dim
  for (int i = 0; i < cpu_tensor.dim(); i++) {
    RCLCPP_INFO(this->get_logger(), "Tensor dim %d: %ld", i,
                cpu_tensor.size(i));
  }

  // Permute and flatten for PCA
  // [B, C, H, W] -> [B, H*W, C]
  auto flattened =
      cpu_tensor.permute({0, 2, 3, 1}).reshape({-1, cpu_tensor.size(1)});

  // Compute the mean of each feature
  auto mean = flattened.mean(0, /*keepdim=*/true);

  // Center the data
  auto centered = flattened - mean;

  // Compute the covariance matrix
  auto covariance_matrix =
      at::mm(centered.t(), centered) / (flattened.size(0) - 1);

  // Perform eigen decomposition
  auto eigen = torch::linalg::eigh(covariance_matrix, "U");
  auto eigenvalues = std::get<0>(eigen);
  auto eigenvectors = std::get<1>(eigen);

  // Select the top 'components' eigenvectors
  auto pca_matrix = eigenvectors.narrow(1, 0, components);

  // Project the data onto the new space
  auto reduced_data = at::mm(centered, pca_matrix);

  return std::make_tuple(reduced_data, pca_matrix);
}

void LSMapNode::run() {
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
  sensor_msgs::msg::Image::SharedPtr image_msg;
  // 1 - Check if there are images and point clouds to process
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (cloud_queue_.empty() || image_queue_.empty()) {
      return;
    }
    auto cloud_time = cloud_queue_.front()->header.stamp;
    auto image_time = image_queue_.front()->header.stamp;
    // Convert to nanoseconds and check if the timestamps are within 100
    // milliseconds
    int64_t cloud_time_ns = cloud_time.sec * 1e9 + cloud_time.nanosec;
    int64_t image_time_ns = image_time.sec * 1e9 + image_time.nanosec;
    if (std::abs(cloud_time_ns - image_time_ns) < 100 * 1e6) {
      cloud_msg = cloud_queue_.front();
      cloud_queue_.pop();
      image_msg = image_queue_.front();
      image_queue_.pop();
    } else {
      // Drop the older message
      if (cloud_time_ns < image_time_ns) {
        cloud_queue_.pop();
      } else {
        image_queue_.pop();
      }
    }
  }
  if (!cloud_msg || !image_msg) {
    return;
  }

  auto start = std::chrono::high_resolution_clock::now();
  // 2 - Project point clouds to image space
  RCLCPP_INFO(this->get_logger(), "Projecting point cloud to image space");
  auto inputs = projection(cloud_msg, image_msg);

  /* BEGIN TEST LOADING */
  // // Load image and p2p tensors from a torch .pt file
  // std::string fp = "/lift-splat-map-realtime/data_dict.pt";
  // // Step 1: Load the data dictionary
  // std::map<std::string, torch::Tensor> data_dict;

  // torch::Tensor image_tensor;
  // torch::Tensor p2p_tensor;
  // try {
  //   // Read the file content into a vector of char
  //   std::ifstream file_stream(fp, std::ios::binary);
  //   std::vector<char> buffer((std::istreambuf_iterator<char>(file_stream)),
  //                            std::istreambuf_iterator<char>());

  //   // Load the data using pickle_load
  //   torch::IValue loaded_data = torch::pickle_load(buffer);

  //   // Convert the IValue to a dictionary
  //   auto dict = loaded_data.toGenericDict();

  //   // Access tensors from the dictionary
  //   if (dict.contains("rgbd") && dict.contains("p2p")) {
  //     image_tensor = dict.at("rgbd").toTensor();
  //     p2p_tensor = dict.at("p2p").toTensor();

  //     // Optionally, move tensors to GPU
  //     if (torch::cuda::is_available()) {
  //       image_tensor = image_tensor.to(torch::kCUDA);
  //       p2p_tensor = p2p_tensor.to(torch::kCUDA);
  //     }
  //     for (int64_t i = 0; i < p2p_tensor.size(2); i++) {
  //       for (int64_t j = 0; j < p2p_tensor.size(3); j++) {
  //         RCLCPP_INFO(this->get_logger(), "p2p[%ld][%ld]: %0.6f", i, j,
  //                     p2p_tensor[0][0][i][j].item<float>());
  //       }
  //     }

  //     // Print tensor sizes to verify
  //     std::cout << "Image Tensor Size: " << image_tensor.sizes() <<
  //     std::endl; std::cout << "P2P Tensor Size: " << p2p_tensor.sizes() <<
  //     std::endl;
  //   } else {
  //     std::cerr << "Tensors 'rgbd' or 'p2p' not found in the dictionary."
  //               << std::endl;
  //   }
  // } catch (const c10::Error& e) {
  //   std::cerr << "Error loading the PyTorch file: " << e.what() << std::endl;
  //   return;
  // }

  // // Step 4: Create a tuple from the tensors
  // // auto inputs = std::make_tuple(image_tensor, p2p_tensor);
  // std::tuple<torch::Tensor, torch::Tensor> inputs(image_tensor,
  //                                                 std::get<1>(test_inputs));
  /* END TEST LOADING */

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> inference_time = end - start;
  RCLCPP_INFO(this->get_logger(), "Projection time: %f seconds",
              inference_time.count());

  // 3 - Perform model inference
  start = std::chrono::high_resolution_clock::now();
  auto output = model_.forward(inputs);
  end = std::chrono::high_resolution_clock::now();
  inference_time = end - start;
  RCLCPP_INFO(this->get_logger(), "Model Inference time: %f seconds",
              inference_time.count());

  // 4 - Process elevation and semantic predictions
  start = std::chrono::high_resolution_clock::now();
  std::unordered_map<std::string, torch::Tensor> tensor_map;
  tensor_map["elevation"] = output.at("elevation_preds");
  tensor_map["traversability"] = output.at("traversability_preds_full");

  // Save traversability tensor
  auto pickled = torch::pickle_save(tensor_map["traversability"]);
  std::ofstream fout("traversability.pt", std::ios::out | std::ios::binary);
  fout.write(pickled.data(), pickled.size());
  fout.close();

  // Create the GridMap with the "elevation" layer
  grid_map::GridMap map({"elevation", "traversability"});
  map.setFrameId("os_sensor");
  map.setTimestamp(this->now().nanoseconds());

  // // semantic
  // auto [pca_result, principal_components] = computePCA(semantic, 3);
  // pca_result = (pca_result - pca_result.min()) / (pca_result.max() -
  //       pca_result.min());
  // auto rgb_tensor = pca_result.reshape({semantic.size(0), 3,
  // semantic.size(2), semantic.size(3)}); // [B, 3, H, W]
  // // Convert tensor to rgb image
  // cv::Mat rgb_mat(semantic.size(2), semantic.size(3), CV_32FC3,
  // rgb_tensor.data_ptr<float>());

  // Convert tensor data to GridMap elevation
  tensorToGridMap(tensor_map, map);

  // Ensure the map contains valid elevation data
  if (map.exists("elevation")) {
    // Convert elevation data to a ROS2 GridMap message
    auto grid_map_msg_ptr = grid_map::GridMapRosConverter::toMessage(map);

    // Publish the grid map message
    grid_map_publisher_->publish(*grid_map_msg_ptr);
  } else {
    RCLCPP_WARN(this->get_logger(),
                "Elevation layer not found in the GridMap.");
  }

  // Stop timing the process and log the processing time
  end = std::chrono::high_resolution_clock::now();
  inference_time = end - start;
  RCLCPP_INFO(this->get_logger(), "Map Processing time: %f seconds",
              inference_time.count());
}

}  // namespace lsmap