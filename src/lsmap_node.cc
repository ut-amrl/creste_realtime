#include "lsmap_node.h"

namespace lsmap
{

LSMapNode::LSMapNode(const std::string& model_path)
  : nh_("~")
  , model_(model_path /* pass a logger if your LSMapModel expects it*/)
{
  ROS_INFO("LSMapNode initialized.");

  // Subscription to PointCloud2 topic
  pointcloud_subscriber_ = nh_.subscribe<sensor_msgs::PointCloud2>(
      "/ouster/points", 10, &LSMapNode::pointcloud_callback, this);

  // Subscription to Image topic
  image_subscriber_ = nh_.subscribe<sensor_msgs::Image>(
      "/stereo/left", 10, &LSMapNode::image_callback, this);

  // Subscription to CameraInfo
  camera_info_subscriber_ = nh_.subscribe<sensor_msgs::CameraInfo>(
      "/camera_info", 10, &LSMapNode::camera_info_callback, this);

  // Subscription to pixel-to-point data
  p2p_subscriber_ = nh_.subscribe<std_msgs::Float32MultiArray>(
      "/p2p", 10, &LSMapNode::p2p_callback, this);

  // Create fov mask (example: 256 x 256). Adjust as needed.
  fov_mask_ = createTrapezoidalFovMask(256, 256);

  // Publishers
  image_publisher_ = nh_.advertise<sensor_msgs::Image>("/lsmap/rgbd", 10);
  grid_map_publisher_ = nh_.advertise<grid_map_msgs::GridMap>("/lsmap/grid_map", 10);
}

void LSMapNode::pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);
  ROS_INFO("Received PointCloud2 message");
  cloud_queue_.push(msg);

  // Convert ROS PointCloud2 message to PCL point cloud
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  pcl::fromROSMsg(*msg, pcl_cloud);

  // Example processing
  ROS_INFO("PointCloud size: %lu", pcl_cloud.size());
}

void LSMapNode::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(queue_mutex_);
  ROS_INFO("Received Image message");
  image_queue_.push(msg);
}

void LSMapNode::p2p_callback(const std_msgs::Float32MultiArrayConstPtr& msg)
{
  ROS_INFO("Received pixel-to-point message");
  pixel_to_point_ = *msg;
}

void LSMapNode::camera_info_callback(const sensor_msgs::CameraInfoConstPtr& msg)
{
  ROS_INFO("Received CameraInfo message");
  camera_info_ = *msg;
}

void LSMapNode::save_depth_image(const cv::Mat& depthMatrix, const std::string& filename)
{
  if (depthMatrix.empty())
  {
    ROS_ERROR("Invalid input matrix for save_depth_image.");
    return;
  }

  ROS_INFO("Saving depth image of size [%d x %d]", depthMatrix.rows, depthMatrix.cols);
  cv::Mat normDepthImage = cv::Mat::zeros(depthMatrix.size(), CV_8UC1);
  double min_depth, max_depth;
  cv::minMaxLoc(depthMatrix, &min_depth, &max_depth);
  for (int i = 0; i < depthMatrix.rows; ++i)
  {
    for (int j = 0; j < depthMatrix.cols; ++j)
    {
      float depth = depthMatrix.at<float>(i, j);
      uchar normDepth = static_cast<uchar>(255 * (depth - min_depth) / (max_depth - min_depth));
      normDepthImage.at<uchar>(i, j) = normDepth;
    }
  }

  if (!cv::imwrite(filename, normDepthImage))
  {
    ROS_ERROR("Failed to save depth image to %s", filename.c_str());
  }
  else
  {
    ROS_INFO("Depth image saved to %s", filename.c_str());
  }
}

std::tuple<torch::Tensor, torch::Tensor> LSMapNode::projection(
    const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
    const sensor_msgs::ImageConstPtr& image_msg)
{
  // Convert ROS PointCloud2 message to PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

  // Convert ROS Image to OpenCV image (BGR)
  cv::Mat bgr_image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;

  // Extract P(3x4) from camera_info_.p
  Eigen::Matrix<float, 3, 4, Eigen::RowMajor> pt2pixel;
  for (size_t i = 0; i < 3; ++i)
  {
    for (size_t j = 0; j < 4; ++j)
    {
      pt2pixel(i, j) = camera_info_.p[i * 4 + j];
    }
  }

  // Convert BGR to RGB
  cv::Mat rgb_image;
  cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

  // Normalize to float in [0,1]
  rgb_image.convertTo(rgb_image, CV_32FC3, 1.0 / 255.0);

  // Initialize depth image with min float
  cv::Mat depth_image(rgb_image.rows, rgb_image.cols, CV_32FC1, std::numeric_limits<float>::lowest());

  size_t num_points = pcl_cloud->points.size();
  Eigen::MatrixXf points(4, num_points);
  for (size_t i = 0; i < num_points; ++i)
  {
    points(0, i) = pcl_cloud->points[i].x;
    points(1, i) = pcl_cloud->points[i].y;
    points(2, i) = pcl_cloud->points[i].z;
    points(3, i) = 1.0f;
  }

  // Project point cloud to image space
  Eigen::MatrixXf pixels = pt2pixel * points;
  for (size_t i = 0; i < num_points; ++i)
  {
    float u = pixels(0, i) / pixels(2, i);
    float v = pixels(1, i) / pixels(2, i);
    float z = pixels(2, i);

    if (u >= 0 && u < rgb_image.cols && v >= 0 && v < rgb_image.rows && z > 0.0)
    {
      size_t pixel_x = static_cast<size_t>(u);
      size_t pixel_y = static_cast<size_t>(v);
      float& depth_val = depth_image.at<float>(pixel_y, pixel_x);
      // Keep the furthest (max Z) if needed, or nearest (min Z).
      depth_val = std::max(depth_val, z);
    }
  }

  // Replace any "lowest()" with 0.0
  depth_image.setTo(0.0, depth_image == std::numeric_limits<float>::lowest());

  // Convert to millimeters
  depth_image *= 1000.0f;

  // Downsample by 2 for faster inference
  cv::resize(rgb_image, rgb_image, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
  cv::resize(depth_image, depth_image, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);

  // Merge RGB + Depth -> 4 channels
  cv::Mat rgbd_image;
  std::vector<cv::Mat> channels;
  cv::split(rgb_image, channels);  // [R, G, B]
  channels.push_back(depth_image); // add depth as 4th channel
  cv::merge(channels, rgbd_image); // [H, W, 4]

  // Convert to torch tensor: [1, 1, 4, H, W] on GPU (if available)
  torch::Tensor rgbd_tensor = torch::from_blob(
      rgbd_image.data, {rgbd_image.rows, rgbd_image.cols, 4}, torch::kFloat32);
  rgbd_tensor = rgbd_tensor.permute({2, 0, 1}); // [4, H, W]
  rgbd_tensor = rgbd_tensor.unsqueeze(0).unsqueeze(0); // [B, #cams, 4, H, W]

  if (torch::cuda::is_available())
  {
    rgbd_tensor = rgbd_tensor.to(torch::kCUDA);
  }

  // Create pixel2point matrix (4x4) from pixel_to_point_
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> pixel2point;
  for (size_t i = 0; i < 4; i++)
  {
    for (size_t j = 0; j < 4; j++)
    {
      pixel2point(i, j) = pixel_to_point_.data[i * 4 + j];
    }
  }

  torch::Tensor pixel2point_tensor =
      torch::from_blob(pixel2point.data(), {4, 4}, torch::kFloat32).clone();
  pixel2point_tensor = pixel2point_tensor.unsqueeze(0).unsqueeze(0); // [B, #cams, 4, 4]

  if (torch::cuda::is_available())
  {
    pixel2point_tensor = pixel2point_tensor.to(torch::kCUDA);
  }

  return std::make_tuple(rgbd_tensor, pixel2point_tensor);
}

// void LSMapNode::tensorToGridMap(const std::unordered_map<std::string, torch::Tensor>& output,
//                                 grid_map::GridMap& map)
// {
//   bool is_map_set = false;
//   int grid_height = 0;
//   int grid_width = 0;

//   for (const auto& item : output)
//   {
//     // Key (layer name) and value (tensor)
//     const std::string& field = item.first;
//     torch::Tensor tensor = item.second;

//     // If your grid_map already has these layers, you can check or create them
//     if (!map.exists(field))
//       continue;

//     if (!is_map_set)
//     {
//       // Set geometry from tensor size
//       float grid_res = 0.1f;
//       float map_height = tensor.size(2) * grid_res;
//       float map_width = tensor.size(3) * grid_res;
//       map.setGeometry(grid_map::Length(map_width, map_height),
//                       grid_res,
//                       grid_map::Position(0.0, 0.0));

//       grid_height = map.getSize().y();
//       grid_width = map.getSize().x();
//       is_map_set = true;
//     }

//     // Move tensor to CPU, float32
//     torch::Tensor tensor_cpu = tensor.to(torch::kCPU).to(torch::kFloat32);
//     const float* tensor_data = tensor_cpu.data_ptr<float>();

//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < grid_height; ++i)
//     {
//       for (int j = 0; j < grid_width; ++j)
//       {
//         float grid_val = tensor_data[i * grid_width + j];
//         // Optionally use your fov_mask_
//         if (!fov_mask_.empty() && i < (int)fov_mask_.size() && j < (int)fov_mask_[i].size())
//         {
//           if (!fov_mask_[i][j])
//             grid_val = 0.0f;
//         }
//         map.at(field, grid_map::Index(i, j)) = grid_val;
//       }
//     }
//   }
// }

std::tuple<at::Tensor, at::Tensor> LSMapNode::computePCA(const at::Tensor& input_tensor, int components)
{
  // Example PCA code. Logging replaced with ROS_INFO where needed.

  at::Tensor cpu_tensor = input_tensor.to(at::kCPU).to(torch::kFloat32);

  // [B, C, H, W] -> flatten
  auto flattened = cpu_tensor.permute({0, 2, 3, 1}).reshape({-1, cpu_tensor.size(1)});
  auto mean = flattened.mean(0, /*keepdim=*/true);
  auto centered = flattened - mean;
  auto covariance_matrix = at::mm(centered.t(), centered) / (flattened.size(0) - 1);

  auto eigen = torch::linalg::eigh(covariance_matrix, "U");
  auto eigenvalues = std::get<0>(eigen);
  auto eigenvectors = std::get<1>(eigen);

  // Top 'components'
  auto pca_matrix = eigenvectors.narrow(1, 0, components);
  auto reduced_data = at::mm(centered, pca_matrix);

  return std::make_tuple(reduced_data, pca_matrix);
}

void LSMapNode::run()
{
  sensor_msgs::PointCloud2ConstPtr cloud_msg;
  sensor_msgs::ImageConstPtr image_msg;

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (cloud_queue_.empty() || image_queue_.empty())
      return;

    ros::Time cloud_time = cloud_queue_.front()->header.stamp;
    ros::Time image_time = image_queue_.front()->header.stamp;

    // Convert to nanoseconds
    int64_t cloud_time_ns = static_cast<int64_t>(cloud_time.sec) * 1000000000LL + cloud_time.nsec;
    int64_t image_time_ns = static_cast<int64_t>(image_time.sec) * 1000000000LL + image_time.nsec;

    // Check if timestamps are within 100 ms
    if (std::abs(cloud_time_ns - image_time_ns) < 100LL * 1000000LL)
    {
      cloud_msg = cloud_queue_.front();
      cloud_queue_.pop();
      image_msg = image_queue_.front();
      image_queue_.pop();
    }
    else
    {
      // Drop the older message
      if (cloud_time_ns < image_time_ns)
        cloud_queue_.pop();
      else
        image_queue_.pop();
    }
  }

  if (!cloud_msg || !image_msg)
    return;

  // 1) Project
  ros::Time start = ros::Time::now();
  ROS_INFO("Projecting point cloud to image space...");
  auto inputs = projection(cloud_msg, image_msg);
  ros::Time end = ros::Time::now();
  ROS_INFO("Projection time: %f seconds", (end - start).toSec());

  // 2) Model Inference
  start = ros::Time::now();
  auto output = model_.forward(inputs);
  end = ros::Time::now();
  ROS_INFO("Model Inference time: %f seconds", (end - start).toSec());

  // 3) Process the resulting map(s)
  start = ros::Time::now();
  std::unordered_map<std::string, torch::Tensor> tensor_map;
  tensor_map["elevation"] = output.at("elevation_preds");
  tensor_map["traversability"] = output.at("traversability_preds_full");

  // Example: Save traversability to a file (optional)
  auto pickled = torch::pickle_save(tensor_map["traversability"]);
  std::ofstream fout("traversability.pt", std::ios::out | std::ios::binary);
  fout.write(pickled.data(), pickled.size());
  fout.close();

  // Create the GridMap
  // grid_map::GridMap map({"elevation", "traversability"});
  // map.setFrameId("os_sensor");
  // map.setTimestamp(ros::Time::now().toNSec());

  // // Populate from the model output
  // tensorToGridMap(tensor_map, map);

  // // Publish the grid map
  // if (map.exists("elevation"))
  // {
  //   grid_map_msgs::GridMap grid_map_msg;
  //   grid_map::GridMapRosConverter::toMessage(map, grid_map_msg);
  //   grid_map_publisher_.publish(grid_map_msg);
  // }
  // else
  // {
  //   ROS_WARN("Elevation layer not found in the GridMap!");
  // }

  end = ros::Time::now();
  ROS_INFO("Map Processing time: %f seconds", (end - start).toSec());
}

}  // namespace lsmap
