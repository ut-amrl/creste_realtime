#include "lsmap_node.h"

namespace lsmap {

LSMapNode::LSMapNode(const std::string& config_path)
    : nh_("~"), model_(nullptr), model_outputs_(nullptr), carrot_planner_(nullptr) {
  ROS_INFO("LSMapNode initialized.");

  YAML::Node config;
  try {
    config = YAML::LoadFile(config_path);
    ROS_INFO("Loaded config file: %s", config_path.c_str());
  } catch (const std::exception& e) {
    ROS_ERROR("Failed to load config file %s: %s", config_path.c_str(),
              e.what());
    // You can throw or return if the config is essential
    return;
  }

  ROS_INFO("--- Loading parameters ---");
  if (!config["model_params"] || !config["planner_params"] ||
    !config["input_topics"] || !config["output_topics"]) {
    ROS_ERROR("Missing required fields in config file.");
    return;
  }
  std::string model_path = config["model_params"]["weights_path"].as<std::string>();
  std::string image_topic = config["input_topics"]["image"].as<std::string>();
  std::string pointcloud_topic = config["input_topics"]["pointcloud"].as<std::string>();
  std::string camera_info_topic = config["input_topics"]["camera_info"].as<std::string>();
  std::string output_depth_topic =
      config["output_topics"]["depth"].as<std::string>();
  std::string output_traversability_topic =
      config["output_topics"]["traversability"].as<std::string>();

  // Load Model
  model_ = std::make_shared<LSMapModel>(model_path);
  // Load LiDAR Camera extrinsics
  LoadCalibParams(config);
  carrot_planner_ = std::make_shared<CarrotPlanner>(config); // Initialize CarrotPlanner


  // Subscription to PointCloud2 topic
  ROS_INFO("--- Subscribing to topics ---");
  pointcloud_subscriber_ = nh_.subscribe<sensor_msgs::PointCloud2>(
      pointcloud_topic, 10, &LSMapNode::PointCloudCallback, this);

  // Subscription to Image topic
  image_subscriber_ = nh_.subscribe<sensor_msgs::Image>(
      image_topic, 10, &LSMapNode::ImageCallback, this);

  // Subscription to CameraInfo
  camera_info_subscriber_ = nh_.subscribe<sensor_msgs::CameraInfo>(
      camera_info_topic, 10, &LSMapNode::CameraInfoCallback, this);

  // Create fov mask (example: 256 x 256). Adjust as needed.
  fov_mask_ = createTrapezoidalFovMask(256, 256);
  if (torch::cuda::is_available()) {
    fov_mask_ = fov_mask_.to(torch::kCUDA);
  }

  // Publishers
  image_publisher_ =
      nh_.advertise<sensor_msgs::Image>("/creste/stereo/left/image_rect", 10);
  depth_publisher_ = nh_.advertise<sensor_msgs::Image>(output_depth_topic, 10);
  traversability_publisher_ =
      nh_.advertise<sensor_msgs::Image>(output_traversability_topic, 10);

  has_rectification_ = false;
}

void LSMapNode::LoadCalibParams(const YAML::Node& config) {
  {
    const auto& node = config["pt2pix"];
    if (!node) {
      std::cerr << "ERROR: No 'pt2pix' entry in YAML!\n";
      return;
    }

    // 'col' in the snippet might be a typo; let's assume you meant 'cols'.
    // If the YAML key is literally "col", use node["col"].as<int>() instead.
    pt2pix_.rows = node["rows"].as<int>();
    // If your YAML says "col" instead of "cols", switch to node["col"] here:
    pt2pix_.cols = node["cols"].as<int>();
    pt2pix_.data = node["data"].as<std::vector<float>>();
  }

  // Parse "pix2pt" data
  {
    const auto& node = config["pix2pt"];
    if (!node) {
      std::cerr << "ERROR: No 'pix2pt' entry in YAML!\n";
      return;
    }
    pix2pt_.rows = node["rows"].as<int>();
    pix2pt_.cols = node["cols"].as<int>();
    pix2pt_.data = node["data"].as<std::vector<float>>();
  }
}

void LSMapNode::PointCloudCallback(
    const sensor_msgs::PointCloud2ConstPtr& msg) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  ROS_INFO("Received PointCloud2 message");
  cloud_queue_.push(msg);

  // Convert ROS PointCloud2 message to PCL point cloud
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  pcl::fromROSMsg(*msg, pcl_cloud);

  // Example processing
  ROS_INFO("PointCloud size: %lu", pcl_cloud.size());
}

void LSMapNode::ImageCallback(const sensor_msgs::ImageConstPtr& msg) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  ROS_INFO("Received Image message");
  image_queue_.push(msg);
}

void LSMapNode::CameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
  ROS_INFO("Received CameraInfo message");

  camera_info_ = *msg;

  // Compute rectification matrix
  if (!has_rectification_) {
    cv::Size size(camera_info_.width, camera_info_.height);
    // Copy and convert camera_info float64 fields
    cv::Mat Kd(3, 3, CV_64F, (void*)camera_info_.K.data());
    cv::Mat Dd(1, camera_info_.D.size(), CV_64F, (void*)camera_info_.D.data());
    cv::Mat Rd(3, 3, CV_64F, (void*)camera_info_.R.data());
    cv::Mat Pd(3, 4, CV_64F, (void*)camera_info_.P.data());
    cv::Mat Kf, Df, Rf, Pf;
    Kd.convertTo(Kf, CV_32F);
    Dd.convertTo(Df, CV_32F);
    Rd.convertTo(Rf, CV_32F);
    Pd.convertTo(Pf, CV_32F);

    cv::initUndistortRectifyMap(Kf, Df, Rf, Pf, size, CV_32FC1, map1_, map2_);
    ROS_INFO("Rectification maps computed (%d x %d).", size.width, size.height);

    has_rectification_ = true;
  }
}

std::tuple<torch::Tensor, torch::Tensor> LSMapNode::ProcessInputs(
    const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
    const sensor_msgs::ImageConstPtr& image_msg) {
  // Convert ROS PointCloud2 message to PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

  // Convert ROS Image to OpenCV image (BGR)
  cv::Mat bgr_image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;

  // Extract P(3x4) from camera_info_.P
  Eigen::Matrix<float, 3, 4, Eigen::RowMajor> pt2pixel;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      pt2pixel(i, j) = pt2pix_.data[i * 4 + j];
    }
  }

  // Convert BGR to RGB and rectify
  cv::Mat rgb_image;
  cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);
  cv::remap(rgb_image, rgb_image, map1_, map2_, cv::INTER_LINEAR);

  // Publish iamge for visualization
  // sensor_msgs::ImagePtr rectified_image_msg =
  //     cv_bridge::CvImage(std_msgs::Header(), "rgb8", rgb_image).toImageMsg();
  // image_publisher_.publish(rectified_image_msg);

  // Normalize to float in [0,1]
  rgb_image.convertTo(rgb_image, CV_32FC3, 1.0 / 255.0);

  // Initialize depth image with min float
  cv::Mat depth_image(rgb_image.rows, rgb_image.cols, CV_32FC1,
                      std::numeric_limits<float>::lowest());

  size_t num_points = pcl_cloud->points.size();
  Eigen::MatrixXf points(4, num_points);
  for (size_t i = 0; i < num_points; ++i) {
    points(0, i) = pcl_cloud->points[i].x;
    points(1, i) = pcl_cloud->points[i].y;
    points(2, i) = pcl_cloud->points[i].z;
    points(3, i) = 1.0f;
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
      float& depth_val = depth_image.at<float>(pixel_y, pixel_x);
      // Keep the furthest (max Z) if needed, or nearest (min Z).
      depth_val = std::max(depth_val, z);
    }
  }

  // Replace any "lowest()" with 0.0
  depth_image.setTo(0.0, depth_image == std::numeric_limits<float>::lowest());

  // Convert to millimeters
  depth_image *= 1000.0f;

  // Merge RGB + Depth -> 4 channels
  cv::Mat rgbd_image;
  std::vector<cv::Mat> channels;
  cv::split(rgb_image, channels);   // [R, G, B]
  channels.push_back(depth_image);  // add depth as 4th channel
  cv::merge(channels, rgbd_image);  // [H, W, 4]

  // Convert to torch tensor: [1, 1, 4, H, W] on GPU (if available)
  torch::Tensor rgbd_tensor = torch::from_blob(
      rgbd_image.data, {rgbd_image.rows, rgbd_image.cols, 4}, torch::kFloat32);
  rgbd_tensor = rgbd_tensor.permute({2, 0, 1});         // [4, H, W]
  rgbd_tensor = rgbd_tensor.unsqueeze(0).unsqueeze(0);  // [B, #cams, 4, H, W]

  if (torch::cuda::is_available()) {
    rgbd_tensor = rgbd_tensor.to(torch::kCUDA);
  }

  // Create pixel2point matrix (4x4) from pixel_to_point_
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> pixel2point;
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      pixel2point(i, j) = pix2pt_.data[i * 4 + j];
    }
  }

  torch::Tensor pixel2point_tensor =
      torch::from_blob(pixel2point.data(), {4, 4}, torch::kFloat32).clone();
  pixel2point_tensor =
      pixel2point_tensor.unsqueeze(0).unsqueeze(0);  // [B, #cams, 4, 4]

  if (torch::cuda::is_available()) {
    pixel2point_tensor = pixel2point_tensor.to(torch::kCUDA);
  }

  return std::make_tuple(rgbd_tensor, pixel2point_tensor);
}

bool LSMapNode::CarrotPlannerCallback(CarrotPlannerSrv::Request& req,
                             CarrotPlannerSrv::Response& res) {
  const auto& carrot = req.carrot;    

  // Plan path to carrot
  if (model_outputs_ == nullptr) {
    ROS_ERROR("Model outputs not available.");
    res.success.data = false;
    return true;
  }

  // Get the model outputs
  const auto& traversability_map = model_outputs_->at("traversability");
  std::vector<std::vector<float>> traversability_vec;
  TensorToVec2D(traversability_map.squeeze(), traversability_vec);
  
  // Time the time to plan the path
  
  Pose2D carrot_pose(carrot.pose.position.x, carrot.pose.position.y, 0.0f);
  const auto &path = carrot_planner_->PlanPath(traversability_vec, carrot_pose);

  // Copy path to response
  nav_msgs::Path path_ros;
  path_ros.header.frame_id = "base_link";
  path_ros.header.stamp = ros::Time::now();
  for (const auto& path_pose : path.poses) {
    geometry_msgs::PoseStamped posestamped;
    posestamped.pose.position.x = path_pose.x;
    posestamped.pose.position.y = path_pose.y;
    posestamped.pose.position.z = 0.0;
    posestamped.pose.orientation.w = std::cos(path_pose.theta / 2.0);
    posestamped.pose.orientation.z = std::sin(path_pose.theta / 2.0);
    res.path.poses.push_back(posestamped);
  }
  res.success.data = true;

  ROS_INFO("Planned path with %lu waypoints.", path.poses.size());

  return true;
}

void LSMapNode::run() {
  sensor_msgs::PointCloud2ConstPtr cloud_msg;
  sensor_msgs::ImageConstPtr image_msg;

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (cloud_queue_.empty() || image_queue_.empty()) return;

    ros::Time cloud_time = cloud_queue_.front()->header.stamp;
    ros::Time image_time = image_queue_.front()->header.stamp;

    // Convert to nanoseconds
    int64_t cloud_time_ns =
        static_cast<int64_t>(cloud_time.sec) * 1000000000LL + cloud_time.nsec;
    int64_t image_time_ns =
        static_cast<int64_t>(image_time.sec) * 1000000000LL + image_time.nsec;

    // Check if timestamps are within 100 ms
    if (std::abs(cloud_time_ns - image_time_ns) < 100LL * 1000000LL) {
      cloud_msg = cloud_queue_.front();
      cloud_queue_.pop();
      image_msg = image_queue_.front();
      image_queue_.pop();
    } else {
      // Drop the older message
      if (cloud_time_ns < image_time_ns)
        cloud_queue_.pop();
      else
        image_queue_.pop();
    }
  }

  if (!cloud_msg || !image_msg || !has_rectification_) return;

  // 1) Project
  ros::Time start = ros::Time::now();
  ROS_INFO("Projecting point cloud to image space...");
  auto inputs = ProcessInputs(cloud_msg, image_msg);
  ros::Time end = ros::Time::now();
  ROS_INFO("Projection time: %f seconds", (end - start).toSec());

  // 2) Model Inference
  start = ros::Time::now();
  auto output = model_->forward(inputs);
  end = ros::Time::now();
  ROS_INFO("Model Inference time: %f seconds", (end - start).toSec());

  // 3) Process the resulting map(s)
  start = ros::Time::now();
  std::unordered_map<std::string, torch::Tensor> tensor_map;
  tensor_map["elevation"] = output.at("elevation_preds");
  tensor_map["static_sem"] = output.at("inpainting_sam_preds");  // [1, F, H, W]
  tensor_map["dynamic_sem"] =
      output.at("inpainting_sam_dynamic_preds");                // [1, F, H, W]
  tensor_map["depth_preds"] = output.at("depth_preds_metric");  // [1, Hd, Wd]
  tensor_map["traversability"] =
      output.at("traversability_preds_full");  // [1, 1, H, W]

  // Mask out the FOV
  tensor_map["traversability"] = tensor_map["traversability"] * fov_mask_;
  tensor_map["static_sem"] = tensor_map["static_sem"] * fov_mask_;
  tensor_map["dynamic_sem"] = tensor_map["dynamic_sem"] * fov_mask_;
  tensor_map["elevation"] = tensor_map["elevation"] * fov_mask_;

  // Store the model outputs
  {
    std::lock_guard<std::mutex> lock(model_outputs_mutex_);
    model_outputs_ = std::make_shared<std::unordered_map<std::string, torch::Tensor>>(
        tensor_map);
  }

  // Publish model predictions
  PublishCompletedDepth(tensor_map, "depth_preds", depth_publisher_);
  PublishTraversability(tensor_map, "traversability",
                        traversability_publisher_);
  end = ros::Time::now();
  ROS_INFO("Map Processing time: %f seconds", (end - start).toSec());
}

}  // namespace lsmap
