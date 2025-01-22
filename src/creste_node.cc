#include "creste_node.h"

using amrl_msgs::CostmapSrv;
using std::vector;

namespace creste {

CresteNode::CresteNode(const std::string& config_path,
                     const std::string& weights_path)
    : nh_("~"),
      model_(nullptr),
      model_outputs_(nullptr),
      semantic_history_idx_(0),
      viz_3d_(false),
      carrot_planner_(nullptr) {
  ROS_INFO("CresteNode initialized.");

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
      !config["input_topics"] || !config["output_topics"] ||
      !config["viz_params"] || !config["map_params"]) {
    ROS_ERROR("Missing required fields in config file.");
    return;
  }
  std::string model_path =
      config["model_params"]["weights_path"].as<std::string>();
  model_path = model_path.empty() ? weights_path : model_path;
  std::string image_topic = config["input_topics"]["image"].as<std::string>();
  std::string pointcloud_topic =
      config["input_topics"]["pointcloud"].as<std::string>();
  std::string camera_info_topic =
      config["input_topics"]["camera_info"].as<std::string>();
  std::string output_depth_topic =
      config["output_topics"]["depth"].as<std::string>();
  std::string output_traversability_topic =
      config["output_topics"]["traversability"].as<std::string>();
  std::string output_semantic_elevation_topic =
      config["output_topics"]["semantic_elevation"].as<std::string>();

  // Load map to world extrinsics
  map_to_base_link_ = config["map_params"]["map_to_base_link"].as<std::vector<float>>();

  // Load Model
  model_ = std::make_shared<CresteModel>(model_path);
  // Load LiDAR Camera extrinsics
  LoadCalibParams(config);
  carrot_planner_ =
      std::make_shared<CarrotPlanner>(config);  // Initialize CarrotPlanner

  // ServiceServers
  carrot_planner_service = nh_.advertiseService(
      "/navigation/deep_cost_map_service", &CresteNode::CostmapCallback, this);

  // Subscription to PointCloud2 topic
  ROS_INFO("--- Subscribing to topics ---");
  pointcloud_subscriber_ = nh_.subscribe<sensor_msgs::PointCloud2>(
      pointcloud_topic, 10, &CresteNode::PointCloudCallback, this);

  // Subscription to Image topic
  image_subscriber_ = nh_.subscribe<sensor_msgs::CompressedImage>(
      image_topic, 10, &CresteNode::CompressedImageCallback, this);

  // Subscription to CameraInfo
  camera_info_subscriber_ = nh_.subscribe<sensor_msgs::CameraInfo>(
      camera_info_topic, 10, &CresteNode::CameraInfoCallback, this);

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
  semantic_elevation_publisher_ = 
      nh_.advertise<sensor_msgs::Image>(output_semantic_elevation_topic, 10);

  // Initialize semantic map queue
  int history_window = config["viz_params"]["history_window"].as<int>();
  int static_output_dim = config["model_params"]["static_output_dim"].as<int>();
  semantic_history_ = torch::zeros({history_window, static_output_dim, 256, 256}).to(torch::kCUDA);
  viz_3d_ = config["viz_params"]["viz_3d"].as<bool>();

  has_rectification_ = false;
}

void CresteNode::LoadCalibParams(const YAML::Node& config) {
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

void CresteNode::PointCloudCallback(
    const sensor_msgs::PointCloud2ConstPtr& msg) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  // ROS_INFO("Received PointCloud2 message");
  cloud_queue_.push(msg);

  // Convert ROS PointCloud2 message to PCL point cloud
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  pcl::fromROSMsg(*msg, pcl_cloud);

  // Example processing
  // ROS_INFO("PointCloud size: %lu", pcl_cloud.size());
}

void CresteNode::CompressedImageCallback(
    const sensor_msgs::CompressedImageConstPtr& msg) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  // ROS_INFO("Received Image message");
  image_queue_.push(msg);
}

void CresteNode::CameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
  // ROS_INFO("Received CameraInfo message");

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
    // ROS_INFO("Rectification maps computed (%d x %d).", size.width,
    // size.height);

    has_rectification_ = true;
  }
}

std::tuple<torch::Tensor, torch::Tensor> CresteNode::ProcessInputs(
    const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
    const sensor_msgs::CompressedImageConstPtr& image_msg) {
  // Convert ROS PointCloud2 message to PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

  // Convert Compressed ROS Image to OpenCV image (BGR)
  cv::Mat bgr_image = cv::imdecode(cv::Mat(image_msg->data), cv::IMREAD_COLOR);
  // cv::Mat bgr_image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;

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

bool CresteNode::CostmapCallback(CostmapSrv::Request& req,
                                CostmapSrv::Response& res) {
  // 1 Perform model inference (Cached)
  std::shared_ptr<std::unordered_map<std::string, torch::Tensor>> model_outputs;
  {
    std::lock_guard<std::mutex> lock(model_outputs_mutex_);
    model_outputs = model_outputs_;
  }

  if (model_outputs == nullptr) {
    ROS_ERROR("Model outputs not available.");
    res.success.data = false;
    return true;
  }

  const auto& traversability_map = model_outputs->at("traversability_cost");

  // 2 Create image costmap
  cv_bridge::CvImage cv_img;
  cv_img.header.stamp = ros::Time::now();  // Timestamp
  cv_img.header.frame_id = "base_link";    // Set frame_id (can be customized)
  cv_img.encoding = sensor_msgs::image_encodings::TYPE_8UC1;  // 8-bit grayscale
  cv_img.image = TensorToMat(traversability_map);

  // Convert CvImage to Image message
  sensor_msgs::Image img_msg;
  cv_img.toImageMsg(img_msg);
  res.costmap = img_msg;
  res.success.data = true;

  return true;
}

void CresteNode::inference() {
  sensor_msgs::PointCloud2ConstPtr cloud_msg;
  sensor_msgs::CompressedImageConstPtr image_msg;

  {
    std::lock_guard<std::mutex> lock(latest_msg_mutex_);
    if (!latest_cloud_msg_ || !latest_image_msg_ || !has_rectification_) return;
    cloud_msg = latest_cloud_msg_;
    image_msg = latest_image_msg_;
  }

  // 1) Project
  ros::Time start = ros::Time::now();
  // ROS_INFO("Projecting point cloud to image space...");
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

  // 4) Construct normalized cost map
  torch::Tensor cost_map = -tensor_map["traversability"];
  float min_cost = cost_map.min().item<float>();
  float max_cost = cost_map.max().item<float>();

  if (max_cost > min_cost) {  // Avoid division by zero
    cost_map = (cost_map - min_cost) / (max_cost - min_cost);
  } else {
    ROS_WARN("Cost map has uniform values; setting to 1.");
    cost_map.fill_(1.0f);  // Uniform cost map
  }

  // Set cost outside FOV to 1.0f
  cost_map.masked_fill_(fov_mask_.logical_not(), 1.0f);
  tensor_map["traversability_cost"] = cost_map;

  // Mask out the FOV
  tensor_map["static_sem"] = tensor_map["static_sem"] * fov_mask_;
  tensor_map["dynamic_sem"] = tensor_map["dynamic_sem"] * fov_mask_;
  tensor_map["elevation"] = tensor_map["elevation"] * fov_mask_;

  {
    // 1) Move costmap to CPU & convert [0,1] float -> [0,255] uint8
    torch::Tensor cost_map_8u = cost_map.detach()
                                        .cpu()               // ensure on CPU
                                        .mul(255.0f)         // scale [0,1] => [0,255]
                                        .clamp_(0, 255)      // clamp to valid range
                                        .to(torch::kU8);     // convert to uint8
    cost_map_8u = cost_map_8u.squeeze();  // [B, 1, H, W] -> [H, W]

    // 2) Wrap cost_map_8u in an OpenCV Mat (CV_8UC1).
    int rows = cost_map_8u.size(0);
    int cols = cost_map_8u.size(1);
    cv::Mat cost_map_mat(rows, cols, CV_8UC1, cost_map_8u.data_ptr<uint8_t>());

    // 3) Build affine transform: rotate 4° clockwise & shift 10 px left.
    //    - Use angle = -4 for clockwise rotation in OpenCV.
    //    - Then adjust the last column of the rotation matrix to shift left.
    cv::Point2f center(cols / 2.0f, rows / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(center, map_to_base_link_[2], 1.0);
    // Decrease the X translation by 10.0 to shift left
    rot.at<double>(0, 2) += map_to_base_link_[0];
    // Add Y Translation
    rot.at<double>(1, 2) += map_to_base_link_[1];

    // 4) warpAffine to rotate + shift
    cv::Mat rotated_8u;
    cv::warpAffine(cost_map_mat, rotated_8u, rot, cost_map_mat.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue*/ 255);

    // 5) Convert rotated CV_8UC1 back to float [0,1] in a torch::Tensor
    torch::Tensor rotated_tensor = torch::from_blob(
        rotated_8u.data, {rotated_8u.rows, rotated_8u.cols}, torch::kUInt8);
    // clone() so we own the memory (from_blob() just references existing data)
    rotated_tensor = rotated_tensor.clone().to(torch::kFloat32).div_(255.0f);
    rotated_tensor = rotated_tensor.unsqueeze(0).unsqueeze(0);  // [1,1,H,W]

    // 6) Overwrite the costmap in the tensor_map
    tensor_map["traversability_cost"] = rotated_tensor;
}

  // Store the model outputs
  {
    std::lock_guard<std::mutex> lock(model_outputs_mutex_);
    model_outputs_ =
        std::make_shared<std::unordered_map<std::string, torch::Tensor>>(
            tensor_map);
  }

  // Publish model predictions
  PublishCompletedDepth(tensor_map, "depth_preds", depth_publisher_);
  PublishTraversability(tensor_map, "traversability_cost",
                        traversability_publisher_);
  
  if (viz_3d_) {
  // Prepare 3D visualization
  printf("Computing PCA for semantic elevation map...");
  semantic_history_[semantic_history_idx_] = tensor_map["static_sem"].index({0}); // [1, F, H, W] -> [F, H, W]

  const auto& sem_rgb_window = computePCA(semantic_history_); // [B, 3, H, W]
  auto sem_rgb_th = sem_rgb_window[semantic_history_idx_].unsqueeze(0); // [1, 3, H, W]
  sem_rgb_th.masked_fill_(fov_mask_.logical_not().unsqueeze(0), -0.5f); // Mask out FOV
  printf("Computed PCA for semantic elevation map.\n");
  vector<vector<RGBColor> > elevation_rgb_vec;

  const auto& rel_elevation = tensor_map["elevation"].index({0, 0}).unsqueeze(0).unsqueeze(0); // [1, 1, H, W]
  printf("rel_elevation ndims: %ld\n", rel_elevation.dim());
  printf("rel_elevation dims: %ld, %ld, %ld\n", rel_elevation.size(0), rel_elevation.size(1), rel_elevation.size(2));
  // TensorToColorMap(rel_elevation, elevation_rgb_vec);
   TensorToColorMap(sem_rgb_th, elevation_rgb_vec);

  vector<vector<float> > elevation_vec;
  // Extract only the elevation tensor from [B, 2, H, W] to [H, W]
  TensorToVec2D(rel_elevation, elevation_vec);
  printf("Converted tensors to vectors.\n");
  // Publish the 3D visualization
  creste::GenerateAndPublishHeightMapImageStructuredGrid(
    elevation_vec, elevation_rgb_vec, semantic_elevation_publisher_
  );
  semantic_history_idx_ = (semantic_history_idx_ + 1) % semantic_history_.sizes()[0];
  }
  // Upsample depth iamge
  // const int target_height = camera_info_.height;
  // const int target_width = camera_info_.width;
  // tensor_map["depth_full_preds"] = UpsampleDepthImage(target_height, target_width, tensor_map["depth_preds"]);
  // PublishCompletedDepth(tensor_map, "depth_full_preds", depth_publisher_);
  
  end = ros::Time::now();
  ROS_INFO("Map Processing time: %f seconds", (end - start).toSec());
}

void CresteNode::run() {
  sensor_msgs::PointCloud2ConstPtr cloud_msg;
  sensor_msgs::CompressedImageConstPtr image_msg;

  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (cloud_queue_.empty() || image_queue_.empty()) return;

    // Get the current time
    ros::Time current_time = ros::Time::now();

    // Get the timestamp of the first messages in the queues
    ros::Time cloud_time = cloud_queue_.front()->header.stamp;
    ros::Time image_time = image_queue_.front()->header.stamp;

    // Convert timestamps to nanoseconds
    int64_t cloud_time_ns =
        static_cast<int64_t>(cloud_time.sec) * 1000000000LL + cloud_time.nsec;
    int64_t image_time_ns =
        static_cast<int64_t>(image_time.sec) * 1000000000LL + image_time.nsec;
    int64_t current_time_ns =
        static_cast<int64_t>(current_time.sec) * 1000000000LL +
        current_time.nsec;

    // Check if the messages are older than 300ms
    if (std::abs(current_time_ns - cloud_time_ns) > 300LL * 1000000LL) {
      cloud_queue_.pop();  // Drop the cloud message if it's older than 300ms
    }

    if (std::abs(current_time_ns - image_time_ns) > 300LL * 1000000LL) {
      image_queue_.pop();  // Drop the image message if it's older than 300ms
    }

    // After dropping older messages, if both queues are still not empty,
    // compare timestamps
    if (!cloud_queue_.empty() && !image_queue_.empty()) {
      cloud_time = cloud_queue_.front()->header.stamp;
      image_time = image_queue_.front()->header.stamp;

      cloud_time_ns =
          static_cast<int64_t>(cloud_time.sec) * 1000000000LL + cloud_time.nsec;
      image_time_ns =
          static_cast<int64_t>(image_time.sec) * 1000000000LL + image_time.nsec;

      // Check if the timestamps are within 100ms
      int64_t time_diff_ns = std::abs(cloud_time_ns - image_time_ns);

      if (time_diff_ns < 100LL * 1000000LL) {
        // If within 100ms, pop both messages
        cloud_msg = cloud_queue_.front();
        cloud_queue_.pop();
        image_msg = image_queue_.front();
        image_queue_.pop();
      } else {
        // If not within 100ms, drop the older message
        if (cloud_time_ns < image_time_ns) {
          ROS_INFO("Dropping cloud message with timestamp %ld", cloud_time_ns);
          cloud_queue_.pop();  // Drop the older cloud message
        } else {
          ROS_INFO("Dropping image message with timestamp %ld", image_time_ns);
          image_queue_.pop();  // Drop the older image message
        }
        
        return; // early return if we dropped a message
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(latest_msg_mutex_);
    latest_cloud_msg_ = cloud_msg;
    latest_image_msg_ = image_msg;
  }

  if (!cloud_msg || !image_msg || !has_rectification_) return;

  this->inference();
}

}  // namespace creste
