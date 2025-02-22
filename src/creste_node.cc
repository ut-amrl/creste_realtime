#include "creste_node.h"

using std::vector;

namespace creste {

CresteNode::CresteNode(const std::string& config_path,
                        const std::string& weights_path)
    : nh_("~"),
      model_(nullptr),
      model_outputs_(nullptr),
      semantic_history_idx_(0),
      viz_3d_(false) {
  ROS_INFO("CresteNode initialized.");

  YAML::Node config;
  try {
    config = YAML::LoadFile(config_path);
    ROS_INFO("Loaded config file: %s", config_path.c_str());
  } catch (const std::exception& e) {
    ROS_ERROR("Failed to load config file %s: %s", config_path.c_str(), e.what());
    return;
  }

  // Read parameters from config
  std::string left_image_topic  = config["input_topics"]["left_image"].as<std::string>("");
  std::string right_image_topic = config["input_topics"]["right_image"].as<std::string>("");
  std::string pointcloud_topic  = config["input_topics"]["pointcloud"].as<std::string>("");
  std::string left_info_topic   = config["input_topics"]["left_camera_info"].as<std::string>("");
  std::string right_info_topic  = config["input_topics"]["right_camera_info"].as<std::string>("");
  modality_ = config["model_params"]["modality"].as<std::string>("rgbd");

  std::string output_depth_topic =
      config["output_topics"]["depth"].as<std::string>();
  std::string output_traversability_topic =
      config["output_topics"]["traversability"].as<std::string>();
  std::string output_semantic_elevation_topic =
      config["output_topics"]["semantic_elevation"].as<std::string>();

  // Load map to world extrinsics
  map_to_base_link_ = config["map_params"]["map_to_base_link"].as<std::vector<float>>();

  // Decide which sensors to enable
  enable_cloud_ = !pointcloud_topic.empty();

  // Build up a vector of cameras we want to handle
  if (!left_image_topic.empty()) {
    auto cam = std::make_unique<CameraHandler>();
    cam->enabled      = true;
    cam->camera_name  = "left";
    cam->image_topic  = left_image_topic;
    cam->info_topic   = left_info_topic;
    cameras_.push_back(std::move(cam));
  }

  if (!right_image_topic.empty()) {
    auto cam = std::make_unique<CameraHandler>();
    cam->enabled      = true;
    cam->camera_name  = "right";
    cam->image_topic  = right_image_topic;
    cam->info_topic   = right_info_topic;
    cameras_.push_back(std::move(cam));
  }
  // If only 1 camera is desired, you'll have just one in cameras_.  
  // If you want stereo, you have 2, etc.

  // Subscribe to pointcloud if needed
  if (enable_cloud_) {
    pointcloud_subscriber_ = nh_.subscribe<sensor_msgs::PointCloud2>(
        pointcloud_topic, 10, &CresteNode::PointCloudCallback, this);
  }

  // For each camera in cameras_, subscribe to both image & camera_info
  for (size_t i = 0; i < cameras_.size(); i++) {
    auto& cam = *(cameras_[i]);  // dereference the unique_ptr to get a reference
    if (!cam.enabled) continue;

    if (!cam.image_topic.empty()) {
      cam.image_sub = nh_.subscribe<sensor_msgs::CompressedImage>(
          cam.image_topic, 10,
          boost::bind(&CresteNode::CameraImageCallback, this, _1, i));
      ROS_INFO("Subscribed to image topic: %s", cam.image_topic.c_str());
    }
    if (!cam.info_topic.empty()) {
      cam.info_sub = nh_.subscribe<sensor_msgs::CameraInfo>(
          cam.info_topic, 10,
          boost::bind(&CresteNode::CameraInfoCallback, this, _1, i));
      ROS_INFO("Subscribed to info topic: %s", cam.info_topic.c_str());
    }
  }

  // Load model
  std::string model_path = config["model_params"]["weights_path"].as<std::string>("");
  if (!weights_path.empty()) {
    model_path = weights_path;  // override if needed
  }
  model_ = std::make_shared<CresteModel>(model_path);

  // Advertise a service
  carrot_planner_service = nh_.advertiseService(
      "/navigation/deep_cost_map_service", &CresteNode::CostmapCallback, this);

  // Load extrinsics, calibrations, FOV mask, etc.
  LoadCalibParams(config);
  fov_mask_ = createTrapezoidalFovMask(256, 256);
  if (torch::cuda::is_available()) {
    fov_mask_ = fov_mask_.to(torch::kCUDA);
  }

  // Publishers
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

// Example helper to load calibration
void CresteNode::LoadCalibParams(const YAML::Node& config) {
  {
    const auto& node = config["pt2pix"];
    if (!node) {
      std::cerr << "ERROR: No 'pt2pix' entry in YAML!\n";
      return;
    }
    pt2pix_.rows = node["rows"].as<int>();
    pt2pix_.cols = node["cols"].as<int>();
    pt2pix_.data = node["data"].as<std::vector<float>>();
  }

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

// --------------------------------------------------------------------------
//                               Callbacks
// --------------------------------------------------------------------------
void CresteNode::PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
  std::lock_guard<std::mutex> lk(cloud_queue_mutex_);
  cloud_queue_.push(msg);
}

void CresteNode::CameraImageCallback(
    const sensor_msgs::CompressedImageConstPtr& msg,
    size_t cam_idx) {
  // Lock this camera's queue
  auto& cam = *(cameras_[cam_idx]);
  std::lock_guard<std::mutex> lk(cam.queue_mutex);
  cam.image_queue.push(msg);
}

void CresteNode::CameraInfoCallback(
    const sensor_msgs::CameraInfoConstPtr& msg,
    size_t cam_idx) {
  auto& cam = *(cameras_[cam_idx]);
  cam.camera_info = *msg;

  if (!cam.has_rectification) {
    // Initialize rectification maps for this camera
    cv::Size size(msg->width, msg->height);

    cv::Mat Kd(3, 3, CV_64F, (void*)msg->K.data());
    cv::Mat Dd(1, msg->D.size(), CV_64F, (void*)msg->D.data());
    cv::Mat Rd(3, 3, CV_64F, (void*)msg->R.data());
    cv::Mat Pd(3, 4, CV_64F, (void*)msg->P.data());

    cv::Mat Kf, Df, Rf, Pf;
    Kd.convertTo(Kf, CV_32F);
    Dd.convertTo(Df, CV_32F);
    Rd.convertTo(Rf, CV_32F);
    Pd.convertTo(Pf, CV_32F);

    cv::initUndistortRectifyMap(Kf, Df, Rf, Pf, size, CV_32FC1, cam.map1, cam.map2);
    cam.has_rectification = true;
    ROS_INFO("Initialized rectification for camera [%s]", cam.camera_name.c_str());
  }
}

// --------------------------------------------------------------------------
//                                run()
// --------------------------------------------------------------------------
void CresteNode::run() {
  // Try to grab and sync a batch of camera images + LiDAR (if enabled).

  // 1) Gather front messages from each camera's queue (if enabled).
  std::vector<sensor_msgs::CompressedImageConstPtr> front_images(cameras_.size(),
                                                                  nullptr);
  bool all_cameras_have_data = true;

  // Store timestamps in a vector to see if they match
  std::vector<int64_t> camera_timestamps_ns(cameras_.size(), 0);

  for (size_t i = 0; i < cameras_.size(); i++) {
    auto& cam = *(cameras_[i]);
    if (!cam.enabled) continue;

    {
      std::lock_guard<std::mutex> lk(cam.queue_mutex);
      if (cam.image_queue.empty()) {
        all_cameras_have_data = false;
        break;
      }
      front_images[i] = cam.image_queue.front();
    }
    // Get its timestamp
    ros::Time t = front_images[i]->header.stamp;
    camera_timestamps_ns[i] = static_cast<int64_t>(t.sec) * 1000000000LL + t.nsec;
  }

  if (!all_cameras_have_data) return;  // can't sync yet

  // 2) Check pointcloud queue if enabled
  sensor_msgs::PointCloud2ConstPtr front_cloud = nullptr;
  if (enable_cloud_) {
    std::lock_guard<std::mutex> lk(cloud_queue_mutex_);
    if (cloud_queue_.empty()) {
      return;  // no cloud => can't sync
    }
    front_cloud = cloud_queue_.front();
  }

  // 3) Compare timestamps
  int64_t min_ts = INT64_MAX;
  int64_t max_ts = INT64_MIN;
  for (auto ns_val : camera_timestamps_ns) {
    if (ns_val == 0) continue;
    min_ts = std::min(min_ts, ns_val);
    max_ts = std::max(max_ts, ns_val);
  }

  int64_t cloud_ts = 0;
  if (enable_cloud_ && front_cloud) {
    ros::Time t = front_cloud->header.stamp;
    cloud_ts = static_cast<int64_t>(t.sec) * 1000000000LL + t.nsec;
    min_ts = std::min(min_ts, cloud_ts);
    max_ts = std::max(max_ts, cloud_ts);
  }

  static constexpr int64_t kSyncThresholdNs = 100LL * 1000000LL;  // 100 ms
  if ((max_ts - min_ts) > kSyncThresholdNs) {
    // Timestamps not matched => drop the oldest
    for (size_t i = 0; i < cameras_.size(); i++) {
      auto& cam = *(cameras_[i]);
      if (!cam.enabled) continue;
      if (camera_timestamps_ns[i] == min_ts) {
        std::lock_guard<std::mutex> lk(cam.queue_mutex);
        cam.image_queue.pop();
        ROS_WARN("Dropping old camera [%s] frame", cam.camera_name.c_str());
      }
    }
    if (enable_cloud_ && cloud_ts == min_ts) {
      std::lock_guard<std::mutex> lk(cloud_queue_mutex_);
      cloud_queue_.pop();
      ROS_WARN("Dropping old LiDAR msg");
    }
    return; // try again next run()
  }

  // We have matched frames => pop from queues
  for (size_t i = 0; i < cameras_.size(); i++) {
    auto& cam = *(cameras_[i]);
    if (!cam.enabled) continue;
    {
      std::lock_guard<std::mutex> lk(cam.queue_mutex);
      cam.image_queue.pop();
    }
  }
  if (enable_cloud_) {
    std::lock_guard<std::mutex> lk(cloud_queue_mutex_);
    cloud_queue_.pop();
  }

  // Store in latest_*_msg_
  {
    std::lock_guard<std::mutex> lk(latest_msg_mutex_);
    latest_camera_msgs_ = front_images;
    latest_cloud_msg_   = front_cloud;
  }

  // Launch inference
  this->inference();
}

// --------------------------------------------------------------------------
//                            inference()
// --------------------------------------------------------------------------
void CresteNode::inference() {
  std::vector<sensor_msgs::CompressedImageConstPtr> camera_imgs;
  sensor_msgs::PointCloud2ConstPtr cloud_msg;
  {
    std::lock_guard<std::mutex> lk(latest_msg_mutex_);
    if (enable_cloud_ && !latest_cloud_msg_) {
      return;
    }
    for (size_t i = 0; i < cameras_.size(); i++) {
      auto& cam = *(cameras_[i]);
      if (cam.enabled && !latest_camera_msgs_[i]) {
        return;  // not ready
      }
    }
    camera_imgs = latest_camera_msgs_;
    cloud_msg   = latest_cloud_msg_;
  }

  if (!model_) return;

  // 1) Projection
  ros::Time start = ros::Time::now();
  auto inputs = ProcessInputs(cloud_msg, camera_imgs);
  ros::Time end = ros::Time::now();
  ROS_INFO("Projection time: %f seconds", (end - start).toSec());

  // 2) Inference
  start = ros::Time::now();
  auto output = model_->forward(inputs);
  end = ros::Time::now();
  ROS_INFO("Model Inference time: %f seconds", (end - start).toSec());

  // 3) Post-processing (construct cost map, etc.)
  start = ros::Time::now();
  std::unordered_map<std::string, torch::Tensor> tensor_map;
  if (viz_3d_) {
    tensor_map["elevation"]  = output.at("elevation_preds");
    tensor_map["static_sem"] = output.at("inpainting_sam_preds");
    tensor_map["dynamic_sem"] = output.at("inpainting_sam_dynamic_preds");
  }
  tensor_map["depth_preds"]       = output.at("depth_preds_metric");
  tensor_map["traversability"]    = output.at("traversability_preds_full");

  torch::Tensor cost_map = -tensor_map["traversability"];
  float min_cost = cost_map.min().item<float>();
  float max_cost = cost_map.max().item<float>();

  if (max_cost > min_cost) {
    cost_map = (cost_map - min_cost) / (max_cost - min_cost);
  } else {
    ROS_WARN("Cost map uniform. Setting to 1.0");
    cost_map.fill_(1.0f);
  }
  cost_map.clamp_(0.0f, 244.0f/255.0f);
  cost_map.masked_fill_(fov_mask_.logical_not(), 1.0f);
  tensor_map["traversability_cost"] = cost_map;

  if (viz_3d_) {
    tensor_map["static_sem"]  = tensor_map["static_sem"]  * fov_mask_;
    tensor_map["dynamic_sem"] = tensor_map["dynamic_sem"] * fov_mask_;
    tensor_map["elevation"]   = tensor_map["elevation"]   * fov_mask_;
  }

  // Convert costmap to CPU, rotate, shift, etc.
  {
    torch::Tensor cost_map_8u = cost_map.detach()
                                        .cpu()
                                        .mul(255.0f)
                                        .clamp_(0,255)
                                        .to(torch::kU8)
                                        .squeeze();
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
    ROS_INFO("Rotated cost map.");
    // 5) Convert rotated CV_8UC1 back to float [0,1] in a torch::Tensor
    torch::Tensor rotated_tensor = torch::from_blob(
        rotated_8u.data, {rotated_8u.rows, rotated_8u.cols}, torch::kUInt8);
    // clone() so we own the memory (from_blob() just references existing data)
    rotated_tensor = rotated_tensor.clone().to(torch::kFloat32).div_(255.0f);
    rotated_tensor = rotated_tensor.unsqueeze(0).unsqueeze(0);  // [1,1,H,W]
    ROS_INFO("Converted rotated cost map to tensor.");
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
  ROS_INFO("Stored model outputs.");

  // Publish model predictions
  PublishCompletedDepth(tensor_map, "depth_preds", depth_publisher_);
  PublishTraversability(tensor_map, "traversability_cost",
                        traversability_publisher_);
  ROS_INFO("Published model predictions.");
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
                                    

std::tuple<torch::Tensor, torch::Tensor> CresteNode::ProcessInputs(
  const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
  const std::vector<sensor_msgs::CompressedImageConstPtr>& camera_imgs) 
{
  // Final outputs
  torch::Tensor final_image_tensor;
  torch::Tensor final_p2p_tensor;

  // CPU float by default
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32);

  //===================================================
  // 1) "rgbd" => Single camera + LiDAR => 4 channels
  //===================================================
  if (modality_ == "rgbd") {
    // Expect exactly 1 camera + LiDAR
    if (camera_imgs.empty()) {
      ROS_ERROR("rgbd mode but no camera image found!");
      return {torch::Tensor(), torch::Tensor()};
    }
    if (!cloud_msg) {
      ROS_ERROR("rgbd mode but no LiDAR data found!");
      return {torch::Tensor(), torch::Tensor()};
    }

    // Convert ROS -> PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

    // Decode compressed => BGR
    cv::Mat bgr_image = cv::imdecode(cv::Mat(camera_imgs[0]->data), cv::IMREAD_COLOR);
    if (bgr_image.empty()) {
      ROS_ERROR("Empty camera image in rgbd mode!");
      return {torch::Tensor(), torch::Tensor()};
    }

    // BGR -> RGB
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    // If you want per-camera rectification for camera[0]:
    //   cameras_[0]->map1, cameras_[0]->map2
    // This line changed from cameras_[0].map1 => cameras_[0]->map1
    cv::remap(rgb_image, rgb_image,
              cameras_[0]->map1,  // <--- pointer dereference
              cameras_[0]->map2,
              cv::INTER_LINEAR);

    // Normalize to float [0..1]
    rgb_image.convertTo(rgb_image, CV_32FC3, 1.0 / 255.0);

    // Prepare a depth image
    cv::Mat depth_image(
        rgb_image.rows, rgb_image.cols, CV_32FC1,
        std::numeric_limits<float>::lowest());

    // Build pt2pixel (3x4) from pt2pix_
    Eigen::Matrix<float, 3, 4, Eigen::RowMajor> pt2pixel;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        pt2pixel(i, j) = pt2pix_.data[i * 4 + j];
      }
    }

    // Collect 3D points
    size_t num_points = pcl_cloud->points.size();
    Eigen::MatrixXf points(4, num_points);
    for (size_t i = 0; i < num_points; i++) {
      points(0, i) = pcl_cloud->points[i].x;
      points(1, i) = pcl_cloud->points[i].y;
      points(2, i) = pcl_cloud->points[i].z;
      points(3, i) = 1.0f;
    }

    // Project => (u,v,w)
    Eigen::MatrixXf pixels = pt2pixel * points;
    for (size_t i = 0; i < num_points; i++) {
      float w = pixels(2, i);
      if (w <= 0.0f) continue;  // behind camera
      float u = pixels(0, i) / w;
      float v = pixels(1, i) / w;
      float z = w;

      if (u >= 0 && u < rgb_image.cols &&
          v >= 0 && v < rgb_image.rows &&
          z > 0.0f) {
        int px = static_cast<int>(u);
        int py = static_cast<int>(v);
        float& dval = depth_image.at<float>(py, px);
        // Keep the furthest (max Z)
        dval = std::max(dval, z);
      }
    }

    // Replace "lowest()" with 0
    depth_image.setTo(0.0f, depth_image == std::numeric_limits<float>::lowest());

    // Convert depth to mm
    depth_image *= 1000.0f;

    // Merge (R,G,B,Depth) => [H,W,4]
    std::vector<cv::Mat> channels;
    cv::split(rgb_image, channels);  // => [R, G, B]
    channels.push_back(depth_image); // => Depth
    cv::Mat rgbd_image;
    cv::merge(channels, rgbd_image); // => [H,W,4]

    // Torch => [4,H,W] => [1,1,4,H,W]
    torch::Tensor rgbd_tensor =
        torch::from_blob(rgbd_image.data,
                        {rgbd_image.rows, rgbd_image.cols, 4},
                        options_f32);
    rgbd_tensor = rgbd_tensor.permute({2, 0, 1});         // => [4,H,W]
    rgbd_tensor = rgbd_tensor.unsqueeze(0).unsqueeze(0);  // => [1,1,4,H,W]

    // Build pixel2point => [1,1,4,4]
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> pixel2point;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        pixel2point(i, j) = pix2pt_.data[i * 4 + j];
      }
    }
    torch::Tensor pixel2point_tensor =
        torch::from_blob(pixel2point.data(), {4, 4}, torch::kFloat32)
              .clone();  // must clone so we own memory
    pixel2point_tensor = pixel2point_tensor.unsqueeze(0).unsqueeze(0); // => [1,1,4,4]

    // GPU if needed
    if (torch::cuda::is_available()) {
      rgbd_tensor = rgbd_tensor.cuda();
      pixel2point_tensor = pixel2point_tensor.cuda();
    }

    final_image_tensor = rgbd_tensor;
    final_p2p_tensor   = pixel2point_tensor;
  }

  //===========================================================
  // 2) MONOCULAR / STEREO => [1,#cams,3,H,W], no LiDAR depth
  //===========================================================
  else {
    if (camera_imgs.empty()) {
      ROS_ERROR("monocular/stereo mode but no camera images!");
      return {torch::Tensor(), torch::Tensor()};
    }

    // For each camera => rectify => build [3,H,W], then stack
    std::vector<torch::Tensor> cam_tensors;
    cam_tensors.reserve(camera_imgs.size());

    for (size_t i = 0; i < camera_imgs.size(); i++) {
      // Decode => BGR
      cv::Mat bgr = cv::imdecode(cv::Mat(camera_imgs[i]->data), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        ROS_WARN("Empty camera image at index %lu! Skipping...", i);
        continue;
      }

      // BGR->RGB
      cv::cvtColor(bgr, bgr, cv::COLOR_BGR2RGB);

      // Rectify with camera i's map
      if (i < cameras_.size()) {
        cv::remap(bgr, bgr,
                  cameras_[i]->map1,  // pointer
                  cameras_[i]->map2,
                  cv::INTER_LINEAR);
      }

      // Normalize
      bgr.convertTo(bgr, CV_32FC3, 1.0 / 255.0);

      // => Torch
      torch::Tensor cam_tensor =
          torch::from_blob(bgr.data, {bgr.rows, bgr.cols, 3}, options_f32);
      cam_tensor = cam_tensor.permute({2, 0, 1});  // => [3,H,W]
      cam_tensors.push_back(cam_tensor.clone());
    }

    if (cam_tensors.empty()) {
      ROS_ERROR("No valid images after decoding for monocular/stereo!");
      return {torch::Tensor(), torch::Tensor()};
    }

    // Stack => [#cams,3,H,W], then [1,#cams,3,H,W]
    torch::Tensor stacked = torch::stack(cam_tensors, 0);
    stacked = stacked.unsqueeze(0);

    // pixel2point => [1,1,4,4], from e.g. left camera or global
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> pixel2point;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        pixel2point(i, j) = pix2pt_.data[i * 4 + j];
      }
    }
    torch::Tensor p2p_tensor =
        torch::from_blob(pixel2point.data(), {4, 4}, torch::kFloat32).clone();
    p2p_tensor = p2p_tensor.unsqueeze(0).unsqueeze(0);  // => [1,1,4,4]

    if (torch::cuda::is_available()) {
      stacked = stacked.cuda();
      p2p_tensor = p2p_tensor.cuda();
    }

    final_image_tensor = stacked;    // => [1,#cams,3,H,W]
    final_p2p_tensor   = p2p_tensor; // => [1,1,4,4]
  }

  // Return final
  return std::make_tuple(final_image_tensor, final_p2p_tensor);
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

}  // namespace creste
