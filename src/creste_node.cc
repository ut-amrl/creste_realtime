#include "creste_node.h"

#include "utils.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

// For time difference in seconds
#define TIME_DIFF(start, end) ((end - start).seconds())

using std::vector;

namespace creste {

CresteNode::CresteNode(const std::string& config_path,
                       const std::string& weights_path, NodeHandleType node)
    : node_(node),
      model_(nullptr),
      model_outputs_(nullptr),
      semantic_history_idx_(0),
      viz_3d_(false) {
  // #endif
  LOG_INFO("CresteNode constructor starting.");

  YAML::Node config;
  try {
    config = YAML::LoadFile(config_path);
    LOG_INFO("Loaded config file: %s", config_path.c_str());
  } catch (const std::exception& e) {
    LOG_ERROR("Failed to load config file %s: %s", config_path.c_str(),
              e.what());
    return;
  }

  modality_ = config["model_params"]["modality"].as<std::string>("rgbd");
  stale_cutoff_ms_ = config["stale_cutoff_ms"].as<double>(1000.0);
  sync_cutoff_ms_ = config["sync_cutoff_ms"].as<double>(200.0);

  // Loop through dictionaries in input topics field
  if (config["input_topics"] && config["input_topics"].IsSequence()) {
    const auto& topics_list = config["input_topics"];
    for (size_t i = 0; i < topics_list.size(); ++i) {
      const auto& topic_cfg = topics_list[i];

      std::string sensor_type = topic_cfg["type"].as<std::string>("");
      if (sensor_type == "image") {
        // Create a CameraHandler
        auto cam = std::make_unique<CameraHandler>();
        cam->enabled = true;
        cam->camera_name = topic_cfg["camera_id"].as<std::string>("camera");

        // Parse fields from YAML
        cam->image_topic = topic_cfg["image_topic"].as<std::string>("");
        cam->info_topic = topic_cfg["info_topic"].as<std::string>("");
        cam->target_shape = topic_cfg["target_shape"].as<std::vector<int>>();
        cam->ds_factor = topic_cfg["ds_factor"].as<std::vector<float>>();
        cam->tf_frames = topic_cfg["tf_frames"].as<std::vector<std::string>>();
        cam->queue_size =
            topic_cfg["queue_size"].as<int>(10);  // default queue size

        // Create subscribers
        if (!cam->image_topic.empty()) {
          cam->image_sub = node_->create_subscription<CompressedImage>(
              cam->image_topic, 10,
              [this, idx = cameras_.size()](const CompressedImageConstPtr msg) {
                CameraImageCallback(msg, idx);
              });
          RCLCPP_INFO(node_->get_logger(), "Subscribed to image: %s",
                      cam->image_topic.c_str());
        }
        if (!cam->info_topic.empty()) {
          cam->info_sub = node_->create_subscription<CameraInfo>(
              cam->info_topic, 10,
              [this, idx = cameras_.size()](const CameraInfoConstPtr msg) {
                CameraInfoCallback(msg, idx);
              });
          RCLCPP_INFO(node_->get_logger(), "Subscribed to camera info: %s",
                      cam->info_topic.c_str());
        }

        // Push into cameras_ vector
        cameras_.push_back(std::move(cam));
      } else if (sensor_type == "pointcloud") {
        // Create a CloudHandler
        auto cloud = std::make_unique<CloudHandler>();
        cloud->enabled = true;
        cloud->cloud_topic = topic_cfg["cloud_topic"].as<std::string>("");
        cloud->tf_frame = topic_cfg["tf_frame"].as<std::string>("");

        // Create subscriber
        if (!cloud->cloud_topic.empty()) {
          cloud->cloud_sub = node_->create_subscription<PointCloud2>(
              cloud->cloud_topic, 10, [this](const PointCloud2ConstPtr msg) {
                PointCloudCallback(msg);
              });
          RCLCPP_INFO(node_->get_logger(), "Subscribed to pointcloud: %s",
                      cloud->cloud_topic.c_str());
        }

        // Push into cameras_ vector
        cloud_ = std::move(cloud);
      } else {
        LOG_ERROR("Unknown sensor type in input topics: %s",
                  sensor_type.c_str());
      }
    }
  } else {
    LOG_ERROR("Unknown sensor intialization error in input topics: %s",
              config["input_topics"].as<std::string>("").c_str());
  }

  auto qos = rclcpp::QoS(10).transient_local();
  tf_subscriber_ = node_->create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf_static", qos, [this](const tf2_msgs::msg::TFMessage::SharedPtr msg) {
        TFCallback(msg);
      });
  LOG_INFO("Subscribed to TF messages.");

  // Load map -> base_link extrinsics
  map_to_base_link_ =
      config["map_params"]["map_to_base_link"].as<std::vector<float>>();

  // Load model
  std::string model_path =
      config["model_params"]["weights_path"].as<std::string>("");
  if (!weights_path.empty()) {
    model_path = weights_path;  // override
  }
  model_ = std::make_shared<CresteModel>(model_path);

  // Advertise a service
  costmap_service_ = node_->create_service<CostmapSrv>(
      "/navigation/deep_cost_map_service",
      [this](const std::shared_ptr<CostmapSrv::Request> req,
             std::shared_ptr<CostmapSrv::Response> res) {
        return CostmapCallback(req, res);
      });

  // Load extrinsics, calibrations, FOV mask, etc.
  fov_mask_ = createTrapezoidalFovMask(192, 192);
  if (torch::cuda::is_available()) {
    fov_mask_ = fov_mask_.cuda();
  }

  // Publishers
  if (config["output_topics"] && config["output_topics"].IsSequence()) {
    const auto& topics_list = config["output_topics"];
    for (size_t i = 0; i < topics_list.size(); ++i) {
      const auto& topic_cfg = topics_list[i];
      std::string topic_name = topic_cfg["topic_name"].as<std::string>("");
      std::string topic_type = topic_cfg["type"].as<std::string>("");
      std::string topic_key = topic_cfg["key"].as<std::string>("");

      // Store configurations
      predictions_[topic_key] = {topic_type, topic_key, topic_name};

      if (topic_type == "depth") {
        depth_publishers_[topic_key] =
            node_->create_publisher<sensor_msgs::msg::Image>(topic_name, 10);
      } else if (topic_type == "traversability") {
        traversability_publisher_ =
            node_->create_publisher<sensor_msgs::msg::Image>(topic_name, 10);
      } else if (topic_type == "semantic_elevation") {
        semantic_elevation_publisher_ =
            node_->create_publisher<sensor_msgs::msg::Image>(topic_name, 10);
      }
    }
  }

  // Initialize semantic map queue
  int history_window = config["viz_params"]["history_window"].as<int>(5);
  int static_output_dim =
      config["model_params"]["static_output_dim"].as<int>(3);
  semantic_history_ =
      torch::zeros({history_window, static_output_dim, 192, 192}).cuda();
  viz_3d_ = config["viz_params"]["viz_3d"].as<bool>(false);

  LOG_INFO("CresteNode constructor finished.");
}

void CresteNode::TFCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
  // For each transform in the TF message
  for (auto& transform_stamped : msg->transforms) {
    const std::string& parent = transform_stamped.header.frame_id;
    const std::string& child = transform_stamped.child_frame_id;

    // Now loop through each camera; see if it matches their src->target frames
    for (auto& cam_ptr : cameras_) {
      if (!cam_ptr->enabled) continue;
      if (cam_ptr->tf_frames.size() < 2) continue;  // must have [src, target]
      const std::string& src = cam_ptr->tf_frames[0];
      const std::string& tgt = cam_ptr->tf_frames[1];

      if (!((parent == src && child == tgt) ||
            (parent == tgt && child == src))) {
        continue;
      }

      const auto& tf = transform_stamped.transform;
      Eigen::Quaternionf q(tf.rotation.w, tf.rotation.x, tf.rotation.y,
                           tf.rotation.z);
      Eigen::Matrix3f R = q.toRotationMatrix();
      Eigen::Vector3f t(tf.translation.x, tf.translation.y, tf.translation.z);

      Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
      M.block<3, 3>(0, 0) = R;
      M.block<3, 1>(0, 3) = t;
      LOG_INFO("Transform [%s] -> [%s]", parent.c_str(), child.c_str());
      if (parent == src && child == tgt) {
        // It's a direct transform: src->tgt
        cam_ptr->extrinsics = M;
      } else if (parent == tgt && child == src) {
        // It's the inverse transform: we want src->tgt, so invert it
        cam_ptr->extrinsics = M.inverse();
      } else {
        LOG_ERROR("Transform [%s] -> [%s] not found in camera frames [%s]",
                  parent.c_str(), child.c_str(), cam_ptr->tf_frames[0].c_str());
      }
    }
  }
}

void CresteNode::PointCloudCallback(const PointCloud2ConstPtr msg) {
  auto& cloud = *(cloud_);
  std::lock_guard<std::mutex> lk(cloud.queue_mutex);
  // LOG_INFO("PointCloud callback");
  while (int(cloud.cloud_queue.size()) >= cloud.queue_size) {
    LOG_WARN("Cloud queue full. Dropping old data.");
    cloud.cloud_queue.pop();
  }

  cloud.cloud_queue.push(msg);
}

void CresteNode::CameraImageCallback(const CompressedImageConstPtr msg,
                                     size_t cam_idx) {
  auto& cam = *(cameras_[cam_idx]);
  std::lock_guard<std::mutex> lk(cam.queue_mutex);
  // LOG_INFO("Camera [%s] with timestamp [%lf] callback",
  // cam.camera_name.c_str(),
  //          msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);
  // Prune old data if queue is full
  while (int(cam.image_queue.size()) >= cam.queue_size) {
    LOG_WARN("Camera [%s] queue full. Dropping old data.",
             cam.camera_name.c_str());
    cam.image_queue.pop();
  }

  cam.image_queue.push(msg);
}

void CresteNode::CameraInfoCallback(const CameraInfoConstPtr msg,
                                    size_t cam_idx) {
  auto& cam = *(cameras_[cam_idx]);
  cam.camera_info = *msg;
  if (!cam.has_rectification) {
    cv::Size size(msg->width, msg->height);

    cv::Mat Kd(3, 3, CV_64F, (void*)msg->k.data());
    cv::Mat Dd(1, msg->d.size(), CV_64F, (void*)msg->d.data());
    cv::Mat Rd(3, 3, CV_64F, (void*)msg->r.data());
    cv::Mat Pd(3, 4, CV_64F, (void*)msg->p.data());

    cv::Mat Kf, Df, Rf, Pf;
    Kd.convertTo(Kf, CV_32F);
    Dd.convertTo(Df, CV_32F);
    Rd.convertTo(Rf, CV_32F);
    Pd.convertTo(Pf, CV_32F);

    // Store the new camera matrix
    cam.new_K = cv::getOptimalNewCameraMatrix(Kf, Df, size, 0, size);

    cv::initUndistortRectifyMap(Kf, Df, Rf, Pf, size, CV_32FC1, cam.map1,
                                cam.map2);
    cam.has_rectification = true;
    // LOG_INFO("Initialized rectification for camera [%s]",
    //          cam.camera_name.c_str());
  }
}

void CresteNode::run() {
  static bool kDebug = FLAGS_v > 1;

  // 0) Drop old data
  auto now = GET_TIME;
  int64_t now_ns = to_nanoseconds(now);
  int64_t cutoff_ns = now_ns - (stale_cutoff_ms_ * 1000000LL);  // 1000 ms

  if (stale_cutoff_ms_ > 0.0) {
    if (kDebug) LOG_INFO("Dropping old data. Cutoff time: %lf seconds", stale_cutoff_ms_);
    // Clear old camera messages
    for (auto& cam_ptr : cameras_) {
      if (!cam_ptr->enabled) continue;
      std::lock_guard<std::mutex> lk(cam_ptr->queue_mutex);
      while (!cam_ptr->image_queue.empty()) {
        auto front_msg = cam_ptr->image_queue.front();
        int64_t front_ts_ns = to_nanoseconds(front_msg->header.stamp);
        // LOG_INFO("Camera [%s] front_ts_ns: %ld", cam_ptr->camera_name.c_str(),
        //          front_ts_ns);
        // LOG_INFO("Now_seconds: %lf", now_ns * 1e-9);
        if (front_ts_ns < cutoff_ns) {
          if (kDebug) {
            LOG_INFO("Dropping old camera [%s] frame",
                    cam_ptr->camera_name.c_str());
          }
          cam_ptr->image_queue.pop();
        } else {
          break;
        }
      }
    }

    // Cloud queue
    if (cloud_ && cloud_->enabled) {
      std::lock_guard<std::mutex> lk(cloud_->queue_mutex);
      while (!cloud_->cloud_queue.empty()) {
        auto front_msg = cloud_->cloud_queue.front();
        int64_t front_ts_ns = to_nanoseconds(front_msg->header.stamp);
        if (front_ts_ns < cutoff_ns) {
          cloud_->cloud_queue.pop();
        } else {
          break;
        }
      }
    }
  } else {
    if (kDebug) LOG_INFO("Cutoff time not set. Not dropping old data.");
  }

  // 1) Gather front images
  std::vector<CompressedImageConstPtr> cam_images(cameras_.size(), nullptr);
  bool all_cameras_ok = true;
  std::vector<int64_t> camera_ts_ns(cameras_.size(), 0);
  for (size_t i = 0; i < cameras_.size(); i++) {
    auto& cam = *(cameras_[i]);
    if (!cam.enabled) continue;
    std::lock_guard<std::mutex> lk(cam.queue_mutex);
    if (cam.image_queue.empty() || !cam.has_rectification) {
      all_cameras_ok = false;
      if (kDebug) {
        LOG_INFO("Camera [%s] not ready", cam.camera_name.c_str());
        LOG_INFO("Queue size: %ld", cam.image_queue.size());
        LOG_INFO("Rectification: %d", cam.has_rectification);
      }
      break;
    }
    cam_images[i] = cam.image_queue.front();
    camera_ts_ns[i] = to_nanoseconds(cam_images[i]->header.stamp);
  }

  if (!all_cameras_ok) return;
  if (kDebug) LOG_INFO("All cameras ready");

  // 2) Check cloud if needed
  PointCloud2ConstPtr front_cloud = nullptr;
  int64_t cloud_ts = 0;
  if (cloud_ && cloud_->enabled) {
    std::lock_guard<std::mutex> lk(cloud_->queue_mutex);
    if (cloud_->cloud_queue.empty()) return;
    front_cloud = cloud_->cloud_queue.front();
    cloud_ts = to_nanoseconds(front_cloud->header.stamp);
  }

  // 3) Compare timestamps
  int64_t min_ts = INT64_MAX, max_ts = INT64_MIN;
  for (auto ts : camera_ts_ns) {
    if (ts == 0) continue;
    min_ts = std::min(min_ts, ts);
    max_ts = std::max(max_ts, ts);
  }
  if (cloud_ && cloud_->enabled && front_cloud) {
    min_ts = std::min(min_ts, cloud_ts);
    max_ts = std::max(max_ts, cloud_ts);
  }

  int64_t kSyncThresholdNs = sync_cutoff_ms_ * 1000000LL;  // ms->ns
  if ((max_ts - min_ts) > kSyncThresholdNs) {
    // Drop oldest
    for (size_t i = 0; i < cameras_.size(); ++i) {
      if (!cameras_[i]->enabled) continue;
      if (camera_ts_ns[i] == min_ts) {
        std::lock_guard<std::mutex> lk(cameras_[i]->queue_mutex);
        cameras_[i]->image_queue.pop();
        LOG_WARN("Dropping old camera [%s] frame",
                 cameras_[i]->camera_name.c_str());
      }
    }
    if (cloud_ && cloud_->enabled && cloud_ts == min_ts) {
      std::lock_guard<std::mutex> lk(cloud_->queue_mutex);
      cloud_->cloud_queue.pop();
      RCLCPP_WARN(node_->get_logger(), "Dropping old cloud message");
    }
    return;
  }

  // We have matched data => pop them
  for (auto& cam_ptr : cameras_) {
    if (!cam_ptr->enabled) continue;
    std::lock_guard<std::mutex> lk(cam_ptr->queue_mutex);
    if (!cam_ptr->image_queue.empty()) cam_ptr->image_queue.pop();
  }
  if (cloud_ && cloud_->enabled) {
    std::lock_guard<std::mutex> lk(cloud_->queue_mutex);
    if (!cloud_->cloud_queue.empty()) cloud_->cloud_queue.pop();
  }

  // Store in latest
  {
    std::lock_guard<std::mutex> lk(latest_msg_mutex_);
    latest_camera_msgs_ = cam_images;
    latest_cloud_msg_ = front_cloud;
  }

  if (kDebug) {
    LOG_INFO("Running inference");
  }
  // Launch inference
  inference();
}

void CresteNode::inference() {
  // Acquire snapshots
  if (!model_) return;
  static bool kDebug = FLAGS_v > 0;

  std::vector<CompressedImageConstPtr> camera_imgs;
  PointCloud2ConstPtr cloud_msg;
  {
    std::lock_guard<std::mutex> lk(latest_msg_mutex_);
    if (cloud_ && cloud_->enabled && !latest_cloud_msg_) {
      LOG_INFO("Inference(): Cloud not ready");
      return;
    }
    for (size_t i = 0; i < cameras_.size(); ++i) {
      if (cameras_[i]->enabled && !latest_camera_msgs_[i]) {
        LOG_INFO("Inference(): Camera [%s] not ready",
                 cameras_[i]->camera_name.c_str());
        return;
      }
    }
    camera_imgs = latest_camera_msgs_;
    cloud_msg = latest_cloud_msg_;
  }

  auto start = GET_TIME;
  auto inputs = ProcessInputs(cloud_msg, camera_imgs);
  auto end = GET_TIME;

  if (kDebug) LOG_INFO("Inference(): Input processing time: %f seconds", TIME_DIFF(start, end));

  if (inputs.empty()) {
    LOG_ERROR("Inference(): Failed to process inputs");
    return;
  }

  // Inference
  start = GET_TIME;
  auto tensor_map = model_->forward(inputs);
  end = GET_TIME;
  if (kDebug) {
    LOG_INFO("Inference(): Model Inference time: %f seconds",
           TIME_DIFF(start, end));
  }
  // 3) Post-processing (construct cost map, etc.)
  start = GET_TIME;

  // if (viz_3d_) {
  //   tensor_map["elevation"] = output.at("elevation_preds");
  //   tensor_map["static_sem"] = output.at("inpainting_sam_preds");
  //   tensor_map["dynamic_sem"] = output.at("inpainting_sam_dynamic_preds");
  // }
  // tensor_map["depth_preds"] = output.at("depth_preds_metric");
  // tensor_map["traversability"] = output.at("traversability_preds_full");

  torch::Tensor cost_map = -tensor_map["traversability_preds_full"];
  float min_cost = cost_map.min().item<float>();
  float max_cost = cost_map.max().item<float>();

  if (max_cost > min_cost) {
    cost_map = (cost_map - min_cost) / (max_cost - min_cost);
  } else {
    LOG_WARN("Cost map uniform. Setting to 1.0");
    cost_map.fill_(1.0f);
  }
  cost_map.clamp_(0.0f, 244.0f / 255.0f);
  cost_map.masked_fill_(fov_mask_.logical_not(), 1.0f);
  tensor_map["traversability_cost"] = cost_map;

  if (viz_3d_) {
    tensor_map["static_sem"] = tensor_map["static_sem"] * fov_mask_;
    tensor_map["dynamic_sem"] = tensor_map["dynamic_sem"] * fov_mask_;
    tensor_map["elevation"] = tensor_map["elevation"] * fov_mask_;
  }

  // print keys in tensor_map
  // LOG_INFO("Tensor map keys:");
  // for (const auto& pair : tensor_map) {
  //   LOG_INFO("Key: %s", pair.first.c_str());
  // }

  for (const auto& pred_pair : predictions_) {
    const PredictionHandler& pred = pred_pair.second;
    // Use pred.key to fetch the corresponding tensor.
    if (tensor_map.find(pred.key) == tensor_map.end()) {
      LOG_INFO("Prediction key [%s] not found in output.", pred.key.c_str());
      continue;
    }
    torch::Tensor pred_tensor = tensor_map[pred.key];

    // Based on the type, select the appropriate processing function.
    // For example, if type "depth", process as a depth image;
    // if "traversability", process accordingly; etc.
    // LOG_INFO("Processing prediction type [%s] for key [%s]",
    // pred.type.c_str(),
    //          pred.key.c_str());
    if (pred.type == "depth") {
      // For depth, use the corresponding publisher from depth_publishers_.
      if (depth_publishers_.find(pred.key) != depth_publishers_.end()) {
        PublishCompletedDepth(tensor_map, pred.key,
                              depth_publishers_[pred.key]);
      } else {
        RCLCPP_WARN(node_->get_logger(),
                    "No publisher found for depth key [%s]", pred.key.c_str());
      }
    } else if (pred.type == "traversability") {
      // Use the traversability publisher.
      PublishTraversability(tensor_map, pred.key, traversability_publisher_);
    } else if (pred.type == "semantic_elevation") {
      // PublishSemanticElevation(tensor_map, pred.key,
      //                          semantic_elevation_publisher_);
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "Unknown prediction type [%s] for key [%s]",
                  pred.type.c_str(), pred.key.c_str());
    }
  }

  // Convert costmap to CPU, rotate, shift, etc.
  // {
  //   torch::Tensor cost_map_8u = cost_map.detach()
  //                                   .cpu()
  //                                   .mul(255.0f)
  //                                   .clamp_(0, 255)
  //                                   .to(torch::kU8)
  //                                   .squeeze();
  //   cost_map_8u = cost_map_8u.squeeze();  // [B, 1, H, W] -> [H, W]

  //   // 2) Wrap cost_map_8u in an OpenCV Mat (CV_8UC1).
  //   int rows = cost_map_8u.size(0);
  //   int cols = cost_map_8u.size(1);
  //   cv::Mat cost_map_mat(rows, cols, CV_8UC1,
  //   cost_map_8u.data_ptr<uint8_t>());

  //   // 3) Build affine transform: rotate 4° clockwise & shift 10 px left.
  //   //    - Use angle = -4 for clockwise rotation in OpenCV.
  //   //    - Then adjust the last column of the rotation matrix to shift left.
  //   cv::Point2f center(cols / 2.0f, rows / 2.0f);
  //   cv::Mat rot = cv::getRotationMatrix2D(center, map_to_base_link_[2], 1.0);
  //   // Decrease the X translation by 10.0 to shift left
  //   rot.at<double>(0, 2) += map_to_base_link_[0];
  //   // Add Y Translation
  //   rot.at<double>(1, 2) += map_to_base_link_[1];

  //   // 4) warpAffine to rotate + shift
  //   cv::Mat rotated_8u;
  //   cv::warpAffine(cost_map_mat, rotated_8u, rot, cost_map_mat.size(),
  //                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue*/
  //                  255);
  //   LOG_INFO("Rotated cost map.");
  //   // 5) Convert rotated CV_8UC1 back to float [0,1] in a torch::Tensor
  //   torch::Tensor rotated_tensor = torch::from_blob(
  //       rotated_8u.data, {rotated_8u.rows, rotated_8u.cols}, torch::kUInt8);
  //   // clone() so we own the memory (from_blob() just references existing
  //   data) rotated_tensor =
  //   rotated_tensor.clone().to(torch::kFloat32).div_(255.0f);
  //   rotated_tensor = rotated_tensor.unsqueeze(0).unsqueeze(0);  // [1,1,H,W]
  //   LOG_INFO("Converted rotated cost map to tensor.");
  //   // 6) Overwrite the costmap in the tensor_map
  //   tensor_map["traversability_cost"] = rotated_tensor;
  // }

  // Store the model outputs
  {
    std::lock_guard<std::mutex> lock(model_outputs_mutex_);
    model_outputs_ =
        std::make_shared<std::unordered_map<std::string, torch::Tensor>>(
            tensor_map);
  }
  if (kDebug) {
    LOG_INFO("Stored model outputs.");

    end = GET_TIME;
    LOG_INFO("Map Processing time: %f seconds", TIME_DIFF(start, end));
  }
  // // Publish model predictions
  // PublishCompletedDepth(tensor_map, "depth_preds", depth_publisher_);
  // PublishTraversability(tensor_map, "traversability_cost",
  //                       traversability_publisher_);
  // LOG_INFO("Published model predictions.");
  // if (viz_3d_) {
  //   // Prepare 3D visualization
  //   printf("Computing PCA for semantic elevation map...");
  //   semantic_history_[semantic_history_idx_] =
  //       tensor_map["static_sem"].index({0});  // [1, F, H, W] -> [F, H, W]

  //   const auto& sem_rgb_window = computePCA(semantic_history_);  // [B, 3, H,
  //   W] auto sem_rgb_th =
  //       sem_rgb_window[semantic_history_idx_].unsqueeze(0);  // [1, 3, H, W]
  //   sem_rgb_th.masked_fill_(fov_mask_.logical_not().unsqueeze(0),
  //                           -0.5f);  // Mask out FOV
  //   printf("Computed PCA for semantic elevation map.\n");
  //   vector<vector<RGBColor>> elevation_rgb_vec;

  //   const auto& rel_elevation =
  //       tensor_map["elevation"].index({0, 0}).unsqueeze(0).unsqueeze(
  //           0);  // [1, 1, H, W]
  //   printf("rel_elevation ndims: %ld\n", rel_elevation.dim());
  //   printf("rel_elevation dims: %ld, %ld, %ld\n", rel_elevation.size(0),
  //          rel_elevation.size(1), rel_elevation.size(2));
  //   // TensorToColorMap(rel_elevation, elevation_rgb_vec);
  //   TensorToColorMap(sem_rgb_th, elevation_rgb_vec);

  //   vector<vector<float>> elevation_vec;
  //   // Extract only the elevation tensor from [B, 2, H, W] to [H, W]
  //   TensorToVec2D(rel_elevation, elevation_vec);
  //   printf("Converted tensors to vectors.\n");
  //   // Publish the 3D visualization
  //   creste::GenerateAndPublishHeightMapImageStructuredGrid(
  //       elevation_vec, elevation_rgb_vec, semantic_elevation_publisher_);
  //   semantic_history_idx_ =
  //       (semantic_history_idx_ + 1) % semantic_history_.sizes()[0];
  // }
  // // Upsample depth iamge
  // // const int target_height = camera_info_.height;
  // // const int target_width = camera_info_.width;
  // // tensor_map["depth_full_preds"] = UpsampleDepthImage(target_height,
  // // target_width, tensor_map["depth_preds"]);
  // PublishCompletedDepth(tensor_map, "depth_full_preds", depth_publisher_);
}

Eigen::Matrix4f CresteNode::GetPix2PtMatrix(const CameraHandler& cam_handler,
                                            float ds_factor) {
  static bool kDebug = FLAGS_v > 2;

  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> T_cam_world =
      cam_handler.extrinsics;
  Eigen::Matrix3f K =
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          cam_handler.camera_info.k.data())
          .cast<float>();

  // Use camera_info.r, which is stored as a std::array<double,9>.
  Eigen::Matrix3f R =
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          cam_handler.camera_info.r.data())
          .cast<float>();

  if (kDebug) {
    printf("Camera [%s] ds_factor: %f\n", cam_handler.camera_name.c_str(),
           ds_factor);
    printf("Old Camera Matrix:\n");
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        printf("%8.3f ", K(r, c));
      }
      printf("\n");
    }
  }

  // Downsample K by scaling focal lengths and principal point.
  K(0, 0) *= ds_factor;  // fx
  K(1, 1) *= ds_factor;  // fy
  K(0, 2) *= ds_factor;  // cx
  K(1, 2) *= ds_factor;  // cy

  if (kDebug) {
    printf("New Camera Matrix:\n");
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        printf("%8.3f ", K(r, c));
      }
      printf("\n");
    }
    printf("Camera extrinsics:\n");
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        printf("%8.3f ", T_cam_world(r, c));
      }
      printf("\n");
    }
  }

  // Build a 4x4 T_canon with R.transpose() in the upper-left block.
  Eigen::Matrix4f T_canon = Eigen::Matrix4f::Identity();
  T_canon.block<3, 3>(0, 0) = R.transpose();

  // Build a 4x4 P_pix_cam with K.inverse() in the upper-left block.
  Eigen::Matrix4f P_pix_cam = Eigen::Matrix4f::Identity();
  P_pix_cam.block<3, 3>(0, 0) = K.inverse();

  // Compute the final transform.
  Eigen::Matrix4f P_pix_world = T_cam_world * T_canon * P_pix_cam;

  return P_pix_world;
}

Eigen::Matrix4f CresteNode::GetPt2PixMatrix(const CameraHandler& cam_handler,
                                            float ds_factor) {
  // Create a 3x4 matrix P from camera_info.p (assumed to have 12 elements).
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> T_cam_world =
      cam_handler.extrinsics;
  Eigen::Matrix3f K =
      Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
          cam_handler.new_K.ptr<float>());

  Eigen::Matrix3f R =
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          cam_handler.camera_info.r.data())
          .cast<float>();

  K(0, 0) *= ds_factor;  // fx
  K(1, 1) *= ds_factor;  // fy
  K(0, 2) *= ds_factor;  // cx
  K(1, 2) *= ds_factor;  // cy

  Eigen::Matrix4f T_canon = Eigen::Matrix4f::Identity();
  T_canon.block<3, 3>(0, 0) = R;

  Eigen::Matrix4f P_cam_pix = Eigen::Matrix4f::Identity();
  P_cam_pix.block<3, 3>(0, 0) = K;

  Eigen::Matrix4f result = P_cam_pix * T_canon * T_cam_world.inverse();

  // Cast the result back to float and return.
  return result;
}

std::vector<torch::Tensor> CresteNode::ProcessInputs(
    const PointCloud2ConstPtr& cloud_msg,
    const std::vector<CompressedImageConstPtr>& camera_imgs) {
  static bool kDebug = FLAGS_v > 2;
  // Assummed that all camera imgs are valid
  // Final outputs
  std::vector<torch::Tensor> inputs;

  // CPU float by default
  auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32);

  // Construct pix2pt transform
  std::unordered_map<std::string, torch::Tensor> pix2pt_map;
  std::unordered_map<std::string, torch::Tensor> pt2pix_map;
  for (size_t i = 0; i < cameras_.size(); ++i) {
    auto& cam = *(cameras_[i]);
    auto pix2pt = GetPix2PtMatrix(cam, cam.ds_factor[1]);  // Used for output
    auto pt2pix = GetPt2PixMatrix(cam, cam.ds_factor[0]);  // Used for input

    // Transpose to convert from column to row-major (torch::Tensor)
    pix2pt_map[cam.camera_name] =
        torch::from_blob(pix2pt.data(), {4, 4}, options_f32)
            .clone()
            .t();  // [4,4]
    pt2pix_map[cam.camera_name] =
        torch::from_blob(pt2pix.data(), {4, 4}, options_f32)
            .clone()
            .t();  // [4,4]

    if (kDebug) {
      // print pix2pt matrix
      printf("Camera [%s] pix2pt matrix:\n", cam.camera_name.c_str());
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          printf("%8.3f ", pix2pt(r, c));
        }
        printf("\n");
      }
      printf("Map Camera [%s] pt2pix matrix:\n", cam.camera_name.c_str());
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          printf("%8.3f ", pix2pt_map[cam.camera_name][r][c].item<float>());
        }
        printf("\n");
      }
    }
  }

  //===================================================
  // 1) "rgbd" => Single camera + LiDAR => 4 channels
  //===================================================
  if (modality_ == "rgbd") {
    // Expect exactly 1 camera + LiDAR
    if (camera_imgs.empty()) {
      LOG_ERROR("rgbd mode but no camera image found!");
      return inputs;
    }
    if (!cloud_msg) {
      LOG_ERROR("rgbd mode but no LiDAR data found!");
      return inputs;
    }

    // Convert ROS -> PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

    // Decode compressed => BGR
    cv::Mat bgr_image =
        cv::imdecode(cv::Mat(camera_imgs[0]->data), cv::IMREAD_COLOR);
    if (bgr_image.empty()) {
      LOG_ERROR("Empty camera image in rgbd mode!");
      return inputs;
    }

    // BGR -> RGB
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    cv::remap(rgb_image, rgb_image, cameras_[0]->map1, cameras_[0]->map2,
              cv::INTER_LINEAR);

    // Normalize to float [0..1]
    rgb_image.convertTo(rgb_image, CV_32FC3, 1.0 / 255.0);

    // Prepare a depth image
    cv::Mat depth_image(rgb_image.rows, rgb_image.cols, CV_32FC1,
                        std::numeric_limits<float>::lowest());

    // Obtain pt2pix matrix
    auto pt2pix = pt2pix_map[cameras_[0]->camera_name].clone();  // [4,4]
    Eigen::Matrix4f pt2pix_eigen =
        Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(
            pt2pix.data_ptr<float>(), 4, 4);

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
    Eigen::MatrixXf pixels = pt2pix_eigen * points;
    for (size_t i = 0; i < num_points; i++) {
      float w = pixels(2, i);
      if (w <= 0.0f) continue;  // behind camera
      float u = pixels(0, i) / w;
      float v = pixels(1, i) / w;
      float z = w;

      if (u >= 0 && u < rgb_image.cols && v >= 0 && v < rgb_image.rows &&
          z > 0.0f) {
        int px = static_cast<int>(u);
        int py = static_cast<int>(v);
        float& dval = depth_image.at<float>(py, px);
        // Keep the furthest (max Z)
        dval = std::max(dval, z);
      }
    }

    // Replace "lowest()" with 0
    depth_image.setTo(0.0f,
                      depth_image == std::numeric_limits<float>::lowest());

    // Convert depth to mm
    depth_image *= 1000.0f;

    // Merge (R,G,B,Depth) => [H,W,4]
    std::vector<cv::Mat> channels;
    cv::split(rgb_image, channels);   // => [R, G, B]
    channels.push_back(depth_image);  // => Depth
    cv::Mat rgbd_image;
    cv::merge(channels, rgbd_image);  // => [H,W,4]

    // Torch => [4,H,W] => [1,1,4,H,W]
    torch::Tensor rgbd_tensor = torch::from_blob(
        rgbd_image.data, {rgbd_image.rows, rgbd_image.cols, 4}, options_f32);
    rgbd_tensor = rgbd_tensor.permute({2, 0, 1});         // => [4,H,W]
    rgbd_tensor = rgbd_tensor.unsqueeze(0).unsqueeze(0);  // => [1,1,4,H,W]

    // Build pixel2point => [1,1,4,4]
    auto pix2pt_tensor = pix2pt_map[cameras_[0]->camera_name].clone();  // [4,4]
    pix2pt_tensor = pix2pt_tensor.unsqueeze(0).unsqueeze(0);  // => [1,1,4,4]

    // GPU if needed
    if (torch::cuda::is_available()) {
      rgbd_tensor = rgbd_tensor.cuda();
      pix2pt_tensor = pix2pt_tensor.cuda();
    }

    inputs.push_back(rgbd_tensor);
    inputs.push_back(pix2pt_tensor);
  }

  //===========================================================
  // 2) MONOCULAR / STEREO => [1,#cams,3,H,W], no LiDAR depth
  //===========================================================
  else {
    if (camera_imgs.empty()) {
      LOG_ERROR("monocular/stereo mode but no camera images!");
      return inputs;
    }

    // For each camera => rectify => build [3,H,W], then stack
    torch::Tensor p2p_tensor = torch::empty({1, 0, 4, 4}, options_f32);
    for (size_t i = 0; i < camera_imgs.size(); i++) {
      // Decode => BGR
      cv::Mat bgr =
          cv::imdecode(cv::Mat(camera_imgs[i]->data), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        LOG_WARN("Empty camera image at index %lu! Skipping...", i);
        continue;
      }

      // BGR->RGB
      cv::cvtColor(bgr, bgr, cv::COLOR_BGR2RGB);

      // Rectify with camera i's map
      if (i < cameras_.size()) {
        cv::remap(bgr, bgr,
                  cameras_[i]->map1,  // pointer
                  cameras_[i]->map2, cv::INTER_LINEAR);
      }

      // Resize to target shape
      CHECK_GE(cameras_[i]->target_shape.size(), 2)
          << "Target shape must have at least two dimensions.";
      int target_h = cameras_[i]->target_shape[0];
      int target_w = cameras_[i]->target_shape[1];
      cv::resize(bgr, bgr, cv::Size(target_w, target_h));


      // Save image for debugging
      if (kDebug) {
        std::string img_name = "cam_" + std::to_string(i) + ".png";
        cv::imwrite(img_name, bgr);
        LOG_INFO("Saved image: %s", img_name.c_str());
      }

      // Normalize
      bgr.convertTo(bgr, CV_32FC3, 1.0 / 255.0);

      // => Torch
      torch::Tensor cam_tensor =
          torch::from_blob(bgr.data, {bgr.rows, bgr.cols, 3}, options_f32);
      cam_tensor = cam_tensor.permute({2, 0, 1})
                       .unsqueeze(0)
                       .unsqueeze(0)
                       .clone();  // => [1,1,3,H,W]
      if (torch::cuda::is_available()) {
        cam_tensor = cam_tensor.cuda();
      }
      inputs.push_back(cam_tensor);

      const auto& pix2pt =
          pix2pt_map[cameras_[i]->camera_name].clone().unsqueeze(0).unsqueeze(
              0);  // [1,1,4,4]

      p2p_tensor = torch::cat({p2p_tensor, pix2pt}, 1);
      // => [1,#cams,4,4]
      // for (size_t i = 0; i < (size_t)p2p_tensor.dim(); i++) {
      //   LOG_INFO("p2p_tensor dim[%lu]: %ld", i, p2p_tensor.size(i));
      // }
    }

    if (inputs.empty()) {
      LOG_ERROR("No valid images after decoding for monocular/stereo!");
      return inputs;
    }

    if (torch::cuda::is_available()) {
      p2p_tensor = p2p_tensor.cuda();
    }
    inputs.push_back(p2p_tensor);  // => [1,#cams,4,4]
  }
  // Return final
  return inputs;
}

bool CresteNode::CostmapCallback(const std::shared_ptr<CostmapSrv::Request> req,
  std::shared_ptr<CostmapSrv::Response> res) {
  // 1) Retrieve cached model outputs
  std::shared_ptr<std::unordered_map<std::string, torch::Tensor>> model_outputs;
  {
    std::lock_guard<std::mutex> lock(model_outputs_mutex_);
    model_outputs = model_outputs_;
  }

  if (!model_outputs) {
    LOG_ERROR("Model outputs not available for costmap service.");
    res->success.data = false;
    return true;  // Return from the service callback
  }

  // 2) Get the traversability map (8-bit cost image) from the model outputs
  if (model_outputs->find("traversability_cost") == model_outputs->end()) {
    LOG_ERROR("Key 'traversability_cost' not found in model outputs!");
    res->success.data = false;
    return true;
  }
  const auto& traversability_map = model_outputs->at("traversability_cost");

  // 3) Create an 8-bit cost map image using cv_bridge
  cv_bridge::CvImage cv_img;
  cv_img.header.stamp = node_->now();                // Use ROS 2 node clock
  cv_img.header.frame_id = "base_link";
  cv_img.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
  cv_img.image = TensorToMat(traversability_map);

  // 4) Convert that to a ROS 2 Image message
  sensor_msgs::msg::Image out_img;
  cv_img.toImageMsg(out_img);

  // 5) Populate the service response
  res->costmap = out_img;
  res->success.data = true;

  LOG_INFO("Handled costmap service request successfully.");
  return true;
}


}  // namespace creste