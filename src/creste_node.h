#ifndef CRESTE_NODE_H
#define CRESTE_NODE_H

#include <cv_bridge/cv_bridge.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "amrl_msgs/srv/costmap_srv.hpp"

using CostmapSrv = amrl_msgs::srv::CostmapSrv;
using Image = sensor_msgs::msg::Image;
using CompressedImage = sensor_msgs::msg::CompressedImage;
using CompressedImageConstPtr = sensor_msgs::msg::CompressedImage::SharedPtr;
using CameraInfo = sensor_msgs::msg::CameraInfo;
using CameraInfoConstPtr = sensor_msgs::msg::CameraInfo::SharedPtr;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using PointCloud2ConstPtr = sensor_msgs::msg::PointCloud2::SharedPtr;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using Path = nav_msgs::msg::Path;
using TransformStamped = geometry_msgs::msg::TransformStamped;
using NodeHandleType = rclcpp::Node::SharedPtr;

#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "creste.h"
#include "planner.h"
#include "utils.h"
#include "visualization.h"

namespace creste {

/** Handler for a single camera. */
struct CameraHandler {
  bool enabled = false;
  std::string camera_name;
  std::string image_topic;
  std::string info_topic;
  std::vector<int> target_shape;       // e.g. [H, W]
  std::vector<float> ds_factor;        // e.g. [1, 1]
  std::vector<std::string> tf_frames;  // e.g. [src_frame, target_frame]
  Eigen::Matrix4f extrinsics =
      Eigen::Matrix4f::Identity();  // camera extrinsics

  rclcpp::Subscription<CompressedImage>::SharedPtr image_sub;
  rclcpp::Subscription<CameraInfo>::SharedPtr info_sub;

  // Queues
  std::queue<CompressedImageConstPtr> image_queue;
  std::mutex queue_mutex;
  int queue_size;

  // Camera info & rectification data
  CameraInfo camera_info;
  cv::Mat new_K;  // new camera matrix
  bool has_rectification{false};
  cv::Mat map1, map2;  // rectification maps
};

/** Handler for LiDAR/pointcloud data. */
struct CloudHandler {
  bool enabled = false;
  std::string cloud_topic;
  std::string tf_frame;

  rclcpp::Subscription<PointCloud2>::SharedPtr cloud_sub;

  // Queues
  std::queue<PointCloud2ConstPtr> cloud_queue;
  std::mutex queue_mutex;
  int queue_size;
};

struct PredictionHandler {
  std::string type;
  std::string key;
  std::string topic;
};

class CresteNode {
 public:
  CresteNode(const std::string& config_path, const std::string& weights_path,
             NodeHandleType node);

  void run();
  void inference();

 private:
  // === Callbacks ===
  void PointCloudCallback(const PointCloud2ConstPtr msg);
  void CameraImageCallback(const CompressedImageConstPtr msg,
                           size_t sensor_idx);
  void CameraInfoCallback(const CameraInfoConstPtr msg, size_t sensor_idx);

  // NEW: /tf callback
  void TFCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);

  // === Helper functions ===
  bool CostmapCallback(const std::shared_ptr<CostmapSrv::Request> req,
                       std::shared_ptr<CostmapSrv::Response> res);

  void LoadCalibParams(const YAML::Node& config);

  /**
   * Gathers matched camera + cloud data, builds model input,
   * uses the transforms from tf_frames if needed
   */
  std::vector<torch::Tensor> ProcessInputs(
      const PointCloud2ConstPtr& cloud_msg,
      const std::vector<CompressedImageConstPtr>& camera_imgs);

  bool is_cell_visible(int i, int j, int grid_height, int grid_width);

  Eigen::Matrix4f GetPix2PtMatrix(const CameraHandler& cam_handler,
                                  float ds_factor);

  Eigen::Matrix4f GetPt2PixMatrix(const CameraHandler& cam_handler,
                                  float ds_factor);

 private:
  NodeHandleType node_;

  // Service
  rclcpp::Service<CostmapSrv>::SharedPtr costmap_service_;

  // Subscriptions
  rclcpp::Subscription<PointCloud2>::SharedPtr pointcloud_subscriber_;

  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_subscriber_;
  // Store transforms in a map: child_frame -> TransformStamped
  std::mutex tf_mutex_;
  std::unordered_map<std::string, TransformStamped> tf_map_;

  // Publishers
  std::unordered_map<std::string, rclcpp::Publisher<Image>::SharedPtr>
      depth_publishers_;
  rclcpp::Publisher<Image>::SharedPtr traversability_publisher_;
  rclcpp::Publisher<Image>::SharedPtr semantic_elevation_publisher_;

  // Services, planners
  std::shared_ptr<CarrotPlanner> carrot_planner_;

  // LiDAR / Cloud
  std::unique_ptr<CloudHandler> cloud_;

  // Cameras
  std::vector<std::unique_ptr<CameraHandler>> cameras_;

  // Cut off all messages older than this relative time offset
  double stale_cutoff_ms_;
  double sync_cutoff_ms_;

  // For the final matched messages each cycle:
  PointCloud2ConstPtr latest_cloud_msg_;
  std::vector<CompressedImageConstPtr> latest_camera_msgs_;
  std::mutex latest_msg_mutex_;

  // Model / inference
  std::shared_ptr<CresteModel> model_;
  std::mutex model_outputs_mutex_;
  std::shared_ptr<std::unordered_map<std::string, torch::Tensor>>
      model_outputs_;
  std::string modality_;
  std::unordered_map<std::string, PredictionHandler> predictions_;

  torch::Tensor fov_mask_;

  torch::Tensor semantic_history_;
  int semantic_history_idx_{0};
  bool viz_3d_{false};

  std::vector<float> map_to_base_link_;
};

}  // namespace creste

#endif  // CRESTE_NODE_H
