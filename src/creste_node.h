#ifndef CRESTE_NODE_H
#define CRESTE_NODE_H

#include <torch/script.h>
#include <torch/torch.h>
#include <cv_bridge/cv_bridge.h>

// AMRL Service messages
#include "amrl_msgs/CostmapSrv.h"

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include "creste.h"
#include "planner.h"
#include "utils.h"
#include "visualization.h"

using amrl_msgs::CostmapSrv;

// Forward declare a struct for storing camera data.
namespace creste {

struct CameraHandler {
  bool enabled = false;
  std::string camera_name;  // e.g. "left" or "right"
  std::string image_topic;
  std::string info_topic;

  // Subscribers
  ros::Subscriber image_sub;
  ros::Subscriber info_sub;

  // Queues
  std::queue<sensor_msgs::CompressedImageConstPtr> image_queue;
  std::mutex queue_mutex;  // Protects image_queue

  // Camera info & rectification data
  sensor_msgs::CameraInfo camera_info;
  bool has_rectification{false};
  cv::Mat map1, map2;  // rectification maps

  // For more advanced usage, you could also keep a separate queue for
  // CameraInfo messages if you expect them to change frequently.
};

class CresteNode {
 public:
  CresteNode(const std::string& config_path, const std::string& weights_path);
  void run();
  void inference();

 private:
  // === Callbacks ===
  void PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

  // Unified camera callbacks:
  void CameraImageCallback(const sensor_msgs::CompressedImageConstPtr& msg,
                           size_t cam_idx);
  void CameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg,
                          size_t cam_idx);

  // === Helper functions ===
  bool CostmapCallback(amrl_msgs::CostmapSrv::Request& req,
                       amrl_msgs::CostmapSrv::Response& res);
  void LoadCalibParams(const YAML::Node& config);

  // Example function that processes camera+LiDAR into model inputs
  std::tuple<torch::Tensor, torch::Tensor> ProcessInputs(
      const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
      const std::vector<sensor_msgs::CompressedImageConstPtr>& camera_imgs);

  // Others...
  bool is_cell_visible(const int i, const int j,
                       const int grid_height, const int grid_width);

 private:
  ros::NodeHandle nh_;

  // Services, planners
  ros::ServiceServer carrot_planner_service;
  std::shared_ptr<CarrotPlanner> carrot_planner_;
  bool CostmapCallbackInternal(amrl_msgs::CostmapSrv::Request& req,
                               amrl_msgs::CostmapSrv::Response& res);

  // === LiDAR / Cloud ===
  bool enable_cloud_{false};
  ros::Subscriber pointcloud_subscriber_;
  std::queue<sensor_msgs::PointCloud2ConstPtr> cloud_queue_;
  std::mutex cloud_queue_mutex_;

  // === Cameras ===
  // Instead of separate left/right variables, we keep them in a vector.
  std::vector<std::unique_ptr<CameraHandler>> cameras_;

  // For the final matched messages each cycle:
  sensor_msgs::PointCloud2ConstPtr latest_cloud_msg_;
  std::vector<sensor_msgs::CompressedImageConstPtr> latest_camera_msgs_;
  std::mutex latest_msg_mutex_;

  // === Publishers ===
  ros::Publisher depth_publisher_;
  ros::Publisher traversability_publisher_;
  ros::Publisher semantic_elevation_publisher_;

  // === Model / Inference ===
  std::shared_ptr<CresteModel> model_;
  std::mutex model_outputs_mutex_;
  std::shared_ptr<std::unordered_map<std::string, torch::Tensor>> model_outputs_;
  std::string modality_;

  // FOV mask, calibration
  // cv::Mat map1_, map2_;  // if you want global rectification data, or you store per-camera
  bool has_rectification_{false};

  CalibInfo pt2pix_, pix2pt_;  // LiDAR <-> camera calibration, etc.
  torch::Tensor fov_mask_;

  // For large 3D visualization / semantic mapping
  torch::Tensor semantic_history_;
  int semantic_history_idx_{0};
  bool viz_3d_{false};

  // Additional config
  std::vector<float> map_to_base_link_;
};

}  // namespace creste

#endif  // CRESTE_NODE_H
