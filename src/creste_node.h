#ifndef CRESTE_NODE_H
#define CRESTE_NODE_H

#include <cv_bridge/cv_bridge.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

// Conditional includes for ROS1 vs. ROS2
#ifdef ROS1
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>

#include "amrl_msgs/CostmapSrv.h"

// Type aliases for ROS1
using CostmapSrv = amrl_msgs::CostmapSrv;
using Image = sensor_msgs::Image;
using CompressedImage = sensor_msgs::CompressedImage;
using CompressedImagePtr = sensor_msgs::CompressedImagePtr;
using CompressedImageConstPtr = sensor_msgs::CompressedImageConstPtr;
using CameraInfo = sensor_msgs::CameraInfo;
using CameraInfoConstPtr = sensor_msgs::CameraInfoConstPtr;
using PointCloud2 = sensor_msgs::PointCloud2;
using PointCloud2ConstPtr = sensor_msgs::PointCloud2ConstPtr;
using PoseStamped = geometry_msgs::PoseStamped;
using Path = nav_msgs::Path;
// etc. for other message types if needed

using NodeHandleType = ros::NodeHandle;

#else  // ROS2
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "amrl_msgs/srv/costmap_srv.hpp"

// Type aliases for ROS2
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

using NodeHandleType = rclcpp::Node::SharedPtr;
#endif

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

struct CameraHandler {
  bool enabled = false;
  std::string camera_name;  // e.g. "left" or "right"
  std::string image_topic;
  std::string info_topic;

#ifdef ROS1
  ros::Subscriber image_sub;
  ros::Subscriber info_sub;
#else
  rclcpp::Subscription<CompressedImage>::SharedPtr image_sub;
  rclcpp::Subscription<CameraInfo>::SharedPtr info_sub;
#endif

  // Queues
  std::queue<CompressedImageConstPtr> image_queue;
  std::mutex queue_mutex;  // Protects image_queue

  // Camera info & rectification data
  CameraInfo camera_info;
  bool has_rectification{false};
  cv::Mat map1, map2;  // rectification maps
};

class CresteNode {
 public:
#ifdef ROS1
  CresteNode(const std::string& config_path, const std::string& weights_path,
             const ros::NodeHandle& nh = ros::NodeHandle());
#else
  // For ROS2, we pass a NodeHandleType which is rclcpp::Node::SharedPtr
  CresteNode(const std::string& config_path, const std::string& weights_path,
             NodeHandleType node);
#endif

  void run();
  void inference();

 private:
  // === Callbacks ===
  void PointCloudCallback(const PointCloud2ConstPtr msg);

  void CameraImageCallback(const CompressedImageConstPtr msg, size_t cam_idx);
  void CameraInfoCallback(const CameraInfoConstPtr msg, size_t cam_idx);

  // === Helper functions ===
#ifdef ROS1
  bool CostmapCallback(amrl_msgs::CostmapSrv::Request& req,
                       amrl_msgs::CostmapSrv::Response& res);
#else
  // ROS2 service callback signature typically uses shared_ptr for
  // request/response
  bool CostmapCallback(const std::shared_ptr<CostmapSrv::Request> req,
                       std::shared_ptr<CostmapSrv::Response> res);
#endif

  void LoadCalibParams(const YAML::Node& config);

  std::tuple<torch::Tensor, torch::Tensor> ProcessInputs(
      const PointCloud2ConstPtr& cloud_msg,
      const std::vector<CompressedImageConstPtr>& camera_imgs);

  bool is_cell_visible(int i, int j, int grid_height, int grid_width);

 private:
#ifdef ROS1
  ros::NodeHandle nh_;
  // Service server
  ros::ServiceServer costmap_service_;
  // Subscriber
  ros::Subscriber pointcloud_subscriber_;
  // Publishers
  ros::Publisher depth_publisher_;
  ros::Publisher traversability_publisher_;
  ros::Publisher semantic_elevation_publisher_;
#else
  NodeHandleType node_;
  // Service
  rclcpp::Service<CostmapSrv>::SharedPtr costmap_service_;
  // Subscriber
  rclcpp::Subscription<PointCloud2>::SharedPtr pointcloud_subscriber_;
  // Publishers
  rclcpp::Publisher<Image>::SharedPtr depth_publisher_;
  rclcpp::Publisher<Image>::SharedPtr traversability_publisher_;
  rclcpp::Publisher<Image>::SharedPtr semantic_elevation_publisher_;
#endif

  // Services, planners
  std::shared_ptr<CarrotPlanner> carrot_planner_;

  // LiDAR / Cloud
  bool enable_cloud_{false};
  std::queue<PointCloud2ConstPtr> cloud_queue_;
  std::mutex cloud_queue_mutex_;

  // Cameras
  std::vector<std::unique_ptr<CameraHandler>> cameras_;

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

  CalibInfo pt2pix_, pix2pt_;  // LiDAR <-> camera calibration
  torch::Tensor fov_mask_;

  torch::Tensor semantic_history_;
  int semantic_history_idx_{0};
  bool viz_3d_{false};

  std::vector<float> map_to_base_link_;
};

}  // namespace creste

#endif  // CRESTE_NODE_H
