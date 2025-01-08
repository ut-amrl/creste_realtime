#ifndef LSMAP_NODE_H
#define LSMAP_NODE_H

#include <omp.h>
#include <torch/script.h>
#include <torch/torch.h>

// AMRL Service messages
#include "amrl_msgs/CarrotPlannerSrv.h"

// ROS 1 headers
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "glog/logging.h"
#include "yaml-cpp/yaml.h"

// STL / OpenCV
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

// Local headers
#include "lsmap.h"
#include "planner.h"
#include "shared/util/timer.h"
#include "utils.h"

using amrl_msgs::CarrotPlannerSrv;

namespace lsmap {

class LSMapNode {
 public:
  LSMapNode(const std::string& config_path, const std::string& weights_path);

  /// \brief Main processing function called periodically in main()
  void run();
  void inference();

 private:
  // === Callbacks ===
  void PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void ImageCallback(const sensor_msgs::ImageConstPtr& msg);
  void CameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);

  // === Helper functions ===
  bool CarrotPlannerCallback(CarrotPlannerSrv::Request& req,
                             CarrotPlannerSrv::Response& res);
  void LoadCalibParams(const YAML::Node& config);

  bool is_cell_visible(const int i, const int j, const int grid_height,
                       const int grid_width);

  // Projection function for combining point cloud & image
  std::tuple<torch::Tensor, torch::Tensor> ProcessInputs(
      const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
      const sensor_msgs::ImageConstPtr& image_msg);

 private:
  // === ROS 1 NodeHandle ===
  ros::NodeHandle nh_;

  // === Services ===
  ros::ServiceServer carrot_planner_service;

  // === Subscribers ===
  ros::Subscriber pointcloud_subscriber_;
  ros::Subscriber image_subscriber_;
  ros::Subscriber camera_info_subscriber_;
  ros::Subscriber p2p_subscriber_;

  // === Publishers ===
  ros::Publisher image_publisher_;
  ros::Publisher depth_publisher_;
  ros::Publisher traversability_publisher_;

  // === ROS messages ===
  sensor_msgs::CameraInfo camera_info_;
  CalibInfo pt2pix_, pix2pt_;

  // Whether we have computed our remap matrices yet
  bool has_rectification_{false};

  // Remap matrices for OpenCV
  cv::Mat map1_, map2_;

  // === Buffers/queues ===
  std::queue<sensor_msgs::PointCloud2ConstPtr> cloud_queue_;
  std::queue<sensor_msgs::ImageConstPtr> image_queue_;
  std::mutex queue_mutex_;

  // === Misc ===
  sensor_msgs::PointCloud2ConstPtr latest_cloud_msg_;
  sensor_msgs::ImageConstPtr latest_image_msg_;
  std::mutex latest_msg_mutex_;

  // === Model Inference ===
  std::shared_ptr<lsmap::LSMapModel> model_;
  std::mutex model_outputs_mutex_;
  std::shared_ptr<std::unordered_map<std::string, torch::Tensor>>
      model_outputs_;
  torch::Tensor fov_mask_;

  // === Planners ===
  std::shared_ptr<CarrotPlanner> carrot_planner_;
  std::vector<PathPoint> latest_path_;
  std::vector<std::vector<float>> traversability_vec_;
};
}  // namespace lsmap

#endif  // LSMAP_NODE_H
