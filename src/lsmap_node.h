#ifndef LSMAP_NODE_H
#define LSMAP_NODE_H

#include <omp.h>
#include <torch/script.h>
#include <torch/torch.h>

// ROS 1 headers
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

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
#include "utils.h"

namespace lsmap {

struct CalibInfo {
  int rows;
  int cols;
  std::vector<float> data;
};

class LSMapNode {
 public:
  LSMapNode(const std::string& config_path);

  /// \brief Main processing function called periodically in main()
  void run();

 private:
  // === Callbacks ===
  void PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void ImageCallback(const sensor_msgs::ImageConstPtr& msg);
  void CameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);

  // === Helper functions ===
  void LoadCalibInfo(const YAML::Node& config);
  void save_depth_image(const cv::Mat& depthMatrix,
                        const std::string& filename);
  //   void tensorToGridMap(const std::unordered_map<std::string,
  //   torch::Tensor>& output,
  //                        grid_map::GridMap& map);
  bool is_cell_visible(const int i, const int j, const int grid_height,
                       const int grid_width);

  // Projection function for combining point cloud & image
  std::tuple<torch::Tensor, torch::Tensor> projection(
      const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
      const sensor_msgs::ImageConstPtr& image_msg);

 private:
  // === ROS 1 NodeHandle ===
  ros::NodeHandle nh_;

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
  std::shared_ptr<lsmap::LSMapModel> model_;
  torch::Tensor fov_mask_;
};
}  // namespace lsmap

#endif  // LSMAP_NODE_H
