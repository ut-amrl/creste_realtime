#ifndef LSMAP_NODE_H
#define LSMAP_NODE_H

#include <omp.h>
#include <torch/torch.h>
#include <torch/script.h>

// ROS 1 headers
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float32MultiArray.h>
#include <cv_bridge/cv_bridge.h>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Grid Map
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_cv/grid_map_cv.hpp>

// STL / OpenCV
#include <opencv2/opencv.hpp>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

// Local headers
#include "lsmap.h"
#include "utils.h"

namespace lsmap
{
class LSMapNode
{
public:
  LSMapNode(const std::string& model_path);

  /// \brief Main processing function called periodically in main()
  void run();

private:
  // === Callbacks ===
  void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void image_callback(const sensor_msgs::ImageConstPtr& msg);
  void p2p_callback(const std_msgs::Float32MultiArrayConstPtr& msg);
  void camera_info_callback(const sensor_msgs::CameraInfoConstPtr& msg);

  // === Helper functions ===
  void save_depth_image(const cv::Mat& depthMatrix, const std::string& filename);
//   void tensorToGridMap(const std::unordered_map<std::string, torch::Tensor>& output,
//                        grid_map::GridMap& map);
  bool is_cell_visible(const int i, const int j, const int grid_height, const int grid_width);
  std::tuple<torch::Tensor, torch::Tensor> computePCA(const torch::Tensor& tensor, int components);

  // Projection function for combining point cloud & image
  std::tuple<torch::Tensor, torch::Tensor> projection(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
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
  ros::Publisher grid_map_publisher_;

  // === ROS messages ===
  sensor_msgs::CameraInfo camera_info_;
  std_msgs::Float32MultiArray pixel_to_point_;

  // === Buffers/queues ===
  std::queue<sensor_msgs::PointCloud2ConstPtr> cloud_queue_;
  std::queue<sensor_msgs::ImageConstPtr> image_queue_;
  std::mutex queue_mutex_;

  // === Misc ===
  lsmap::LSMapModel model_;
  std::vector<std::vector<bool>> fov_mask_;
};
}  // namespace lsmap

#endif  // LSMAP_NODE_H
