#include <omp.h>
#include <torch/torch.h>
#include <torch/script.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <grid_map_cv/grid_map_cv.hpp>

#include "lsmap.h"

using std::placeholders::_1;

namespace lsmap {
class LSMapNode : public rclcpp::Node {
public:
    LSMapNode(const std::string model_path) : Node("lsmap_node"), model_(model_path, this->get_logger()) {
        RCLCPP_INFO(this->get_logger(), "LSMapNode initialized.");
        // Subscription to PointCloud2 topic
        pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ouster/points", 10, std::bind(&LSMapNode::pointcloud_callback, this, _1));

        // Subscription to Image topic
        image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/stereo/left", 10, std::bind(&LSMapNode::image_callback, this, _1));

        camera_info_subscriber_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info", 10, std::bind(&LSMapNode::camera_info_callback, this, _1));

        // Publisher for Image topic
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/lsmap/rgbd", 10);
        grid_map_publisher_ = this->create_publisher<grid_map_msgs::msg::GridMap>("/lsmap/grid_map", 10);
    }

    void run();
private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    void save_depth_image(const cv::Mat &depthMatrix, const std::string &filename);

    void tensorToGridMap(const torch::Tensor& elevation_tensor, const torch::Tensor& rgb_tensor,  grid_map::GridMap& map);

    std::tuple<torch::Tensor, torch::Tensor> computePCA(const torch::Tensor& tensor, int components);

    std::tuple<torch::Tensor, torch::Tensor> projection(
        sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,                   sensor_msgs::msg::Image::SharedPtr image_msg
    );

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscriber_;
    sensor_msgs::msg::CameraInfo camera_info_;
    
    //Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr grid_map_publisher_;

    lsmap::LSMapModel model_;
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> cloud_queue_;
    std::queue<sensor_msgs::msg::Image::SharedPtr> image_queue_;
    std::mutex queue_mutex_;
};
} // namespace lsmap