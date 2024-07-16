#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/left/rgbd", 10);
    }

    void run();
private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    void projection(sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,                          
        sensor_msgs::msg::Image::SharedPtr image_msg);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    sensor_msgs::msg::CameraInfo camera_info_;
    lsmap::LSMapModel model_;
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> cloud_queue_;
    std::queue<sensor_msgs::msg::Image::SharedPtr> image_queue_;
    std::mutex queue_mutex_;
};
} // namespace lsmap