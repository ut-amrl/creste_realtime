#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using std::placeholders::_1;

namespace lsmap {
class LSMapNode : public rclcpp::Node {
public:
    LSMapNode() : Node("lsmap_node") {
        RCLCPP_INFO(this->get_logger(), "Hello, world!");
        // Subscription to PointCloud2 topic
        pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "pointcloud_topic", 10, std::bind(&LSMapNode::pointcloud_callback, this, _1));

        // Subscription to Image topic
        image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
        "image_topic", 10, std::bind(&LSMapNode::image_callback, this, _1));

        // Publisher for Image topic
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("output_image_topic", 10);
    }
private:
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message");

        // Convert ROS2 PointCloud2 message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);

        // Process the PCL point cloud (example: just log the size)
        RCLCPP_INFO(this->get_logger(), "PointCloud size: %lu", pcl_cloud.size());
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received Image message");
        
        // Convert ROS2 Image to OpenCV image
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Process the OpenCV image
        cv::Mat image = cv_ptr->image;

        // For example, convert to grayscale
        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

        // Convert OpenCV image back to ROS2 Image and publish
        auto output_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", gray_image).toImageMsg();
        image_publisher_->publish(*output_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
};
} // namespace lsmap