#include "lsmap_node.h"

namespace lsmap {
    void LSMapNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) 
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message");
        cloud_queue_.push(msg);

        // Convert ROS2 PointCloud2 message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);

        // Process the PCL point cloud (example: just log the size)
        RCLCPP_INFO(this->get_logger(), "PointCloud size: %lu", pcl_cloud.size());
    }

    void LSMapNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        RCLCPP_INFO(this->get_logger(), "Received Image message");
        image_queue_.push(msg);

        // // Convert ROS2 Image to OpenCV image
        // cv_bridge::CvImagePtr cv_ptr;
        // try {
        //     cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        // } catch (cv_bridge::Exception& e) {
        //     RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        //     return;
        // }

        // // Process the OpenCV image
        // cv::Mat image = cv_ptr->image;

        // // For example, convert to grayscale
        // cv::Mat gray_image;
        // cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

        // // Convert OpenCV image back to ROS2 Image and publish
        // auto output_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", gray_image).toImageMsg();
        // image_publisher_->publish(*output_msg);
    }

    void LSMapNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) 
    {
        RCLCPP_INFO(this->get_logger(), "Received CameraInfo message");
        camera_info_ = *msg;
    }

    void LSMapNode::projection(sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,                          
        sensor_msgs::msg::Image::SharedPtr image_msg)
    {
        // Convert ROS2 PointCloud2 message to PCL point cloud

        // Convert ROS2 Image to OpenCV image

        // Project point cloud to image space

        // Create RGBD input image

        // Create point2pixel matrix for inference

    }

    void LSMapNode::run()
    {
        sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
        sensor_msgs::msg::Image::SharedPtr image_msg;
        //1 - Check if there are images and point clouds to process
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (cloud_queue_.empty() || image_queue_.empty()) {
                return;
            }
            auto cloud_time = cloud_queue_.front()->header.stamp;
            auto image_time = image_queue_.front()->header.stamp;
            // Convert to nanoseconds and check if the timestamps are within 100 milliseconds
            int64_t cloud_time_ns = cloud_time.sec * 1e9 + cloud_time.nanosec;
            int64_t image_time_ns = image_time.sec * 1e9 + image_time.nanosec;
            if (std::abs(cloud_time_ns - image_time_ns) < 100 * 1e6) {
                cloud_msg = cloud_queue_.front();
                cloud_queue_.pop();
                image_msg = image_queue_.front();
                image_queue_.pop();
            } else {
                // Drop the older message
                if (cloud_time_ns < image_time_ns) {
                    cloud_queue_.pop();
                } else {
                    image_queue_.pop();
                }
            }
        }
        if (!cloud_msg || !image_msg) {
            return;
        }
        
        //2 - Project point clouds to image space
        RCLCPP_INFO(this->get_logger(), "Projecting point cloud to image space");
        projection(cloud_msg, image_msg);

        //3 - Perform model inference

        //4 - Process elevation and semantic predictions

        //5 - Publish the results

    }


} // namespace lsmap