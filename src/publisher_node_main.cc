#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <sensor_msgs/msg/camera_info.hpp>

/**
Example usage:
./build/publisher_node /images/5 /point_clouds/5 0 10
**/

namespace fs = boost::filesystem;

namespace LSMap {
class DataPublisher : public rclcpp::Node {
public:
    DataPublisher(const std::string& image_dir, const std::string& pointcloud_dir, int sequence, int start_frame, double rate)
    : Node("data_publisher"), image_dir_(image_dir), pointcloud_dir_(pointcloud_dir), sequence_(sequence), start_frame_(start_frame), rate_(rate), frame_count_(start_frame) {
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/left", 10);
        pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ouster/points", 10);
        camera_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/camera_info", 10);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // Add sequence prefix to image and pc dir
        image_dir_ = image_dir_ + "/" + std::to_string(sequence_);
        pointcloud_dir_ = pointcloud_dir_ + "/" + std::to_string(sequence_);

        // Initialize camera intrinsics (example values, adjust accordingly)
        camera_info_.header.frame_id = "stereo_left";
        camera_info_.width = 612;
        camera_info_.height = 512;
        camera_info_.k = {364.36645, 0.0, 313.01115, 0.0, 364.50625, 265.94215, 0.0, 0.0, 1.0}; 
        camera_info_.p = {
            303.97445679,   -386.56228638,    -44.00563049,     20.68536568,
            210.73265076,     -0.73953837,   -412.18536377,    -46.30334091,
            0.99126321,     -0.01760194,     -0.13071881,     -0.02998733
        };

        timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / rate), std::bind(&DataPublisher::publish_data, this));
    }

private:
    void publish_data() {
        std::stringstream img_ss, pc_ss;
        img_ss << image_dir_ << "/" << "2d_rect_cam0_" << sequence_ << "_" << frame_count_ << ".png";
        pc_ss << pointcloud_dir_ << "/" << "3d_comp_os1_" << sequence_ << "_" << frame_count_ << ".bin";

        fs::path img_path(img_ss.str());
        fs::path pc_path(pc_ss.str());

        if (!fs::exists(img_path) || !fs::exists(pc_path)) {
            RCLCPP_WARN(this->get_logger(), "Frame %d does not exist. Stopping publisher.", frame_count_);
            rclcpp::shutdown();
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Publishing frame %d", frame_count_);
        // Publish image
        cv::Mat image = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (!image.empty()) {
            auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
            img_msg->header.frame_id = "camera_frame";
            img_msg->header.stamp = this->now();
            image_publisher_->publish(*img_msg);

            // Publish camera intrinsics
            camera_info_.header.stamp = this->now();
            camera_info_publisher_->publish(camera_info_);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to load image: %s", img_path.string().c_str());
        }

        // Publish point cloud
        pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
        std::ifstream ifs(pc_path.string(), std::ios::binary);
        if (ifs.is_open()) {
            while (true) {
                pcl::PointXYZRGB point;
                float intensity;
                if (!ifs.read(reinterpret_cast<char*>(&point.x), sizeof(float))) break;
                if (!ifs.read(reinterpret_cast<char*>(&point.y), sizeof(float))) break;
                if (!ifs.read(reinterpret_cast<char*>(&point.z), sizeof(float))) break;
                if (!ifs.read(reinterpret_cast<char*>(&intensity), sizeof(float))) break;

                if (point.x <= 0) continue;

                // Project 3d point to camera frame using camera projection matrix P
                // convert double array to float array
                std::vector<float> p;
                for (size_t i = 0; i < camera_info_.p.size(); i++) {
                    p.push_back(static_cast<float>(camera_info_.p[i]));
                }
                Eigen::Matrix<float, 3, 4, Eigen::RowMajor> P;
                for (size_t i = 0; i < 3; i++) {
                    for (size_t j = 0; j < 4; j++) {
                        P(i, j) = static_cast<float>(p[i * 4 + j]);
                    }
                }

                Eigen::Vector4f point3d(point.x, point.y, point.z, 1);
                Eigen::Vector3f pixel_uv = P * point3d;
                pixel_uv /= pixel_uv(2);

                // Check if point is within image bounds
                if (pixel_uv(0) < 0 || pixel_uv(0) >= camera_info_.width || pixel_uv(1) < 0 || pixel_uv(1) >= camera_info_.height) {
                    continue;
                }
                point.r = image.at<cv::Vec3b>(pixel_uv(1), pixel_uv(0))[2];
                point.g = image.at<cv::Vec3b>(pixel_uv(1), pixel_uv(0))[1];
                point.b = image.at<cv::Vec3b>(pixel_uv(1), pixel_uv(0))[0];

                pointcloud.push_back(point);
            }
            ifs.close();

            sensor_msgs::msg::PointCloud2 pc_msg;
            pcl::toROSMsg(pointcloud, pc_msg);
            pc_msg.header.frame_id = "os_sensor";
            pc_msg.header.stamp = this->now();
            pointcloud_publisher_->publish(pc_msg);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to load point cloud: %s", pc_path.string().c_str());
        }

        // Broadcast transforms
        broadcast_transforms();

        frame_count_++;
    }

    void broadcast_transforms() {
        // Broadcast sensor frame transform
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = this->now();
        transform.header.frame_id = "world";
        transform.child_frame_id = "os_sensor";
        transform.transform.translation.x = 0.0;
        transform.transform.translation.y = 0.0;
        transform.transform.translation.z = 0.0;
        transform.transform.rotation.x = 0.0;
        transform.transform.rotation.y = 0.0;
        transform.transform.rotation.z = 0.0;
        transform.transform.rotation.w = 1.0;
        tf_broadcaster_->sendTransform(transform);

        // Broadcast camera extrinsics transform
        geometry_msgs::msg::TransformStamped camera_transform;
        camera_transform.header.stamp = this->now();
        camera_transform.header.frame_id = "os_sensor";
        camera_transform.child_frame_id = "stereo_left";
        camera_transform.transform.translation.x = 0.0; // Example translation
        camera_transform.transform.translation.y = 0.0;
        camera_transform.transform.translation.z = 0.0;
        camera_transform.transform.rotation.x = 0.0;
        camera_transform.transform.rotation.y = 0.0;
        camera_transform.transform.rotation.z = 0.0;
        camera_transform.transform.rotation.w = 1.0;
        tf_broadcaster_->sendTransform(camera_transform);

        // Broadcast lidar extrinsics transform
        geometry_msgs::msg::TransformStamped lidar_transform;
        lidar_transform.header.stamp = this->now();
        lidar_transform.header.frame_id = "stereo_left";
        lidar_transform.child_frame_id = "os_sensor";
        lidar_transform.transform.translation.x = 0.0; // Example translation
        lidar_transform.transform.translation.y = 0.0;
        lidar_transform.transform.translation.z = 0.0;
        lidar_transform.transform.rotation.x = 0.0;
        lidar_transform.transform.rotation.y = 0.0;
        lidar_transform.transform.rotation.z = 0.0;
        lidar_transform.transform.rotation.w = 1.0;
        tf_broadcaster_->sendTransform(lidar_transform);

        // Broadcast camera info transform
        sensor_msgs::msg::CameraInfo camera_info = camera_info_;
        camera_info.header.stamp = this->now();
        camera_info.header.frame_id = "stereo_left";
        camera_info_publisher_->publish(camera_info);
    }

    std::string image_dir_;
    std::string pointcloud_dir_;
    int sequence_;
    int start_frame_;
    double rate_;
    int frame_count_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;

    sensor_msgs::msg::CameraInfo camera_info_;
};
} // namespace LSMap

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);

    if (argc != 6) {
        std::cerr << "Usage: data_publisher <image_directory> <pointcloud_directory> <sequence> <start_frame> <rate>" << std::endl;
        return 1;
    }

    std::string image_dir = argv[1];
    std::string pointcloud_dir = argv[2];
    int sequence = std::stoi(argv[3]);
    int start_frame = std::stoi(argv[4]);
    double rate = std::stod(argv[5]);

    auto node = std::make_shared<LSMap::DataPublisher>(image_dir, pointcloud_dir, sequence, start_frame, rate);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
