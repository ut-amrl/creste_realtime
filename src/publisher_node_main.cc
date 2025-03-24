#include "utils.h"

#ifdef ROS1
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Header.h>
#include <tf/transform_broadcaster.h>

using CameraInfo = sensor_msgs::CameraInfo;
using Image = sensor_msgs::Image;
using PointCloud2 = sensor_msgs::PointCloud2;
using Float32MultiArray = std_msgs::Float32MultiArray;
using Header = std_msgs::Header;
using TransformStamped = geometry_msgs::TransformStamped;
using tf::TransformBroadcaster;

using NodeHandle = ros::NodeHandle;
using Time = ros::Time;
using Publisher = ros::Publisher;
using Timer = ros::Timer;
using TimerEvent = ros::TimerEvent;

#define PUBLISH(pub, msg) (pub).publish(msg)
#define INIT(argc, argv, name) ros::init(argc, argv, name)
#define SPIN() ros::spin()
#define SHUTDOWN() ros::shutdown()

// For ROS1, we use the standard member function pointer syntax:
#else  // ROS2

#include <tf2_ros/transform_broadcaster.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/header.hpp>

using geometry_msgs::msg::TransformStamped;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::PointCloud2;
using std_msgs::msg::Float32MultiArray;
using std_msgs::msg::Header;
using tf2_ros::TransformBroadcaster;

using NodeHandle = std::shared_ptr<rclcpp::Node>;
using Time = rclcpp::Time;
using Publisher = rclcpp::PublisherBase::SharedPtr;
using Timer = rclcpp::TimerBase::SharedPtr;

// Dummy TimerEvent for signature compatibility:
struct TimerEvent {};

#define PUBLISH(pub, msg)                                           \
  do {                                                              \
    using TMsgType = std::remove_reference<decltype(*(msg))>::type; \
    auto typed_pub = std::static_pointer_cast<rclcpp::Publisher<TMsgType>>(pub); \
    typed_pub->publish(*(msg));                                     \
  } while (0)

#define INIT(argc, argv, name) rclcpp::init(argc, argv)
#define SPIN() rclcpp::spin(node)
#define SHUTDOWN() rclcpp::shutdown()

// In ROS2 we will use a lambda to wrap our member function call.
#endif  // ROS1/ROS2

// ------------------------------------------------------------------
// Common includes
// ------------------------------------------------------------------
#include <cv_bridge/cv_bridge.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = boost::filesystem;

namespace creste {

class DataPublisher {
 public:
  // Constructor signature is the same for ROS1 and ROS2.
  DataPublisher(NodeHandle& nh, const std::string& image_dir,
                const std::string& pointcloud_dir, int sequence,
                int start_frame, double rate)
      : nh_(nh),
        image_dir_(image_dir),
        pointcloud_dir_(pointcloud_dir),
        sequence_(sequence),
        start_frame_(start_frame),
        rate_(rate),
        frame_count_(start_frame) {
#ifdef ROS2
    // In ROS2, create the static transform broadcaster if not already constructed.
    if (!g_broadcaster) {
      g_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(nh_);
    }
#endif

    // Publishers
#ifdef ROS1
    image_pub_ = nh_.advertise<Image>("/stereo/left", 10);
    pointcloud_pub_ = nh_.advertise<PointCloud2>("/ouster/points", 10);
    camera_info_pub_ = nh_.advertise<CameraInfo>("/camera_info", 10);
    pixel_to_point_pub_ = nh_.advertise<Float32MultiArray>("/p2p", 10);
#else
    image_pub_ = nh_->create_publisher<Image>("/stereo/left", 10);
    pointcloud_pub_ = nh_->create_publisher<PointCloud2>("/ouster/points", 10);
    camera_info_pub_ = nh_->create_publisher<CameraInfo>("/camera_info", 10);
    pixel_to_point_pub_ = nh_->create_publisher<Float32MultiArray>("/p2p", 10);
#endif

    // Append sequence to directory paths
    image_dir_ += "/" + std::to_string(sequence_);
    pointcloud_dir_ += "/" + std::to_string(sequence_);

    // Load calibration files
    std::string intrinsic_path = "/lift-splat-map-realtime/data/calibrations/" +
                                 std::to_string(sequence_) +
                                 "/calib_cam0_intrinsics.yaml";
    std::string extrinsic_path = "/lift-splat-map-realtime/data/calibrations/" +
                                 std::to_string(sequence_) +
                                 "/calib_os1_to_cam0.yaml";
    loadCalibration(intrinsic_path, extrinsic_path);

    // Create timer with a fixed period (1.0 / rate_)
#ifdef ROS1
    timer_ = nh_.createTimer(ros::Duration(1.0 / rate_),
                             &DataPublisher::publishData, this);
#else
    timer_ = nh_->create_wall_timer(std::chrono::duration<double>(1.0 / rate_),
                                    [this]() { this->publishData(TimerEvent{}); });
#endif
  }

 private:
  // Timer callback: In ROS1 it receives a TimerEvent; in ROS2 the event is unused.
  void publishData(const TimerEvent&) {
    // Build file paths for image and point cloud
    std::stringstream img_ss, pc_ss;
    img_ss << image_dir_ << "/2d_rect_cam0_" << sequence_ << "_" << frame_count_ << ".png";
    pc_ss << pointcloud_dir_ << "/3d_comp_os1_" << sequence_ << "_" << frame_count_ << ".bin";

    fs::path img_path(img_ss.str());
    fs::path pc_path(pc_ss.str());

    // Check if files exist
    if (!fs::exists(img_path) || !fs::exists(pc_path)) {
      LOG_WARN("Frame " << frame_count_ << " does not exist. Stopping publisher.");
      SHUTDOWN();
      return;
    }

    LOG_INFO("Publishing frame " << frame_count_);

    // Read and publish image
    cv::Mat image = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    if (!image.empty()) {
      auto img_msg = cv_bridge::CvImage(Header(), "bgr8", image).toImageMsg();
      img_msg->header.frame_id = "camera_frame";
      img_msg->header.stamp = GET_TIME();

      PUBLISH(image_pub_, img_msg);

      camera_info_.header.stamp = GET_TIME();
      PUBLISH(camera_info_pub_, std::make_shared<CameraInfo>(camera_info_));
    } else {
      LOG_ERROR("Failed to load image: " << img_path.string());
    }

    // Read and publish point cloud
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

        // Simple filter: skip points with non-positive x
        if (point.x <= 0) continue;

        // Project the point to image space for coloring
        Eigen::Matrix<float, 3, 4, Eigen::RowMajor> P;
        for (size_t i = 0; i < 3; i++) {
          for (size_t j = 0; j < 4; j++) {
            P(i, j) = static_cast<float>(camera_info_.P[i * 4 + j]);
          }
        }
        Eigen::Vector4f pt3d(point.x, point.y, point.z, 1.0f);
        Eigen::Vector3f pixel_uv = P * pt3d;
        if (pixel_uv(2) == 0) continue;
        pixel_uv /= pixel_uv(2);

        // If the projected pixel is within the image, sample its color
        if (pixel_uv(0) < 0 || pixel_uv(0) >= camera_info_.width ||
            pixel_uv(1) < 0 || pixel_uv(1) >= camera_info_.height) {
          continue;
        }
        int px = static_cast<int>(pixel_uv(0));
        int py = static_cast<int>(pixel_uv(1));
        cv::Vec3b color = image.at<cv::Vec3b>(py, px);
        point.r = color[2];
        point.g = color[1];
        point.b = color[0];

        pointcloud.push_back(point);
      }
      ifs.close();

      PointCloud2 pc_msg;
      pcl::toROSMsg(pointcloud, pc_msg);
      pc_msg.header.frame_id = "os_sensor";
      pc_msg.header.stamp = GET_TIME();
      PUBLISH(pointcloud_pub_, std::make_shared<PointCloud2>(pc_msg));
      PUBLISH(pixel_to_point_pub_, std::make_shared<Float32MultiArray>(pixel_to_point_));
    } else {
      LOG_ERROR("Failed to load point cloud: " << pc_path.string());
    }

    // Broadcast transforms and update frame counter
    broadcastTransforms();
    frame_count_++;
  }

  void loadCalibration(const std::string& intrinsic_file, const std::string& extrinsic_file) {
    try {
      YAML::Node iconfig = YAML::LoadFile(intrinsic_file);
      YAML::Node econfig = YAML::LoadFile(extrinsic_file);

      int ds = 1;
      int ds_depth = 8;

      camera_info_.header.frame_id = "stereo_left";
      camera_info_.width = iconfig["image_width"].as<int>() / ds;
      camera_info_.height = iconfig["image_height"].as<int>() / ds;

      // Load intrinsic matrix
      auto K = iconfig["camera_matrix"]["data"].as<std::vector<double>>();
      for (size_t i = 0; i < 9; ++i) {
        camera_info_.K[i] = K[i];
        if (i < 6)
          camera_info_.K[i] /= ds;
        LOG_INFO("K[" << i << "]: " << camera_info_.K[i]);
      }

      // Load projection matrix
      auto P = econfig["projection_matrix"]["data"].as<std::vector<double>>();
      for (size_t i = 0; i < 12; ++i) {
        camera_info_.P[i] = P[i];
        if (i < 3)
          camera_info_.P[i] /= ds;
        if (i >= 4 && i < 7)
          camera_info_.P[i] /= ds;
        LOG_INFO("P[" << i << "]: " << camera_info_.P[i]);
      }

      // Load extrinsic and rectification matrices, then compute the pixel-to-point transform
      Eigen::Matrix4f lidar2cam = Eigen::Matrix4f::Identity();
      auto lidar2cam_data = econfig["extrinsic_matrix"]["data"].as<std::vector<double>>();
      for (size_t i = 0; i < 12; ++i)
        lidar2cam(i / 4, i % 4) = static_cast<float>(lidar2cam_data[i]);

      Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
      auto R_data = iconfig["rectification_matrix"]["data"].as<std::vector<double>>();
      for (size_t i = 0; i < 9; ++i)
        R(i / 3, i % 3) = static_cast<float>(R_data[i]);

      Eigen::Matrix3f K_new = Eigen::Matrix3f::Identity();
      auto K_new_data = iconfig["projection_matrix"]["data"].as<std::vector<double>>();
      K_new(0, 0) = K_new_data[0] / ds_depth;
      K_new(1, 1) = K_new_data[5] / ds_depth;
      K_new(0, 2) = K_new_data[2] / ds_depth;
      K_new(1, 2) = K_new_data[6] / ds_depth;

      Eigen::Matrix4f pixel2pts = getPixelToPtsTransform(lidar2cam, R, K_new);
      for (size_t i = 0; i < 16; ++i)
        pixel_to_point_.data.push_back(pixel2pts(i / 4, i % 4));
    } catch (const std::exception& e) {
      LOG_ERROR("Failed to load camera calibration: " << e.what());
    }
  }

  Eigen::Matrix4f getPixelToPtsTransform(const Eigen::Matrix4f& T_lidar_cam,
                                         const Eigen::Matrix3f& R,
                                         const Eigen::Matrix3f& K_new) {
    Eigen::Matrix4f T_cam_lidar = T_lidar_cam.inverse();
    Eigen::Matrix4f T_canon = Eigen::Matrix4f::Identity();
    T_canon.block<3, 3>(0, 0) = R.transpose();
    Eigen::Matrix4f P_pix_cam = Eigen::Matrix4f::Identity();
    P_pix_cam.block<3, 3>(0, 0) = K_new.inverse();
    return T_cam_lidar * T_canon * P_pix_cam;
  }

  void broadcastTransforms() {
    TransformStamped transform;
    transform.header.stamp = GET_TIME();
    transform.header.frame_id = "world";
    transform.child_frame_id = "os_sensor";
    transform.transform.translation.x = 0.0;
    transform.transform.translation.y = 0.0;
    transform.transform.translation.z = 0.0;
    transform.transform.rotation.x = 0.0;
    transform.transform.rotation.y = 0.0;
    transform.transform.rotation.z = 0.0;
    transform.transform.rotation.w = 1.0;
#ifdef ROS1
    g_broadcaster.sendTransform(transform);
#else
    g_broadcaster->sendTransform(transform);
#endif

    TransformStamped camera_transform;
    camera_transform.header.stamp = GET_TIME();
    camera_transform.header.frame_id = "os_sensor";
    camera_transform.child_frame_id = "stereo_left";
    camera_transform.transform.translation.x = 0.0;
    camera_transform.transform.translation.y = 0.0;
    camera_transform.transform.translation.z = 0.0;
    camera_transform.transform.rotation.x = 0.0;
    camera_transform.transform.rotation.y = 0.0;
    camera_transform.transform.rotation.z = 0.0;
    camera_transform.transform.rotation.w = 1.0;
#ifdef ROS1
    g_broadcaster.sendTransform(camera_transform);
#else
    g_broadcaster->sendTransform(camera_transform);
#endif

    TransformStamped lidar_transform;
    lidar_transform.header.stamp = GET_TIME();
    lidar_transform.header.frame_id = "stereo_left";
    lidar_transform.child_frame_id = "os_sensor";
    lidar_transform.transform.translation.x = 0.0;
    lidar_transform.transform.translation.y = 0.0;
    lidar_transform.transform.translation.z = 0.0;
    lidar_transform.transform.rotation.x = 0.0;
    lidar_transform.transform.rotation.y = 0.0;
    lidar_transform.transform.rotation.z = 0.0;
    lidar_transform.transform.rotation.w = 1.0;
#ifdef ROS1
    g_broadcaster.sendTransform(lidar_transform);
#else
    g_broadcaster->sendTransform(lidar_transform);
#endif

    CameraInfo camera_info_copy = camera_info_;
    camera_info_copy.header.stamp = GET_TIME();
    camera_info_copy.header.frame_id = "stereo_left";
    PUBLISH(camera_info_pub_, std::make_shared<CameraInfo>(camera_info_copy));
  }

 private:
  NodeHandle& nh_;
  std::string image_dir_;
  std::string pointcloud_dir_;
  int sequence_;
  int start_frame_;
  double rate_;
  int frame_count_;

  Publisher image_pub_;
  Publisher pointcloud_pub_;
  Publisher pixel_to_point_pub_;
  Publisher camera_info_pub_;
  Timer timer_;

  CameraInfo camera_info_;
  Float32MultiArray pixel_to_point_;
};

}  // namespace creste

int main(int argc, char* argv[]) {
  INIT(argc, argv, "data_publisher");

#ifdef ROS1
  ros::NodeHandle nh;
  if (argc != 6) {
    std::cerr << "Usage: data_publisher <image_directory> <pointcloud_directory> <sequence> <start_frame> <rate>\n";
    return 1;
  }
#else
  auto nh = std::make_shared<rclcpp::Node>("data_publisher");
  if (argc != 6) {
    std::cerr << "Usage: data_publisher <image_directory> <pointcloud_directory> <sequence> <start_frame> <rate>\n";
    return 1;
  }
#endif

  std::string image_dir = argv[1];
  std::string pointcloud_dir = argv[2];
  int sequence = std::stoi(argv[3]);
  int start_frame = std::stoi(argv[4]);
  double rate = std::stod(argv[5]);

  creste::DataPublisher publisher(nh, image_dir, pointcloud_dir, sequence, start_frame, rate);

  SPIN();
  return 0;
}
