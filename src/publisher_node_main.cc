#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf/transform_broadcaster.h>  // TF1 (ROS 1)
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
Example usage:
rosrun creste_realtime data_publisher <image_directory> <pointcloud_directory>
<sequence> <start_frame> <rate>
**/

namespace fs = boost::filesystem;

namespace creste {

class DataPublisher {
 public:
  DataPublisher(ros::NodeHandle& nh, const std::string& image_dir,
                const std::string& pointcloud_dir, int sequence,
                int start_frame, double rate)
      : nh_(nh),
        image_dir_(image_dir),
        pointcloud_dir_(pointcloud_dir),
        sequence_(sequence),
        start_frame_(start_frame),
        rate_(rate),
        frame_count_(start_frame) {
    // Publishers
    image_pub_ = nh_.advertise<sensor_msgs::Image>("/stereo/left", 10);
    pointcloud_pub_ =
        nh_.advertise<sensor_msgs::PointCloud2>("/ouster/points", 10);
    camera_info_pub_ =
        nh_.advertise<sensor_msgs::CameraInfo>("/camera_info", 10);
    pixel_to_point_pub_ =
        nh_.advertise<std_msgs::Float32MultiArray>("/p2p", 10);

    // Add sequence prefix
    image_dir_ = image_dir_ + "/" + std::to_string(sequence_);
    pointcloud_dir_ = pointcloud_dir_ + "/" + std::to_string(sequence_);

    // Load calibrations
    std::string intrinsic_path = "/lift-splat-map-realtime/data/calibrations/" +
                                 std::to_string(sequence_) +
                                 "/calib_cam0_intrinsics.yaml";
    std::string extrinsic_path = "/lift-splat-map-realtime/data/calibrations/" +
                                 std::to_string(sequence_) +
                                 "/calib_os1_to_cam0.yaml";
    loadCalibration(intrinsic_path, extrinsic_path);

    // Create a timer to publish at the specified rate
    timer_ = nh_.createTimer(ros::Duration(1.0 / rate_),
                             &DataPublisher::publishData, this);
  }

 private:
  void loadCalibration(const std::string& intrinsic_file,
                       const std::string& extrinsic_file) {
    try {
      YAML::Node iconfig = YAML::LoadFile(intrinsic_file);
      YAML::Node econfig = YAML::LoadFile(extrinsic_file);

      int ds = 1;
      int ds_depth = 8;

      camera_info_.header.frame_id = "stereo_left";
      camera_info_.width = iconfig["image_width"].as<int>() / ds;
      camera_info_.height = iconfig["image_height"].as<int>() / ds;

      // Load intrinsic matrix (K)
      auto K = iconfig["camera_matrix"]["data"].as<std::vector<double>>();
      for (size_t i = 0; i < 9; ++i) {
        camera_info_.K[i] = K[i];
        if (i < 6) {
          camera_info_.K[i] /= ds;
        }
        ROS_INFO("K[%ld]: %f", i, camera_info_.K[i]);
      }

      // Load projection matrix (P)
      auto P = econfig["projection_matrix"]["data"].as<std::vector<double>>();
      for (size_t i = 0; i < 12; ++i) {
        camera_info_.P[i] = P[i];
        // scale fx, fy, cx, cy by ds
        if (i < 3) camera_info_.P[i] /= ds;
        if (i >= 4 && i < 7) camera_info_.P[i] /= ds;
        ROS_INFO("P[%ld]: %f", i, camera_info_.P[i]);
      }

      // extrinsic_matrix => T_lidar_cam
      Eigen::Matrix4f lidar2cam = Eigen::Matrix4f::Identity();
      auto lidar2cam_data =
          econfig["extrinsic_matrix"]["data"].as<std::vector<double>>();
      for (size_t i = 0; i < 12; ++i) {
        lidar2cam(i / 4, i % 4) = static_cast<float>(lidar2cam_data[i]);
      }

      // rectification_matrix => R
      Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
      auto R_data =
          iconfig["rectification_matrix"]["data"].as<std::vector<double>>();
      for (size_t i = 0; i < 9; ++i) {
        R(i / 3, i % 3) = static_cast<float>(R_data[i]);
      }

      // new projection_matrix => K_new
      Eigen::Matrix3f K_new = Eigen::Matrix3f::Identity();
      auto K_new_data =
          iconfig["projection_matrix"]["data"].as<std::vector<double>>();
      K_new(0, 0) = K_new_data[0] / ds_depth;  // fx
      K_new(1, 1) = K_new_data[5] / ds_depth;  // fy
      K_new(0, 2) = K_new_data[2] / ds_depth;  // cx
      K_new(1, 2) = K_new_data[6] / ds_depth;  // cy

      // Compute pixel2point transform
      Eigen::Matrix4f pixel2pts = getPixelToPtsTransform(lidar2cam, R, K_new);
      for (size_t i = 0; i < 16; ++i) {
        pixel_to_point_.data.push_back(pixel2pts(i / 4, i % 4));
      }
    } catch (const std::exception& e) {
      ROS_ERROR("Failed to load camera calibration: %s", e.what());
    }
  }

  Eigen::Matrix4f getPixelToPtsTransform(const Eigen::Matrix4f& T_lidar_cam,
                                         const Eigen::Matrix3f& R,
                                         const Eigen::Matrix3f& K_new) {
    // T_cam_lidar
    Eigen::Matrix4f T_cam_lidar = T_lidar_cam.inverse();
    // rectification
    Eigen::Matrix4f T_canon = Eigen::Matrix4f::Identity();
    T_canon.block<3, 3>(0, 0) = R.transpose();
    // inverse K
    Eigen::Matrix4f P_pix_cam = Eigen::Matrix4f::Identity();
    P_pix_cam.block<3, 3>(0, 0) = K_new.inverse();
    // final
    Eigen::Matrix4f T_rect_to_lidar = T_cam_lidar * T_canon * P_pix_cam;
    return T_rect_to_lidar;
  }

  void publishData(const ros::TimerEvent&) {
    // Build file paths
    std::stringstream img_ss, pc_ss;
    img_ss << image_dir_ << "/2d_rect_cam0_" << sequence_ << "_" << frame_count_
           << ".png";
    pc_ss << pointcloud_dir_ << "/3d_comp_os1_" << sequence_ << "_"
          << frame_count_ << ".bin";

    fs::path img_path(img_ss.str());
    fs::path pc_path(pc_ss.str());

    // Check if files exist
    if (!fs::exists(img_path) || !fs::exists(pc_path)) {
      ROS_WARN("Frame %d does not exist. Stopping publisher.", frame_count_);
      ros::shutdown();
      return;
    }

    ROS_INFO("Publishing frame %d", frame_count_);

    // Publish image
    cv::Mat image = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    if (!image.empty()) {
      // Convert to ROS Image message
      sensor_msgs::ImagePtr img_msg =
          cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
      img_msg->header.frame_id = "camera_frame";
      img_msg->header.stamp = ros::Time::now();
      image_pub_.publish(img_msg);

      // Publish camera intrinsics
      camera_info_.header.stamp = ros::Time::now();
      camera_info_pub_.publish(camera_info_);
    } else {
      ROS_ERROR("Failed to load image: %s", img_path.string().c_str());
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
        if (!ifs.read(reinterpret_cast<char*>(&intensity), sizeof(float)))
          break;

        // Example filter: discard negative x points
        if (point.x <= 0) continue;

        // Project to image to color the point
        Eigen::Matrix<float, 3, 4, Eigen::RowMajor> P;
        for (size_t i = 0; i < 3; i++) {
          for (size_t j = 0; j < 4; j++) {
            P(i, j) = static_cast<float>(camera_info_.P[i * 4 + j]);
          }
        }
        Eigen::Vector4f pt3d(point.x, point.y, point.z, 1.0f);
        Eigen::Vector3f pixel_uv = P * pt3d;
        if (pixel_uv(2) == 0) continue;  // avoid division by zero
        pixel_uv /= pixel_uv(2);

        // Check if within image
        if (pixel_uv(0) < 0 || pixel_uv(0) >= camera_info_.width ||
            pixel_uv(1) < 0 || pixel_uv(1) >= camera_info_.height) {
          continue;
        }

        // Color from image
        auto px = static_cast<int>(pixel_uv(0));
        auto py = static_cast<int>(pixel_uv(1));
        cv::Vec3b color = image.at<cv::Vec3b>(py, px);
        point.r = color[2];
        point.g = color[1];
        point.b = color[0];

        pointcloud.push_back(point);
      }
      ifs.close();

      // Convert to ROS PointCloud2
      sensor_msgs::PointCloud2 pc_msg;
      pcl::toROSMsg(pointcloud, pc_msg);
      pc_msg.header.frame_id = "os_sensor";
      pc_msg.header.stamp = ros::Time::now();
      pointcloud_pub_.publish(pc_msg);

      // Publish pixel-to-point transform
      pixel_to_point_pub_.publish(pixel_to_point_);
    } else {
      ROS_ERROR("Failed to load point cloud: %s", pc_path.string().c_str());
    }

    // Broadcast transforms
    broadcastTransforms();

    frame_count_++;
  }

  void broadcastTransforms() {
    static tf::TransformBroadcaster br;

    // 1) sensor frame transform (world -> os_sensor)
    {
      geometry_msgs::TransformStamped transform;
      transform.header.stamp = ros::Time::now();
      transform.header.frame_id = "world";
      transform.child_frame_id = "os_sensor";
      transform.transform.translation.x = 0.0;
      transform.transform.translation.y = 0.0;
      transform.transform.translation.z = 0.0;
      transform.transform.rotation.x = 0.0;
      transform.transform.rotation.y = 0.0;
      transform.transform.rotation.z = 0.0;
      transform.transform.rotation.w = 1.0;
      br.sendTransform(transform);
    }

    // 2) camera extrinsics transform (os_sensor -> stereo_left)
    {
      geometry_msgs::TransformStamped camera_transform;
      camera_transform.header.stamp = ros::Time::now();
      camera_transform.header.frame_id = "os_sensor";
      camera_transform.child_frame_id = "stereo_left";
      camera_transform.transform.translation.x = 0.0;
      camera_transform.transform.translation.y = 0.0;
      camera_transform.transform.translation.z = 0.0;
      camera_transform.transform.rotation.x = 0.0;
      camera_transform.transform.rotation.y = 0.0;
      camera_transform.transform.rotation.z = 0.0;
      camera_transform.transform.rotation.w = 1.0;
      br.sendTransform(camera_transform);
    }

    // 3) lidar extrinsics transform (stereo_left -> os_sensor) [example]
    {
      geometry_msgs::TransformStamped lidar_transform;
      lidar_transform.header.stamp = ros::Time::now();
      lidar_transform.header.frame_id = "stereo_left";
      lidar_transform.child_frame_id = "os_sensor";
      lidar_transform.transform.translation.x = 0.0;
      lidar_transform.transform.translation.y = 0.0;
      lidar_transform.transform.translation.z = 0.0;
      lidar_transform.transform.rotation.x = 0.0;
      lidar_transform.transform.rotation.y = 0.0;
      lidar_transform.transform.rotation.z = 0.0;
      lidar_transform.transform.rotation.w = 1.0;
      br.sendTransform(lidar_transform);
    }

    // 4) Publish camera info again, if needed
    sensor_msgs::CameraInfo camera_info_copy = camera_info_;
    camera_info_copy.header.stamp = ros::Time::now();
    camera_info_copy.header.frame_id = "stereo_left";
    camera_info_pub_.publish(camera_info_copy);
  }

 private:
  ros::NodeHandle nh_;

  std::string image_dir_;
  std::string pointcloud_dir_;
  int sequence_;
  int start_frame_;
  double rate_;
  int frame_count_;

  // Publishers
  ros::Publisher image_pub_;
  ros::Publisher pointcloud_pub_;
  ros::Publisher pixel_to_point_pub_;
  ros::Publisher camera_info_pub_;

  // Timer
  ros::Timer timer_;

  // Camera / calibration info
  sensor_msgs::CameraInfo camera_info_;
  std_msgs::Float32MultiArray pixel_to_point_;
};

}  // namespace creste

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "data_publisher");

  if (argc != 6) {
    std::cerr << "Usage: data_publisher <image_directory> "
                 "<pointcloud_directory> <sequence> <start_frame> <rate>\n";
    return 1;
  }

  ros::NodeHandle nh;

  std::string image_dir = argv[1];
  std::string pointcloud_dir = argv[2];
  int sequence = std::stoi(argv[3]);
  int start_frame = std::stoi(argv[4]);
  double rate = std::stod(argv[5]);

  creste::DataPublisher publisher(nh, image_dir, pointcloud_dir, sequence,
                                 start_frame, rate);

  ros::spin();
  return 0;
}
