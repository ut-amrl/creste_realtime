#include "gflags/gflags.h"
#include "glog/logging.h"

// Define GFlags for config, weights
DEFINE_string(config_path, "./config/creste.yaml",
              "Path to the configuration file");
DEFINE_string(weights_path, "", "Path to the model weights file");

#ifdef ROS1
// ROS1 includes
#include <ros/ros.h>
// If your creste_node.h used ros::NodeHandle, etc. be sure it #ifdef's as well
#else
// ROS2 includes
#include <rclcpp/rclcpp.hpp>
#endif

#include "creste_node.h"

int main(int argc, char** argv) {
  // Parse GFlags
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

#ifdef ROS1
  ros::init(argc, argv, "creste_node");
  ros::NodeHandle nh;
  // Possibly create a ros::NodeHandle if needed inside CresteNode
  // or pass in any extra arguments
  creste::CresteNode node(FLAGS_config_path, FLAGS_weights_path, nh);

  ros::Rate loop_rate(100);  // e.g. 100 Hz
  while (ros::ok()) {
    ros::spinOnce();
    node.run();
    loop_rate.sleep();
  }

#else  // ROS2
  rclcpp::init(argc, argv);

  // Optionally create an rclcpp Node if your CresteNode needs it
  auto rclcpp_node = rclcpp::Node::make_shared("creste_node");

  creste::CresteNode node(FLAGS_config_path, FLAGS_weights_path, rclcpp_node);
  // e.g. node.initialize(rclcpp_node);

  rclcpp::Rate loop_rate(100.0);  // ~100 Hz
  while (rclcpp::ok()) {
    rclcpp::spin_some(rclcpp_node);
    node.run();
    loop_rate.sleep();
  }
  rclcpp::shutdown();

#endif

  return 0;
}
