#include "gflags/gflags.h"
#include "glog/logging.h"

// Define GFlags for config, weights
DEFINE_string(config_path, "./config/creste.yaml",
              "Path to the configuration file");
DEFINE_string(weights_path, "", "Path to the model weights file");

// ROS2 includes
#include <rclcpp/rclcpp.hpp>

#include "creste_node.h"

int main(int argc, char** argv) {
  // Parse GFlags
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

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

  return 0;
}
