#include <ros/ros.h>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "creste_node.h"

// Define a command-line flag for the configuration path
DEFINE_string(config_path, "./config/creste.yaml",
              "Path to the configuration file");
DEFINE_string(weights_path, "",
              "Path to the model weights file");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  ros::init(argc, argv, "creste_node");
  // Private node handle or global node handle is fine; we do it inside
  // CresteNode's constructor if desired. Just ensure it’s consistent.

  creste::CresteNode node(FLAGS_config_path, FLAGS_weights_path);

  // Loop
  ros::Rate loop_rate(100);  // 10 Hz, for example
  while (ros::ok()) {
    ros::spinOnce();
    node.run();
    loop_rate.sleep();
  }
  return 0;
}