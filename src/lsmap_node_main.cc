#include <iostream>
#include <rclcpp/rclcpp.hpp>

#include "lsmap_node.h"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lsmap::LSMapNode>());
  rclcpp::shutdown();
  return 0;
}