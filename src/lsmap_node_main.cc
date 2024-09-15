#include <iostream>
#include <rclcpp/rclcpp.hpp>

#include "lsmap_node.h"

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<lsmap::LSMapNode>("/lift-splat-map-realtime/traversability_model_trace.pt");
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->run();
    }
    rclcpp::shutdown();
    return 0;
}