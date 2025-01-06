#include <ros/ros.h>
#include "lsmap_node.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "lsmap_node");
  // Private node handle or global node handle is fine; we do it inside LSMapNode’s constructor
  // if desired. Just ensure it’s consistent.

  lsmap::LSMapNode node("/lift-splat-map-realtime/traversability_model_trace.pt");

  // Loop
  ros::Rate loop_rate(10); // 10 Hz, for example
  while (ros::ok())
  {
    ros::spinOnce();
    node.run();
    loop_rate.sleep();
  }
  return 0;
}
