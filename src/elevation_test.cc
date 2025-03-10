#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>

#include "visualization.h"

// Suppose fillHeightAndColorData fills your heights and colors
void fillHeightAndColorData(
    std::vector<std::vector<float>>& heights,
    std::vector<std::vector<creste::RGBColor>>& colors)
{
    // Example: 10x10
    int rows = 100, cols = 100;
    heights.resize(rows, std::vector<float>(cols, 0.0f));
    colors.resize(rows, std::vector<creste::RGBColor>(cols, {255, 255, 255}));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            heights[i][j] = static_cast<float>(i + j); // simple slope
            colors[i][j] = {(unsigned char)(25*i), (unsigned char)(25*j), 128};
        }
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "structured_grid_heightmap_node");
    ros::NodeHandle nh;

    ros::Publisher imagePub = nh.advertise<sensor_msgs::Image>("height_map_image", 2);

    // Prepare data
    std::vector<std::vector<float>> heights;
    std::vector<std::vector<creste::RGBColor>> colors;

    // Publishin a loop at 1 hz
    ros::Rate rate(1);
    while (ros::ok()) {
        fillHeightAndColorData(heights, colors);
        ROS_INFO("Starting rendering...");
        creste::GenerateAndPublishHeightMapImageStructuredGrid(heights, colors, imagePub);
        // GenerateAndPublishPolygonImage(imagePub);
        // PublishGaussianHeightMap(imagePub);
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}
