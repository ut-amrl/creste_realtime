#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <torch/torch.h>

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkDataSetMapper.h>
#include <vtkGeometryFilter.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolygon.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkUnsignedCharArray.h>
#include <vtkWindowToImageFilter.h>

// OpenCV Include (if you want bridging or other CV ops)
#include <opencv2/opencv.hpp>

// Conditional includes for ROS1 vs ROS2
#ifdef ROS1
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
// We can alias them for code consistency:
using ImageMsg = sensor_msgs::Image;
using ImagePublisher = ros::Publisher;
using Header = std_msgs::Header;
#else
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
// We can alias them for code consistency:
using ImageMsg = sensor_msgs::msg::Image;
using ImagePublisher = rclcpp::Publisher<ImageMsg>::SharedPtr;
using Header = std_msgs::msg::Header;
#endif

namespace creste {

struct RGBColor {
  unsigned char r;
  unsigned char g;
  unsigned char b;
};

// Simple copy function
template <typename T>
std::vector<std::vector<T>> CopyTopHalf(
    const std::vector<std::vector<T>>& original);

// Convert a PyTorch tensor to a color map (rgb)
void TensorToColorMap(const torch::Tensor& tensor,
                      std::vector<std::vector<RGBColor>>& colors);

// Example function that uses VTK to generate a height map, then publish as an
// Image
void GenerateAndPublishHeightMapImageStructuredGrid(
    const std::vector<std::vector<float>>& heights,
    const std::vector<std::vector<RGBColor>>& colors,
    const ImagePublisher& publisher);

// Example function that draws a polygon shape & publishes
void GenerateAndPublishPolygonImage(const ImagePublisher& publisher);

// Example function that publishes a simple Gaussian height map
void PublishGaussianHeightMap(const ImagePublisher& publisher);

}  // namespace creste

#endif  // VISUALIZATION_H
