#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <torch/torch.h>  // <-- or #include <ATen/ATen.h>

// VTK Includes
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkPoints.h>
#include <vtkPolygon.h>         // Include vtkPolygon header
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>
#include <vtkGeometryFilter.h>
#include <vtkDataSetMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkWindowToImageFilter.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkImageData.h>
#include <vtkRenderWindowInteractor.h>

#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>
#include <vtkGeometryFilter.h>
#include <vtkDataSetMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkWindowToImageFilter.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkImageData.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolygon.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>         // ADD THIS LINE

// OpenCV Include (if you want it for bridging)
#include <opencv2/opencv.hpp>

namespace creste {

struct RGBColor {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};


template <typename T>
std::vector<std::vector<T>> CopyTopHalf(const std::vector<std::vector<T>>& original);

void TensorToColorMap(const torch::Tensor & tensor, 
    std::vector<std::vector<RGBColor>>& colors);

void GenerateAndPublishHeightMapImageStructuredGrid(
    const std::vector<std::vector<float>>& heights,
    const std::vector<std::vector<RGBColor>>& colors,
    ros::Publisher& publisher);

void GenerateAndPublishPolygonImage(ros::Publisher& publisher);

void PublishGaussianHeightMap(ros::Publisher& publisher);

}  // namespace creste

#endif  // VISUALIZATION_H