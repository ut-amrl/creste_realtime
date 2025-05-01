#include "visualization.h"

#include "utils.h"
namespace creste {

template <typename T>
std::vector<std::vector<T>> CopyTopHalf(
    const std::vector<std::vector<T>>& original) {
  // Determine the midpoint (top half of the rows)
  size_t numRows = original.size();
  size_t midPoint = (numRows + 1) / 2;  // Include the middle row if odd

  // Create a new vector containing only the top half
  std::vector<std::vector<T>> result(original.begin(),
                                     original.begin() + midPoint);
  return result;
}

// Explicit template instantiation for the code referencing CopyTopHalf<float>
// etc.
template std::vector<std::vector<float>> CopyTopHalf<float>(
    const std::vector<std::vector<float>>& original);
template std::vector<std::vector<RGBColor>> CopyTopHalf<RGBColor>(
    const std::vector<std::vector<RGBColor>>& original);

void TensorToColorMap(const torch::Tensor& tensor,
                      std::vector<std::vector<RGBColor>>& colors) {
  // Ensure the tensor has at least 4 dims: e.g. [B, C, H, W]
  if (tensor.dim() < 4) {
    LOG_WARN("TensorToColorMap: Input tensor must have at least 4 dimensions.");
    return;
  }
  int64_t C = tensor.size(1), H = tensor.size(2), W = tensor.size(3);

  // Move to CPU and remove batch dim => [C, H, W]
  torch::Tensor tensor_cpu = tensor.to(torch::kCPU).squeeze(0);

  // Compute min/max per-channel
  at::Tensor max_vals = torch::amax(tensor_cpu, {1, 2}, /*keepdim=*/true);
  at::Tensor min_vals = torch::amin(tensor_cpu, {1, 2}, /*keepdim=*/true);

  // Normalize: (x - min)/(max - min+1e-6)*255
  tensor_cpu = (tensor_cpu - min_vals) / (max_vals - min_vals + 1e-6) * 255.0f;
  tensor_cpu = tensor_cpu.clamp(0.0f, 255.0f).to(torch::kUInt8);

  cv::Mat colorMapped(H, W, CV_8UC3);
  if (C == 1) {
    // Single-channel => grayscale => applyColorMap
    cv::Mat mat(H, W, CV_8UC1);
    auto accessor = tensor_cpu.accessor<uint8_t, 3>();  // shape [1, H, W]
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        mat.at<uint8_t>(i, j) = accessor[0][i][j];
      }
    }
    // Use OpenCV colormap
    cv::applyColorMap(mat, colorMapped, cv::COLORMAP_TURBO);
  } else if (C >= 3) {
    // If we have 3 channels => copy to BGR
    auto accessor = tensor_cpu.accessor<uint8_t, 3>();
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        cv::Vec3b pixel;
        // BGR
        pixel[0] = accessor[2][i][j];
        pixel[1] = accessor[1][i][j];
        pixel[2] = accessor[0][i][j];
        colorMapped.at<cv::Vec3b>(i, j) = pixel;
      }
    }
  } else {
    LOG_WARN("TensorToColorMap: Unexpected #channels: %ld", C);
    return;
  }

  colors.resize(H, std::vector<RGBColor>(W));
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      cv::Vec3b pixel = colorMapped.at<cv::Vec3b>(i, j);
      // BGR -> RGB
      colors[i][j] = {pixel[2], pixel[1], pixel[0]};
    }
  }
}

// 3D height map image via vtkStructuredGrid
void GenerateAndPublishHeightMapImageStructuredGrid(
    const std::vector<std::vector<float>>& heights_in,
    const std::vector<std::vector<RGBColor>>& colors_in,
    const ImagePublisher& publisher) {
  LOG_INFO("GenerateAndPublishHeightMapImageStructuredGrid() start.");
  if (heights_in.empty() || heights_in[0].empty()) {
    LOG_WARN("Height map is empty. Nothing to render.");
    return;
  }

  auto heights = CopyTopHalf(heights_in);
  auto colors = CopyTopHalf(colors_in);

  int rows = static_cast<int>(heights.size());
  int cols = static_cast<int>(heights[0].size());

  // 1) Make structured grid
  vtkSmartPointer<vtkStructuredGrid> structuredGrid =
      vtkSmartPointer<vtkStructuredGrid>::New();
  structuredGrid->SetDimensions(cols, rows, 1);

  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  points->SetDataTypeToFloat();
  points->SetNumberOfPoints(rows * cols);

  vtkSmartPointer<vtkUnsignedCharArray> colorArray =
      vtkSmartPointer<vtkUnsignedCharArray>::New();
  colorArray->SetName("Colors");
  colorArray->SetNumberOfComponents(3);
  colorArray->SetNumberOfTuples(rows * cols);
  LOG_INFO("Setting points and colors...");
  vtkIdType idx = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      float x = static_cast<float>(j);
      float y = static_cast<float>(i);
      float z = heights[i][j] * 10.f;  // scale
      idx = i * cols + j;
      points->SetPoint(idx, x, y, z);

      RGBColor c = colors[i][j];
      unsigned char rgb[3] = {c.r, c.g, c.b};
      colorArray->SetTypedTuple(idx, rgb);
    }
  }
  structuredGrid->SetPoints(points);
  structuredGrid->GetPointData()->SetScalars(colorArray);

  // 2) geometry filter to polygon
  vtkSmartPointer<vtkGeometryFilter> geometryFilter =
      vtkSmartPointer<vtkGeometryFilter>::New();
  geometryFilter->SetInputData(structuredGrid);
  geometryFilter->Update();
  vtkSmartPointer<vtkPolyData> polyData = geometryFilter->GetOutput();

  // 3) mapper, actor, renderer, offscreen
  vtkSmartPointer<vtkDataSetMapper> mapper =
      vtkSmartPointer<vtkDataSetMapper>::New();
  mapper->SetInputData(polyData);
  mapper->SetColorModeToDirectScalars();
  mapper->ScalarVisibilityOn();

  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(actor);
  renderer->SetBackground(1.0, 1.0, 1.0);

  vtkSmartPointer<vtkRenderWindow> renderWindow =
      vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->OffScreenRenderingOn();
  renderWindow->AddRenderer(renderer);
  renderWindow->SetSize(512, 312);

  // camera
  vtkCamera* camera = renderer->GetActiveCamera();
  camera->SetPosition(cols / 2.0, rows / 2.0, 60);
  camera->SetFocalPoint(cols / 2.0, 0.0, 0.0);
  camera->SetViewUp(0, 0, -1);
  renderer->ResetCamera();
  renderer->ResetCameraClippingRange();
  camera->Zoom(2.2);

  renderWindow->Render();
  LOG_INFO("RenderWindow rendered!");
  // 4) capture
  vtkSmartPointer<vtkWindowToImageFilter> w2i =
      vtkSmartPointer<vtkWindowToImageFilter>::New();
  w2i->SetInput(renderWindow);
  w2i->SetInputBufferTypeToRGB();
  w2i->ReadFrontBufferOff();
  w2i->Update();

//   vtkImageData* vtkImg = w2i->GetOutput();
//   int dims[3];
//   vtkImg->GetDimensions(dims);
//   int width = dims[0];
//   int height = dims[1];
//   const int channels = 3;
//   LOG_INFO("Copying VTK image to OpenCV...");
//   cv::Mat cvImg(height, width, CV_8UC3);
//   for (int y = 0; y < height; y++) {
//     unsigned char* vtkRow =
//         static_cast<unsigned char*>(vtkImg->GetScalarPointer(0, y, 0));
//     memcpy(cvImg.ptr(y), vtkRow, width * channels);
//   }

//   // Convert to ROS image
//   auto rosImgMsg = cv_bridge::CvImage(Header(), "bgr8", cvImg).toImageMsg();
// #ifdef ROS1
//   publisher.publish(*rosImgMsg);
// #else
//   publisher->publish(*rosImgMsg);
// #endif
  LOG_INFO("StructuredGrid height map image published!");
}

void GenerateAndPublishPolygonImage(const ImagePublisher& publisher) {
  // 1) create simple polygon
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  points->InsertNextPoint(100.0, 100.0, 0.0);
  points->InsertNextPoint(300.0, 100.0, 0.0);
  points->InsertNextPoint(300.0, 300.0, 0.0);
  points->InsertNextPoint(100.0, 300.0, 0.0);

  vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();
  polygon->GetPointIds()->SetNumberOfIds(4);
  for (unsigned int i = 0; i < 4; ++i) {
    polygon->GetPointIds()->SetId(i, i);
  }

  vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();
  polygons->InsertNextCell(polygon->GetPointIds());

  vtkSmartPointer<vtkPolyData> polygonPolyData =
      vtkSmartPointer<vtkPolyData>::New();
  polygonPolyData->SetPoints(points);
  polygonPolyData->SetPolys(polygons);

  // mapper/actor
  vtkSmartPointer<vtkPolyDataMapper> mapper =
      vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputData(polygonPolyData);

  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(1.0, 0.0, 0.0);
  actor->GetProperty()->EdgeVisibilityOn();
  actor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);

  // renderer
  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(actor);
  renderer->SetBackground(1.0, 1.0, 1.0);

  // offscreen
  vtkSmartPointer<vtkRenderWindow> renderWindow =
      vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->OffScreenRenderingOn();
  renderWindow->AddRenderer(renderer);
  renderWindow->SetSize(400, 400);
  renderWindow->Render();

  // capture
  vtkSmartPointer<vtkWindowToImageFilter> w2i =
      vtkSmartPointer<vtkWindowToImageFilter>::New();
  w2i->SetInput(renderWindow);
  w2i->SetInputBufferTypeToRGB();
  w2i->ReadFrontBufferOff();
  w2i->Update();

  vtkImageData* vtkImg = w2i->GetOutput();
  int dims[3];
  vtkImg->GetDimensions(dims);
  int width = dims[0];
  int height = dims[1];
  int channels = 3;
  cv::Mat cvImg(height, width, CV_8UC3);
  for (int y = 0; y < height; ++y) {
    auto* vtkRow =
        static_cast<unsigned char*>(vtkImg->GetScalarPointer(0, y, 0));
    memcpy(cvImg.ptr(y), vtkRow, width * channels);
  }

  auto rosImgMsg = cv_bridge::CvImage(Header(), "bgr8", cvImg).toImageMsg();
#ifdef ROS1
  publisher.publish(*rosImgMsg);
#else
  publisher->publish(*rosImgMsg);
#endif
  LOG_INFO("Polygon image published!");
}

void PublishGaussianHeightMap(const ImagePublisher& publisher) {
  const int rows = 256, cols = 256;
  std::vector<std::vector<float>> heights(rows, std::vector<float>(cols, 0.0f));
  std::vector<std::vector<RGBColor>> colors(
      rows, std::vector<RGBColor>(cols, {255, 255, 255}));

  float centerX = cols / 2.f;
  float centerY = 0.f;
  float sigma = 20.f;
  float amplitude = 10.f;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float dx = j - centerX;
      float dy = i - centerY;
      float val =
          amplitude * expf(-(dx * dx + dy * dy) / (2.f * sigma * sigma));
      heights[i][j] = val;
      unsigned char cval = static_cast<unsigned char>(128 + val);
      colors[i][j] = {cval, 128, 128};
    }
  }

  // create structured grid
  vtkSmartPointer<vtkStructuredGrid> grid =
      vtkSmartPointer<vtkStructuredGrid>::New();
  grid->SetDimensions(cols, rows, 1);

  vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
  pts->SetDataTypeToFloat();
  pts->SetNumberOfPoints(rows * cols);

  vtkSmartPointer<vtkUnsignedCharArray> cArray =
      vtkSmartPointer<vtkUnsignedCharArray>::New();
  cArray->SetName("Colors");
  cArray->SetNumberOfComponents(3);
  cArray->SetNumberOfTuples(rows * cols);

  vtkIdType idx = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++, ++idx) {
      float x = j, y = i, z = heights[i][j];
      pts->SetPoint(idx, x, y, z);
      RGBColor cc = colors[i][j];
      unsigned char rgb[3] = {cc.r, cc.g, cc.b};
      cArray->SetTypedTuple(idx, rgb);
    }
  }
  grid->SetPoints(pts);
  grid->GetPointData()->SetScalars(cArray);

  // geometry filter
  vtkSmartPointer<vtkGeometryFilter> gf =
      vtkSmartPointer<vtkGeometryFilter>::New();
  gf->SetInputData(grid);
  gf->Update();
  vtkSmartPointer<vtkPolyData> poly = gf->GetOutput();

  vtkSmartPointer<vtkDataSetMapper> mapper =
      vtkSmartPointer<vtkDataSetMapper>::New();
  mapper->SetInputData(poly);
  mapper->SetColorModeToDirectScalars();
  mapper->ScalarVisibilityOn();

  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  vtkSmartPointer<vtkRenderer> ren = vtkSmartPointer<vtkRenderer>::New();
  ren->AddActor(actor);
  ren->SetBackground(1.0, 1.0, 1.0);

  vtkSmartPointer<vtkRenderWindow> rw = vtkSmartPointer<vtkRenderWindow>::New();
  rw->OffScreenRenderingOn();
  // auto rw = vtkSmartPointer<vtkOSOpenGLRenderWindow>::New();
  // rw->SetOffScreenRendering(true);
  rw->AddRenderer(ren);
  rw->SetSize(1024, 1024);

  // camera
  auto camera = ren->GetActiveCamera();
  float cX = cols / 2.f;
  float cY = rows / 2.f;
  double cameraHeight = 50.0;
  camera->SetPosition(cX, cY, cameraHeight);
  camera->SetFocalPoint(cX, 0.0, 0.0);
  camera->SetViewUp(0, 0, -1);
  ren->ResetCamera();
  ren->ResetCameraClippingRange();
  camera->Zoom(1.7);

  rw->Render();

  vtkSmartPointer<vtkWindowToImageFilter> w2i =
      vtkSmartPointer<vtkWindowToImageFilter>::New();
  w2i->SetInput(rw);
  w2i->SetInputBufferTypeToRGB();
  w2i->ReadFrontBufferOff();
  w2i->Update();

  vtkImageData* vImg = w2i->GetOutput();
  int dims[3];
  vImg->GetDimensions(dims);
  int width = dims[0];
  int height = dims[1];
  int channels = 3;
  cv::Mat cvImg(height, width, CV_8UC3);
  for (int y = 0; y < height; y++) {
    auto* rowPtr = static_cast<unsigned char*>(vImg->GetScalarPointer(0, y, 0));
    memcpy(cvImg.ptr(y), rowPtr, width * channels);
  }

  auto rosImgMsg = cv_bridge::CvImage(Header(), "bgr8", cvImg).toImageMsg();
#ifdef ROS1
  publisher.publish(*rosImgMsg);
#else
  publisher->publish(*rosImgMsg);
#endif
  LOG_INFO("Published Gaussian height map image.");
}

}  // namespace creste
