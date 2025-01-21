#include "visualization.h"

#include "utils.h"

namespace creste {

void TensorToColorMap(const torch::Tensor & tensor, 
    std::vector<std::vector<RGBColor>>& colors) {
    // Ensure the tensor has at least 4 dimensions
    if (tensor.dim() < 4) {
        ROS_WARN("TensorToColorMap: Input tensor must have at least 4 dimensions.");
        return;
    }
    int64_t C = tensor.size(1), H = tensor.size(2), W = tensor.size(3);

    // Move tensor to CPU and squeeze out the batch dimension: [C, H, W]
    torch::Tensor tensor_cpu = tensor.to(torch::kCPU).squeeze(0);

    // Normalize tensor values across channels to [0, 255]
    // Compute per-channel min and max over the entire tensor shape [C, H, W]
    at::Tensor max_vals = torch::amax(tensor_cpu, {1, 2}, /*keepdim=*/true); // [C, 1, 1]
    at::Tensor min_vals = torch::amin(tensor_cpu, {1, 2}, /*keepdim=*/true); // [C, 1, 1]

    // Broadcast subtraction and division across height and width dimensions
    tensor_cpu = (tensor_cpu - min_vals) / (max_vals - min_vals + 1e-6) * 255.0f;
    tensor_cpu = tensor_cpu.clamp(0.0f, 255.0f).to(torch::kUInt8);

    cv::Mat colorMapped(H, W, CV_8UC3);
    if (C == 1) {
        // Copy data from the tensor to a single-channel OpenCV mat
        cv::Mat mat(H, W, CV_8UC1);
        auto accessor = tensor_cpu.accessor<uint8_t, 3>();  // shape [1, H, W] after squeeze
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                mat.at<uint8_t>(i, j) = accessor[0][i][j];
            }
        }
        printf("TensorToColorMap() Converted tensor to mat\n");

        // Apply a colormap to convert grayscale to BGR
        cv::applyColorMap(mat, colorMapped, cv::COLORMAP_TURBO);
        printf("TensorToColorMap() Applied color map\n");
    } else if (C >= 3) {
        // If tensor has 3 or more channels, copy first 3 channels to BGR image
        auto accessor = tensor_cpu.accessor<uint8_t, 3>();
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                cv::Vec3b pixel;
                // OpenCV default channel order is BGR
                pixel[0] = accessor[2][i][j];  // Blue
                pixel[1] = accessor[1][i][j];  // Green
                pixel[2] = accessor[0][i][j];  // Red
                colorMapped.at<cv::Vec3b>(i, j) = pixel;
            }
        }
    } else {
        ROS_WARN("TensorToColorMap: Unexpected number of channels: %ld", C);
        return;
    }

    // Resize the colors vector to match image dimensions using H and W
    colors.resize(H, std::vector<RGBColor>(W));

    // Copy the color-mapped pixels into the colors 2D vector, converting BGR to RGB
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            cv::Vec3b pixel = colorMapped.at<cv::Vec3b>(i, j);
            // Convert BGR to RGB
            colors[i][j] = { pixel[2], pixel[1], pixel[0] };
        }
    }
}


template <typename T>
std::vector<std::vector<T>> CopyTopHalf(const std::vector<std::vector<T>>& original) {
    // Determine the midpoint (top half of the rows)
    size_t numRows = original.size();
    size_t midPoint = (numRows + 1) / 2;  // Include the middle row for odd row counts

    // Create a new vector containing only the top half of the rows
    std::vector<std::vector<T>> result(original.begin(), original.begin() + midPoint);
    return result;
}

/**
 * @brief Create and publish a 3D height map image using vtkStructuredGrid.
 *
 * @param heights 2D array of height values [rows x cols].
 * @param colors  2D array of RGB colors [rows x cols].
 * @param publisher ROS Publisher to publish the final sensor_msgs::Image.
 */
void GenerateAndPublishHeightMapImageStructuredGrid(
    const std::vector<std::vector<float>>& heights_in,
    const std::vector<std::vector<RGBColor>>& colors_in,
    ros::Publisher& publisher)
{
    printf("GenerateAndPublishHeightMapImageStructuredGrid()\n");
    if (heights_in.empty() || heights_in[0].empty()) {
        ROS_WARN("Height map is empty. Nothing to render.");
        return;
    }

    // Copy top half rows and all columns
    const auto& heights = CopyTopHalf(heights_in);
    const auto& colors = CopyTopHalf(colors_in);

    // Dimensions
    const int rows = static_cast<int>(heights.size());
    const int cols = static_cast<int>(heights[0].size());
    printf("GenerateAndPublishHeightMapImageStructuredGrid() Non empty heights\n");
    // -------------------------------------------------------------------------
    // 1. Create a vtkStructuredGrid to store points and color
    // -------------------------------------------------------------------------
    vtkSmartPointer<vtkStructuredGrid> structuredGrid =
        vtkSmartPointer<vtkStructuredGrid>::New();

    // The grid has 'rows' in one direction (Y) and 'cols' in the other direction (X).
    // We'll keep Z = 1, because it's a 2D slice in terms of structured indexing.
    structuredGrid->SetDimensions(cols, rows, 1);

    // Create a vtkPoints to hold the coordinates (x, y, z)
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataTypeToFloat();  // store float coordinates

    // Create a color array (3 components => RGB)
    vtkSmartPointer<vtkUnsignedCharArray> colorArray =
        vtkSmartPointer<vtkUnsignedCharArray>::New();
    colorArray->SetName("Colors");
    colorArray->SetNumberOfComponents(3); // R, G, B

    // Reserve memory for all points
    points->SetNumberOfPoints(rows * cols);
    colorArray->SetNumberOfTuples(rows * cols);

    // Fill points & colors.  
    // VTK's structured grid indexing: (xId, yId, zId).
    // We'll treat: xId = j (column), yId = i (row), zId = 0 (since it's a single layer).
    vtkIdType idx = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float x = static_cast<float>(j);
            float y = static_cast<float>(i);
            float z = heights[i][j] * 10; // Convert from meters to cm

            // Set the point in the points array
            idx = i * cols + j;  
            points->SetPoint(idx, x, y, z);

            // Set the color
            RGBColor c = colors[i][j];
            unsigned char rgb[3] = { c.r, c.g, c.b };
            colorArray->SetTypedTuple(idx, rgb);
        }
    }
    printf("GenerateAndPublishHeightMapImageStructuredGrid() Instantiated vtk grid\n");
    // Attach points to the structured grid
    structuredGrid->SetPoints(points);
    // Attach color data as point data
    structuredGrid->GetPointData()->SetScalars(colorArray);

    // -------------------------------------------------------------------------
    // 2. Convert StructuredGrid to a polygonal data set (surface) via a filter
    // -------------------------------------------------------------------------
    // A structured grid is not directly polygonal. Use vtkGeometryFilter to extract a surface.
    vtkSmartPointer<vtkGeometryFilter> geometryFilter = 
        vtkSmartPointer<vtkGeometryFilter>::New();
    geometryFilter->SetInputData(structuredGrid);
    geometryFilter->Update();
    
    // Now we have a polygonal representation of the structured grid
    vtkSmartPointer<vtkPolyData> polyData = geometryFilter->GetOutput();
    printf("GenerateAndPublishHeightMapImageStructuredGrid() Creating mapper\n");
    // -------------------------------------------------------------------------
    // 3. Setup a mapper, actor, renderer, and off-screen render window
    // -------------------------------------------------------------------------
    // Use vtkDataSetMapper (or vtkPolyDataMapper) to map the polyData
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(polyData);
    // Tells the mapper to treat the "Colors" array as RGB
    mapper->SetColorModeToDirectScalars();
    mapper->ScalarVisibilityOn();

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);

    // Set background color to white instead of dark gray
    renderer->SetBackground(1.0, 1.0, 1.0);  

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->OffScreenRenderingOn();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(512, 312);

    //  double cameraHeight = 50.0;
    // camera->SetPosition(centerX, centerY, cameraHeight);

    // // Set the focal point to the middle of the top row (where the Gaussian blob is)
    // // Since the Gaussian center was set at (cols/2, 0), we target that point at ground level
    // camera->SetFocalPoint(centerX, 0.0, 0.0);

    // // Set the view up direction as the world Z-axis (assuming Z is up)
    // camera->SetViewUp(0, 0, -1);

    // Camera setup (just a basic overhead or angled view)
    vtkCamera* camera = renderer->GetActiveCamera();
    camera->SetPosition(cols / 2.0, rows / 2.0, 60);
    camera->SetFocalPoint(cols / 2.0, 0.0, 0.0);
    camera->SetViewUp(0, 0, -1);
    renderer->ResetCamera();
    renderer->ResetCameraClippingRange();

    camera->Zoom(2.2);

    printf("GenerateAndPublishHeightMapImageStructuredGrid() rendering\n");
    // Render
    renderWindow->Render();
    printf("GenerateAndPublishHeightMapImageStructuredGrid() Capturing render\n");
    // -------------------------------------------------------------------------
    // 4. Capture the rendered image (VTK -> cv::Mat -> ROS)
    // -------------------------------------------------------------------------
    vtkSmartPointer<vtkWindowToImageFilter> w2i =
        vtkSmartPointer<vtkWindowToImageFilter>::New();
    w2i->SetInput(renderWindow);
    w2i->SetInputBufferTypeToRGB();
    w2i->ReadFrontBufferOff();
    w2i->Update();
    printf("GenerateAndPublishHeightMapImageStructuredGrid() Read render\n");
    vtkImageData* vtkImg = w2i->GetOutput();
    int dims[3];
    vtkImg->GetDimensions(dims);
    const int width = dims[0];
    const int height = dims[1];
    const int channels = 3; // R, G, B
    printf("GenerateAndPublishHeightMapImageStructuredGrid() SAved render\n");
    // Copy to cv::Mat
    cv::Mat cvImg(height, width, CV_8UC3);
    for (int y = 0; y < height; y++) {
        // VTK row pointer
        unsigned char* vtkRow = static_cast<unsigned char*>(
            vtkImg->GetScalarPointer(0, y, 0)
        );
        // Copy row into cv::Mat
        memcpy(cvImg.ptr(y), vtkRow, width * channels);
    }
    printf("GenerateAndPublishHeightMapImageStructuredGrid() Copied render to cv2 iamge\n");
    // If you need to flip the image vertically (since VTK often has origin at bottom-left):
    // cv::flip(cvImg, cvImg, 0);

    // Convert to ROS Image message
    sensor_msgs::ImagePtr rosImgMsg;
    rosImgMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvImg).toImageMsg();
    printf("GenerateAndPublishHeightMapImageStructuredGrid() Publishing render\n");
    // Publish
    publisher.publish(rosImgMsg);
    ROS_INFO("StructuredGrid height map image published!");
}

void GenerateAndPublishPolygonImage(ros::Publisher& publisher) {
    // Create points for a simple polygon (e.g., a square)
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->InsertNextPoint(100.0, 100.0, 0.0);
    points->InsertNextPoint(300.0, 100.0, 0.0);
    points->InsertNextPoint(300.0, 300.0, 0.0);
    points->InsertNextPoint(100.0, 300.0, 0.0);

    // Create a polygon from the points
    vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();
    polygon->GetPointIds()->SetNumberOfIds(4); // 4 vertices for a quadrilateral
    for (unsigned int i = 0; i < 4; ++i) {
        polygon->GetPointIds()->SetId(i, i);
    }

    // Create a cell array to store the polygon
    vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();
    polygons->InsertNextCell(polygon->GetPointIds());

    // Create a polydata to store everything
    vtkSmartPointer<vtkPolyData> polygonPolyData = vtkSmartPointer<vtkPolyData>::New();
    polygonPolyData->SetPoints(points);
    polygonPolyData->SetPolys(polygons);

    // Mapper and Actor
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polygonPolyData);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(1.0, 0.0, 0.0);  // Set polygon color to red
    actor->GetProperty()->EdgeVisibilityOn();
    actor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);  // Black edges for contrast

    // Renderer setup
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);
    renderer->SetBackground(1.0, 1.0, 1.0);  // White background

    // Render window setup for offscreen rendering
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->OffScreenRenderingOn();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(400, 400);  // Set desired image size

    // Render the scene
    renderWindow->Render();

    // Capture the rendered image
    vtkSmartPointer<vtkWindowToImageFilter> w2i = vtkSmartPointer<vtkWindowToImageFilter>::New();
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

    // Copy VTK image to OpenCV matrix
    cv::Mat cvImg(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        unsigned char* vtkRow = static_cast<unsigned char*>(vtkImg->GetScalarPointer(0, y, 0));
        memcpy(cvImg.ptr(y), vtkRow, width * channels);
    }

    // Publish the image as a ROS message
    sensor_msgs::ImagePtr rosImgMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvImg).toImageMsg();
    publisher.publish(rosImgMsg);
    ROS_INFO("Polygon image published!");
}

void PublishGaussianHeightMap(ros::Publisher& publisher) {
    // Define grid size
    const int rows = 256;
    const int cols = 256;
    
    // Create height and color arrays for the grid
    std::vector<std::vector<float>> heights(rows, std::vector<float>(cols, 0.0f));
    std::vector<std::vector<RGBColor>> colors(rows, std::vector<RGBColor>(cols, {255, 255, 255}));
    
    // Parameters for Gaussian bump
    float centerX = cols / 2.0f;
    float centerY = 0; //rows / 2.0f;
    float sigma = 20.0f;
    float amplitude = 10.0f;
    
    // Fill the height and color arrays with Gaussian bump data
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float dx = j - centerX;
            float dy = i - centerY;
            // Gaussian formula: amplitude * exp(-(dx^2 + dy^2)/(2*sigma^2))
            heights[i][j] = amplitude * exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
            // For simplicity, colors are set to a constant; adjust as needed
            colors[i][j] = { static_cast<unsigned char>(128 + heights[i][j]), 
                             static_cast<unsigned char>(128), 
                             static_cast<unsigned char>(128) };
        }
    }
    
    // Set up a vtkStructuredGrid using the heights and colors
    vtkSmartPointer<vtkStructuredGrid> structuredGrid = vtkSmartPointer<vtkStructuredGrid>::New();
    structuredGrid->SetDimensions(cols, rows, 1);
    
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataTypeToFloat();
    points->SetNumberOfPoints(rows * cols);
    
    vtkSmartPointer<vtkUnsignedCharArray> colorArray = vtkSmartPointer<vtkUnsignedCharArray>::New();
    colorArray->SetName("Colors");
    colorArray->SetNumberOfComponents(3);
    colorArray->SetNumberOfTuples(rows * cols);
    
    vtkIdType idx = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j, ++idx) {
            float x = static_cast<float>(j);
            float y = static_cast<float>(i);
            float z = heights[i][j];
            points->SetPoint(idx, x, y, z);
            RGBColor c = colors[i][j];
            unsigned char rgb[3] = { c.r, c.g, c.b };
            colorArray->SetTypedTuple(idx, rgb);
        }
    }
    
    structuredGrid->SetPoints(points);
    structuredGrid->GetPointData()->SetScalars(colorArray);
    
    // Convert structured grid to polygonal data
    vtkSmartPointer<vtkGeometryFilter> geometryFilter = vtkSmartPointer<vtkGeometryFilter>::New();
    geometryFilter->SetInputData(structuredGrid);
    geometryFilter->Update();
    vtkSmartPointer<vtkPolyData> polyData = geometryFilter->GetOutput();
    
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(polyData);
    mapper->SetColorModeToDirectScalars();
    mapper->ScalarVisibilityOn();
    
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);
    renderer->SetBackground(1.0, 1.0, 1.0);  // White background
    
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->OffScreenRenderingOn();
    renderWindow->AddRenderer(renderer);
    renderWindow->SetSize(1024, 1024);
    
    // Calculate center of the grid in x-y plane
    centerX = cols / 2.0;
    centerY = rows / 2.0;

    // Set up camera: positioned above the grid center, looking toward the top middle where the Gaussian is
    vtkCamera* camera = renderer->GetActiveCamera();

    // Place the camera directly above the grid center at a specified height
    double cameraHeight = 50.0;
    camera->SetPosition(centerX, centerY, cameraHeight);

    // Set the focal point to the middle of the top row (where the Gaussian blob is)
    // Since the Gaussian center was set at (cols/2, 0), we target that point at ground level
    camera->SetFocalPoint(centerX, 0.0, 0.0);

    // Set the view up direction as the world Z-axis (assuming Z is up)
    camera->SetViewUp(0, 0, -1);

    // Optionally reset the camera to ensure all geometry fits in view, particularly the top corners
    renderer->ResetCamera();
    renderer->ResetCameraClippingRange();

    // Apply zoom to magnify the scene
    camera->Zoom(1.7);  // Zoom in by a factor of 2
    
    renderWindow->Render();
    
    vtkSmartPointer<vtkWindowToImageFilter> w2i = vtkSmartPointer<vtkWindowToImageFilter>::New();
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
        unsigned char* vtkRow = static_cast<unsigned char*>(vtkImg->GetScalarPointer(0, y, 0));
        memcpy(cvImg.ptr(y), vtkRow, width * channels);
    }
    
    sensor_msgs::ImagePtr rosImgMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvImg).toImageMsg();
    publisher.publish(rosImgMsg);
    ROS_INFO("Published Gaussian height map image.");
}

}