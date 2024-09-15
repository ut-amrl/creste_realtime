#include "lsmap_node.h"

namespace lsmap {
    void LSMapNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) 
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message");
        cloud_queue_.push(msg);

        // Convert ROS2 PointCloud2 message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);

        // Process the PCL point cloud (example: just log the size)
        RCLCPP_INFO(this->get_logger(), "PointCloud size: %lu", pcl_cloud.size());
    }

    void LSMapNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        RCLCPP_INFO(this->get_logger(), "Received Image message");
        image_queue_.push(msg);

        // // Convert ROS2 Image to OpenCV image
        // cv_bridge::CvImagePtr cv_ptr;
        // try {
        //     cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        // } catch (cv_bridge::Exception& e) {
        //     RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        //     return;
        // }

        // // Process the OpenCV image
        // cv::Mat image = cv_ptr->image;

        // // For example, convert to grayscale
        // cv::Mat gray_image;
        // cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

        // // Convert OpenCV image back to ROS2 Image and publish
        // auto output_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", gray_image).toImageMsg();
        // image_publisher_->publish(*output_msg);
    }

    void LSMapNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) 
    {
        RCLCPP_INFO(this->get_logger(), "Received CameraInfo message");
        camera_info_ = *msg;
    }

    void LSMapNode::save_depth_image(const cv::Mat &depthMatrix, const std::string &filename) {
        if (depthMatrix.empty())
        {
            std::cerr << "Invalid input matrix." << std::endl;
            return;
        }

        std::cout << "depth image size: " << depthMatrix.size() << std::endl;
        cv::Mat normDepthImage = cv::Mat::zeros(depthMatrix.size(), CV_8UC1);
        double min_depth, max_depth;
        cv::minMaxLoc(depthMatrix, &min_depth, &max_depth);
        for (int i = 0; i < depthMatrix.rows; ++i) {
            for (int j = 0; j < depthMatrix.cols; ++j) {
                float depth = depthMatrix.at<float>(i, j);
                uchar normDepth = static_cast<uchar>(255 * (depth - min_depth) / (max_depth - min_depth));

                normDepthImage.at<uchar>(i, j) = normDepth;
            }
        }

        if (!cv::imwrite(filename, normDepthImage)) {
            std::cerr << "Failed to save depth image." << std::endl;
        } else {
            std::cout << "Depth image saved to " << filename << std::endl;
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> LSMapNode::projection(sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg,                          
        sensor_msgs::msg::Image::SharedPtr image_msg)
    {
        // Convert ROS2 PointCloud2 message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

        // Convert ROS2 Image to OpenCV image
        cv::Mat rgb_image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;
        Eigen::Matrix<float, 3, 4> pt2pixel;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                pt2pixel(i, j) = camera_info_.p[i * 4 + j];
            }
        }
        // // Save RGB iamge
        // cv::imwrite("rgb_image.png", rgb_image);
        rgb_image.convertTo(rgb_image, CV_32FC3, 1.0 / 255.0);

        cv::Mat depth_image(rgb_image.rows, rgb_image.cols, CV_32FC1, std::numeric_limits<float>::max());

        size_t num_points = pcl_cloud->points.size();
        Eigen::MatrixXf points(4, num_points);
        for (size_t i = 0; i < num_points; ++i) {
            points(0, i) = pcl_cloud->points[i].x;
            points(1, i) = pcl_cloud->points[i].y;
            points(2, i) = pcl_cloud->points[i].z;
            points(3, i) = 1;
        }

        // Project point cloud to image space
        Eigen::MatrixXf pixels = pt2pixel * points;
        for (size_t i = 0; i < num_points; ++i) {
            float u = pixels(0, i) / pixels(2, i);
            float v = pixels(1, i) / pixels(2, i);
            float z = pixels(2, i);

            if (u >=0 && u < rgb_image.cols && v >= 0 && v < rgb_image.rows && z > 0.0) {
                size_t pixel_x = static_cast<size_t>(u);
                size_t pixel_y = static_cast<size_t>(v);
                depth_image.at<float>(pixel_y, pixel_x) = \
                    std::min(depth_image.at<float>(pixel_y, pixel_x), z);
            }
        }
        depth_image.setTo(0, depth_image == std::numeric_limits<float>::max());
        // save_depth_image(depth_image, "depth_image.png");

        // Create RGBD input image for jit trace
        cv::Mat rgbd_image;
        std::vector<cv::Mat> channels;
        cv::split(rgb_image, channels);
        channels.push_back(depth_image);
        cv::merge(channels, rgbd_image);
        torch::Tensor rgbd_tensor = torch::from_blob(rgbd_image.data, {rgbd_image.rows, rgbd_image.cols, 4}, torch::kFloat32);
        rgbd_tensor = rgbd_tensor.permute({2, 0, 1}); // [H, W, C] -> [C, H, W]
        rgbd_tensor = rgbd_tensor.unsqueeze(0).unsqueeze(0); // [C, H, W] -> [B, #cams, C, H, W]
        rgbd_tensor = rgbd_tensor.to(torch::kCUDA);

        // Create point2pixel matrix for inference
        Eigen::Matrix<float, 4, 4> pixel2point;
        pixel2point << -0.00019714, -0.00137023, 1.09675157, 0.01807708,
                       -0.01049331,  0.00010702, 0.79876536, 0.07945613,
                       -0.00008196, -0.01040510, 0.55928874, -0.10325236,
                       -0.00000000,  0.00000000, 0.00000000, 0.99999994;
        torch::Tensor pixel2point_tensor = torch::from_blob(pixel2point.data(), {4, 4}, torch::kFloat32);
        pixel2point_tensor = pixel2point_tensor.unsqueeze(0).unsqueeze(0); // [4, 4] -> [B, #cams, 4, 4]
        pixel2point_tensor = pixel2point_tensor.to(torch::kCUDA);

        // Create tuple for model input
        auto inputs = std::make_tuple(rgbd_tensor, pixel2point_tensor);
        return inputs;
    }

    void LSMapNode::tensorToGridMap(const torch::Tensor& elevation_tensor, const torch::Tensor& rgb_tensor,  grid_map::GridMap& map) {
        // Ensure the tensor is on CPU and of type float
        // torch::Tensor cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat32); // [B, 1, H, W]
        torch::Tensor elevation_cpu_tensor = elevation_tensor.to(torch::kCPU).to(torch::kFloat32); // [B, 1, H, W]
        torch::Tensor rgb_cpu_tensor = rgb_tensor.to(torch::kCPU).to(torch::kFloat32); // [B, 3, H, W]

        // Get tensor dimensions
        auto height = elevation_cpu_tensor.size(2);
        auto width = elevation_cpu_tensor.size(3);

        // Create a grid map with the appropriate dimensions
        map.setGeometry(grid_map::Length(width, height), 1.0); // Adjust the resolution as needed

        // Get pointers to the tensor data
        const float* elevation_data = elevation_cpu_tensor.data_ptr<float>();
        const float* rgb_data = rgb_cpu_tensor.data_ptr<float>();

        // Populate the grid map with elevation data
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                map.at("elevation", grid_map::Index(i, j)) = elevation_data[i * width + j];
            }
        }

        // Create a cv::Mat to hold the image data
        cv::Mat rgb_image(height, width, CV_32FC3, (void*)rgb_data);

        // Convert the float image to 8-bit image
        cv::Mat rgb_image_8u;
        rgb_image.convertTo(rgb_image_8u, CV_8UC3, 255.0);

        grid_map::GridMapCvConverter::addColorLayerFromImage<unsigned char, 3>(rgb_image_8u, "semantics", map);
    }

    std::tuple<at::Tensor, at::Tensor> LSMapNode::computePCA(const at::Tensor& input_tensor, int components) {
        // Ensure the tensor is on CPU and of type float
        at::Tensor cpu_tensor = input_tensor.to(at::kCPU).to(torch::kFloat32);

        // Flatten the tensor to 2D (samples x features)
        auto flattened = cpu_tensor.flatten(1);

        // Compute the mean of each feature
        auto mean = flattened.mean(0, /*keepdim=*/true);

        // Center the data
        auto centered = flattened - mean;

        // Compute the covariance matrix
        auto covariance_matrix = at::mm(centered.t(), centered) / (flattened.size(0) - 1);

        // Perform eigen decomposition
        auto eigen = torch::linalg::eigh(covariance_matrix, "U");
        auto eigenvalues = std::get<0>(eigen);
        auto eigenvectors = std::get<1>(eigen);

        // Select the top 'components' eigenvectors
        auto pca_matrix = eigenvectors.narrow(1, 0, components);

        // Project the data onto the new space
        auto reduced_data = at::mm(centered, pca_matrix);

        return std::make_tuple(reduced_data, pca_matrix);
    }

    void LSMapNode::run()
    {
        sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
        sensor_msgs::msg::Image::SharedPtr image_msg;
        //1 - Check if there are images and point clouds to process
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (cloud_queue_.empty() || image_queue_.empty()) {
                return;
            }
            auto cloud_time = cloud_queue_.front()->header.stamp;
            auto image_time = image_queue_.front()->header.stamp;
            // Convert to nanoseconds and check if the timestamps are within 100 milliseconds
            int64_t cloud_time_ns = cloud_time.sec * 1e9 + cloud_time.nanosec;
            int64_t image_time_ns = image_time.sec * 1e9 + image_time.nanosec;
            if (std::abs(cloud_time_ns - image_time_ns) < 100 * 1e6) {
                cloud_msg = cloud_queue_.front();
                cloud_queue_.pop();
                image_msg = image_queue_.front();
                image_queue_.pop();
            } else {
                // Drop the older message
                if (cloud_time_ns < image_time_ns) {
                    cloud_queue_.pop();
                } else {
                    image_queue_.pop();
                }
            }
        }
        if (!cloud_msg || !image_msg) {
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();
        //2 - Project point clouds to image space
        RCLCPP_INFO(this->get_logger(), "Projecting point cloud to image space");
        auto inputs = projection(cloud_msg, image_msg);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> inference_time = end - start;
        RCLCPP_INFO(this->get_logger(), "Projection time: %f seconds", inference_time.count());

        //3 - Perform model inference
        start = std::chrono::high_resolution_clock::now();
        // Log inference time
        auto output = model_.forward(inputs);
        end = std::chrono::high_resolution_clock::now();
        inference_time = end - start;
        RCLCPP_INFO(this->get_logger(), "Model Inference time: %f seconds", inference_time.count());

        // for (const auto& item : output) {
        //     // Convert the key to a string and print
        //     if (item.key().isString()) {
        //         std::string key_str = item.key().toStringRef();
        //         std::cout << "Key: " << key_str << std::endl;
        //     } else {
        //         std::cout << "Key is not a string." << std::endl;
        //     }
        // }


        //4 - Process elevation and semantic predictions
        start = std::chrono::high_resolution_clock::now();
        const auto& elevation = output.at("elevation_preds").toTensor();
        const auto& semantic = output.at("inpainting_sam_preds").toTensor();
        const auto& traversability = output.at("traversability_preds_full").toTensor();

        //elevation

        //semantic
        auto [pca_result, principal_components] = computePCA(semantic, 3);
        pca_result = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min());
        auto rgb_tensor = pca_result.reshape({semantic.size(0), 3, semantic.size(2), semantic.size(3)});
        // Convert the tensor to a ROS2 Image message and publish
        // cv::Mat rgb_mat(rgb_image.size(2), rgb_image.size(3), CV_8UC3, rgb_image.data_ptr());
        // auto rgb_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", rgb_mat).toImageMsg();
        // image_publisher_->publish(*rgb_msg);
        // Save rgb mat to image if debug flag enabled

        grid_map::GridMap map({"elevation", "semantics"});
        map.setFrameId("os_sensor");
        map.setTimestamp(this->now().nanoseconds());
        tensorToGridMap(elevation, rgb_tensor, map);


        //5 - Publish the results
        start = std::chrono::high_resolution_clock::now();
        const auto& grid_map_msg_ptr = grid_map::GridMapRosConverter::toMessage(map);
        grid_map_publisher_->publish(*grid_map_msg_ptr);
        end = std::chrono::high_resolution_clock::now();
        inference_time = end - start;
        RCLCPP_INFO(this->get_logger(), "Map Processing time: %f seconds", inference_time.count());
    }


} // namespace lsmap