#include <string>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <torch/script.h> // One-stop header.

namespace lsmap {

class LSMapModel {
public:
    LSMapModel(const std::string& model_path, rclcpp::Logger logger) : logger_(logger) {
        // Initialize model
        initialize_model(model_path);
    }
private:
    torch::jit::script::Module model_;
    rclcpp::Logger logger_;

    void initialize_model(const std::string& model_path);

    // Function takes in cv2 rgbd and matrix and return jit traces named dictionary
    torch::jit::IValue inference(const cv::Mat& image, const cv::Mat& matrix);

};

} // namespace lsmap