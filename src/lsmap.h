#include <string>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <torch/script.h> // One-stop header.

namespace lsmap {

class LSMapModel {
public:
    LSMapModel(const std::string& model_path, rclcpp::Logger logger) : logger_(logger) {
        try {
            // Load the TorchScript model
            model_ = torch::jit::load(model_path, torch::kCUDA);
            RCLCPP_INFO(logger_, "Model loaded successfully.");
        } catch (const c10::Error& e) {
            RCLCPP_ERROR(logger_, "Error loading the model: %s", e.what());
        }
    }

    // Function takes in cv2 rgbd and matrix and return jit traces named dictionary
    c10::Dict<c10::IValue, c10::IValue> forward(const std::tuple<torch::Tensor, torch::Tensor>& inputs) {
        c10::IValue output = model_.forward({inputs});
        return output.toGenericDict();
    }
private:
    torch::jit::script::Module model_;
    rclcpp::Logger logger_;
};

} // namespace lsmap