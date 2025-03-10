#pragma once

#include <ros/ros.h>       // For ROS_XXX logging macros
#include <torch/script.h>  // One-stop header for TorchScript

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>

namespace creste {

class CresteModel {
 public:
  CresteModel(const std::string& model_path) {
    try {
      // Load the TorchScript model onto CUDA
      model_ = torch::jit::load(model_path, torch::kCUDA);
      ROS_INFO("Model loaded successfully from %s", model_path.c_str());
    } catch (const c10::Error& e) {
      ROS_ERROR("Error loading the model: %s", e.what());
    }
  }

  /**
   * Forward function takes a tuple of Tensors (e.g., [rgbd_tensor,
   * pixel2point_tensor]), runs a forward pass of the TorchScript model, and
   * returns a dictionary of named Tensors.
   */
  std::unordered_map<std::string, torch::Tensor> forward(
      const std::tuple<torch::Tensor, torch::Tensor>& inputs) {
    // Run the model
    c10::IValue output = model_.forward({inputs});

    // Convert the IValue output (which should be a dictionary) to a GenericDict
    auto dict = output.toGenericDict();

    // Copy each (string key, Tensor value) pair into an std::unordered_map
    std::unordered_map<std::string, torch::Tensor> output_map;
    for (const auto& item : dict) {
      std::string key = item.key().toString()->string();
      torch::Tensor value = item.value().toTensor();
      output_map[key] = value;
    }
    return output_map;
  }

 private:
  torch::jit::script::Module model_;
};

}  // namespace creste
