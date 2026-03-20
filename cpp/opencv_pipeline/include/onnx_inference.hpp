#pragma once
/**
 * @file onnx_inference.hpp
 * @brief ONNX model inference via OpenCV DNN module.
 */

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace rt_pipeline {

/**
 * @struct Detection
 * @brief Single detection result.
 */
struct Detection {
    int class_id;
    float confidence;
    cv::Rect bbox;
};

/**
 * @class ONNXInference
 * @brief Runs ONNX model inference using OpenCV's DNN backend.
 */
class ONNXInference {
public:
    /**
     * @param model_path   Path to the ONNX model file.
     * @param conf_thresh  Minimum confidence threshold.
     * @param nms_thresh   NMS IoU threshold.
     * @param input_width  Model input width.
     * @param input_height Model input height.
     */
    ONNXInference(const std::string& model_path,
                  float conf_thresh = 0.45f,
                  float nms_thresh = 0.50f,
                  int input_width = 640,
                  int input_height = 640);

    /** Load the ONNX model. Returns true on success. */
    bool load();

    /**
     * Run inference on a preprocessed blob.
     * @param blob 4D input blob.
     * @param orig_width  Original frame width (for rescaling).
     * @param orig_height Original frame height.
     * @return Vector of detections after NMS.
     */
    std::vector<Detection> infer(const cv::Mat& blob,
                                 int orig_width,
                                 int orig_height);

    /** Get the last inference time in milliseconds. */
    double lastInferenceMs() const { return last_inference_ms_; }

private:
    std::string model_path_;
    float conf_thresh_;
    float nms_thresh_;
    int input_width_;
    int input_height_;
    double last_inference_ms_ = 0.0;

    cv::dnn::Net net_;

    /** Apply NMS and scale boxes back to original image coords. */
    std::vector<Detection> postprocess(const cv::Mat& output,
                                       int orig_width,
                                       int orig_height);
};

} // namespace rt_pipeline
