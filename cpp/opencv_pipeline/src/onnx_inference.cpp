/**
 * @file onnx_inference.cpp
 * @brief Implementation of the ONNXInference class using OpenCV DNN.
 */

#include "onnx_inference.hpp"
#include <iostream>
#include <chrono>
#include <opencv2/imgproc.hpp>

namespace rt_pipeline {

ONNXInference::ONNXInference(const std::string& model_path,
                             float conf_thresh,
                             float nms_thresh,
                             int input_width,
                             int input_height)
    : model_path_(model_path),
      conf_thresh_(conf_thresh),
      nms_thresh_(nms_thresh),
      input_width_(input_width),
      input_height_(input_height) {}

bool ONNXInference::load() {
    try {
        net_ = cv::dnn::readNetFromONNX(model_path_);
        std::cout << "[ONNXInference] Model loaded: " << model_path_ << std::endl;

        // Prefer CUDA if available, else CPU
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[ONNXInference] ERROR loading model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<Detection> ONNXInference::infer(const cv::Mat& blob,
                                            int orig_width,
                                            int orig_height) {
    net_.setInput(blob);

    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat output = net_.forward();
    auto t1 = std::chrono::high_resolution_clock::now();

    last_inference_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return postprocess(output, orig_width, orig_height);
}

std::vector<Detection> ONNXInference::postprocess(const cv::Mat& output,
                                                  int orig_width,
                                                  int orig_height) {
    // YOLOv8 output: (1, 84, 8400) — need to transpose to (8400, 84)
    // output.size = [1, 84, N]
    const int rows = output.size[2];     // number of detections (8400)
    const int cols = output.size[1];     // 4 + num_classes (84)

    // Reshape to 2D: (84, 8400)
    cv::Mat data = output.reshape(1, cols);  // (84, 8400)
    cv::Mat transposed;
    cv::transpose(data, transposed);          // (8400, 84)

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float sx = static_cast<float>(orig_width) / input_width_;
    float sy = static_cast<float>(orig_height) / input_height_;

    for (int i = 0; i < transposed.rows; ++i) {
        const float* row = transposed.ptr<float>(i);

        // Columns 0-3: cx, cy, w, h
        float cx = row[0];
        float cy = row[1];
        float w  = row[2];
        float h  = row[3];

        // Columns 4+: class scores
        float max_score = 0.0f;
        int max_class = 0;
        for (int c = 4; c < cols; ++c) {
            if (row[c] > max_score) {
                max_score = row[c];
                max_class = c - 4;
            }
        }

        if (max_score < conf_thresh_) continue;

        // Scale to original image coords
        float x1 = (cx - w / 2.0f) * sx;
        float y1 = (cy - h / 2.0f) * sy;
        float bw  = w * sx;
        float bh  = h * sy;

        boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1),
                           static_cast<int>(bw), static_cast<int>(bh));
        confidences.push_back(max_score);
        class_ids.push_back(max_class);
    }

    // NMS
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, nms_indices);

    std::vector<Detection> results;
    results.reserve(nms_indices.size());
    for (int idx : nms_indices) {
        Detection det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.bbox = boxes[idx];
        results.push_back(det);
    }

    return results;
}

} // namespace rt_pipeline
