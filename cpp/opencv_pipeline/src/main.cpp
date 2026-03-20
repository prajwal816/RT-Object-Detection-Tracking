/**
 * @file main.cpp
 * @brief C++ real-time object detection pipeline entry point.
 *
 * Usage:
 *   rt_pipeline <source> <model.onnx> [conf_thresh] [nms_thresh]
 *
 * Example:
 *   rt_pipeline 0 models/yolov8n.onnx 0.45 0.5
 *   rt_pipeline video.mp4 models/yolov8n.onnx
 */

#include "frame_reader.hpp"
#include "preprocessor.hpp"
#include "onnx_inference.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// ── Color palette for drawing (BGR) ─────────────────────────────────────
const cv::Scalar COLORS[] = {
    {75, 25, 230},  {48, 130, 245}, {25, 225, 255},
    {60, 245, 210}, {75, 180, 60},  {200, 130, 0},
    {240, 110, 70}, {230, 50, 240}, {190, 75, 190},
    {80, 190, 230},
};
constexpr int NUM_COLORS = sizeof(COLORS) / sizeof(COLORS[0]);

void drawDetections(cv::Mat& frame,
                    const std::vector<rt_pipeline::Detection>& dets) {
    for (const auto& det : dets) {
        auto color = COLORS[det.class_id % NUM_COLORS];

        cv::rectangle(frame, det.bbox, color, 2);

        std::ostringstream oss;
        oss << "cls " << det.class_id << " "
            << std::fixed << std::setprecision(2) << det.confidence;
        std::string label = oss.str();

        int baseline = 0;
        auto text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                         0.5, 1, &baseline);
        cv::rectangle(frame,
                      cv::Point(det.bbox.x, det.bbox.y - text_size.height - 6),
                      cv::Point(det.bbox.x + text_size.width + 4, det.bbox.y),
                      color, -1);
        cv::putText(frame, label,
                    cv::Point(det.bbox.x + 2, det.bbox.y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

void drawHUD(cv::Mat& frame, double fps, int num_dets, double latency_ms) {
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Point(8, 8), cv::Point(240, 90),
                  cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.55, frame, 0.45, 0, frame);

    std::ostringstream fps_ss, det_ss, lat_ss;
    fps_ss << "FPS:       " << std::fixed << std::setprecision(1) << fps;
    det_ss << "Detections: " << num_dets;
    lat_ss << "Latency:   " << std::fixed << std::setprecision(1) << latency_ms << " ms";

    cv::putText(frame, fps_ss.str(), cv::Point(16, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 200), 1, cv::LINE_AA);
    cv::putText(frame, det_ss.str(), cv::Point(16, 52),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 200), 1, cv::LINE_AA);
    cv::putText(frame, lat_ss.str(), cv::Point(16, 74),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 200), 1, cv::LINE_AA);
}

} // anonymous namespace


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: rt_pipeline <source> <model.onnx> "
                     "[conf_thresh=0.45] [nms_thresh=0.50]\n";
        return 1;
    }

    std::string source = argv[1];
    std::string model_path = argv[2];
    float conf_thresh = (argc > 3) ? std::stof(argv[3]) : 0.45f;
    float nms_thresh  = (argc > 4) ? std::stof(argv[4]) : 0.50f;

    constexpr int INPUT_W = 640;
    constexpr int INPUT_H = 640;

    // ── Initialise components ───────────────────────────────────────────
    rt_pipeline::FrameReader reader(source);
    if (!reader.open()) return 1;

    rt_pipeline::Preprocessor preprocessor(INPUT_W, INPUT_H);

    rt_pipeline::ONNXInference inference(model_path, conf_thresh, nms_thresh,
                                         INPUT_W, INPUT_H);
    if (!inference.load()) return 1;

    std::cout << "[main] Pipeline ready. Press 'q' to quit.\n";

    // ── Frame loop ──────────────────────────────────────────────────────
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (reader.read(frame)) {
        frame_count++;

        // Preprocess
        cv::Mat blob = preprocessor.preprocess(frame);

        // Inference
        auto dets = inference.infer(blob, reader.width(), reader.height());

        // Draw
        drawDetections(frame, dets);

        // FPS calculation
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double fps = (elapsed > 0) ? frame_count / elapsed : 0.0;

        drawHUD(frame, fps, static_cast<int>(dets.size()),
                inference.lastInferenceMs());

        cv::imshow("RT Pipeline (C++)", frame);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
    }

    reader.release();
    cv::destroyAllWindows();

    auto total_time = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "\n[main] Done. Processed " << frame_count << " frames in "
              << std::fixed << std::setprecision(2) << total_time << "s ("
              << std::setprecision(1) << frame_count / total_time << " FPS)\n";

    return 0;
}
