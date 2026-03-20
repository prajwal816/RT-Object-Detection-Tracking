/**
 * @file frame_reader.cpp
 * @brief Implementation of the FrameReader class.
 */

#include "frame_reader.hpp"
#include <iostream>

namespace rt_pipeline {

FrameReader::FrameReader(const std::string& source)
    : source_(source) {}

FrameReader::~FrameReader() {
    release();
}

bool FrameReader::open() {
    // Try to parse as integer (camera index)
    try {
        int cam_idx = std::stoi(source_);
        cap_.open(cam_idx);
    } catch (const std::exception&) {
        // Treat as file path
        cap_.open(source_);
    }

    if (!cap_.isOpened()) {
        std::cerr << "[FrameReader] ERROR: Cannot open source: "
                  << source_ << std::endl;
        return false;
    }

    std::cout << "[FrameReader] Opened source: " << source_
              << " (" << width() << "x" << height()
              << " @ " << fps() << " FPS)" << std::endl;
    return true;
}

bool FrameReader::read(cv::Mat& frame) {
    return cap_.read(frame);
}

void FrameReader::release() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool FrameReader::isOpened() const {
    return cap_.isOpened();
}

int FrameReader::width() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int FrameReader::height() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double FrameReader::fps() const {
    return cap_.get(cv::CAP_PROP_FPS);
}

} // namespace rt_pipeline
