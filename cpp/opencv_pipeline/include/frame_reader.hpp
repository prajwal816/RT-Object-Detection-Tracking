#pragma once
/**
 * @file frame_reader.hpp
 * @brief Video frame capture and streaming using OpenCV VideoCapture.
 */

#include <string>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>

namespace rt_pipeline {

/**
 * @class FrameReader
 * @brief Reads frames from a video file or camera device.
 */
class FrameReader {
public:
    /**
     * @param source Camera index (as string) or video file path.
     */
    explicit FrameReader(const std::string& source);
    ~FrameReader();

    /** Open the video source. Returns true on success. */
    bool open();

    /** Read the next frame. Returns false at end of stream. */
    bool read(cv::Mat& frame);

    /** Release the capture device. */
    void release();

    /** Check if the source is currently open. */
    bool isOpened() const;

    /** Get the frame width. */
    int width() const;

    /** Get the frame height. */
    int height() const;

    /** Get the source FPS (may be 0 for cameras). */
    double fps() const;

private:
    std::string source_;
    cv::VideoCapture cap_;
};

} // namespace rt_pipeline
