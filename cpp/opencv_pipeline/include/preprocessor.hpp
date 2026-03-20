#pragma once
/**
 * @file preprocessor.hpp
 * @brief Frame preprocessing — resize, normalize, blob creation for DNN.
 */

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace rt_pipeline {

/**
 * @class Preprocessor
 * @brief Converts raw BGR frames into DNN-ready blobs.
 */
class Preprocessor {
public:
    /**
     * @param input_width  Model input width (e.g. 640).
     * @param input_height Model input height (e.g. 640).
     * @param scale_factor Pixel value scale (e.g. 1/255).
     * @param swap_rb      Swap Red and Blue channels.
     */
    Preprocessor(int input_width = 640,
                 int input_height = 640,
                 double scale_factor = 1.0 / 255.0,
                 bool swap_rb = true);

    /**
     * Create a 4D blob from a BGR frame.
     * @param frame Input BGR image.
     * @return Preprocessed blob ready for DNN inference.
     */
    cv::Mat preprocess(const cv::Mat& frame) const;

    int inputWidth() const { return input_width_; }
    int inputHeight() const { return input_height_; }

private:
    int input_width_;
    int input_height_;
    double scale_factor_;
    bool swap_rb_;
};

} // namespace rt_pipeline
