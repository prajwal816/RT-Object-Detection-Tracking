/**
 * @file preprocessor.cpp
 * @brief Implementation of the Preprocessor class.
 */

#include "preprocessor.hpp"
#include <opencv2/imgproc.hpp>

namespace rt_pipeline {

Preprocessor::Preprocessor(int input_width,
                           int input_height,
                           double scale_factor,
                           bool swap_rb)
    : input_width_(input_width),
      input_height_(input_height),
      scale_factor_(scale_factor),
      swap_rb_(swap_rb) {}

cv::Mat Preprocessor::preprocess(const cv::Mat& frame) const {
    cv::Mat blob = cv::dnn::blobFromImage(
        frame,
        scale_factor_,
        cv::Size(input_width_, input_height_),
        cv::Scalar(0, 0, 0),   // mean subtraction
        swap_rb_,               // swap R & B
        false                   // crop
    );
    return blob;
}

} // namespace rt_pipeline
