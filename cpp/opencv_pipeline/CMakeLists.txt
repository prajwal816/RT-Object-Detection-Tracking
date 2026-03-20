# C++ OpenCV Pipeline — Subdirectory CMake
cmake_minimum_required(VERSION 3.14)

# This file is consumed by the top-level CMakeLists.txt.
# It is kept here for standalone build support:
#
#   cd cpp/opencv_pipeline
#   cmake -S . -B build && cmake --build build

project(opencv_pipeline LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_executable(rt_pipeline_standalone
    src/frame_reader.cpp
    src/preprocessor.cpp
    src/onnx_inference.cpp
    src/main.cpp
)

target_link_libraries(rt_pipeline_standalone PRIVATE ${OpenCV_LIBS})
