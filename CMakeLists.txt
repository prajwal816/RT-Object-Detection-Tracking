cmake_minimum_required(VERSION 3.14)
project(RTObjectDetectionTracking VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Find OpenCV ──────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

# ── Include directories ──────────────────────────────────────────────────────
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(cpp/opencv_pipeline/include)

# ── Source files ─────────────────────────────────────────────────────────────
set(PIPELINE_SOURCES
    cpp/opencv_pipeline/src/frame_reader.cpp
    cpp/opencv_pipeline/src/preprocessor.cpp
    cpp/opencv_pipeline/src/onnx_inference.cpp
    cpp/opencv_pipeline/src/main.cpp
)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(rt_pipeline ${PIPELINE_SOURCES})
target_link_libraries(rt_pipeline PRIVATE ${OpenCV_LIBS})

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS rt_pipeline DESTINATION bin)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs:    ${OpenCV_LIBS}")
