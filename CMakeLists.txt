cmake_minimum_required(VERSION 3.5)
project(lanedetection)

find_package(CUDA 9.0 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("~/tensorflow-1.5.0/tensorflow/contrib/makefile/downloads/eigen/")
include_directories("~/tensorflow-1.5.0/")
include_directories("~/tensorflow-1.5.0/bazel-genfiles/")
include_directories("~/tensorflow-1.5.0/bazel-genfiles/tensorflow/")
include_directories("~/tensorflow-1.5.0/tensorflow/")
include_directories("~/tensorflow-1.5.0/tensorflow/contrib/makefile/downloads/nsync" )
include_directories("~/tensorflow-1.5.0/tensorflow/contrib/makefile/downloads/nsync/public")
include_directories("~/tensorflow-1.5.0/third-party/")
include_directories("~/protobuf/src/")

set(SOURCE_FILES LaneProcessing.cpp nonzero.cu binary.cpp calibration.cpp erosion_dilation.cpp first_frameprocessing.cpp gray.cpp hsv.cpp imagewrap.cpp nxt_frameprocessing.cpp polyfit.cpp processinga.cpp processingb.cpp resize.cpp sanity_check.cpp sobel.cpp video_processing.cpp yolo.cpp  all_header.h)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

CUDA_ADD_EXECUTABLE(lanedetection ${SOURCE_FILES})

list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61, code=sm_61; -std=c++11")

target_link_libraries(lanedetection "~/tensorflow-1.5.0/bazel-bin/tensorflow/libtensorflow_cc.so" ${OpenCV_LIBS})




