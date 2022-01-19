cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 14)
set(THREADS_PREFER_PTHREAD_FLAG ON)

project(nanodet_demo)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
find_package(realsense2 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(nanodet_demo main.cpp nanodet_openvino.cpp)

target_link_libraries(
    nanodet_demo
    ${InferenceEngine_LIBRARIES}
    ${NGRAPH_LIBRARIES}
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
    Threads::Threads
)
