cmake_minimum_required(VERSION 2.8)

find_package(PkgConfig)

add_subdirectory(userland)

project(OpenRover)

pkg_check_modules(EIGEN REQUIRED eigen3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(userland)
include_directories(userland/interface/vcos/pthreads)
include_directories(userland/interface/vmcs_host/linux)
include_directories(userland/host_applications/linux/libs/bcm_host/include)

add_executable(drive drive.cpp kalman_filter.cpp cam.cpp uWS/Extensions.cpp uWS/Group.cpp uWS/Networking.cpp uWS/Hub.cpp uWS/Node.cpp uWS/WebSocket.cpp uWS/HTTPSocket.cpp uWS/Socket.cpp uWS/Epoll.cpp)
target_link_libraries(drive opencv_core opencv_highgui opencv_video opencv_videoio mmal pigpiod_if2 rt ssl crypto z)

