cmake_minimum_required(VERSION 2.8)

find_package(PkgConfig)

if(LINUX)
  add_subdirectory(userland)
endif()

project(OpenRover)

pkg_check_modules(EIGEN REQUIRED eigen3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -I ~/macports/include") 

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(userland)
include_directories(userland/interface/vcos/pthreads)
include_directories(userland/interface/vmcs_host/linux)
include_directories(userland/host_applications/linux/libs/bcm_host/include)

add_executable(drive drive.cpp kalman_filter.cpp camera_usb.cpp uWS/Extensions.cpp uWS/Group.cpp uWS/Networking.cpp uWS/Hub.cpp uWS/Node.cpp uWS/WebSocket.cpp uWS/HTTPSocket.cpp uWS/Socket.cpp uWS/Epoll.cpp)

target_link_libraries(drive -L/usr/local/lib opencv_core opencv_highgui opencv_video opencv_videoio opencv_imgcodecs ssl crypto z pthread) # Pi: mmal pigpiod_if2 rt uv

