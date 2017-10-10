cmake_minimum_required(VERSION 2.8)

find_package(PkgConfig)

add_subdirectory(userland)

project(OpenRover)

pkg_check_modules(EIGEN REQUIRED eigen3)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(userland)
include_directories(userland/interface/vcos/pthreads)
include_directories(userland/interface/vmcs_host/linux)
include_directories(userland/host_applications/linux/libs/bcm_host/include)

add_executable(drive drive.cpp kalman_filter.cpp cam.cpp)
target_link_libraries(drive opencv_core opencv_highgui opencv_video opencv_videoio mmal pigpiod_if2 rt)

