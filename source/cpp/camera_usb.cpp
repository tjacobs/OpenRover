#include "camera.h"

#include <stdlib.h>
#include <stdio.h>

CameraReceiver::~CameraReceiver() {}

void Camera::ControlCallback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer) {

}

void Camera::BufferCallback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer) {
}

bool Camera::Init(int width, int height, int fps) {

  // Camera is set up
  return true;
}

bool Camera::StartRecord(CameraReceiver *receiver) {

  // We are capturing
  return true;
}

bool Camera::StopRecord() {

  // We have stopped capturing
  return true;
}
