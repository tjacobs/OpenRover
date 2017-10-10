#include "cam.h"

#include <stdlib.h>
#include <stdio.h>

#include "interface/mmal/mmal.h"
#include "interface/mmal/mmal_buffer.h"
#include "interface/mmal/util/mmal_connection.h"
#include "interface/mmal/util/mmal_default_components.h"
#include "interface/mmal/util/mmal_util.h"
#include "interface/mmal/util/mmal_util_params.h"

MMAL_POOL_T *Camera::camera_pool_ = NULL;
MMAL_COMPONENT_T *Camera::camera_ = NULL;
CameraReceiver *Camera::receiver_ = NULL;

CameraReceiver::~CameraReceiver() {}

void Camera::ControlCallback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer) {
  // Print command
  fprintf(stderr, "Camera control callback cmd=0x%08x", buffer->cmd);
  mmal_buffer_header_release(buffer);
}

void Camera::BufferCallback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer) {
  if (buffer->length) {
    if (receiver_ != NULL) {
      // Call OnFrame
      mmal_buffer_header_mem_lock(buffer);
      receiver_->OnFrame(buffer->data, buffer->length);
      mmal_buffer_header_mem_unlock(buffer);
    }
  }

  // Release buffer back to the pool
  mmal_buffer_header_release(buffer);

  // If port is still open
  if (port->is_enabled) {
    MMAL_STATUS_T status;
    MMAL_BUFFER_HEADER_T *new_buffer = mmal_queue_get(camera_pool_->queue);

    // Send buffer back to the port
    if (new_buffer) {
      status = mmal_port_send_buffer(port, new_buffer);
    }
    if (!new_buffer || status != MMAL_SUCCESS)
      fprintf(stderr, "Unable to return the buffer to the camera video port\n");
  }
}

bool Camera::Init(int width, int height, int fps) {

  // Check resolution
  if (width & 31) {
    fprintf(stderr, "Camera: Width must be multiple of 32");
    return false;
  }
  if (height & 15) {
    fprintf(stderr, "Camera: Height must be multiple of 16");
    return false;
  }

  // Create camera
  MMAL_STATUS_T status;
  if ((status = mmal_component_create(MMAL_COMPONENT_DEFAULT_CAMERA, &camera_)) != MMAL_SUCCESS) {
    fprintf(stderr, "Camera: Cannot create mmal camera component\n");
    return false;
  }

  // Enable the camera, and tell it its control callback function
  status = mmal_port_enable(camera_->control, ControlCallback);
  if (status != MMAL_SUCCESS) {
    fprintf(stderr, "Camera: Cannot enable camera\n");
    return false;
  }

  // Configure camera
  MMAL_PARAMETER_CAMERA_CONFIG_T cam_config = { { MMAL_PARAMETER_CAMERA_CONFIG, sizeof(cam_config) }};
  cam_config.max_stills_w = width;
  cam_config.max_stills_h = height;
  cam_config.stills_yuv422 = 0;
  cam_config.one_shot_stills = 0;
  cam_config.max_preview_video_w = width;
  cam_config.max_preview_video_h = height;
  cam_config.num_preview_video_frames = 3;
  cam_config.stills_capture_circular_buffer_height = 0;
  cam_config.fast_preview_resume = 0;
  cam_config.use_stc_timestamp = MMAL_PARAM_TIMESTAMP_MODE_RESET_STC;
  mmal_port_parameter_set(camera_->control, &cam_config.hdr);

  // Configure port
  MMAL_PORT_T *video_port = camera_->output[1];
  video_port->format->encoding = MMAL_ENCODING_I420;
  video_port->format->encoding_variant = MMAL_ENCODING_I420;
  video_port->format->es->video.width = width;
  video_port->format->es->video.height = height;
  video_port->format->es->video.crop.x = 0;
  video_port->format->es->video.crop.y = 0;
  video_port->format->es->video.crop.width = width;
  video_port->format->es->video.crop.height = height;
  video_port->format->es->video.frame_rate.num = fps;
  video_port->format->es->video.frame_rate.den = 1;
  status = mmal_port_format_commit(video_port);
  if (status != MMAL_SUCCESS) {
    fprintf(stderr, "Camera: cannot set video port format\n");
    return false;
  }

  // Ensure there are enough buffers to avoid dropping frames
  if (video_port->buffer_num < 20)
    video_port->buffer_num = 20;

  // Turn the red led on
  status = mmal_component_enable(camera_);
  if (status != MMAL_SUCCESS) {
    fprintf(stderr, "Camera: Cannot enable camera\n");
    return false;
  }
  fprintf(stderr, "Camera: Video port %s: %d buffers x %d bytes\n", video_port->name, video_port->buffer_num, video_port->buffer_size);

  // Create pool of buffer headers for the output port to consume
  camera_pool_ = mmal_port_pool_create(video_port, video_port->buffer_num, video_port->buffer_size);
  if (!camera_pool_) {
    fprintf(stderr, "Camera: Cannot create buffer header pool for video port\n");
    return false;
  }

  // Set up buffer callback
  if (mmal_port_enable(video_port, BufferCallback) != MMAL_SUCCESS) {
    fprintf(stderr, "Camera: Failed to set up camera output\n");
    return false;
  }

  // Set shutter speed to auto
  if (mmal_port_parameter_set_uint32(camera_->control, MMAL_PARAMETER_SHUTTER_SPEED, 0) != MMAL_SUCCESS) {
    fprintf(stderr, "Camera: Cannot set shutter speed\n");
    return false;
  }

  // Send all the buffers to the camera output port
  int qlen = mmal_queue_length(camera_pool_->queue);
  for (int q = 0; q < qlen; q++) {
    MMAL_BUFFER_HEADER_T *buffer = mmal_queue_get(camera_pool_->queue);
    if (!buffer) {
      fprintf(stderr, "Camera: Unable to get a required buffer %d from pool queue\n", q);
    }

    // Send it
    if (mmal_port_send_buffer(video_port, buffer)!= MMAL_SUCCESS) {
      fprintf(stderr, "Camera: Unable to send a buffer to camera output port (%d)\n", q);
    }
  }

  // Camera is set up
  return true;
}

bool Camera::StartRecord(CameraReceiver *receiver) {

  // Save receiver
  receiver_ = receiver;

  // Enable capturing
  MMAL_PORT_T *video_port = camera_->output[1];
  if (mmal_port_parameter_set_boolean(video_port, MMAL_PARAMETER_CAPTURE, 1) != MMAL_SUCCESS) {
    fprintf(stderr, "Camera: Failed to start capture\n");
    return false;
  }

  // We are capturing
  return true;
}

bool Camera::StopRecord() {

  // Disable capturing
  MMAL_PORT_T *video_port = camera_->output[1];
  if (mmal_port_parameter_set_boolean(video_port, MMAL_PARAMETER_CAPTURE, 0) != MMAL_SUCCESS) {
    fprintf(stderr, "Camera: Failed to stop capture\n");
    return false;
  }

  // Close receiver
  receiver_ = NULL;

  // We have stopped capturing
  return true;
}
