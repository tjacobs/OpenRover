#ifndef DRIVE_H_
#define DRIVE_H_

#include <eigen3/Eigen/Dense>
#include <math.h>
#include "kalman_filter.h"

class Drive {
public:
  Drive();

  void Reset();
  void Update(float throttle, float steering, float dt);
  void UpdateCamera(const uint8_t *frame);
  
  void UpdateState(const uint8_t *yuv, size_t yuvlen,
      float throttle_in, float steering_in,
      const Eigen::Vector3f &accel,
      const Eigen::Vector3f &gyro,
      uint8_t servo_pos,
      const uint16_t *wheel_encoders, float dt);
  bool GetControl(float *throttle_out, float *steering_out, float dt);

  EKF kalman_filter;

  bool _first_frame;
};

#endif  // DRIVE_H_