#ifndef DRIVE_H_
#define DRIVE_H_

#include <eigen3/Eigen/Dense>
#include <math.h>
#include "kalman_filter.h"

class Drive {
public:
  Drive();

  void Update(float throttle, float steering, float dt);
  void Reset();
  void UpdateCamera(const uint8_t *frame);

  KalmanFilter kalman_filter;

  bool _first_frame;
};

#endif  // DRIVE_H_