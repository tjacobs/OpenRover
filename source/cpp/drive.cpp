#include <iostream>
#include <sys/time.h>
#include <fcntl.h>
#include <semaphore.h>
#include <pigpiod_if2.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "drive.h"
#include "cam.h"
#include <uWS/uWS.h>

using namespace std;
using Eigen::Matrix2f;
using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::MatrixXf;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::VectorXf;

// Our output throttle value
int8_t throttle_ = 0;

// Our output steering angle
int8_t steering_ = 0;

// The actual current angle of steering (as servos aren't instant)
uint8_t servo_pos_ = 110;

// The current IMU values
Eigen::Vector3f accel_(0, 0, 0), gyro_(0, 0, 0);

// The four wheel encoder values from the four wheels
uint16_t wheel_pos_[4] = {0, 0, 0, 0};
uint16_t last_encoders_[4];

// Time keeping
struct timeval t;
struct timeval _last_t;

// Keyboard key presses
char key;

int pi;

static const float MAX_THROTTLE = 0.8;
static const float SPEED_LIMIT = 5.0;

static const float ACCEL_LIMIT = 4.0;     // Maximum dv/dt (m/s^2)
static const float BRAKE_LIMIT = -100.0;  // Minimum dv/dt
static const float TRACTION_LIMIT = 4.0;  // Maximum v*w product (m/s^2)
static const float kpy = 1.0;
static const float kvy = 2.0;

static const float LANE_OFFSET = 0.0;
static const float LANEOFFSET_PER_K = 0.0;

Drive::Drive() {
  Reset();
}

void Drive::Reset() {
  kalman_filter.Reset();
  _first_frame = true;
}

void Drive::Update(float throttle, float steering, float dt) {

  // Get our current model of the world
  Eigen::VectorXf &_x = kalman_filter.GetState();

  // Freak out if we freaked out
  if(isinf(_x[0]) || isnan(_x[0])) {
    fprintf(stderr, "Caution: Massive freakout currently underweigh.\n");
    Reset();
    return;
  }

  // Start off sensibly
  if(_first_frame) {
    _first_frame = false;
  }

  // After 'dt' milliseconds since we last thought about it, what would we expect the world to look like now? 
  kalman_filter.Predict(dt, throttle, steering);

  // Log
  std::cout << "World state after our prediction: " << _x.transpose() << std::endl;

}

static inline float clip(float x, float min, float max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}



// Asynchronously flush files to disk
struct FlushEntry {
  int fd_;
  uint8_t *buf_;
  size_t len_;

  FlushEntry() { buf_ = NULL; }
  FlushEntry(int fd, uint8_t *buf, size_t len):
    fd_(fd), buf_(buf), len_(len) {}

  void flush() {
    if (len_ == -1) {
      fprintf(stderr, "FlushThread: closing fd %d\n", fd_);
      close(fd_);
    }
    if (buf_ != NULL) {
      if (write(fd_, buf_, len_) != len_) {
        perror("FlushThread write");
      }
      delete[] buf_;
      buf_ = NULL;
    }
  }
};

class FlushThread {
 public:
  FlushThread() {
    pthread_mutex_init(&mutex_, NULL);
    sem_init(&sem_, 0, 0);
  }

  ~FlushThread() {
  }

  bool Init() {
    if (pthread_create(&thread_, NULL, thread_entry, this) != 0) {
      perror("FlushThread: pthread_create");
      return false;
    }
    return true;
  }

  void AddEntry(int fd, uint8_t *buf, size_t len) {
    static int count = 0;
    pthread_mutex_lock(&mutex_);
    flush_queue_.push_back(FlushEntry(fd, buf, len));
    size_t siz = flush_queue_.size();
    pthread_mutex_unlock(&mutex_);
    sem_post(&sem_);
    count++;
    if (count >= 15) {
      if (siz > 2) {
        fprintf(stderr, "[FlushThread %d]\r", siz);
        fflush(stderr);
      }
      count = 0;
    }
  }

 private:
  static void* thread_entry(void* arg) {

    FlushThread *self = reinterpret_cast<FlushThread*>(arg);
    for (;;) {
      sem_wait(&self->sem_);
      pthread_mutex_lock(&self->mutex_);
      if (!self->flush_queue_.empty()) {
        FlushEntry e = self->flush_queue_.front();
        self->flush_queue_.pop_front();
        pthread_mutex_unlock(&self->mutex_);
        e.flush();
      } else {
        pthread_mutex_unlock(&self->mutex_);
      }
    }
  }

  std::deque<FlushEntry> flush_queue_;
  pthread_mutex_t mutex_;
  pthread_t thread_;
  sem_t sem_;
};

FlushThread flush_thread_;

// At what y value down the image do we start looking?
static const int ytop = 100;

// The uxrange is (-56, 55) the uyrange is (2, 59), so x0=-56 and y0-2
static const int ux0 = -56, uy0 = 2;

// Threshold for the difference between track lane and ground U value. Reduce to 15 for shiny floors.
static const int ACTIV_THRESH = 30;

// U size, width and height
static const int uxsiz = 111, uysiz = 57;

// How wide is one pixel in the real world?
static const float pixel_scale_m = 0.025;

// Bucketcount has resolution: (57, 111), (3197, 2)
static const uint16_t bucketcount[uxsiz * uysiz] = {
#include "bucketcount.txt"
};

static const uint16_t floodmap[uxsiz * uysiz] = {
#include "floodmap.txt"
};

static const uint8_t udmask[320*(240-ytop)] = {
#include "udmask.txt"
};

static const int8_t udplane[320*(240-ytop)*2] = {
#include "udplane.txt"
};

bool TophatFilter(const uint8_t *yuv, Vector3f *Bout, float *y_cout, Matrix4f *Rkout) {

  // Input is a 640x480 YUV420 image, create buffer
  int32_t accumbuf[uxsiz * uysiz * 3];
  memset(accumbuf, 0, uxsiz * uysiz * 3 * sizeof(accumbuf[0]));

  // Snapshot at 40 frames in to give the camera time to adjust light
  static int snapshot = 0;
  snapshot++;
  FILE *fp = NULL;
  if (snapshot == 40) {
    fp = fopen("snapshot.bin", "w");
  }

  // For each yuv, remap into detected
  size_t bufidx = ytop*320;
  size_t udidx = 0;
  for (int j = 0; j < 240 - ytop; j++) {
    for (int i = 0; i < 320; i++, bufidx++, udidx++) {
      uint8_t y = yuv[(j+ytop)*2*640 + 2*i];
      uint8_t u = yuv[640*480 + bufidx];
      uint8_t v = yuv[640*480 + 320*240 + bufidx];

      // Snapshot YUV values for this pixel
      if (fp) {
        fwrite(&y, 1, 1, fp);
        fwrite(&u, 1, 1, fp);
        fwrite(&v, 1, 1, fp);
      }

      // What is ud mask?  Skip this pixel if the ud mask says so
      if (!udmask[udidx]) continue;

      // Write at dx dy into our buffer 
      int dx = udplane[udidx*2] - ux0;
      int dy = udplane[udidx*2 + 1] - uy0;
      accumbuf[(uxsiz * dy + dx) * 3] += y;
      accumbuf[(uxsiz * dy + dx) * 3 + 1] += u;
      accumbuf[(uxsiz * dy + dx) * 3 + 2] += v;
    }
  }

  // Take an average
  size_t uidx = 0;
  for (int j = 0; j < uysiz; j++) {
    for (int i = 0; i < uxsiz; uidx++, i++) {
      if (bucketcount[uidx] > 0) {
        accumbuf[uidx*3] /= bucketcount[uidx];
        accumbuf[uidx*3 + 1] /= bucketcount[uidx];
        accumbuf[uidx*3 + 2] /= bucketcount[uidx];
      }

      // Add to snapshot
      if (fp) {
        fwrite(&accumbuf[uidx*3], 4, 3, fp);
      }
    }
  }

  // Flood fill
  uidx = 0;
  for (int j = 0; j < uysiz; j++) {
    for (int i = 0; i < uxsiz; uidx++, i++) {
      if (bucketcount[uidx] == 0) {
        accumbuf[uidx*3] = accumbuf[floodmap[uidx]*3];
        accumbuf[uidx*3 + 1] = accumbuf[floodmap[uidx]*3 + 1];
        accumbuf[uidx*3 + 2] = accumbuf[floodmap[uidx]*3 + 2];
      }

      // Add to snapshot
      if (fp) {
        fwrite(&accumbuf[uidx*3], 4, 3, fp);
      }
    }
  }

  // Take horizontal cumulative sum
  for (int j = 0; j < uysiz; j++) {
    for (int i = 1; i < uxsiz; i++) {
      accumbuf[3*(j*uxsiz + i)] += accumbuf[3*(j*uxsiz + i - 1)];
      accumbuf[3*(j*uxsiz + i) + 1] += accumbuf[3*(j*uxsiz + i - 1) + 1];
      accumbuf[3*(j*uxsiz + i) + 2] += accumbuf[3*(j*uxsiz + i - 1) + 2];
    }
  }

  // Take horizontal convolution with tophat kernel (it looks like a tophat): [-1, -1, 2, 2, -1, -1]
  // And add linear regression points XTX, Xty, yTy, N.
  Matrix3f regXTX = Matrix3f::Zero();
  Vector3f regXTy = Vector3f::Zero();
  double regyTy = 0;
  double regxsum = 0;
  double regwsum = 0;
  int regN = 0;
  for (int j = 0; j < uysiz; j++) {
    for (int i = 0; i < uxsiz-7; i++) {
      int32_t yd =
        -(accumbuf[3*(j*uxsiz + i + 6)] - accumbuf[3*(j*uxsiz + i)])
        + 3*(accumbuf[3*(j*uxsiz + i + 4)] - accumbuf[3*(j*uxsiz + i + 2)]);
      int32_t ud =
        -(accumbuf[3*(j*uxsiz + i + 6) + 1] - accumbuf[3*(j*uxsiz + i) + 1])
        + 3*(accumbuf[3*(j*uxsiz + i + 4) + 1] - accumbuf[3*(j*uxsiz + i + 2) + 1]);
      int32_t vd =
        -(accumbuf[3*(j*uxsiz + i + 6) + 2] - accumbuf[3*(j*uxsiz + i) + 2])
        + 3*(accumbuf[3*(j*uxsiz + i + 4) + 2] - accumbuf[3*(j*uxsiz + i + 2) + 2]);

      // Add to snapshot
      if (fp) {
        fwrite(&yd, 4, 1, fp);
        fwrite(&ud, 4, 1, fp);
        fwrite(&vd, 4, 1, fp);
      }

      // Calculate detected
      int32_t detected = -ud - ACTIV_THRESH;
      if (detected > 0) {

        // Add x, y to linear regression
        float pu = pixel_scale_m * (i + ux0 + 3),
              pv = pixel_scale_m * (j + uy0);

        // Use activation as regression weight
        float w = detected;
        Vector3f regX(w*pv*pv, w*pv, w);
        regxsum += w*pv;
        regwsum += w;
        regXTX.noalias() += regX * regX.transpose();
        regXTy.noalias() += regX * w * pu;
        regyTy += w * w * pu * pu;
        regN += 1;
      }
    }
  }

  // Close snapshot
  if (fp) {
    fclose(fp);
    fp = NULL;
  }

  // Print
  std::cout << "Number of activations: " << regN << "\n";

  // If not enough data, don't even try to do an update
  if (regN < 8) {
    return false;
  }

  // Linear fit B = XTX.inverse * XTy
  Matrix3f XTXinv = regXTX.inverse();
  Vector3f B = XTXinv * regXTy;
  *Bout = B;

  // (XB).T y
  // BT XTy
  float r2 = B.dot(regXTX * B) - 2*B.dot(regXTy) + regyTy;
  r2 *= 100.0 / (regN - 1);

  // Save y_c
  *y_cout = regxsum / regwsum;

#if 0
  std::cout << "XTX\n" << regXTX << "\n";
  std::cout << "XTy " << regXTy.transpose() << "\n";
  std::cout << "yTy " << regyTy << "\n";
  std::cout << "XTXinv\n" << XTXinv << "\n";
#endif

#if 1
  cout << "B " << B.transpose() << "\n";
  cout << "r2 " << r2 << "\n";
  cout << "y_c " << *y_cout << "\n";
#endif

  // All good?
  if (isnan(r2)) {
    return false;
  }

  // What is our covariance?
  (*Rkout).topLeftCorner(3, 3) = XTXinv * r2;
  (*Rkout)(3, 3) = regXTX(1, 1) / regwsum - *y_cout;
  return true;
}

void Drive::UpdateCamera(const uint8_t *frame) {
  Vector3f B;
  Matrix4f Rk = Matrix4f::Zero();
  float y_center;

  // Look for that line
  if (!TophatFilter(frame, &B, &y_center, &Rk)) {
    return;
  }

  // The road centerline's position, angle, and curvature is represented in B, as:
  //    ax^2 + bx + c
  // and the covariance in Rk.

  // The y_center is the vertical center of the original datapoints, where our regression should
  // have the least amount of error.
  // We measure the centerline curvature and angle psi_error at this point,
  // and compute y_error as our perpendicular distance to that line. Simples.
  //
  // 
  //                /
  // psi_error &   /
  // curvature -> |___|  <- y_error
  // at y_center  |
  //               \
  //                \
  //
  // The regression line is x = a*y^2 + b*y^1 + c
  //
  //  xc = a*yc**2 + b*yc + c
  //
  // We've got our linear fit B[0:2] (B[a, b, c]) and our measurement covariance Rk, time to do the sensor fusion step
  kalman_filter.UpdateCenterline(B[0], B[1], B[2], y_center, Rk);
}

void Drive::UpdateState(const uint8_t *yuv, size_t yuvlen,
      float throttle_in, float steering_in,
      const Vector3f &accel, const Vector3f &gyro,
      uint8_t servo_pos, const uint16_t *wheel_encoders, float dt) {
  Eigen::VectorXf &x_ = kalman_filter.GetState();

  if (isinf(x_[0]) || isnan(x_[0])) {
    fprintf(stderr, "WARNING: kalman filter diverged to inf/NaN! resetting!\n");
    Reset();
    return;
  }

  if (_first_frame) {
    memcpy(last_encoders_, wheel_encoders, 4*sizeof(uint16_t));
    _first_frame = false;
  }

  kalman_filter.Predict(dt, throttle_in, steering_in);
  std::cout << "x after predict " << x_.transpose() << std::endl;

  if (yuvlen == 640*480 + 320*240*2) {
    UpdateCamera(yuv);
    cout << "x = v, delta, y_error, psi_error, curvature, ml_1,ml_2,ml_3,ml_4, srv_a,srv_b,srv_r,srvfb_a,srvfb_b, gyro" << endl;
    std::cout << "x after camera: " << x_.transpose() << std::endl;
  } else {
    fprintf(stderr, "Drive::UpdateState: invalid yuvlen %ld, expected %d\n", yuvlen, 640*480 + 320*240*2);
  }

  //kalman_filter.UpdateIMU(gyro[2]);
  //std::cout << "x after IMU (" << gyro[2] << ")" << x_.transpose() << std::endl;

  // Force psi_e to be forward facing
  if (x_[3] > M_PI/2) {
    x_[3] -= M_PI;
  } else if (x_[3] < -M_PI/2) {
    x_[3] += M_PI;
  }

  // read / update servo & encoders
  // use the average of the two rear encoders as we're most interested in the
  // motor speed
  // but we could use all four to get turning radius, etc.
  // since the encoders are just 16-bit counters which wrap frequently, we only
  // track the difference in counts between updates.
  //printf("encoders were: %05d %05d %05d %05d\n"
  //       "      are now: %05d %05d %05d %05d\n",
  //    last_encoders_[0], last_encoders_[1], last_encoders_[2], last_encoders_[3],
  //    wheel_encoders[0], wheel_encoders[1], wheel_encoders[2], wheel_encoders[3]);

  // average ds among wheel encoders which are actually moving
  float ds = 0, nds = 0;
  for (int i = 2; i < 4; i++) {  // only use rear encoders
    if (wheel_encoders[i] != last_encoders_[i]) {
      ds += (uint16_t) (wheel_encoders[i] - last_encoders_[i]);
      nds += 1;
    }
  }
  memcpy(last_encoders_, wheel_encoders, 4*sizeof(uint16_t));

  // and do an kalman_filter update if the wheels are moving.
  if (nds > 0) {
//    kalman_filter.UpdateEncoders(ds/(nds * dt), servo_pos);
    //std::cout << "x after encoders (" << ds/dt << ") " << x_.transpose() << std::endl;
  } else {
//    kalman_filter.UpdateEncoders(0, servo_pos);
    //std::cout << "x after encoders (" << ds/dt << ") " << x_.transpose() << std::endl;
  }

  //std::cout << "P " << kalman_filter.GetCovariance().diagonal().transpose() << std::endl;
}

static float MotorControl(float accel, float k1, float k2, float k3, float k4, float v) {
  float a_thresh = -k3 * v - k4;

  // voltage (1 or 0)
  float V = accel > a_thresh ? 1 : 0;

  // duty cycle
  float DC = clip((accel + k3*v + k4) / (V*k1 - k2*v), 0, 1);
  return V == 1 ? DC : -DC;
}

bool Drive::GetControl(float *throttle_out, float *steering_out, float dt) {
  const Eigen::VectorXf &x_ = kalman_filter.GetState();
  float v = x_[0];
  float delta = x_[1];
  float y_e = x_[2];
  float psi_e = x_[3];
  float kappa = x_[4];
  float ml_1 = x_[5];
  float ml_2 = x_[6];
  float ml_3 = x_[7];
  float ml_4 = x_[8];
  float srv_a = x_[9];
  float srv_b = x_[10];
  float srv_r = x_[11];

  float k1 = exp(ml_1), k2 = exp(ml_2), k3 = exp(ml_3), k4 = exp(ml_4);

  float vmax = fmin(SPEED_LIMIT, (k1 - k4)/(k2 + k3));

  // TODO: race line following w/ particle filter localization
  float lane_offset = clip(LANE_OFFSET + kappa * LANEOFFSET_PER_K, -1.0, 1.0);
  float psi_offset = 0;

  float cpsi = cos(psi_e - psi_offset),
        spsi = sin(psi_e - psi_offset);
  float dx = cpsi / (1.0 - kappa*y_e);

  // it's a little backwards though because our steering is reversed w.r.t. curvature
  float k_target = dx * (-(y_e - lane_offset) * dx * kpy*cpsi - spsi*(-kappa*spsi - kvy*cpsi) + kappa);

  *steering_out = clip((k_target - srv_b) / srv_a, -1, 1);
  if (*steering_out == -1 || *steering_out == 1) {
    // steering is clamped, so we may need to further limit speed
    float w_target = v * k_target;
    float k_limit = srv_a * (*steering_out) + srv_b;
    vmax = fmin(vmax, w_target / k_limit);
  }

  float v_target = fmin(vmax, sqrtf(TRACTION_LIMIT / fabs(k_target)));
  float a_target = clip(v_target - v, BRAKE_LIMIT, ACCEL_LIMIT) / dt;
  if (a_target > 0) {  // accelerate more gently than braking
    a_target /= 4;
  }
  *throttle_out = clip(MotorControl(a_target, k1, k2, k3, k4, v), -1, MAX_THROTTLE);

//  printf("steer_target %f delta %f v_target %f v %f a_target %f lateral_a %f/%f v %f y %f psi %f\n",
//      k_target, delta, v_target, v, a_target, v*v*delta, TRACTION_LIMIT, v, y_e, psi_e);

  //printf("Throttle: %f, Steering: %f\n", *throttle_out, *steering_out);
  return true;
}


class Driver: public CameraReceiver {
 public:
  Driver() {
    output_fd_ = -1;
    frame_ = 0;
    frameskip_ = 0;
    autosteer_ = true;
    gettimeofday(&last_t_, NULL);
  }

  bool StartRecording(const char *fname, int frameskip) {
    frameskip_ = frameskip;
    if (!strcmp(fname, "-")) {
      output_fd_ = fileno(stdout);
    } else {
      output_fd_ = open(fname, O_CREAT|O_TRUNC|O_WRONLY, 0666);
    }
    if (output_fd_ == -1) {
      perror(fname);
      return false;
    }
    return true;
  }

  bool IsRecording() {
    return output_fd_ != -1;
  }

  void StopRecording() {
    if (output_fd_ == -1) {
      return;
    }
    flush_thread_.AddEntry(output_fd_, NULL, -1);
    output_fd_ = -1;
  }

  ~Driver() {
    StopRecording();
  }

  void OnFrame(uint8_t *buf, size_t length) {
    struct timeval t;
    gettimeofday(&t, NULL);
    frame_++;
    if (IsRecording() && frame_ > frameskip_) {
      frame_ = 0;

      // Save U channel below ytop only
      static const int ytop = 100;
      
      // Copy our frame, push it onto a stack to be flushed
      uint32_t flushlen = 55 + (240-ytop) * 320;
      uint8_t *flushbuf = new uint8_t[flushlen];
      memcpy(flushbuf, &flushlen, 4);  // write header length
      memcpy(flushbuf+4, &t.tv_sec, 4);
      memcpy(flushbuf+8, &t.tv_usec, 4);
      memcpy(flushbuf+12, &throttle_, 1);
      memcpy(flushbuf+13, &steering_, 1);
      memcpy(flushbuf+14, &accel_[0], 4);
      memcpy(flushbuf+14+4, &accel_[1], 4);
      memcpy(flushbuf+14+8, &accel_[2], 4);
      memcpy(flushbuf+26, &gyro_[0], 4);
      memcpy(flushbuf+26+4, &gyro_[1], 4);
      memcpy(flushbuf+26+8, &gyro_[2], 4);
      memcpy(flushbuf+38, &servo_pos_, 1);
      memcpy(flushbuf+39, wheel_pos_, 2*4);
      //memcpy(flushbuf+47, wheel_dt_, 2*4);
      memcpy(flushbuf+55, buf + 640*480 + ytop*320, (240-ytop)*320);

      // Check timing
      struct timeval t1;
      gettimeofday(&t1, NULL);
      float dt = t1.tv_sec - t.tv_sec + (t1.tv_usec - t.tv_usec) * 1e-6;
      if (dt > 0.1) {
        fprintf(stderr, "CameraThread::OnFrame: WARNING: alloc/copy took %fs\n", dt);
      }

      // Flush
      flush_thread_.AddEntry(output_fd_, flushbuf, flushlen);
      struct timeval t2;
      gettimeofday(&t2, NULL);
      dt = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1e-6;
      if (dt > 0.1) {
        fprintf(stderr, "CameraThread::OnFrame: WARNING: flush_thread.AddEntry took %fs\n", dt);
      }
    }

    {
    // Check timing
    static struct timeval t0 = {0, 0};
    float dt = t.tv_sec - t0.tv_sec + (t.tv_usec - t0.tv_usec) * 1e-6;
    if (dt > 0.2 && t0.tv_usec != 0) {
      fprintf(stderr, "CameraThread::OnFrame: WARNING: %fs gap between frames?!\n", dt);
    }
    t0 = t;
    }

    // Update Kalman Filter
    float u_a = throttle_ / 127.0;
    float u_s = steering_ / 127.0;
    float dt = t.tv_sec - last_t_.tv_sec + (t.tv_usec - last_t_.tv_usec) * 1e-6;
    controller_.UpdateState(buf, length,
            u_a, u_s,
            accel_, gyro_,
            servo_pos_, wheel_pos_,
            dt);
    last_t_ = t;

    // Output actuations
    if (autosteer_ && controller_.GetControl(&u_a, &u_s, dt)) {
      steering_ = 127 * u_s;
      throttle_ = 127 * u_a;
      int width = max(980, min(1500, steering_+1200));
      set_servo_pulsewidth(pi, 17, 1000);
      set_servo_pulsewidth(pi, 27, width);
      printf("Servo 27: %d\n", width);
      //teensy.SetControls(frame_ & 4 ? 1 : 0, throttle_, steering_);
      // pca.SetPWM(PWMCHAN_STEERING, steering_);
      // pca.SetPWM(PWMCHAN_ESC, throttle_);
    }
  }

  bool autosteer_;
  Drive controller_;
  int frame_;

 private:
  int output_fd_;
  int frameskip_;
  struct timeval last_t_;
};


int main(){
  
  // Start it up
  printf("\nStarting OpenRover.\n");

  uWS::Hub h;
  const char* message = "Websocket";
    
  h.onMessage([](uWS::WebSocket<uWS::SERVER> *ws, char *message, size_t length, uWS::OpCode opCode) {
    ws->send(message, strlen(message), opCode);
  });

  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t length, size_t remainingBytes) {
    res->end("Woo", 3);
  });

  if (h.listen(3000)) {
    h.run();
  }

  // Start PWM output
  pi = pigpio_start(0, 0);
  if (pi < 0) {
    printf("Error connecting to PWM.\n");
    return 1;
  }
  set_PWM_range(pi, 17, 1000);
  set_PWM_range(pi, 27, 1000);
  set_PWM_frequency(pi, 17, 50);
  set_PWM_frequency(pi, 27, 50);

  // Start up our car driver
  Drive drive;

  // Create window
  //cvNamedWindow("Camera_Output", 1);

  // Start cam
  //CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
  //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 320);
  //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);
 
  // Start disk writing thread 
  if (!flush_thread_.Init()) {
    return 1;
  }

  // Start camera
  Driver driver;
  int fps = 30;
  if (!Camera::Init(640, 480, fps))
    return 1;
  if (!Camera::StartRecord(&driver)) {
    return 1;
  }

  // Start recording
  int frameskip = 4;
  int recording_num = 0;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  if (!driver.IsRecording()) {
    char fnamebuf[256];
    snprintf(fnamebuf, sizeof(fnamebuf), "%s-%d.yuv", "Recording", recording_num++);
    if (driver.StartRecording(fnamebuf, frameskip)) {
      fprintf(stdout, "Started recording to %s at %d/%d FPS.\n", fnamebuf, fps, frameskip+1);
    }
  }

  cout << "Starting.\n" << endl;
  for( int i = 0; i < 30 * 10 * 10; i++) {

    // Get frame
    //IplImage* frame = cvQueryFrame(capture);

    // Show frame
    //cvShowImage("Camera_Output", frame);

    usleep(1000);

    // Wait for that esc
    //key = cvWaitKey(10);
    //if (char(key) == 27){
    //    break;
    //}
  }

  printf( "Stopping.\n" );    
  pigpio_stop(pi);
  driver.StopRecording();
  usleep(1 * 1000 * 1000);
  printf( "Done.\n" );    

  // We're done
  //cvReleaseCapture(&capture);
  //cvDestroyWindow("Camera_Output");
  return 0;
}

