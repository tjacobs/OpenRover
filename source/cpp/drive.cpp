#include <iostream>
#include <sys/time.h>
#include <fcntl.h>
#include <semaphore.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include "drive.h"
#include "camera.h"
#include <uWS/uWS.h>
#include <fstream>
#include <sstream>
#include "flushthread.h"

//#define PI
//#define ODROID

#ifdef PI
  #include <pigpiod_if2.h>
#endif

// Using
using namespace std;
using namespace cv;
using Eigen::Matrix2f;
using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::MatrixXf;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::VectorXf;

// Our output throttle value
int8_t _throttle = 0;

// Our output steering angle
int8_t _steering = 0;

// The actual current angle of steering (servos aren't instant)
uint8_t _servo_position = 90;

// The current IMU values
Eigen::Vector3f _accelerometer(0, 0, 0);
Eigen::Vector3f _gyro(0, 0, 0);

// The wheel encoder values 
uint16_t _wheel_encoders[2] = {0, 0};
uint16_t _last_wheel_encoders[2];

// Time keeping
struct timeval t;
struct timeval _last_t;

// Keyboard key presses
char key;

// Pi output device
int8_t pi;

// Consts
static const float MAX_THROTTLE = 0.8;
static const float SPEED_LIMIT = 5.0;
static const float ACCEL_LIMIT = 4.0;     // Maximum dv/dt (m/s^2)
static const float BRAKE_LIMIT = -100.0;  // Minimum dv/dt
static const float TRACTION_LIMIT = 4.0;  // Maximum v*w product (m/s^2)
static const float LANE_OFFSET = 0.0;
static const float LANEOFFSET_PER_K = 0.0;
static const float kpy = 1.0;
static const float kvy = 2.0;

// Drive
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
    fprintf(stderr, "Caution: Massive freakout.\n");
    Reset();
    return;
  }

  // Start off sensibly
  if(_first_frame) {
    _first_frame = false;
  }

  // After 'dt' milliseconds since we last processed, what would we expect the world to look like now? 
  kalman_filter.Predict(dt, throttle, steering);

  // Log
  std::cout << "World state after our prediction: " << _x.transpose() << std::endl;
}

// Clip
static inline float clip(float x, float min, float max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

// At what y value down the image do we start looking?
static const int ytop = 100;

// Buffer for jpg encoded output
vector<uchar> jpg_buffer;
pthread_mutex_t buffer_mutex;


bool TophatFilter(const uint8_t *yuv, Vector3f *Bout, float *y_cout, Matrix4f *Rkout) {

  // Take horizontal convolution with tophat kernel (it looks like a tophat): [-1, -1, 2, 2, -1, -1]
  // And add linear regression points XTX, Xty, yTy, N.
  Matrix3f regXTX = Matrix3f::Zero();
  Vector3f regXTy = Vector3f::Zero();
  double regyTy = 0;
  double regxsum = 0;
  double regwsum = 0;
  int regN = 0;
  
  /*for (int j = 0; j < uysiz; j++) {
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
  } */

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

void Drive::UpdateState(const uint8_t *yuv, size_t yuvlen, float throttle_in, float steering_in,
      const Vector3f &accel, const Vector3f &gyro, uint8_t servo_pos, const uint16_t *wheel_encoders, float dt) {
  Eigen::VectorXf &x_ = kalman_filter.GetState();

  if (isinf(x_[0]) || isnan(x_[0])) {
    fprintf(stderr, "WARNING: kalman filter diverged to inf/NaN! resetting!\n");
    Reset();
    return;
  }

  if (_first_frame) {
    memcpy(_last_wheel_encoders, _wheel_encoders, 2*sizeof(uint16_t));
    _first_frame = false;
  }

  kalman_filter.Predict(dt, throttle_in, steering_in);
  //std::cout << "x after predict " << x_.transpose() << std::endl;

  UpdateCamera(yuv);
  if (yuvlen == 640*480 + 320*240*2) {
    //cout << "x = v, delta, y_error, psi_error, curvature, ml_1,ml_2,ml_3,ml_4, srv_a,srv_b,srv_r,srvfb_a,srvfb_b, gyro" << endl;
    //std::cout << "x after camera: " << x_.transpose() << std::endl;
  } else {
//    fprintf(stderr, "Drive::UpdateState: invalid yuvlen %d, expected %d\n", (int)yuvlen, 640*480 + 320*240*2);
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
  for (int i = 0; i < 2; i++) {
    if (_wheel_encoders[i] != _last_wheel_encoders[i]) {
      ds += (uint16_t) (_wheel_encoders[i] - _last_wheel_encoders[i]);
      nds += 1;
    }
  }
  memcpy(_last_wheel_encoders, _wheel_encoders, 2*sizeof(uint16_t));

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

    /*
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
    */

    // Update Kalman Filter
    float u_a = _throttle / 127.0;
    float u_s = _steering / 127.0;
    float dt = t.tv_sec - last_t_.tv_sec + (t.tv_usec - last_t_.tv_usec) * 1e-6;
    controller_.UpdateState(buf, length,
            u_a, u_s,
            _accelerometer, _gyro,
            _servo_position, _wheel_encoders,
            dt);
    last_t_ = t;

    // Output actuations
    if (autosteer_ && controller_.GetControl(&u_a, &u_s, dt)) {
      _steering = 127 * u_s;
      _throttle = 127 * u_a;
      int width = max(980, min(1500, _steering*10+1200));
      #ifdef PI
        set_servo_pulsewidth(pi, 17, 1000);
        set_servo_pulsewidth(pi, 27, width);
      #endif
      //printf("Servo 27: %d\n", width);
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


// Websockets thread
stringstream indexHtml;
static void* websockets_thread(void* arg) {

  // Log
  cout << "Started websockets thread." << endl;

  // Start HTTP and websockets
  uWS::Hub h;

  // Serve HTTP
  indexHtml << std::ifstream("../web/index.html").rdbuf();
  if (!indexHtml.str().length()) {
    std::cerr << "Failed to load index.html" << std::endl;
    return 0;
  }
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t length, size_t remainingBytes) {
    if (req.getUrl().valueLength == 1) {
      res->end(indexHtml.str().data(), indexHtml.str().length());
    } else {
      stringstream file;
      string filename = "../web" + req.getUrl().toString();
      file << std::ifstream(filename).rdbuf();
      res->end(file.str().data(), file.str().length());
    }
  });

  // Serve websockets
  h.onMessage([](uWS::WebSocket<uWS::SERVER> *ws, char *message, size_t length, uWS::OpCode opCode) {

    // Send jpg image buffer
    pthread_mutex_lock(&buffer_mutex);
    ws->send((char *)(&jpg_buffer[0]), jpg_buffer.size(), uWS::OpCode::BINARY);
    pthread_mutex_unlock(&buffer_mutex);
  });

  // Run websockets
  if (h.listen(8081)) {
    h.run();
  }
  return 0;
}

// Start it up
Drive drive;

Driver driver;

// Camera thread
static void* camera_thread(void* arg) {

  // Start cam
  int video_device_number = 0;
  #ifdef ODROID
	  video_device_number = 6;
  #endif
  cv::VideoCapture cap(video_device_number);
  if (!cap.isOpened()) {
    std::cout << "Failed to open camera." << std::endl;
  }
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  // Loop
  pthread_mutex_init(&buffer_mutex, NULL);
  cv::Mat frame = cv::Mat::zeros(cv::Size(0, 0), CV_8UC3);
  while(true) {

    // Read from camera
    if(!cap.read(frame)) {
      std::cout << "Failed to read from camera. " << std::endl;
    }

    // Convert to YUV
    cvtColor(frame, frame, CV_BGR2YUV);

    // Process
    int channels = 3;
    uchar *input = (uchar*)(frame.data);
    for(int j = 0; j < frame.rows; j++){
        for(int i = 0; i < frame.cols; i++){
            // Read YUV
            uchar y = input[frame.step * j + i*channels + 0];
            uchar u = input[frame.step * j + i*channels + 1];
            uchar v = input[frame.step * j + i*channels + 2];

            // Write BGR
            if(false) {
              // Find red (high v)
              input[frame.step * j + i*channels + 0] = v;
              input[frame.step * j + i*channels + 1] = v;
              input[frame.step * j + i*channels + 2] = v;
            }
            else {
              // Find yellow (low u)
              input[frame.step * j + i*channels + 0] = 255 - u;
              input[frame.step * j + i*channels + 1] = 255 - u;
              input[frame.step * j + i*channels + 2] = 255 - u;
            }
        }
    }

    // Detect edges
    cv::Mat edges = Mat::zeros(Size(0, 0), CV_8UC3);
    cvtColor(frame, edges, CV_BGR2GRAY);
    GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
    Canny(edges, frame, 0, 30, 3);

    // Draw lines
    rectangle(frame, Rect(10, 20, 30, 40), Scalar(200, 100, 0), 2, 8, 0);

    // Draw centerline
    int width = 640;
    int height = 480;
    int top_line = 100;
    int line_height = height - top_line;
    static float centerline_x = width/2;
    static float centerline_m = 0.1;

    centerline_x+=2;
    if(centerline_x > width) {
      centerline_x = 100;
      centerline_m = -0.2;
    }

    line(frame, Point(centerline_x, height), Point(centerline_x + centerline_m*line_height, top_line), Scalar(200, 100, 0), 2, 8, 0);
  /*  std::vector<uchar> vec(edges.rows*edges.cols*3);
    if (frame.isContinuous()) {
      vec.assign(edges.begin<int8_t>(), edges.end<int8_t>());
    }
    else {
      printf("Error, OpenCV buffer not continuous.\n");
    }*/

    // Process frame
//    driver.OnFrame(&vec[0], vec.size());

    // Create jpg
    pthread_mutex_lock(&buffer_mutex);
    imencode(".jpg", frame, jpg_buffer);
    pthread_mutex_unlock(&buffer_mutex);

    // Sleep
    usleep(1000);
  }

  return 0;
}

// Main
int main(){
  
  // Start it up
  printf("Starting OpenRover.\n");

  // Start PWM output
  #ifdef PI
    pi = pigpio_start(0, 0);
    if (pi < 0) {
      printf("Error connecting to PWM.\n");
      return 1;
    }
    set_PWM_range(pi, 17, 1000);
    set_PWM_range(pi, 27, 1000);
    set_PWM_frequency(pi, 17, 50);
    set_PWM_frequency(pi, 27, 50);
  #endif

  // Start disk writing thread 
  if (!flush_thread_.Init()) {
    return 1;
  }

  // Start camera
  int fps = 30;
  if (!Camera::Init(640, 480, fps)) {
    return 1;
  }
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

  // Start camera thread
  pthread_t camera_thread_;
  if (pthread_create(&camera_thread_, NULL, camera_thread, 0) != 0) {
    perror("Failed to create camera thread.");
    return 0;
  }

  // Start websockets thread
  pthread_t websockets_thread_;
  if (pthread_create(&websockets_thread_, NULL, websockets_thread, 0) != 0) {
    perror("Failed to create websockets thread.");
    return 0;
  }

  // Loop
  while(true) {

    // Sleep
    usleep(1000);
  }

  // We're done
  printf( "Stopping...\n" );    
  #ifdef PI
    pigpio_stop(pi);
  #endif
  driver.StopRecording();
  usleep(1 * 1000 * 1000);
  printf( "Done.\n" );    
  //cvReleaseCapture(&capture);
  //cvDestroyWindow("Camera_Output");
  return 0;
}

