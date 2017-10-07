#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "drive.h"
 
using namespace std;
char key;

Drive::Drive() {
  Reset();
}

void Drive::Reset() {
//  kalman_filter.Reset();
  _first_frame = true;
}

void Drive::Update(float throttle, float steering, float dt) {

  // Get our current model of the world
  Eigen::VectorXf _x; //= kalman_filter.GetState();

  // Freak out if we freaked out
  if(isinf(_x[0]) || isnan(_x[0])) {
    fprintf(stderr, "Caution: Massive freakout currently underweigh.\n");
    Reset();
    return;
  }

  // Start off sensibly.
  if(_first_frame) {
    _first_frame = false;
  }

  // After 'dt' milliseconds since we last thought about it, what would we expect the world to look like now? 
  //kalman_filter.Predict(dt, throttle, steering);

  // Log
  std::cout << "World state after our prediction: " << _x.transpose() << std::endl;

}

int main(){

    // Start it up
	printf("\nStarting OpenRover.\n");

    // Create window
    cvNamedWindow("Camera_Output", 1);

    // Start cam
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
    while(1){

        // Get frame
        IplImage* frame = cvQueryFrame(capture);

        // Show frame
        cvShowImage("Camera_Output", frame);



        






        // Wait for that esc
        key = cvWaitKey(10);
        if (char(key) == 27){
            break;
        }
    }

    // We're done
    cvReleaseCapture(&capture);
    cvDestroyWindow("Camera_Output");
    return 0;
}

