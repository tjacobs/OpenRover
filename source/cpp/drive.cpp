#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
 
using namespace std;
char key;

int main(){
	printf("\nStarting OpenRover.\n");
    cvNamedWindow("Camera_Output", 1);    // Create window
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  // Capture using any camera connected to your system
    while(1){

        IplImage* frame = cvQueryFrame(capture); // Create image frames from capture
        cvShowImage("Camera_Output", frame);   // Show image frames on created window

        key = cvWaitKey(10);
        if (char(key) == 27){
            break;
        }
    }
    cvReleaseCapture(&capture); // Release capture
    cvDestroyWindow("Camera_Output"); // Destroy window
    return 0;
}

