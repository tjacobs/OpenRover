# OpenRover.
# Camera.py
# 
# Deals with the USB webcam or Pi Camera.
# Provides an interface for starting the camera and reading frames.
# Runs its own thread so it doesn't slow down your main thread.
#

import time
import os
import sys
import cv2
from threading import Thread
if sys.version_info >= (3, 0):
        from queue import Queue
else:
        from Queue import Queue

# Try some Picamera
try:
        import picamera
        from picamera.array import PiRGBArray
        from picamera import PiCamera
except:
        pass

# Variables
rawCapture = None
picamera = None
cap = None
Q = None
frame = None

# Read a frame from the camera
def read():
        # Return next frame in the queue
        #global Q
        #frame = Q.get()
	
	# Just return the current frame
        global frame
        return frame

def update():
        #global Q 
        global cap
        global frame
        while True:
	    # Read that frame
            #print( "Read camera frame" )
            (grabbed, frame) = cap.read()
            #print( grabbed )
            #if not grabbed:
            #    return
                
            # Ensure the queue has room in it
            #if not Q.full():
                # Read the next frame from the file
                #(grabbed, frame) = cap.read()
                #if not grabbed:
                #        return
                # Add the frame to the queue
                #Q.put(frame)

def startCamera(resolution, cam_number=0):
        global rawCapture, picamera, cap, Q
	
        # First try Pi camera
        print("Starting camera.")
        try:
                picamera = PiCamera()
                picamera.resolution = resolution
                rawCapture = PiRGBArray(picamera, size=resolution)
        except:
                picamera = None

                # Try regular USB webcam
                try:
                    cap = cv2.VideoCapture(cam_number)
                except:
                    try:
                        cap = cv2.VideoCapture(cam_number+1)
                    except:
                        print("Error starting camera.")
                        pass 
                if True: #os.uname()[1] == "beaglebone": 
                    cap.set(3, resolution[0])
                    cap.set(4, resolution[1])
                    #cap.set(15, 0.1) # Exposure, not usually supported

        # Start thread
        #Q = Queue(maxsize=128)
        t = Thread(target=update, args=())
        t.daemon = True
        t.start()

def stopCamera():
        global cap
        if cap:
                cap.release()

def saveFrame(filepath):
        global picamera
        picamera.start_preview()
        picamera.capture(filepath)

def getFrame():
        global rawCapture, picamera, cap
        if rawCapture:
                rawCapture.truncate(0)
                picamera.capture(rawCapture, format="bgr") #, resize=(640, 360))
                return rawCapture.array
        else:
                # Capture frame from regular USB webcam
            ret, frame = cap.read()
            return frame

def showFrame(image):
        cv2.imshow( "Image", image )
        cv2.waitKey(0)
