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
frame = None
running = True
res = (0, 0)

# Read a frame from the camera
def read():
    # Just return the current frame
    global frame
    return frame

def update():
        global cap
        global frame
        global running
        global rawCapture
        global picamera
        while running:
            # Read that frame
            #print( "Read one camera frame" )
            if rawCapture:
                rawCapture.truncate(0)
                picamera.capture(rawCapture, format="bgr") #, resize=(640, 360))
                frame = rawCapture.array
            else:
                # Capture frame from regular USB webcam
                (grabbed, frame2) = cap.read()
                frame = cv2.resize(frame2, res, interpolation = cv2.INTER_CUBIC)
        print("Stopped camera.")

def startCamera(resolution, cam_number=0):
        global rawCapture, picamera, cap, res
        res = resolution 

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
                #cap.set(3, resolution[0])
                #cap.set(4, resolution[1])
                #cap.set(15, 0.1) # Exposure, not usually supported

        # Start thread
        t = Thread(target=update, args=())
        t.daemon = True
        t.start()

def stopCamera():
        global cap
        global running
        running = False
        if cap:
            cap.release()
