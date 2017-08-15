import time
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

rawCapture = None
picamera = None
cap = None
Q = None
#frame = None

def read():
        # return next frame in the queue
        global Q
        frame = Q.get()

        # Get most recent
        #new_frame = Q.get()
        #while new_frame is not None:
        #    frame = new_frame
        #    new_frame = Q.get()

        #global frame
        return frame

def update():
        global Q, cap
#       global frame
        while True:
                # Otherwise, ensure the queue has room in it
                if not Q.full():
                        # Read the next frame from the file
                        (grabbed, frame) = cap.read()
 
                        if not grabbed:
                                return
 
                        # Add the frame to the queue
                        Q.put(frame)

def startCamera(resolution, cam_number=0):
        global rawCapture, picamera, cap, Q

        try:
                picamera = PiCamera()
                picamera.resolution = resolution
                rawCapture = PiRGBArray(picamera, size=resolution)
        except:
                picamera = None

                # Try regular USB webcam
                try:
                        print("Starting webcam capture.")
                        cap = cv2.VideoCapture(cam_number)
                        cap.set(3, resolution[0])
                        cap.set(4, resolution[1])
                        #cap.set(15, 0.1) # Exposure, not usually supported
                        print("Started.")

                except:
                        print("Error starting cam")

        # Start thread
        Q = Queue(maxsize=128)
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
