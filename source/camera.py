import time
import sys
import cv2

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

def startCamera( resolution=(160, 128) ):
	global rawCapture, picamera, cap
	try:
		picamera = PiCamera()
		picamera.resolution = resolution
		rawCapture = PiRGBArray(picamera, size=resolution)
	except:
		picamera = None

		# Try regular USB webcam
		try:
			print("Starting webcam capture.")
			cap = cv2.VideoCapture(0)
			cap.set(3, resolution[0])
			cap.set(4, resolution[1])
			#cap.set(15, 0.1) # Exposure, not usually supported
			print("Started.")

		except:
			print("Error starting cam")

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
