import time
import sys

# Try some OpenCV and Picamera
try:
	import cv2
	import picamera
	from picamera.array import PiRGBArray
	from picamera import PiCamera
except:
	pass

rawCapture = None
picamera = None

def startCamera( resolution=(160, 128) ):
	global rawCapture, picamera
	try:
		picamera = PiCamera()
		picamera.resolution = resolution
		rawCapture = PiRGBArray(picamera, size=resolution)
	except:
		picamera = None

def saveFrame(filepath):
	global picamera
	picamera.start_preview()
	picamera.capture(filepath)

def getFrame():
	global rawCapture, picamera
	if rawCapture:
		rawCapture.truncate(0)
		picamera.capture(rawCapture, format="bgr") #, resize=(640, 360))
		return rawCapture.array

def showFrame(image):
	cv2.imshow( "Image", image )
	cv2.waitKey(0)


