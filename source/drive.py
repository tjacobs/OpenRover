try:
	import cv2
	import camera
	import vision
except:
	print("No OpenCV installed.")
try:
	from PIL import Image   
except:
	print("No Pillow installed.")
try:
	import matplotlib.image as mpimg
except:
	print("No Matplotlib installed.")
import time
import math
import motors
import sys

# Calibrate ESC
print( "Calibrating ESC." )
motors.setPWM(1, 1.0)
motors.startPWM(1, 0.01)
time.sleep(.5)
motors.setPWM(1, 0.0)
time.sleep(.5)
motors.setPWM(1, 0.5)
time.sleep(.5)

def display(string):
    sys.stdout.write("\r\x1b[K" + string)
    sys.stdout.flush()

# Frame processing steps
def process(image):
    # Just put text over it
    cv2.putText(image, "OpenRover", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (155, 155, 255))
    return image

# Create window
#cv2.namedWindow( "preview" )
#cv2.moveWindow( "preview", 10, 10 )

# Start camera
camera.startCamera( (320, 240) )

# Open a sample image
frame = mpimg.imread('test_images/test1.jpg') 
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
#frame = cv2.imread('test_images/test1.jpg') # Use OpenCV instead if matplotlib is giving you trouble

# Loop
i = 0.0
frames_per_second = 0
time_start = time.time()
while True:

    # Get a frame
    frame = camera.read() #getFrame()

    # Run through our machine vision pipeline
    frame = vision.pipeline(frame)

    # Post process
    #processed_frame = process(frame)

    # Move
    if int(i) % 10 == 0:
        display("Jump forward!")
        motors.setPWM(1, 0.56)
    else:
        motors.setPWM(1, 0.5)
    motors.setPWM(2, math.sin(i/100)/2 + 0.25)
    motors.runPWM(2)
    i += 0.20

    # Save image to disk to debug
#    mpimg.imsave('out.png', processed_frame) 
#    img = Image.open('out.png')
#    img.show() 

    # Count frames per second
    frames_per_second += 1
    if( time.time() - time_start > 1.0 ):
        print( "FPS: %.0f\n" % frames_per_second)
        #display("FPS: %.0f" % frames_per_second)
        frames_per_second = 0
        time_start = time.time()

    # Show
#    cv2.imshow( "preview", processed_frame )

    # Esc key hit?
    #key = cv2.waitKey(20)
    #if key == 27:
    #    break

# Close
#cv2.destroyWindow( "preview" )
camera.stopCamera()
motors.servosOff()


