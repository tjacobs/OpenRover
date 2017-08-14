# Drive.

# The main file.
# Run this to start the system.
#
#

try:
	import cv2
except:
	print("No OpenCV installed.")
import camera
import vision
try:
	from PIL import Image   
except:
	print("No Pillow installed.")
try:
	import matplotlib.image as mpimg
except:
	print("No Matplotlib installed.")
import sys
import time
import math
import motors
import video

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
camera.startCamera( (320, 240), 6 )

# Open a sample image
#frame = mpimg.imread('test_images/test1.jpg') 
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
#frame = cv2.imread('test_images/test1.jpg') # Use OpenCV instead if matplotlib is giving you trouble

# Loop
i = 0.0
frames_per_second = 0
time_start = time.time()
while True:

    # Get a frame
    frame = camera.read()

    # Run through our machine vision pipeline
    frame, steer = vision.pipeline(frame)

    # Post process
    frame = process(frame)

    # Pump this frame out so we can see it
    video.send_frame(frame)

    # Steer
    steer = min(max(steer/20, -1), 1)
    display( "Steer: %0.1f\n" % steer)
    motors.setPWM(2, steer)
    motors.runPWM(2)

    # Accellerate
    if int(i) % 10 == 0:
        display("Jump forward!")
        motors.setPWM(1, 0.56)
    else:
        motors.setPWM(1, 0.5)
    i += 0.20

    # Save image to disk to debug
#    mpimg.imsave('out.png', frame) 
#    img = Image.open('out.png')
#    img.show()
#    time.sleep(5)

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


