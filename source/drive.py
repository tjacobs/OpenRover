# Drive.

# The main file. Run this to start OpenRover.
#
#

# ------ Customisation Settings -------
# Differential drive (two independent motors each side, like a tank), else Ackermann steering (steerable front wheels, like a car)
differential = False
# -------------

import os
import sys
import time
import math
print("Starting OpenRover on " + str(os.uname()[1]))
import camera
if os.uname()[1] == "beaglebone":
    camera.startCamera( (320, 240), 0 )
else:
    camera.startCamera( (640, 480), 6 )

# Try importing what we need
try:
    import cv2
except:
    print("No OpenCV installed.")
try:
    import vision
except:
    print("No vision")
try:
    from PIL import Image   
except:
    print("No Pillow installed.")
try:
    import matplotlib.image as mpimg
except:
    print("No Matplotlib installed.")
try:
    import motors
except:
    print("No motors.")
video = None
try:
    import video
except:
    print("No video available.")

# Calibrate ESC
if os.uname()[1] == "beaglebone":
    print( "Calibrating ESC." )
    motors.setPWM(1, 1.0)
    motors.startPWM(1, 0.01)
    time.sleep(.5)
    motors.setPWM(1, 0.0)
    time.sleep(.5)
    motors.setPWM(1, 0.5)
    time.sleep(.5)

# Our outputs
steering = 0.0
acceleration = 0.0

# Video frame post-processing step
frames_per_second = 0
def process(image):
    # Just put text over it
    global frames_per_second, steering, acceleration
    cv2.putText(image, "OpenRover", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 215))
    cv2.putText(image, "FPS: {}".format(frames_per_second), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 195, 195))
    cv2.putText(image, "Steering: {0:.2f}".format(steering), (120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 195, 195))
    cv2.putText(image, "Acceleration: {0:.2f}".format(acceleration), (120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 195, 195))
    cv2.putText(image, "Controls: w a s d".format(), (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (195, 195, 195))
    return image

# Open a test image
#frame = mpimg.imread('test_images/test1.jpg') 
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
frame = cv2.imread('test_images/test1.jpg') # Use OpenCV instead if matplotlib is giving you trouble

# Create window
#cv2.namedWindow( "preview" )
#cv2.moveWindow( "preview", 10, 10 )

# Loop
frames_per_second_so_far = 0
time_start = time.time()
while True:

    # Remote controls
    if video:
        if video.up:
            acceleration += 0.1
        if video.down:
           acceleration -= 0.1
        if video.right:
           steering -= 0.1
        if video.left:
           steering += 0.1
        
    # Slow down
    acceleration *= 0.9
    steering *= 0.9

    # Bonus controls
    #if video.up:
    #    vision.warp = True        
    #if video.down:
    #    vision.warp = False
    #if video.right:
    #    vision.threshold = True
    #if video.left:
    #    vision.threshold = False
    
    # Get a frame
    frame = camera.read()

    # Run through our machine vision pipeline
 #   frame, vision_steering = vision.pipeline(frame)

    # Post process
    frame = process(frame)

    # Pump this frame out so we can see it remotely
    #video.send_frame(frame)

    # Output
    steering = min(max(steering, -1.0), 1.0)
    acceleration = min(max(acceleration, -1.0), 1.0)
    if differential:
        # Steer tank style
        motors.setPWM(1, acceleration + steering)
        motors.setPWM(2, acceleration - steering)
    else:
        # Steer Ackermann style
        motors.setPWM(2, steering)
        motors.runPWM(2)

        # Accellerate
        motors.setPWM(1, acceleration)

    # Save frame to disk to debug
#    mpimg.imsave('out.png', frame) 
#    img = Image.open('out.png')
#    img.show()

    # Count frames per second
    frames_per_second_so_far += 1
    if( time.time() - time_start > 1.0 ):
        frames_per_second = frames_per_second_so_far
        frames_per_second_so_far = 0
        time_start = time.time()
    motors.display("FPS: {}".format(frames_per_second))
    
    # Show frame if we have a GUI
 #   cv2.imshow( "preview", frame )

    # Esc key hit?
    #key = cv2.waitKey(20)
    #if key == 27:
    #    break

# Close
#cv2.destroyWindow( "preview" )
camera.stopCamera()
motors.servosOff()


