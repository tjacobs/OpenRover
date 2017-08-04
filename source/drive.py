try:
	import cv2
	import camera
	import vision
except:
	print("No OpenCV installed.")

from PIL import Image   
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import time
import motors

# Test motors
while True:
    speed = 0.0
    steering = 0.6
    motors.setMotor(1, speed)
    motors.setMotor(2, steering)
    time.sleep(1)
	
    speed = 0.0
    steering = -0.6
    motors.setMotor(1, speed)
    motors.setMotor(2, steering)
    time.sleep(1)

    speed = 0.6
    steering = 0.0
    motors.setMotor(1, speed)
    motors.setMotor(2, steering)
    time.sleep(0.5)
        
    speed = 0.0
    steering = 0.0
    motors.setMotor(1, speed)
    motors.setMotor(2, steering)
    time.sleep(5)

# Test vision
#vision.test_pipeline()
#exit()

# Frame processing steps
def process(image):
    # Just put text over it
    cv2.putText(image, "Rover", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return image

# Create window
#cv2.namedWindow( "preview" )
#cv2.moveWindow( "preview", 10, 10 )

# Start camera
camera.startCamera( (320, 160) )

# Open a sample image
frame = mpimg.imread('test_images/straight_lines1.jpg') 

# Loop
frames_per_second = 0
time_start = time.time()
while True:

    # Get a frame
#    frame = camera.getFrame()

    # De-blue
#    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run through our machine vision pipeline
#    frame = vision.pipeline(frame)

    # Post process
    processed_frame = process(frame)

    # Move
    #speed = 0.1
    #steering = 0
    #motors.setMotor(1, speed)
    #motors.setMotor(2, steering)

    # Save
#    mpimg.imsave('out.png', processed_frame) 

    # Open it to see
#    img = Image.open('out.png')
#    img.show() 

    # Count frames per second
    frames_per_second += 1
    if( time.time() - time_start > 1.0 ):
        print( "FPS: %.0f" % frames_per_second)
        frames_per_second = 0
        time_start = time.time()

    # Show
    cv2.imshow( "preview", processed_frame )

    # Esc key hit?
    key = cv2.waitKey(20)
    if key == 27:
        break

# Close
cv2.destroyWindow( "preview" )
camera.stopCamera( )


