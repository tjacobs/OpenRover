import camera
import vision
import motors
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time                                                                             
from PIL import Image   

# Frame processing steps
def process(image):

    # Just put text over it
    cv2.putText(image, "Rover", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    return image

#vision.test_pipeline()
#exit()

# Create window
#cv2.namedWindow( "preview" )
#cv2.moveWindow( "preview", 10, 10 )

# Start camera
#camera.startCamera( (640, 368) )

# Loop
frames_per_second = 0
time_start = time.time()
while True:

    # Get a frame
    #frame = camera.getFrame()

    # De-blue
#    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Open a sample image
    frame = mpimg.imread('test_images/straight_lines1.jpg') 

    # Run through our machine vision pipeline
    vision_frame = vision.pipeline(frame)

    # Post process
    processed_frame = process(vision_frame)

    # Move
    #speed = 0.1
    #steering = 0
    #motors.setMotorSpeed(1, speed)
    #motors.setMotorSpeed(2, steering)

    # Save
#    mpimg.imsave('out.png', processed_frame) 

    # Open it to see
#    img = Image.open('out.png')
#    img.show() 

#    time.sleep( 5 )

    # Count frames per second
    frames_per_second += 1
    if( time.time() - time_start > 1.0 ):
        print( "FPS: %.0f" % frames_per_second)
        frames_per_second = 0
        time_start = time.time()

    # Show
#    cv2.imshow( "preview", processed_frame )

    # Esc key hit?
    key = cv2.waitKey(20)
    if key == 27:
        break

# Close
cv2.destroyWindow( "preview" )
#camera.stopCamera( )


