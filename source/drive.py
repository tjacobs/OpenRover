# OpenRover
# Drive.py

# The main file. Run this to start OpenRover.

# The basics
import os, sys, time, math

# ------ Settings -------

# Differential drive (two independent motors each side, like a tank), else Ackermann steering (steerable front wheels, like a car)
differential = False
if os.uname()[1] == "odroid":
    differential = True

# Camera capture, vision processing, and video transmission resolution
resolution = (320, 240)
resolution = (640, 480) # Getting about 2 FPS
video_number = 0
if os.uname()[1] == "beaglebone":
    resolution = (320, 240)
    video_number = 0
elif os.uname()[1] == "odroid":
    resolution = (320, 240)  # Getting about 8 FPS
    resolution = (640, 480) # Getting about 2 FPS
    video_number = 6

# -----------------------

# Start
print("Starting OpenRover on " + str(os.uname()[1]))

# Start camera
try:
	import camera
	camera.startCamera(resolution, video_number)
except:
	print("No camera or OpenCV.")
	pass

# Try importing what we need
try:
    import web
except:
    print("No web.")
keys = None
try:
    if os.uname()[1] != "beaglebone" and os.uname()[1] != "Thomass-Air":
        import keys
except:
    print("No keyboard.")
try:
    import cv2
except:
    print("No OpenCV.")
try:
    import vision
except:
    print("No vision.")
try:
    from PIL import Image   
except:
    print("No Pillow.")
try:
    import matplotlib.image as mpimg
except:
    pass
try:
    import motors
except:
    print("No motors.")
video = None
#try:
#    import video
#    video.resolution = resolution
#except:
#    print("No video.")
try:
    import remote
except:
    print("No remote.")

# Calibrate ESC if it's just the one
if False and not differential:
    print( "Starting speed controller." )
    motors.setPWM(1, 1.0)
    motors.startPWM(1, 0.005)
    motors.runPWM(1)
    time.sleep(2)
    motors.setPWM(1, -1.0)
    motors.runPWM(1)
    time.sleep(2)
 
# Our outputs
steering = 0.0
acceleration = 0.0

# Video frame post-processing step
def process(image):
    # Just put text over it
    global frames_per_second, steering, acceleration
    cv2.putText(image, "OpenRover", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 215), 1)
    cv2.putText(image, "FPS: {}".format(frames_per_second), (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 195, 195), 1)
    cv2.putText(image, "Steering: {0:.2f}".format(steering), (140, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 195, 195), 1)
    cv2.putText(image, "Acceleration: {0:.2f}".format(acceleration), (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255, 195, 195), 1)
    cv2.putText(image, "Controls: w a s d".format(), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (195, 195, 195), 1)
    return image

# Open a test image
frame = cv2.imread('test_images/test320.jpg')

# Create window
display = False
if display and os.uname()[1] == "odroid":
    print("Starting window.")
    cv2.namedWindow( "preview" )
    cv2.moveWindow( "preview", 10, 10 )
    cv2.imshow("preview", frame)

# Loop
frames_per_second = 0
frames_per_second_so_far = 0
time_start = time.time()
lastPWM = 0
vision_steering = 0
vision_speed = 0
acceleration_time = 0
sys.stdout.flush()

v = [20, 170, 2]
gy_sum = 0

motors.initMotors()
print("Running.")
while not keys or not keys.esc_key_pressed:
    # Remote controls
    if remote:
        if remote.up:
            acceleration += 0.39
        if remote.down:
           acceleration -= 0.19
        if remote.right:
           steering -= 0.19
        if remote.left:
           steering += 0.19

    # Slow down
    acceleration *= 0.9
    steering *= 0.9
    
    # Get a frame
    newFrame = camera.read()
    if newFrame is None:
        frame = cv2.imread('test_images/test320.jpg')
    else:
        frame = newFrame

    # Run through our machine vision pipeline
    vision_frame1, vision_frame2, vision_steering, vision_speed = vision.pipeline(frame, v)
    vision_steering = 0
    frame = vision_frame1

    # Combine frames for Terminator-vision
    #frame = cv2.addWeighted(frame, 0.7, vision_frame1, 0.3, 0)
 #   frame = cv2.addWeighted(vision_frame1, 0.8, vision_frame2, 0.2, 0)
    
    # Pump the throttle for a second every five seconds
    if( time.time() - acceleration_time < 0.5 ):
        pass
        #acceleration = 0.25
    if( time.time() - acceleration_time > 4.0 ):
        acceleration_time = time.time()

    # Output
    vision_steering /= 4
    steering = min(max(steering - vision_steering, -0.8), 0.8)
    acceleration = min(max(acceleration, 0.0), 0.5)
   
    # Post process
    frame = process(frame)

    # Send this jpg image out to the websocket
    if frame is not None:
        jpg_frame = cv2.imencode(".jpg", frame)[1]
        remote.send_frame(jpg_frame)

    # Pump this frame out so we can see it remotely
    if video:
        video.send_frame(frame)

    if differential:
        # Steer tank style
        motors.setPWM(1, acceleration + steering)
        motors.setPWM(2, acceleration - steering)
    else:
        # Steer Ackermann style
        motors.setPWM(2, steering - 0.3)

        # Accelerate
        motors.setPWM(1, acceleration - 1.0)

    '''
    # Read IMU
    imu = motors.readIMU()
    if imu is not None and imu != 0:
        gy_sum -= imu['ax']
    print(imu)
    spaces = int(gy_sum/10) + 10
    gy_sum *= 0.9
    print(" " * spaces + "*") 
    '''

    # Send the PWM pulse at most every 10ms
    if time.time() > lastPWM + 0.01:
        motors.runPWM(1)
        motors.runPWM(2)
        lastPWM = time.time()

    # Count frames per second
    frames_per_second_so_far += 1
    if( time.time() - time_start > 1.0 ):
        frames_per_second = frames_per_second_so_far
        frames_per_second_so_far = 0
        time_start = time.time()
#    motors.display("FPS: {}".format(frames_per_second))
    
    # Show frame if we have a GUI
    if display and frame is not None:
        cv2.imshow("preview", frame)
        cv2.waitKey(1)

# Close and finish
print("\nStopping.")
motors.setPWM(1, 0)
motors.setPWM(2, 0)
motors.runPWM(1)
motors.runPWM(2)
time.sleep(3)
cv2.destroyWindow( "preview" )
camera.stopCamera()
#motors.servosOff()
#motors.stopMotors()
print("OpenRover stopped.")
