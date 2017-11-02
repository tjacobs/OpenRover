# OpenRover
# Drive.py

# The main file. Run this to start OpenRover.

# The basics
import os, sys, time, math

# ------ Settings -------

# Differential drive (two independent motors each side, like a tank), else Ackermann steering (steerable front wheels, like a car)
differential = True

# Camera capture, vision processing, and video transmission resolution
resolution = (160, 128)
video_number = 0
if os.uname()[1] == "beaglebone":
    video_number = 0
elif os.uname()[1] == "odroid":
#    resolution = (640, 480) #320, 240)
    video_number = 6

# -----------------------

# Start
print("Starting OpenRover on " + str(os.uname()[1]))

# Magnetic sensors
try:
    import sensors
    magneticSensors = sensors.AMS()
    magneticSensors.connect(1)
except:
    print("No sensors.")

# Arm
import arm

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
    if os.uname()[1] != "beaglebone" and os.uname()[1] != "Thomass-Air" and os.uname()[1] != "Thomass-MacBook-Air.local":
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
motors = None
try:
    if os.uname()[1] != "Thomass-Air" and os.uname()[1] != "Thomass-MacBook-Air.local":
        import motors
except:
    print("No motors.")
video = None
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
    cv2.putText(image, "OpenRover", (55, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 215, 215), 1)
    cv2.putText(image, "FPS: {}".format(vision.frames_per_second), (2, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 195, 195), 1)
    cv2.putText(image, "Steer: {0:.2f}".format(steering), (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 195, 195), 1)
    cv2.putText(image, "Accel: {0:.2f}".format(acceleration), (90, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 195, 195), 1)
    #cv2.putText(image, "Controls: w a s d".format(), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (195, 195, 195), 1)
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

# Init
frames_per_second = 0
frames_per_second_so_far = 0
time_start = time.time()
lastPWM = 0
vision_steering = 0
vision_speed = 0
acceleration_time = time.time()
gy_sum = 0
time_now = 0
old_time = 0
dt = 0

# Write log
sys.stdout.flush()

# Start motors
if motors is not None:
    motors.initMotors()

# Loop
print("Running.")
t = 0
while not keys or not keys.esc_key_pressed:
    # Calculate dt   
    time_now = time.time()
    dt = time_now - old_time
    old_time = time_now

    # Remote controls
    if remote:
        if remote.up:
           acceleration += 8 * dt
        if remote.down:
           acceleration -= 5 * dt
        if remote.right:
           steering -= 4.0 * dt
        if remote.left:
           steering += 4.0 * dt

    # Slow down
    acceleration *= (1.0 - (2.5 * dt))
    steering *=     (1.0 - (1.5 * dt))
 
    # Get a frame
    newFrame = camera.read()
    if newFrame is None:
        frame = cv2.imread('test_images/test320.jpg')
    else:
        frame = newFrame

    # Send the frame to vision    
    vision.camera_frame = frame

    # Read the latest processed frame
    frame = vision.frame
    vision_steering = vision.steer
    vision_speed = vision.speed
   
    # Set steering
    #steering += vision_steering/15 
    #acceleration += vision_speed/10

    min_acceleration = 0.0
    max_acceleration = 0.0
    if time.time() - acceleration_time > 6:
       min_acceleration = -0.4
       max_acceleration = 0.25

    # Cap
    steering     = min(max(steering, -0.20), 0.20)
    acceleration = min(max(acceleration, min_acceleration), max_acceleration)
   
    steering_vision_cap     = min(max((vision_steering/2), -0.20), 0.20)
    acceleration_vision_cap = min(max(vision_speed/1, min_acceleration), max_acceleration)

    # Post process
    frame = process(frame)

    # Send this jpg image out to the websocket
    if frame is not None:
        jpg_frame = cv2.imencode(".jpg", frame)[1]
        remote.send_frame(jpg_frame)

    currentAngles = magneticSensors.readCurrentAngles()

    # Balance
    targetAngles = [0, 0, 10, 110]
    targetAngles[2] = math.sin(t/80.0)*20.0 + 10.0
    targetAngles[3] = math.cos(t/80.0)*20.0 + 110.0
    t += 1

#    print("")
#    print(currentAngles)
#    print(targetAngles)
    # Run movement controller to see how fast we should set our motor speeds
    movement = arm.calculateMovement(currentAngles, targetAngles)
#    print(movement)

    motor_capped1 = min(max((movement[2]/2000), -0.50), 0.50)
    motor_capped2 = min(max((movement[3]/2000), -0.50), 0.50)

    # Display target angles and speeds
    print( "Angles: %3d, %3d, Targets: %3d, %3d, Speeds: %3d, %3d. %3f."
                        % (currentAngles[2], currentAngles[3], targetAngles[2], targetAngles[3], movement[2], movement[3], motor_capped1 ))
 
    motors.setPWM(1, motor_capped1)
    motors.setPWM(2, motor_capped2)

    # Output motor commands
    if motors is not None:
        if differential:
            # Steer tank style
            #motors.setPWM(1, acceleration + steering + acceleration_vision_cap + steering_vision_cap)
            #motors.setPWM(2, acceleration - steering + acceleration_vision_cap - steering_vision_cap)
            pass
        else:
            # Steer Ackermann style
            motors.setPWM(2, steering - 0.5)

            # Accelerate
            motors.setPWM(1, acceleration - 1.0)

        # Send the PWM pulse at most every 10ms
        if time.time() > lastPWM + 0.01:
            motors.runPWM(1)
            motors.runPWM(2)
            lastPWM = time.time()

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

    # Count frames per second
    frames_per_second_so_far += 1
    if( time.time() - time_start > 1.0 ):
        frames_per_second = frames_per_second_so_far
        frames_per_second_so_far = 0
        time_start = time.time()
    
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
