# OpenRover.
# Motors.py

# Handles sending commands to motors.
# Supports Beagle Bone Blue robotic cape PWM outputs, and 
# Betaflight control boards.

import time
import sys
import os
import math
import functions

# Import Beagle Bone Blue robot motor controller if present
rcpy = None
if os.uname()[1] == "beaglebone":
    try:
        import rcpy
        import rcpy.servo as servo
        from rcpy.servo import esc1
        from rcpy.servo import servo2
    except:
        print("No rcpy.")
        pass

# Import flight controller motor controller if present
if not rcpy:
    try:
        from controller_board import MultiWii
    except:
        print("No controller board found.")
        pass

# Single line display function
def display(string):
    sys.stdout.write("\r\x1b[K" + string)
    sys.stdout.flush()

# Init motors
board = None
def initMotors():
    global board
    try:
        board = MultiWii("/dev/ttyACM0")
    except:
        try:
            board = MultiWii("/dev/ttyUSB0")
        except:
            try:
                board = MultiWii("/dev/ttyACM1")
            except:
                try:
                    board = MultiWii("/dev/ttyACM2")
                except:
                    print( "\nError: Cannot access USB motors." )
                    sys.stdout.flush()

    # Motor enable pin
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(motorEnablePin, GPIO.OUT)
        GPIO.output(motorEnablePin, GPIO.LOW)
    except:
        pass
#        print( "Error: Cannot access GPIO." )

def stopMotors():
    global board
    motorSpeeds = [0.0] * 9
    motorSpeeds[1] = 0
    motorSpeeds[2] = 0
    sendMotorCommands(motorSpeeds)
    board.close()

    # Finish up
    try:
        import RPi.GPIO as GPIO
        GPIO.cleanup()
    except:
        pass

# Motor speeds can go from -100 to 100
def clampMotorSpeeds(motorSpeeds):
    minSpeed = -100.0
    maxSpeed = 100.0
    for i in range(len(motorSpeeds)):
        motorSpeeds[i] = functions.clamp(motorSpeeds[i], minSpeed, maxSpeed)
    return motorSpeeds

# Send the motor speeds to the motors, and enable the motors if any have any speed
motorEnablePin = 18 # Broadcom 18 = pin 12, 6 from the top corner on the outside of the Pi
goTime = 0
sineOffset = 0
go = False
def sendMotorCommands(motorSpeedsIn, displayChannels=False, displayCommands=False):
    global goTime, board, motorEnablePin, go, sineOffset
    motorSpeeds = [0.0] * 9
        
    # Any motor speeds?
    for i in range(len(motorSpeedsIn)):
        motorSpeeds[i] = motorSpeedsIn[i]
        if motorSpeeds[i] > 1 or motorSpeeds[i] < -1:
            go = True
            goTime = time.time()*1000

    # Leave disabled unless testing
    if False:
        # Test all motor commands
        sineOffset += 10.0
        motorSpeeds[1] = math.sin(sineOffset/100.0)*100.0
        motorSpeeds[2] = math.sin(sineOffset/100.0)*100.0
        motorSpeeds[3] = math.sin(sineOffset/100.0)*100.0
        motorSpeeds[4] = math.sin(sineOffset/100.0)*100.0
        motorSpeeds[5] = math.sin(sineOffset/100.0)*100.0
        motorSpeeds[6] = math.sin(sineOffset/100.0)*100.0
        motorSpeeds[7] = math.sin(sineOffset/100.0)*100.0
        motorSpeeds[8] = math.sin(sineOffset/100.0)*100.0
            
    # Motors been on for two seconds unused?
    if time.time()*1000 > goTime + 2000:
        go = False

    # Display?
    if displayCommands:
        functions.display( "Motors: %3d, %3d, %1f, %1f, %1f, %1f" % (motorSpeeds[1], motorSpeeds[2], motorSpeeds[3], motorSpeeds[4], motorSpeeds[5], motorSpeeds[6] ) )

    # Send
    middle = 1000.0 + 500.0 #+ 5
    scale = 5.0
    motorSpeeds = clampMotorSpeeds(motorSpeeds)
    channels = [int(motorSpeeds[1]*scale+middle),
                int(motorSpeeds[2]*scale+middle),
                int(motorSpeeds[4]*scale+middle), # Why be these flipped, betaflight throttle
                int(motorSpeeds[3]*scale+middle),
                int(motorSpeeds[5]*scale+middle),
                int(motorSpeeds[6]*scale+middle),
                int(motorSpeeds[7]*scale+middle),
                int(motorSpeeds[8]*scale+middle)]
    if displayChannels:
        functions.display( "Channels: " + str( channels ) )
    try:
        board.sendCMD(16, MultiWii.SET_RAW_RC, channels)
    except Exception as error:
#        print( "\n" + str(sys.exc_info()[1]) )
        initMotors()

    # Set enable pin
    try:
        import RPi.GPIO as GPIO
        if go:
            GPIO.output(motorEnablePin, GPIO.HIGH)
        else:
            GPIO.output(motorEnablePin, GPIO.LOW)
    except:
        pass

# Takes a value from -1.5 to 1.5, where 0 is center
# Speed takes 0.0 to 1.0 throttle, and -0.1 is arm
number1 = 0
number2 = 0
def setPWM(number, value):
    global number1, number2
    if rcpy:
        try:
            if number == 1:
                #display( "Setting speed: " + str(value) )
                esc1.set(value)
            elif number == 2:
                #display( "Setting steering: " + str(value) )
                servo2.set(value)
        except:
            pass
    else:
        if number == 1:
            number1 = value
        elif number == 2:
            number2 = value

def servosOff():
    try:
        servo.disable()
    except:
        pass

# Sends a single PWM pulse command to the motors
def runPWM(number):
    # If Beagle Bone Blue robotic cape support
    if rcpy:
        try:
            servo.enable()
            if number == 1:
                esc1.run()
            elif number == 2:
                servo2.run()
        except:
            pass
    else:
        # Else output to Betaflight style motor controller
        global number1, number2
        if number == 1: # Only send if the first motor is pulsed, as we send for all motors in one message
            sendMotorCommands([0, 100.0 * number1, 100.0 * number2], False)

# Starts a periodic automatic pulse for Beagle Bone Blue
def startPWM(number, period):
    try:
        if number == 1:
            return esc1.start(period)
        elif number == 2:
            return servo2.start(period)
    except:
        pass
