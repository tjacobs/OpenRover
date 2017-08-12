import time
import sys
try:
	import rcpy.servo as servo
	from rcpy.servo import esc1
	from rcpy.servo import servo2
except:
	print("Error: No rcpy found.")
	pass

def display(string):
    sys.stdout.write("\r\x1b[K" + string)
    sys.stdout.flush()

# Takes a value from -1.5 to 1.5, where 0 is center
# Speed takes 0.0 to 1.0 throttle, and -0.1 is arm
def setPWM(number, value):
	duty = value 
	if number == 1:
		#display( "Setting speed: " + str(duty) )
		esc1.set(duty)
	elif number == 2:
		#display( "Setting steering: " + str(duty) )
		servo2.set(duty)

def servosOff():
	servo.disable()

def runPWM(number):
	servo.enable()
	if number == 1:
		esc1.run()
	elif number == 2:
		servo2.run()

def startPWM(number, period):
	if number == 1:
		return esc1.start(period)
	elif number == 2:
		return servo2.start(period)

