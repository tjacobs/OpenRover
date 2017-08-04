import time
try:
	import rcpy.motor as motor
	from rcpy.motor import motor1
	from rcpy.motor import motor2
except:
	print("Error: No rcpy found.")
	exit()
	pass

def initMotors():
	pass	

# Takes a speed from -1 to 1
def setMotor(motor, speed):
	if motor == 1:
		motor1.set(speed)
	elif motor == 2:
		motor2.set(speed)

