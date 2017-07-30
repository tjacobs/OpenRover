try:
	import rcpy.motor as motor
	from rcpy.motor import motor1
except:
	pass

# Takes a speed from -1 to 1
def setMotorSpeed(motor, speed):
	if motor == 1:
		motor1.set(speed)
	elif motor == 2:
		motor2.set(speed)
