import time
try:
	import rcpy.servo as servo
	from rcpy.servo import servo1
	from rcpy.servo import servo2
except:
	print("Error: No rcpy found.")
	pass

# Takes a value from -1 to 1, where 0 is center
def setPWM(number, value):
	duty = (value+1) * 500 + 1000
	if number == 1:
		servo1.set(duty)
	elif number == 2:
		servo2.set(duty)

def runPWM(number):
	if number == 1:
		servo1.run()
	elif number == 2:
		servo2.run()
