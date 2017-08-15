# -------------
# Betabot functions
import time
import sys

def display(string):
	sys.stdout.write("\r\x1b[K" + string)
	sys.stdout.flush()

def clamp(n, smallest, largest):
	return max(smallest, min(n, largest))

# Read angles from sensors
calibratedAngles = [None, -10, 170, 0, 0]
STAND_UP = True
if STAND_UP:
	calibratedAngles[1] -= 90
	calibratedAngles[2] += 90
def readCurrentAngles( sensors ):
	currentAngles = [0] * 4
	try:
		for i in range(1, len(currentAngles)+1):
			mag = sensors.getMagnitude(i)
			if mag > 100:
				currentAngles[i] = 360.0 * sensors.getAngle(i) / 16384.0 
				currentAngles[i] = (currentAngles[i] + calibratedAngles[i]) % 360 
	except:
		return currentAngles
	return currentAngles

