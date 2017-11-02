import math
import time
import functions

# PD controller.
numAngles = 4
previousAngles = [0.0] * numAngles
def calculateMovement(currentAngles, targetAngles):
        global previousAngles
        movements = [0.0] * numAngles
        Ps = [0.0] * numAngles
        Ds = [0.0] * numAngles
        P_rate = 5.0
        D_rate = 5.0
        for i in range(0, numAngles):
                # Go the shortest way around
                angle_cw =  targetAngles[i] - currentAngles[i]
                angle_ccw = targetAngles[i] - currentAngles[i] + 360
                angleFromTarget = angle_ccw
                if abs(angle_cw) < abs(angle_ccw):
                        angleFromTarget = angle_cw

                # Calculate difference from last time
                angleDeriv = angleFromTarget - previousAngles[i]
                previousAngles[i] = angleFromTarget

                # Calculate P, D
                Ps[i] = P_rate * angleFromTarget
                Ds[i] = D_rate * angleDeriv
                movements[i] = Ps[i] - Ds[i]
        return movements


