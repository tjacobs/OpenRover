import time

class AMS():
  def __init__(self):
    self.address1 = 0x40
    self.address2 = 0x42
    self.address3 = 0x43
    self.address4 = 0x41
    self.angleReadReg1 = 0xFE
    self.angleReadReg2 = 0xFF
    self.magnitudeReadReg1 = 0xFC
    self.magnitudeReadReg2 = 0xFD

  def connect(self, bus):
    try:
      import smbus
      self.bus = smbus.SMBus(bus)
      time.sleep(0.5)
      return 0
    except:
      print( "Error: Cannot access sensors." ) # Please enable I2C in raspi-config.
      return -1

  def writeAndWait(self, register, value):
    self.bus.write_byte_data(self.address, register, value);
    time.sleep(0.02)

  def readAndWait(self, register, sensorNum):
    res = False
    address = self.address1
    if( sensorNum == 2 ):
        address = self.address2
    if( sensorNum == 3 ):
        address = self.address3
    if( sensorNum == 4 ):
        address = self.address4
    try: 
    	res = self.bus.read_byte_data(address, register)
    except IOError:
        res = 0
    return res

  def getAngle(self, sensorNum):
    angle2 = self.readAndWait(self.angleReadReg1, sensorNum)
    angle1 = self.readAndWait(self.angleReadReg2, sensorNum)
#    print(angle1)
#    print(angle2)  
    return (angle2 << 6) + angle1

  def getMagnitude(self, sensorNum):
    magnitude1 = self.readAndWait(self.magnitudeReadReg1, sensorNum)
    magnitude2 = self.readAndWait(self.magnitudeReadReg2, sensorNum)
    return magnitude2 << 6 + magnitude1

  def signedInt(self, value):
    if value > 127:
      return (256-value) * (-1)
    else:
      return value

  def readCurrentAngles(self):
    calibratedAngles = [0, 0, 0, 0]
    currentAngles = [0] * 4
    try:
        for i in range(0, len(currentAngles)):
            mag = self.getMagnitude(i)
            if mag > 5000:
                currentAngles[i] = 360.0 * self.getAngle(i) / 16384.0
                currentAngles[i] = (currentAngles[i] + calibratedAngles[i]) % 360
                #currentAngles[i] = self.getAngle(i)
    except:
        print("Error reading sensors")
        return currentAngles
    return currentAngles

