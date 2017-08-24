import face_recognition
import camera
import numpy as np
import motors
import random
import os
import time
import cv2
from threading import Thread

# Start camera
camera.startCamera((320, 240), 6)

# Create face detectors
face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_cascade3 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_cascade4 = cv2.CascadeClassifier('haarcascade_profileface.xml')

# Create window
display = False
if os.uname()[1] == "odroid":
    print("Starting window.")
    cv2.namedWindow( "preview" )
    cv2.moveWindow( "preview", 100, 100 )
    display = True

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print( '%s %d ms' % (f.__name__, (time2-time1)*1000.0) )
        return ret
    return wrap

@timing
def detect_faces(frame):
    print("Detecting faces")
    global face_cascade1, face_cascade2, face_cascade3, face_cascade4
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces1 = face_cascade1.detectMultiScale(gray, 1.3, 5)
    faces2 = face_cascade2.detectMultiScale(gray, 1.3, 5)
    faces3 = face_cascade3.detectMultiScale(gray, 1.3, 5)
    faces4 = face_cascade4.detectMultiScale(gray, 1.3, 5)
    faces = []
#    for face_location in faces1:
#        faces += face_location
#    for face_location in faces2:
#        faces += face_location
#    for face_location in faces3:
#        faces += face_location
#    for face_location in faces4:
#        faces += face_location
    faces = faces4
    print( faces )
    return faces

# Dlib method, slow.
#    return face_recognition.face_locations(frame)

# Detect faces loop
frame = None
face_locations = []
face_x = 320/2
face_y = 240/2
x = 0
y = 0
w = 0
h = 0
faces = []
def face_function():
    global face_locations, frame, face_x, face_y, faces, x, y, w, h
    while True:
        if frame is not None:
            x = random.randint(0, 300)
            y = random.randint(0, 200)
            w = 20
            h = 20
            faces =[(x, y, w, h)]

            # Find the faces and in the current frame of video
#            faces = detect_faces(frame)
            if( len(faces) > 0 ):
                print("Found {} faces in image.".format(len(faces)))
            for face_location in faces:
                x, y, w, h = face_location

                # Face center
                face_x = x + w/2
                face_y = y + h/2
                print("Face at {} {}".format(face_x, face_y))
                
                # Just look at the first face found
                break
        time.sleep(1.1)

# Start detection on separate thread
print("Starting face detection.")
thread = Thread(target=face_function, args=())
thread.start()    

# Initialize some variables
lastPWM = 0
motors.setPWM(1, 0)
motors.setPWM(2, 0)
image_center_x = 320/2
image_center_y = 240/2
i = 0
while True:

    # Grab a single frame of video from the camera as a numpy array    
    frame = camera.read()
    
    # Fade to center
#    x -= (x - 320/2)/200
#    y -= (y - 240/2)/200

    face_x = x + w/2
    face_y = y + h/2

    # How far should we move to put the face into the center?
    print( "Face {0:.2f} {0:.2f}".format( face_x, face_y ) )
    movement_x = image_center_x - face_x
    movement_y = image_center_y - face_y
    
    x += movement_x/200
    y += movement_y/200

    # Proportional movement controller
    k_p = 0.002
    p_x = k_p * movement_x
    p_y = k_p * movement_y*2
    print( "Moving {0:.2f} {0:.2f}".format( p_x, p_y ) )
    
    # Move
    motors.setPWM(2, p_x)
    motors.setPWM(1, -p_y)
        
    # Send the PWM pulse at most every 10ms
    if time.time() > lastPWM + 0.1:
        motors.runPWM(1)
        motors.runPWM(2)
        lastPWM = time.time()            

    # Draw a box on faces
#    for (x, y, w, h) in faces:
    cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)

    # Show frame if we have a GUI
    if display and frame is not None:
        cv2.imshow("preview", frame)
        cv2.waitKey(1)
