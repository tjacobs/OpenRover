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

# Create face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
    global face_cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print( faces )
    return faces

# Dlib method, slow.
#    return face_recognition.face_locations(frame)

# Detect faces loop
frame = None
face_locations = []
face_x = 320/2
face_y = 240/2
top = 0
bottom = 0
left = 0
right = 0
faces = []
def face_function():
    global face_locations, frame, face_x, face_y, top, right, bottom, left, faces
    while True:
        if frame is not None:
            #face_locations =[(100, 100+random.randint(0, 100), 100 + random.randint(0, 100), 100)]

            # Find the faces and in the current frame of video
            faces = detect_faces(frame)
            if( len(faces) > 0 ):
                print("Found {} faces in image.".format(len(faces)))
            for face_location in faces:
                top, right, bottom, left = face_location

                # Face center
                face_x = right - left
                face_y = bottom - top
                print("Face at {} {}".format(face_x, face_y))
                
                # Just look at the first face found
                break
        time.sleep(0.1)

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
    face_x -= (face_x - 320/2)/100
    face_y -= (face_y - 240/2)/100

    # How far should we move to put the face into the center?
    movement_x = image_center_x - face_x
    movement_y = image_center_y - face_y

    # Proportional movement controller
    k_p = 0.008
    p_x = k_p * movement_x
    p_y = k_p * movement_y
#    print( "Moving {0:.2f} {0:.2f}".format( p_x, p_y ) )
    
    # Move
    motors.setPWM(2, p_x)
    motors.setPWM(1, 0) #p_y)
        
    # Send the PWM pulse at most every 10ms
    if time.time() > lastPWM + 0.1:
        motors.runPWM(1)
        motors.runPWM(2)
        lastPWM = time.time()            

    # Draw a box on faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show frame if we have a GUI
    if display and frame is not None:
        cv2.imshow("preview", frame)
        cv2.waitKey(1)
