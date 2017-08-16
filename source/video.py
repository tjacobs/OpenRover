# Video.
# Streams video to a website so you can debug settings while seeing what it sees.

# Remote.
# Allows you to control it from a website.

import sys
import os
import shlex
import time
import asyncio
import websockets
import subprocess
from threading import Thread
import cv2
if sys.version_info >= (3, 0):
        from queue import Queue
else:
        from Queue import Queue

Q = None

# Export mouse x and y, and keyboard button press status
left_mouse_down = False
right_mouse_down = False
x = 0
y = 0
left = False
right = False
up = False
down = False

# Finished?
done = False

# Start thread and create new event loop
loop = asyncio.new_event_loop()
def thread_function(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()
thread = Thread(target=thread_function, args=(loop,))
thread.start()

# Coroutine for websocket handling
@asyncio.coroutine
def remote_connect():
    global x, y, left_mouse_down, right_mouse_down, left, right, up, down
    try:
        websocket = yield from websockets.connect("ws://meetzippy.com:8080")
#        print( "Connected to server." )
    except:
        print( "No meetzippy.com connection." )
        return

    # Loop
    try:
        while not done:
            text = yield from websocket.recv()
            print( text )
            if text.startswith( 'left' ):
                left = (len(text.split()) > 1 and text.split()[1] == "down")
            elif text.startswith( 'right' ):
                right = (len(text.split()) > 1 and text.split()[1] == "down")
            elif text.startswith( 'up' ):
                up = (len(text.split()) > 1 and text.split()[1] == "down")
            elif text.startswith( 'down' ):
                down = (len(text.split()) > 1 and text.split()[1] == "down")
            elif text.startswith( 'x' ):
                x = int(text.split()[1])
            elif text.startswith( 'y' ):
                y = int(text.split()[1])
            elif text.startswith( 'left_mouse' ):
                left_mouse_down = (len(text.split()) > 1 and text.split()[1] == "down")
            elif text.startswith( 'right_mouse' ):
                right_mouse_down = (len(test.split()) > 1 and text.split()[1] == "down")
    finally:
        yield from websocket.close()
    print( "Remote done" )

# Run coroutine to listen for keyboard/mouse remote commands via websocket
loop.call_soon_threadsafe(asyncio.async, remote_connect())

ffmpegProcess = None

def send_frame(frame):
    global ffmpegProcess
    try:
        ffmpegProcess.stdin.write(frame.tostring())
    except:
        pass

# Start video transmission
def video_function():

    global Q, ffmpegProcess

    # Wake up raspi camera
    #os.system("sudo modprobe bcm2835-v4l2")

    # Set camera params
    #os.system("v4l2-ctl -c brightness={brightness} -c contrast={contrast} -c saturation={saturation}".format(brightness = 70, contrast = 70, saturation = 70 ))
    
    # Stop ffmpeg
    os.system("sudo killall -9 ffmpeg 2> /dev/null")
   
    # Which /dev/videoWHAT is our camera
    videonum = 6

    # Start camera capture
    if False:
        cap = cv2.VideoCapture(videonum)
        cap.set(3, 640)
        cap.set(4, 480)
        ret, frame = cap.read()
        height, width, ch = frame.shape
        dimension = '{}x{}'.format(width, height)
        fps = str(cap.get(cv2.CAP_PROP_FPS))
    else:
        dimension = '640x480'

    # With sound:
    #commandLine = 'ffmpeg -loglevel error -f alsa -ar 44100 -ac 1 -i hw:1 -f mpegts -codec:a mp2 -f v4l2 -framerate 30 -video_size 640x480 -i /dev/video0 -f mpegts -codec:v mpeg1video -s 640x480 -b:v 200k -bf 0 -muxdelay 0.001 http://meetzippy.com:8081/supersecret'

    # From device directly: 
    commandLine = 'ffmpeg -loglevel error -f mpegts -codec:a mp2 -f v4l2 -framerate 30 -video_size 640x480 -i /dev/video{} -f mpegts -codec:v mpeg1video -s 640x480 -b:v 200k -bf 0 -muxdelay 0.001 http://meetzippy.com:8081/supersecret'.format(videonum)

    # From stdin:
    commandLine = 'ffmpeg -y -f rawvideo -vcodec rawvideo -s {dimension} -pix_fmt bgr24 -i - -an -video_size 640x480 -f mpegts -codec:v mpeg1video -b:v 200k -bf 0 http://meetzippy.com:8081/supersecret'.format(dimension=dimension)

    # Config options
    showCommandLine = True
    showOutput = True

    # Start
    if showCommandLine:
        print(commandLine)
    stderr = subprocess.PIPE
    if showOutput:
        stderr = None
    my_env = os.environ.copy()
    #my_env["LD_PRELOAD"] = "/usr/lib/uv4l/uv4lext/armv6l/libuv4lext.so"
    my_env["LD_LIBRARY_PATH"] = "/usr/local/lib"
    ffmpegProcess = subprocess.Popen(shlex.split(commandLine), stdin=subprocess.PIPE, stderr=stderr, env=my_env)
    print( "Started ffmpeg" )

    # Pipe that pipe
    #while True:
    #frame = Q.get()
        #ret, frame = cap.read()
        #if not ret:
        #    break
        #ffmpegProcess.stdin.write(frame.tostring())

    #cap.release()
    #proc.stdin.close()
    #proc.stderr.close()
    #proc.wait()

# Start thread
thread = Thread(target=video_function, args=())
thread.start()    


