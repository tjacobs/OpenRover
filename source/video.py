# Video.
# Streams video to a website so you can debug settings while seeing what it sees.

# Remote.
# Allows you to remote control the vehicle from the website.

import sys
import os
import shlex
import time
import asyncio
try:
    import websockets
except:
    print("No websockets. Try 'sudo pip install websockets'.")
import subprocess
from threading import Thread
import cv2

# Config options
showCommandLine = False
showOutput = False
resolution = (320, 240)
bitrate = 50 #kbps. 500 is good for 640x480 over the internet.

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
def process(text):
    global x, y, left_mouse_down, right_mouse_down, left, right, up, down
    print(text)
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

# Process incoming websocket connections
async def hello(websocket, path):
    print("Received websocket connection.")
    while True:
        text = await websocket.recv()
        process(text)

start_server = websockets.serve(hello, '10.0.0.15', 8080)
#start_server = websockets.serve(hello, '127.0.0.1', 8080)

# Coroutine for websocket handling
@asyncio.coroutine
def remote_connect():
    try:
        websocket = yield from websockets.connect("ws://meetzippy.com:8080")
        #print( "Connected to server." )
    except:
        print( "No meetzippy.com connection." )
        return

    # Loop
    try:
        while not done:
            text = yield from websocket.recv()
            process(text)
    finally:
        yield from websocket.close()
    print( "Remote done" )

# Run coroutine to listen for keyboard/mouse remote commands via websocket
#loop.call_soon_threadsafe(asyncio.async, remote_connect())
loop.call_soon_threadsafe(asyncio.async, start_server)

# Send a frame
frame_to_send = None
def send_frame(frame):
    global frame_to_send
    frame_to_send = frame

# Start video transmission
ffmpegProcess = None
def video_function():
    global ffmpegProcess, frame_to_send
    global showCommandLine, showOutput
    global resolution, bitrate

    # Wake up raspi camera
    #os.system("sudo modprobe bcm2835-v4l2")

    # Set camera params
    #os.system("v4l2-ctl -c brightness={brightness} -c contrast={contrast} -c saturation={saturation}".format(brightness = 70, contrast = 70, saturation = 70 ))
    
    # Stop ffmpeg
    os.system("sudo killall -9 ffmpeg 2> /dev/null")

    # Set resolution
    dimension = '{}x{}'.format(resolution[0], resolution[1])

    # With sound:
#    commandLine = 'ffmpeg -loglevel error -f alsa -ar 44100 -ac 1 -i hw:1 -f mpegts -codec:a mp2 -f v4l2 -framerate 30 -video_size 640x480 -i /dev/video0 -f mpegts -codec:v mpeg1video -s 640x480 -b:v 200k -bf 0 -muxdelay 0.001 http://meetzippy.com:8081/supersecret'

    # From device directly: 
#    commandLine = 'ffmpeg -loglevel error -f mpegts -codec:a mp2 -f v4l2 -framerate 30 -video_size 640x480 -i /dev/video{} -f mpegts -codec:v mpeg1video -s 640x480 -b:v 200k -bf 0 -muxdelay 0.001 http://meetzippy.com:8081/supersecret'.format(videonum)

    # From stdin:
    commandLine = 'ffmpeg -y -f rawvideo -vcodec rawvideo -s {dimension} -pix_fmt bgr24 -i - -an -f mpegts -codec:v mpeg1video -b:v {bitrate}k -bf 0 http://meetzippy.com:8081/supersecret'.format(dimension=dimension, bitrate=bitrate)

    # Start
    stderr = subprocess.PIPE
    if showOutput: stderr = None
    if showCommandLine: print(commandLine)
    my_env = os.environ.copy()
    #my_env["LD_PRELOAD"] = "/usr/lib/uv4l/uv4lext/armv6l/libuv4lext.so"
    my_env["LD_LIBRARY_PATH"] = "/usr/local/lib"
    ffmpegProcess = subprocess.Popen(shlex.split(commandLine), stdin=subprocess.PIPE, stderr=stderr, env=my_env)

    # Pipe those frames out
    while True:
        try:
            ffmpegProcess.stdin.write(frame_to_send.tostring())
        except:
            pass
        time.sleep(0.1)
 
    #cap.release()
    #proc.stdin.close()
    #proc.stderr.close()
    #proc.wait()

# Start thread
print("Starting video transmission.")
thread = Thread(target=video_function, args=())
thread.start()    

