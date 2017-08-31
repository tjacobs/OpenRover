# OpenRover.
# Video.py
#
# Streams video to a website so you can debug settings while seeing what it sees.
#

import sys
import os
import shlex
import time
import asyncio
import subprocess
from threading import Thread
import cv2

# Config options
showCommandLine = False
showOutput = False
resolution = (320, 240)
bitrate = 500 #kbps. 500 is good for 640x480 over the internet.

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

    # Close
    cap.release()
    proc.stdin.close()
    proc.stderr.close()
    proc.wait()

# Start thread
print("Starting video transmission.")
thread = Thread(target=video_function, args=())
thread.start()    

