## OpenRover
### A Simple Self Driving Car System.

Featuring LaneVision™ Technology.

OpenRover runs on a Raspberry Pi 3, or an ODROID XU4.

It's designed for small scale RC cars. It uses a webcam or Pi Cam and OpenCV to detect lane lines and steer and drive the car.

[Overview and context of the project](https://medium.com/australian-robotics-society/self-racing-cars-down-under-d6223af4fad0).

[How to build your own](https://medium.com/australian-robotics-society/build-your-own-self-racing-car-4204b30dc4d0).

[Example of it running](https://www.instagram.com/p/BXrkQCCB8Qa/?taken-by=australianroboticssociety).

[Example of it driving](https://www.instagram.com/p/BX2N8v0BVw8/?taken-by=australianroboticssociety).

# LaneVision™
LaneVision™ is the system for determining where the lanes of the track are in the camera image, fitting polynomials to them, determining confidence factor with which the lanes have been found, and mapping them into world space.

It works thus:

First, we detect vertical edges.
![](docs/1.gif)

Then, we threshold to be able to see only the detected lines.
![](docs/2.gif)

Then, we run sliding windows up from the bottom of the image, re-adjusting the window position to center around the most detected line pixels.
![](docs/3.gif)

Then we fit a polynomial to the window positions, and generate a centre line. 
![](docs/4.gif)

We use the curve of the centre line to feed into the steering.

# Installation

```
cd install
./install
```

# Running
To start driving, run this.

```
./drive
```

# Development
Currently broadcasts to http://meetzippy.com, where you can turn on and off various OpenCV methods.

