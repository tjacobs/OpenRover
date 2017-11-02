import time
import glob
import cv2
import math
import numpy as np
from PIL import Image   
from threading import Thread

# Globals
M = None
Minv = None

# Timing
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
#        print( '%s %d ms' % (f.__name__, (time2-time1)*1000.0) )
        return ret
    return wrap

# --------- Thresholding functions ----------

# Sharpen
@timing
def sharpen_image(img):
    gb = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)

# Function that thresholds a channel of YUV
@timing
def yuv_select(img, threshold, yuv_option, invert = 0):
    # Convert to YUV color space
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # Apply a threshold to the channel
    u = invert * yuv[:, :, yuv_option]
    binary_output = np.zeros_like(img)
    binary_output[ (u>threshold[0]) & (u<=threshold[1]) ] = 255
    
    # Return a binary image of threshold result
    return binary_output

# Function that applies Sobel x or y, then takes an absolute value and applies a threshold.
#@timing
def sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    
    # Take the derivative in x or y given orient = 'x' or 'y', and absolute
    if orient == 'x':
        x = 1
        y = 0
    else:
        x = 0
        y = 1
    deriv = np.absolute( cv2.Sobel( gray, cv2.CV_64F, x, y ) )
    
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8( 255*deriv / np.max(deriv) )
    
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
    colour = cv2.cvtColor(binary_output, cv2.COLOR_GRAY2RGB)
    return colour

# Warp
@timing
def warp_image(image):
    global M
    return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

@timing
def dewarp_image(image):
    global Minv
    return cv2.warpPerspective(image, Minv, (image.shape[1], image.shape[0]))


    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Smooth track angle and track lateral offset between frames
smoothed_angle = 0
smoothed_offset = 160/2
smoothed_speed = 0

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):

    global smoothed_angle, smoothed_offset, smoothed_speed

    if lines is None:
        smoothed_speed = 0.6

        # Steer extreme one way or another depending on last value
        steer = (160/2 - smoothed_offset)/150 + smoothed_angle/3
        steer = 0
#        if steer > 0: steer = 1
#        else:         steer = -1
        return img, steer, smoothed_speed

    lines_found = []

    # Lines come in needing this offset to draw on the image
    xoffset = 0
    yoffset = 20
    
    # Go through lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1+xoffset, y1+yoffset), (x2+xoffset, y2+yoffset), color, 1)
            m = 0
            try:
                m = ((x1-x2)/(y1-y2)) # Calculate slope
            except:
                pass
#            angle = math.atan2(y1-y2, x1-x2)
            lines_found.append( (m, x1, y1, x2, y2) )

    # Found a line? Go forward
    if len(lines_found) >= 2:
        if smoothed_speed < 1.0:
            smoothed_speed += 0.01
    else:
        if smoothed_speed > 0.7:
            smoothed_speed -= 0.01

    # Find median angle line
    median_x = 0
    median_y = 0
    median_x2 = 0
    median_y2 = 0
    median_angle = 0
    if len(lines_found) >= 1:
        median_angle, median_x, median_y, median_x2, median_y2 = sorted(lines_found)[int(len(lines_found)/2)]

    # Choose higher y value point for bottom of image side of line
    if median_y2 > median_y:
        median_x = median_x2
        median_y = median_y2

    # Position one side of the green steering line at bottom of attention box
    start_x = median_x+xoffset #160/2
    start_y = median_y+yoffset #88
    left_length = 60.0 * smoothed_speed

    # And the other just with the angle, up
    l2x = start_x - left_length * median_angle
    l2y = start_y - left_length * 1

    # Check inf
    if math.isnan(median_angle) or math.isinf(median_angle) or math.isinf(l2x) or math.isinf(l2y):
        return img, 0, smoothed_speed

    # Draw steering line from median x position
    cv2.line(img, (int(median_x+xoffset), int(median_y+yoffset)), (int(l2x), int(l2y)), [0, 200, 200], 1)

    # Update smoothed offset and angle
    smoothed_offset += ((median_x+xoffset - smoothed_offset)/4)
    smoothed_angle += ((median_angle - smoothed_angle)/4)

    # Draw smoothed steering line
    cv2.line(img, (int(smoothed_offset), int(88)), (int(smoothed_offset - 60.0 * smoothed_speed * smoothed_angle), int(88-60.0*smoothed_speed)), [100, 200, 100], 2)

    # Return drawn on image, angle, and speed factor
    steer = (160/2 - smoothed_offset)/120 + smoothed_angle/3
    return img, steer, smoothed_speed


# -------------- The pipeline --------------

steer = 0
speed = 0
def pipeline(image):
    global speed, steer
    global M, Minv
    
    # Blank equals blank
    if image is None: return None, 0, 0

    # Define source, dest and matricies for perspective stretch and stretch back
    if M is None:
        src = np.float32(
           [[image.shape[1] * 0.2, 0], 
            [image.shape[1] * 0.8, 0],
            [-200,                 image.shape[0]],
            [200+image.shape[1],   image.shape[0]]])
        dest = np.float32(
            [[0,              0],
             [image.shape[1], 0],
             [0,              image.shape[0]],
             [image.shape[1], image.shape[0]]])
        M = cv2.getPerspectiveTransform(src, dest)
        Minv = cv2.getPerspectiveTransform(dest, src)
    
    # Warp
#    image = warp_image(image)

    # Save original
    image_original = image

    edge_mode = 4

    # Find Lines Method 1: Blur and Canny. Lots of wiggles.
    if edge_mode == 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 1.5)
        image = cv2.Canny(gray, 30, 3)

    # Find Lines Method 2: Bilateral and Canny. Pretty good.
    if edge_mode == 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        image = cv2.Canny(gray, 30, 200)

    # Find Lines Method 3: Sharpen and Sobel x. Nice.
    if edge_mode == 3:
        image = sharpen_image(image)
        image = sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(20, 160))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Lines Method 4: Filter blue colour
    if edge_mode == 4:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([40, 20, 20])
        upper_blue = np.array([190, 255, 255])
        image = cv2.inRange(hsv, lower_blue, upper_blue)

    # Crop attention area
    #cropped_image = image[20:128-40, 30:160-30]
    cropped_image = image[20:128-40, 0:160-0]
    
    # Find lines and draw them
    lines = hough_lines(cropped_image, 1, np.pi/180, 20, min_line_len=10, max_line_gap=90)

    # New image    
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    lines_image, steer, speed = draw_lines(lines_image, lines, [255, 0, 0], 1)

    # Draw attention box
    cv2.rectangle(lines_image, (0, 20), (160-1, 128-40), (255, 255, 50))

    # Mix
    image = cv2.addWeighted(image_original, 0.3, lines_image, 0.7, 0)
    image = image_original
    return image, steer, speed


# Main loop thread
camera_frame = None
frame = None
frames_per_second_so_far = 0
frames_per_second = 0
time_start = 0
def vision_function():
    global camera_frame, frame, steer, speed
    global frames_per_second_so_far, frames_per_second, time_start
    while True:
        # Process
        frame, steer, speed = pipeline(camera_frame)

        # Count frames per second
        frames_per_second_so_far += 1
        if( time.time() - time_start > 1.0 ):
            frames_per_second = frames_per_second_so_far
            frames_per_second_so_far = 0
            time_start = time.time()

        # Don't hog the CPU
        time.sleep(0.01)

# Start thread
print("Starting vision.")
thread = Thread(target=vision_function, args=())
thread.start() 
