import time
import glob
import cv2
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
@timing
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

# Find those lanes
@timing
def find_lanes(image):

    # Lane positions
    midpoint = np.int(image.shape[1]*0.5)
    leftx_base = np.int(image.shape[1]*0.60)
    rightx_base = np.int(image.shape[1]*0.40)

    # Choose the number of sliding windows
    nwindows = 10

    # Set height of windows
    window_height = np.int(image.shape[0]/nwindows*0.75)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 60

    # Set minimum number of pixels found to recenter window
    minpix = 4

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window+1)*window_height
        win_y_high = image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            cv2.rectangle(image, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 0), 2)
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            cv2.rectangle(image, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 0), 2)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        left_fit = [0, 0, 0]
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        right_fit = [0, 0, 0]

    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Make left and right line points for polygon
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Left lane red, right lane blue. Colour those pixels.
    image[nonzeroy[left_lane_inds],  nonzerox[left_lane_inds]] = [255, 0, 0]
    image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Return image and both x arrays
    return image, left_fitx, right_fitx, ploty


# Function to draw the lanes
@timing
def draw_lanes(image, left_line_x, centre_line_x, right_line_x, y, steer, speed):

    # Create an blank image to draw the lines on
    draw_image = np.zeros_like(image).astype(np.uint8)

    # Draw the road as a big green rectangle onto the blank image
    pts_left = np.array([           np.transpose(np.vstack([left_line_x,  y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line_x, y])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(draw_image, np.int_([pts]), (20, 200, 20))
    image_centre = image.shape[1] / 2.0

    # Draw the centre line as a white line
    right_side = [x+5 for x in centre_line_x]
    left_side = [x-5 for x in centre_line_x]
    pts_left = np.array([           np.transpose(np.vstack([left_side, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_side, y])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(draw_image, np.int_([pts]), (200, 200, 200))

    steering_bar_width = image.shape[1] / 2
    steering_bar_height = 20
    if False:
        # Black bar
        steering = np.array( [[[image_centre - steering_bar_width/2, 10], [image_centre + steering_bar_width/2, image.shape[0] - 50], 
                               [image_centre + steering_bar_width/2, 10+steering_bar_height], [image_centre - steering_bar_width/2, image.shape[0] - 50+steering_bar_height]]], dtype=np.int32 )
        cv2.fillPoly(draw_image, np.int_([steering]), (10, 20, 10))

        # Green indicator
        steering_bar_width = 50
        steering = np.array( [[[steer + image_centre - steering_bar_width/2, 15], [steer + image_centre + steering_bar_width/2, image.shape[0] - 50], 
                               [steer + image_centre + steering_bar_width/2, 5+steering_bar_height], [steer + image_centre - steering_bar_width/2, image.shape[0]- 50 +steering_bar_height]]], dtype=np.int32 )
        cv2.fillPoly(draw_image, np.int_([steering]), (10, 180, 100))

        # Black bar
        speed_bar = np.array( [[[image_centre - 15, 70], [image_centre + 15, 70], 
                               [image_centre + 15, 160], [image_centre - 15, 160]]], dtype=np.int32 )
        cv2.fillPoly(draw_image, np.int_([speed_bar]), (10, 20, 10))

        # Green indicator
        speed_bar = np.array( [[[image_centre - 25, 110-speed], [image_centre + 25, 110-speed], 
                               [image_centre + 25, 160-speed], [image_centre - 25, 160-speed]]], dtype=np.int32 )
        cv2.fillPoly(draw_image, np.int_([speed_bar]), (10, 180, 100))

    # Calculate centre offset
    lane_centre = (left_line_x[-1] + right_line_x[-1]) / 2.0
    offset_distance = lane_centre - image_centre
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
#        cv2.putText(draw_image, "Crosstrack: " + str( round( offset_distance, 2 ) ) + "m", (10, 100), font, 0.8, (255, 255, 255), 3)#, cv2.LINE_AA)
    except:
        pass

    # Done
    return draw_image
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    Draws lines on an image.
    """    
    right_lines = []
    left_lines = []
    lines_found = []

    # If no input, draw big white lines on image from averaged time smoothed globals
    if lines is None:
#        cv2.line(img, (int(left_line_x1), int(left_line_y1)), (int(left_line_x2), int(left_line_y2)), [255,255,255], 12) #draw left line
#        cv2.line(img, (int(right_line_x1), int(right_line_y1)), (int(right_line_x2), int(right_line_y2)), [255,255,255], 12) #draw left line
        return img, 0, 0
    
    # Go through lines, bucket into left lines and right lines based on slope
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1+50, y1+90), (x2+50, y2+90), color, 1)
            m = 0
            try:
                m = ((x1-x2)/(y1-y2)) # Calculate slope
            except:
                pass
            import math
            angle = math.atan2(y1-y2, x1-x2)
            lines_found.append( (m, x1, y1) )
#            print(angle)
#            if m <= LEFT_LANE_SLOPE_MIN and m >= LEFT_LANE_SLOPE_MAX:
#                left_lines.append((m, x1,y1))
#                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
#            elif m >= RIGHT_LANE_SLOPE_MIN and m <= RIGHT_LANE_SLOPE_MAX:
#                right_lines.append((m, x2,y2))
#                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # Find median angle
    median_x = 0
    median_y = 0
    median_angle = 0

    #print( "Lines" )
    #for line in sorted(lines_found):
    #    print( line )

    speed = 0
    if len(lines_found) > 1:
        median_angle, median_x, median_y = sorted(lines_found)[int(len(lines_found)/2)]
    if len(lines_found) > 1:
        speed = 1
    median_x = 160/2
    median_y = 100
    left_length = 60
    l2x = median_x - left_length * median_angle
    l2y = median_y - left_length * 1
    line2 = ((int(median_x), int(median_y)), (int(l2x), int(l2y)))

    # Draw steering line
    cv2.line(img, (int(median_x), int(median_y)), (int(l2x), int(l2y)), [0,255,0], 3)

    return img, median_angle, speed


# -------------- The pipeline --------------

steer = 0
speed = 0
def pipeline(image):
    global speed, steer
    global M, Minv
    global warp_on, threshold_on
    
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
    image = warp_image(image)
    # Save original
    image_original = image


    # Find Lines Method 1: Blur and Canny. Lots of wiggles.
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    gray = cv2.GaussianBlur(gray, (7,7), 1.5)
#    gray = cv2.Canny(gray, 30, 3)
#    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Find Lines Method 2: Bilateral and Canny. Pretty good.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    image = cv2.Canny(gray, 30, 200)
#    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Find Lines Method 3: Sharpen and Sobel x. Nice.
#    image = sharpen_image(image)
#    image = sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(20, 160))
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Lines Method 4: Filter blue colour
#    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#    lower_blue = np.array([90, 10, 10])
#    upper_blue = np.array([120, 135, 135])
#    image = cv2.inRange(hsv, lower_blue, upper_blue)

    cropped_image = image[90:170, 50:320-50]
    
    # Find lines and draw them
    lines = hough_lines(cropped_image, 1, np.pi/180, 20, min_line_len=10, max_line_gap=90)
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    lines_image, steer, speed = draw_lines(lines_image, lines, [255, 0, 0], 1)

    # Find curved lines
#    image2, contours, hi = cv2.findContours(cropped_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    contours = sorted(contours, key = cv2.contourArea, reverse = True)#[:5]
     
#    poly_fit = np.polyfit(y, x, 2)

#    for c in contours: 
#       perimeter_length = cv2.arcLength(c, True)
#       poly = cv2.approxPolyDP(c, 0.02 * perimeter_length, True) 
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    #print("Cont" )
    #print( contours[0] )
    #for c in contours:
    #    print( c[0] )
        #contours[i] = contours[i][0]
    
#    for i in range(len(contours)):
#        cv2.drawContours(image, contours, i, (i*10 % 255, i*50 % 255, 250), 2)

#    image = cv2.addWeighted(image_original, 0.2, image, 0.8, 0)

#    image_out = cv2.addWeighted(image_original, 0.2, lines_image, 0.8, 0)

#    image = dewarp_image(image)
#    lines_image = dewarp_image(lines_image)

    #image = cv2.addWeighted(image, 0.5, image_original, 0.5, 0)
    image = cv2.addWeighted(image_original, 0.3, lines_image, 0.7, 0)
    return image, steer, speed




    # Take a histogram of the bottom half of the image
    histogram = np.sum(processed_image[int(processed_image.shape[0]/2):,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint





    #frame = cv2.addWeighted(frame, 0.7, vision_frame1, 0.3, 0)

    image = sharpen_image(image)

    image_in = image

    # Threshold the image to make the line edges stand out
    if threshold_on:
        image = threshold_image(image, v)
    
    # Stretch the image out so we have a bird's-eye view of the scene
    if warp_on:
        image = warp_image(image)

    # Find the lanes, y goes from 0 to y-max, and left_lane_x and right_lane_x map the lanes
    if threshold_on:
        image_found, left_line_x, right_line_x, y = find_lanes(image)
        centre_line_x = [ int((l_x + r_x) / 2) for l_x, r_x in zip(left_line_x, right_line_x) ]

        # Find the centre line curve shape in the edge detected image
        #centre_line_x, y = find_centre_line_curve(image)

    # Take the bottom of the centre line, and the top of it, and steer that way
    steer = (centre_line_x[-1]- centre_line_x[int(len(centre_line_x)/2)]) / 200
    steer = min(max(steer, -1), 1)

    # Calculate speed from how confident we are with our steering nad how straight it is
    speed = min(max(0, 20 - abs(steer)/4)/20, 1)

    # Draw those lines
    lanes_image = draw_lanes(image, left_line_x, centre_line_x, right_line_x, y, steer, speed)


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
