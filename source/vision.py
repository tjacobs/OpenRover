import cv2
import numpy as np
from PIL import Image   
import time
import glob

# Options
warp_on = True
threshold_on = True

# Globals
M = None
Minv = None

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

# Treshold a channel of HLS
@timing
def hls_select(img, threshold, hls_option, invert = 0):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # Apply a threshold to the channel
    s = invert * hls[:, :, hls_option]
    binary_output = np.zeros_like(img)
    binary_output[ (s>threshold[0]) & (s<=threshold[1]) ] = 255
    
    # Return a binary image of threshold result
    return binary_output

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

# Function that applies Sobel x and y, then computes the direction of the gradient and applies a threshold.
def direction_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Convert to grayscale
    gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    
    # Take the gradient in x and y separately
    grad_x = cv2.Sobel( gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel )
    grad_y = cv2.Sobel( gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel )
    
    # Take the absolute value of the x and y gradients
    grad_x_abs = np.absolute( grad_x )
    grad_y_abs = np.absolute( grad_y )
    
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2( grad_y_abs, grad_x_abs )
    
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like( gray )
    binary_output[ (direction > thresh[0]) & (direction < thresh[1]) ] = 1
    
    # Return this mask binary_output image
    return binary_output

# Function that applies Sobel x and y, then computes the magnitude of the gradient and applies a threshold
def magnitude_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Convert to grayscale
    gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    
    # Take the gradient in x and y separately
    x_grad = cv2.Sobel( gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel )
    y_grad = cv2.Sobel( gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel )
    
    # Calculate the magnitude
    mag = np.sqrt( np.square( x_grad ) + np.square( y_grad ) )
    
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max( mag ) / 255
    scaled = (mag / scale_factor).astype( np.uint8 )

    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like( mag )
    binary_output[ (scaled > mag_thresh[0]) & (scaled < mag_thresh[1]) ] = 1

    # Return this binary_output image
    return binary_output

# Function that applies Sobel x or y, then takes an absolute value and applies a threshold.
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

# Our mega threshold function
@timing
def threshold_image(image, v): # V = settings vector

#    image = hls_select(image, (v[0], v[1]), min(max(v[2], 0), 1), 1)

    image = sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(v[0], v[1]))

    return image

    # Apply each of the thresholding functions
#    ksize = 3
#    gradx_binary = sobel_threshold(image, orient='x', sobel_kernel=ksize, thresh=(80, 255))
#    mag_binary   = magnitude_threshold(image, sobel_kernel=ksize, mag_thresh=(30, 255)) 
#    dir_binary   = direction_threshold(image, sobel_kernel=ksize, thresh=(0.9, 1.2))
#    hls_binary   = hls_select(image, (50, 255), 1)

    # Combine them
#    combined = np.zeros_like(gradx_binary)
#    combined[ ( ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1) ) ] = 1
#    return gradx_binary

# Warp
#@timing
def warp_image(image):
    global M
    return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

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
    
# -------------- The pipeline --------------

# The full pipeline
steering_position = 0
speed = 0
def pipeline(image, v):
    global speed, steering_position
    global M, Minv
    global warp_on, threshold_on
    
    # Blank equals blank
    if image is None: return None, None, 0, 0

    # Define source, dest and matricies for perspective stretch and stretch back
    if M is None:
        src = np.float32(
           [[image.shape[1] * 0.2, 0], 
            [image.shape[1] * 0.8, 0],
            [-500,                 image.shape[0]],
            [500+image.shape[1],   image.shape[0]]])
        dest = np.float32(
            [[0,              0],
             [image.shape[1], 0],
             [0,              image.shape[0]],
             [image.shape[1], image.shape[0]]])
        M = cv2.getPerspectiveTransform(src, dest)
        Minv = cv2.getPerspectiveTransform(dest, src)

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

    return warp_image(image_in), image, steer, speed
 
