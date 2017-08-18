import cv2
import numpy as np
from PIL import Image   
import time
import glob

# Options
warp_on = False
threshold_on = False

# Globals
M = None
Minv = None

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print( '%s %d ms' % (f.__name__, (time2-time1)*1000.0) )
        return ret
    return wrap

# --------- Thresholding functions ----------

# Function that thresholds a channel of HLS
@timing
def hls_select(img, threshold, hls_option):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # Apply a threshold to the channel
    s = hls[:, :, hls_option]
    binary_output = np.zeros_like(s)
    binary_output[ (s>threshold[0]) & (s<=threshold[1]) ] = 1
    
    # Return a binary image of threshold result
    return binary_output

# Function that thresholds a channel of YUV
@timing
def yuv_select(img, threshold, yuv_option, invert = 1):
    # Convert to YUV color space
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # Apply a threshold to the channel
    u = invert * yuv[:, :, yuv_option]
    binary_output = np.zeros_like(u)
    binary_output[ (u>threshold[0]) & (u<=threshold[1]) ] = 1
    
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
    gray = img #cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    
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
    return scaled_sobel
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


# Our mega threshold function
@timing
def threshold(image):
    # Apply each of the thresholding functions
    ksize = 3
    gradx_binary = sobel_threshold(image, orient='x', sobel_kernel=ksize, thresh=(80, 255))
    mag_binary   = magnitude_threshold(image, sobel_kernel=ksize, mag_thresh=(30, 255)) 
    dir_binary   = direction_threshold(image, sobel_kernel=ksize, thresh=(0.9, 1.2))
    hls_binary   = hls_select(image, (50, 255), 1)
#    yuv_binary = yuv_select(image, (50, 95), 0)

    # Combine them
    combined = np.zeros_like(gradx_binary)
    combined[ ( ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1) ) ] = 1
    return gradx_binary

# Warp
#@timing
def warp_image(image):
    global M
    return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

# Find those lanes
@timing
def find_lanes(image, orig):
    
    windows_image = np.zeros_like(image)

    # Take a histogram of the bottom half of the image
    histogram = np.sum(image[int(image.shape[0]/2):, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 5

    # Set height of windows
    window_height = np.int(image.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 20

    # Set minimum number of pixels found to recenter window
    minpix = 2

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

        # Draw the windows on the visualization image
        cv2.rectangle(orig, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 0), 2)
        cv2.rectangle(orig, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

#    return orig
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

 #   return image

    # Fit a second order polynomial to each
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        left_fit = [0,0,0]
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        right_fit = [0,0,0]

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

    new_image = np.dstack((image, image, image))*255

    # Create an output image to draw on and visualize the result. Colouring time. Colour the lanes.
    try:
        if windows_image is not None:
            lanes_image = np.dstack((image, windows_image, image))*255
            lanes_image[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            lanes_image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        else:
            lanes_image = np.zeros_like(new_image)
    except:
        lanes_image = np.zeros_like(new_image)

    coloured_image = np.zeros_like(new_image)

    # Draw the lane onto the warped blank image
 #   cv2.fillPoly(coloured_image, np.int_([left_line_pts]), (255, 0, 0))
  #  cv2.fillPoly(coloured_image, np.int_([right_line_pts]), (0, 0, 255))

    # Combine
    combined_image = cv2.addWeighted(new_image, 1, coloured_image, 1, 0)

    return windows_image, lanes_image, coloured_image, combined_image, left_fitx, right_fitx, ploty

# Fit those curves
ym_per_pix = 30 / 1440 # meters per pixel in y dimension (height of warped image)
xm_per_pix = 3.7 / 740 # meters per pixel in x dimension (midpoint in road, 740-540 = 200. (200 + 1280) / 2 = 740 )
def fit_curves( leftx, rightx, ploty ):

    # Fit a second order polynomial to pixel positions in each lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Fit new polynomials to x, y in world space
    try:
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        y_eval = np.max(ploty)/1.0
        left_curverad =  ((1 + (2 *  left_fit_cr[0] * y_eval * ym_per_pix +  left_fit_cr[1]) **2) **1.5) / (2 *  left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) **2) **1.5) / (2 * right_fit_cr[0])        
    except:
        left_curverad = None
        right_curverad = None

    # Now our radius of curvature is in meters
    return left_curverad, right_curverad, left_fitx, right_fitx, ploty

# Define function to draw the road
@timing
def draw_lanes(image, processed_image, lanes_image, left_fitx, right_fitx, ploty, steering_position, speed):
    # Create an image to draw the lines on
    greyscale_blank_image = np.zeros_like(processed_image).astype(np.uint8)
    coloured_image = np.dstack((greyscale_blank_image, greyscale_blank_image, greyscale_blank_image))

    # Draw the road as a big green rectangle onto the blank image
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(coloured_image, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(coloured_image, Minv, (image.shape[1], image.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    # Add lanes
    #newwarp = cv2.warpPerspective(lanes_image, Minv, (image.shape[1], image.shape[0])) 
    #result = cv2.addWeighted(result, 0.8, newwarp, 1, 0)

    # Calculate centre offset
    lane_centre = (left_fitx[-1] + right_fitx[-1]) / 2.0
    image_centre = image.shape[1] / 2.0
    offset_distance = lane_centre - image_centre
    offset_distance *= xm_per_pix

    # Black bar
    steering_bar_width = image.shape[1] / 2
    steering = np.array( [[[image_centre - steering_bar_width/2, 10], [image_centre + steering_bar_width/2, 10], 
                           [image_centre + steering_bar_width/2, 60], [image_centre - steering_bar_width/2, 60]]], dtype=np.int32 )
    cv2.fillPoly(result, np.int_([steering]), (10, 20, 10))

    # Green indicator
    steering_bar_width = 50
    steering = np.array( [[[steering_position + image_centre - steering_bar_width/2, 15], [steering_position + image_centre + steering_bar_width/2, 15], 
                           [steering_position + image_centre + steering_bar_width/2, 55], [steering_position + image_centre - steering_bar_width/2, 55]]], dtype=np.int32 )
    cv2.fillPoly(result, np.int_([steering]), (10, 180, 100))

    # Black bar
    speed_bar = np.array( [[[image_centre - 25, 70], [image_centre + 25, 70], 
                           [image_centre + 25, 160], [image_centre - 25, 160]]], dtype=np.int32 )
    cv2.fillPoly(result, np.int_([speed_bar]), (10, 20, 10))

    # Green indicator
    speed_bar = np.array( [[[image_centre - 25, 110-speed], [image_centre + 25, 110-speed], 
                           [image_centre + 25, 160-speed], [image_centre - 25, 160-speed]]], dtype=np.int32 )
    cv2.fillPoly(result, np.int_([speed_bar]), (10, 180, 100))

    # Write curve radius
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        cv2.putText(result, "Curve Radius: " + str( int( curve_radius ) ) + "m", (10, 150), font, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(result, "Offset Distance: " + str( round( offset_distance, 2 ) ) + "m", (10, 200), font, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    except:
        pass

    # Done
    return result
    
# -------------- The pipeline --------------

# The full pipeline
steering_position = 0
speed = 0
def pipeline( image ):
    global speed, steering_position
    global M, Minv
    global warp_on, threshold_on

    # Define source, dest and matricies for perspective stretch and stretch back
    if M is None:
        src = np.float32(
           [[image.shape[1] * 0.2, 0], 
            [image.shape[1] * 0.8, 0],
            [-400,                 image.shape[0]],
            [400+image.shape[1],   image.shape[0]]])
        dest = np.float32(
            [[0,              0],
             [image.shape[1], 0],
             [0,              image.shape[0]],
             [image.shape[1], image.shape[0]]])
        M = cv2.getPerspectiveTransform(src, dest)
        Minv = cv2.getPerspectiveTransform(dest, src)
 
    # Threshold the image to make the line edges stand out
#    image = threshold(image)
    if threshold_on:
        image = sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(2, 200))

    # Stretch the image out so we have a bird's-eye view of the scene
    if warp_on:
        image = warp_image(image)

    # Find the lanes, y goes from 0 to y-max, and left_lane_x and right_lane_x map the lanes
#    if threshold_on:
#        _, _, vimage, aimage, left_lane_x, right_lane_x, y = find_lanes(image, image)
#        image = find_lanes(image, image)

#    image = th_image #np.dstack((th_image, th_image, th_image))

    return image, 0
    
    # Find centre line x co-ords
    centre = [ int(l_x + r_x / 2) for l_x, r_x in zip(left_lane_x, right_lane_x) ]

    # After all that, just take the bottom of the centre line, and the top of it, and steer that way
    steer = centre[0]- centre[-1]

    # Return
    return image, steer / 20

    # Update smoothed curve radius frame by frame, 20% each time
#    if left_curverad and right_curverad:
#        curve_radius = (left_curverad + right_curverad) / 2
#    else:
#        curve_radius = 0

    # Figure out possible new steering position from current curve_radius
#    if curve_radius > 900 or curve_radius < -900:
#        new_steering_position = 0
#    elif curve_radius == 0:
#        new_steering_position = 0
#    else:
#        new_steering_position = 50000.0 / curve_radius

    # Update steering position 20% each time
#    if new_steering_position == 0 or new_steering_position > 100 or new_steering_position < -100:
#        pass
#    else:
#        steering_position += ( (new_steering_position - steering_position) / 20 )

    # Test speed
#    speed = max(0, 20 - abs(steering_position)/4)

    # Draw those lines
    #image_out = draw_lanes(image, processed_image, lanes_image, left_fitx, right_fitx, ploty, steering_position, speed)
 
