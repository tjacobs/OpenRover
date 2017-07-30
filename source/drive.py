import camera
import vision
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image                                                                                

# Open a sample image
image = mpimg.imread('test_images/straight_lines1.jpg') 

# Run our vision image processing pipeline
pipelined_image = vision.pipeline(image)

# Save
mpimg.imsave('out.png', pipelined_image) 

# Open it to see
img = Image.open('out.png')
img.show() 

exit()


def process( image ):
    cv2.putText(image, "Rover", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return image

# Create window
#cv2.namedWindow( "preview" )
#cv2.moveWindow( "preview", 10, 10 )

# Start camera
camera.startCamera( (640, 368) )

# Loop
while True:

    # Get a frame
    frame = camera.getFrame()
    
    # Process
    processed_image = process( frame )

    break

    # Show
#    cv2.imshow( "preview", processed_image )

    # Esc key hit?
 #   key = cv2.waitKey(20)
 #   if key == 27:
 #       break

# Close
#cv2.destroyWindow( "preview" )


#processed_image = None   
# Run
#windows_image, lanes_image, coloured_image, combined_image, left_fit, right_fit, ploty = find_lanes(processed_image)
#plot( processed_image, windows_image, "Windows" )
#plot( processed_image, lanes_image, "Lane pixels" )
#plot( processed_image, combined_image, 'Coloured over lanes')

# Plot lanes
#right_img = plot( processed_image, lanes_image, "Fit lanes" )
#right_img.plot(left_fit, ploty, color='yellow')
#right_img.plot(right_fit, ploty, color='yellow')

# Run
#left_curverad, right_curverad, left_fitx, right_fitx, ploty = fit_curves(left_fit, right_fit, ploty)
#image_out = draw_lanes( image, lanes_image, left_fitx, right_fitx, ploty, left_curverad )
#plot_image = plot( image, image_out, "The Long Green Road" )

# Create video
#vision.create_video( vision.pipeline, "video.mp4", "output.mp4" )

