import numpy as np
import cv2
import rgbConversions
from visualize import *
import os
import matplotlib as plt
import matplotlib.gridspec as gridspec

def ConvertColourSpace(input_image, colourspace):
    '''
    Converts an RGB image into a specified color space, visualizes the
    color channels and returns the image in its new color space.

    Colorspace options:
      opponent
      rgb -> for normalized RGB
      hsv
      ycbcr
      gray

    P.S: Do not forget the visualization part!
    '''
    
    # Convert the image into double precision for conversions
    input_image = input_image.astype(np.float32)
    

    if colourspace.lower() == 'opponent':
        # fill in the rgb2opponent function
        new_image = rgbConversions.rgb2opponent(input_image)

    elif colourspace.lower() == 'rgb':
        # fill in the rgb2opponent function
        new_image = rgbConversions.rgb2normedrgb(input_image)

    elif colourspace.lower() == 'hsv':
        # use built-in function from opencv
        new_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)

    elif colourspace.lower() == 'ycbcr':
        # use built-in function from opencv
        new_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
        temp = np.array(new_image[:, :, 1])
        new_image[:, :, 1] = new_image[:, :, 2]
        new_image[:, :, 2] = temp

    elif colourspace.lower() == 'gray':
        # fill in the rgb2opponent function
        new_image = rgbConversions.rgb2grays(input_image)

    else:
        print('Error: Unknown colorspace type [%s]...' % colourspace)
        new_image = input_image

    visualize(new_image)

    return new_image


if __name__ == '__main__':

    # Replace the image name with a valid image
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    img_path = cur_dir + '/images/dog.jpg'
    print(img_path)
    # Read with opencv
    I = cv2.imread(img_path)

    # Convert from BGR to RGB
    # This is a shorthand.
    I = I[:, :, ::-1]

    # All the function has been implimented, please uncomment 
    # to view each type of output
    # For specific gray image need to uncomment the functions 
    # in the rgbConversions.py file
    #out_img = ConvertColourSpace(I, 'opponent')
    #out_img = ConvertColourSpace(I, 'rgb')
    #out_img = ConvertColourSpace(I, 'hsv')
    out_img = ConvertColourSpace(I, 'ycbcr')
    #out_img = ConvertColourSpace(I, 'gray')

