import numpy as np
import cv2
from getColourChannels import getColourChannels

def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods
    r,b,g = getColourChannels(input_image)
    new_image = np.empty_like(input_image[:,:,0])
    
    # ligtness method
    new_image = (np.max(input_image,axis=-1,keepdims=1)+np.min(input_image,axis=-1,keepdims=1))/2
    
    # average method
    #new_image = (r + g + b)/3
    
    # luminosity method
    #new_image = (0.21 * r) + (0.72 * g) + (0.07 * b)
    
    # built-in opencv function 
    #new_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    return new_image


def rgb2opponent(input_image):
    # copy the shape from existing image
    new_image = np.empty_like(input_image)
    # get the red, blue and green channel using given function. 
    r,b,g = getColourChannels(input_image)

    # converts an RGB image into opponent colour space by given formula 

    new_image[:,:,0] = (r - g)/np.sqrt(2)
    new_image[:,:,1] = (r + g - 2 * b)/np.sqrt(6)
    new_image[:,:,2] = (r + g + b)/ np.sqrt(3)

    return new_image


def rgb2normedrgb(input_image):
    # copy the shape from existing image
    new_image = np.empty_like(input_image)
    # get the red, blue and green channel using given function. 
    r,b,g = getColourChannels(input_image)
    
    # converts an RGB image into normalized rgb colour space
    np.seterr(divide='ignore', invalid='ignore')
    new_image[:,:,0] = r / (r + b + g)
    new_image[:,:,1] = g / (r + b + g)
    new_image[:,:,2] = b / (r + b + g)

    return new_image
