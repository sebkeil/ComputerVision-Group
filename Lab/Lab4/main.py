import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import os 
from keypoint_matching import *
from RANSAC import * 

print('Libraries loaded successfully... ')

if __name__ == "__main__":
    
    # Load Images for Image Alignment 

    path = os.path.dirname(os.path.abspath(__file__))
    img1_path = path + '/boat1.pgm'
    img2_path = path + '/boat2.pgm'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # find key points of given images
    kp1, kp2, matches  = keypoint(img1, img2)
    
    # Draw matches ked points 
    plot_matched_points(img1, img2, kp1, kp2, matches)

    # Image alignment
    print('Trainsforming images...')
    N = 1
    P = 3
    transform_x = RANSAC(img1, kp1, kp2, matches, N=N, P=P)
    # trans_img = transformation(img1, transform_x)

    builtin_img = built_in(img1)
    # plot images with target image
    plot_images(img1, img2, trans_img, builtin_img)

