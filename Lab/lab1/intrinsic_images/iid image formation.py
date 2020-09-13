import numpy as np
import cv2
import os
import matplotlib.pyplot as plt 

print('Image Formation')


def iid_image_formation(image_dir='' ):
    # Importing albedo and shade of the ball 
    ball_albedo = cv2.imread(image_dir+'/ball_albedo.png')
    ball_shade = cv2.imread(image_dir+'/ball_shading.png')

    # converting into numpy array and diving by 256 to normalize pixel value
    ball_albedo = np.array(ball_albedo)/256
    ball_shade = np.array(ball_shade)/256
    # I(x) = R(x) x S(x)
    ball_image = ball_albedo * ball_shade

    plt.imshow(ball_image,  cmap='gray')
    plt.show()


if __name__ == '__main__':
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    iid_image_formation(cur_dir)