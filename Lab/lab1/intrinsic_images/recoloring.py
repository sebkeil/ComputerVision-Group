import numpy as np
import cv2
import os
import matplotlib.pyplot as plt 

print('Image Coloring')

def image_recoloring(image_dir='' ):
    # Importing albedo and shade and ball images
    ball = cv2.imread(image_dir+'/ball.png') 
    ball_albedo = cv2.imread(image_dir+'/ball_albedo.png')
    ball_shade = cv2.imread(image_dir+'/ball_shading.png')
    ball_albedo = np.array(ball_albedo)
    ball_shade = np.array(ball_shade)

    # True material colour of the ball in RGB
    # we found the albedo have three unique value beside zero
    # considering this unique value as the true RGB value of the albedo 
    r =np.unique(ball_albedo)[1]
    g =np.unique(ball_albedo)[2]
    b =np.unique(ball_albedo)[3]
    
    # Re-colour the ball with pure green (0,255,0)
    ball_albedo[ball_albedo == r] = 0
    ball_albedo[ball_albedo == g] = 255
    ball_albedo[ball_albedo == b] = 0
    
    # normalize the albedo and shading value
    ball_albedo = ball_albedo/256
    ball_shade = ball_shade/256

    ball_image = ball_albedo * ball_shade

    fig = plt.figure()
    plt.title('Original ball vs Green colored')
    fig.add_subplot(1,2, 1)
    plt.imshow(ball)
    fig.add_subplot(1,2, 2)
    plt.imshow(ball_image)
    plt.show(block=True)


if __name__ == '__main__':
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    image_recoloring(cur_dir)