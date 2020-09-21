import numpy as np
from numpy import pi, exp, sqrt
import matplotlib.pylab as plt
import cv2
from scipy.signal import convolve2d
import os


def compute_LoG(image, LOG_type=1):
    if LOG_type == 1:

        #method 1
        sigma = 0.5      
        size = 5      
        img = cv2.GaussianBlur(image, (size,size), sigma)
        img = cv2.Laplacian(img, ddepth= cv2.CV_16S, ksize=size)
  
        return img

    elif LOG_type == 2:
        #method 2
        size = 5
        sigma = 0.5
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        g = g/g.sum()
        
        outimg = convolve2d(image, g, mode='same')

            
        return outimg

    elif LOG_type == 3:
        #method 3
        size = 5
        sigma = 0.5
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g1 = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        g1 = g1/g1.sum()
        
        size = 5
        sigma = 1.0
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g2 = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        g2 = g2/g2.sum()
        gd = g1 - g2 
        
        outimg = convolve2d(image, gd, mode='same')
        
        return outimg

	#return imOut

      
  
def Gaussian_filter(image, size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    h = g * (1 / (2 * sigma * sigma * np.pi))

    blur_img = convolve2d(image, h)
    return blur_img


if __name__ == '__main__':
    # Load image2.jpg from image folder
    current_folder = os.path.dirname(os.path.realpath(__file__))
    image_path = '/images/image2.jpg'
    img = cv2.imread(current_folder + image_path)

    # Convert data to grayscale
    img = img[:, :, ::-1]
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    img = img.astype(np.float32)
    gray_image = 0.2989 * r + 0.5870 * g + 0.1140 * b

    output_image1 = compute_LoG(gray_image, LOG_type=1)
    output_image2 = compute_LoG(gray_image, LOG_type=2)
    output_image3 = compute_LoG(gray_image, LOG_type=3)

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].set_title('Original Image')
    ax[0, 0].imshow(gray_image, cmap='gray')
    ax[0, 1].set_title('Method 1')
    ax[0, 1].imshow(output_image1, cmap='gray')
    ax[1, 0].set_title('Method 2')
    ax[1, 0].imshow(output_image2, cmap='gray')
    ax[1, 1].set_title('Method 3')
    ax[1, 1].imshow(output_image3, cmap='gray')

    plt.show()

    