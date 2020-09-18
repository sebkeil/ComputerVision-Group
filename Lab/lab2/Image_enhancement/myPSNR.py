import cv2
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

orig_image = cv2.imread('images/image1_gaussian.jpg')
approx_image = cv2.imread('images/image1.jpg')
orig_image2 = cv2.imread('images/image1_saltpepper.jpg')

"""
print(orig_image.shape, approx_image.shape)
cv2.imshow('original', orig_image)
cv2.imshow('clean', approx_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

def myPSNR(orig_image, approx_image):
    orig_pixels = orig_image.astype('float32')
    approx_pixels = approx_image.astype('float32')
    orig_pixels /= 255.0        # normalize images to scale (0,1)
    approx_pixels /= 255.0
    mse = np.mean((orig_pixels - approx_pixels) ** 2)
    imax = np.amax(orig_pixels)
    PSNR = 20 * np.log10(imax/np.sqrt(mse))
    return PSNR


sp_error = myPSNR(orig_image2, approx_image)
gaussian_error = myPSNR(orig_image, approx_image)

comp = cv2.PSNR(orig_image2, approx_image)

print("PSNR for Salt-Pepper Image: {} ".format(sp_error))
print("PSNR for Gaussian Image: {} ".format(gaussian_error))

