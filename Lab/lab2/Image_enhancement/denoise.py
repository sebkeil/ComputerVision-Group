import cv2
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

gaussian = cv2.imread('images/image1_gaussian.jpg')
salt_pepper = cv2.imread('images/image1_saltpepper.jpg')

"""
print(orig_image.shape, approx_image.shape)
cv2.imshow('original', orig_image)
cv2.imshow('clean', approx_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


def denoise(image, kernel_type, **kwargs):
    if kernel_type == 'box':
        imOut = cv2.blur(image, **kwargs)       # e.g. ksize=(3,3)
    elif kernel_type == 'median':
        imOut = cv2.medianBlur(image, **kwargs)
    elif kernel_type == 'gaussian':
        imOut = cv2.GaussianBlur(image, **kwargs)
    else:
        imOut = image
        print('Please specify kernel type!')
    return imOut

gaussian_3x3 = denoise(gaussian, kernel_type='box', ksize=(3,3))
gaussian_5x5 = denoise(gaussian, kernel_type='box', ksize=(5,5))
gaussian_7x7 = denoise(gaussian, kernel_type='box', ksize=(5,5))


cv2.imshow('gaussian', gaussian)
cv2.imshow('gaussian_3x3', gaussian_3x3)
cv2.imshow('gaussian_5', gaussian_5x5)
cv2.imshow('gaussian_7', gaussian_7x7)
cv2.waitKey(0)
cv2.destroyAllWindows()

