import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve2d


def compute_gradient(image):
    print('Computing Gradient...\n')

    # Defining Gaussian filters
    filter_x = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [3, 0, -3]])
    filter_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])

    # Sobel kernels  in the x and y directions
    Gx = convolve2d(gray_image, filter_x, mode='same')
    Gy = convolve2d(gray_image, filter_y, mode='same')

    # Computing image magnitude and direction
    im_magnitude = np.sqrt(np.square(Gx) + np.square(Gy))
    im_direction = np.arctan(np.divide(Gy, Gx))
    # Threshold
    # im_direction[im_direction > 50 ] = 100

    return Gx, Gy, im_magnitude, im_direction


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
    # img = img.astype(np.float32)
    gray_image = 0.2989 * r + 0.5870 * g + 0.1140 * b

    Gx, Gy, magnitude, direction = compute_gradient(gray_image)

    fig = plt.figure(figsize=(10, 8))

    fig.add_subplot(2, 2, 1)
    plt.imshow(Gx, cmap='gray')
    plt.axis("off")
    fig.add_subplot(2, 2, 2)
    plt.imshow(Gy, cmap='gray')
    plt.axis("off")
    fig.add_subplot(2, 2, 3)
    plt.imshow(magnitude, cmap='gray')
    plt.axis("off")
    fig.add_subplot(2, 2, 4)
    plt.imshow(direction, cmap='gray')
    plt.axis("off")
    plt.show()
