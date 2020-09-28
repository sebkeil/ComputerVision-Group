import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter


def load_img(path='/person_toy/00000001.jpg'):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.astype(np.float32)

    return img, gray_img

def harris_corner_detector(img, window_size=29, threshold=600, sigma=5, kernel_size=5):

    # Creating Gaussina filter based on kernel size
    centre = int(kernel_size/2)
    kernel = []
    val = -centre
    if kernel_size %2 != 0: 
        for i in range(kernel_size):
            kernel.append(val)
            val +=1 
    else: 
        for i in range(kernel_size):
            kernel.append(val)
            val +=1 

    kernel = np.asarray(kernel)
    kernel = kernel.reshape(1, kernel_size)

    # Calculate Ix and Iy 
    Ix = convolve2d(img, kernel, mode='same')
    Iy = convolve2d(img, kernel.T, mode='same')
    
    # Calculating Harris matrix 
    A = gaussian_filter(np.square(Ix), sigma=sigma)
    B = gaussian_filter(Ix*Iy, sigma=sigma)
    C = gaussian_filter(np.square(Iy), sigma=sigma)
    H = (A*C-B**2) - 0.04*(A+C)**2

    # Find corners 
    r = []
    c = []
    row_no, col_no = gray_img.shape
    offset = np.int(window_size/2)

    for rx in range(offset, row_no-offset):
        for cx in range(offset, col_no-offset):
            w = H[rx - offset:rx + offset, cx - offset: cx + offset]
            if H[rx,cx] == np.max(w) and H[rx,cx] > threshold:
                r.append(rx)
                c.append(cx)

    return H, r, c, Ix, Iy

def plot_figures(img, Ix, Iy, r, c):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figwidth(15)
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Orginal Image')
    ax1.plot(c,r, '.r', markersize=4)
    ax2.imshow(Ix, cmap='gray')
    ax2.set_title('Image derivatives Ix')
    ax3.imshow(Iy, cmap='gray')
    ax3.set_title('Image derivatives Iy')
    plt.show()


if __name__ == "__main__":
    
    # load image(s)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.getcwd()
    img_path = '/person_toy/00000001.jpg'
    img, gray_img = load_img(cwd+img_path)

    # Perameters 
    kernel_size = 5
    sigma = 15
    window_size = 21
    threshold = 300
    H, r, c, Ix, Iy = harris_corner_detector(gray_img, window_size=window_size, threshold=threshold, sigma=sigma, kernel_size = kernel_size)
    plot_figures(img,Ix, Iy, r, c) 

    # Perameters 
    kernel_size = 5
    sigma = 10
    window_size = 25
    threshold = 20

    img_path = '/pingpong/0000.jpeg'
    img, gray_img = load_img(cwd+img_path)
    H, r, c, Ix, Iy = harris_corner_detector(gray_img, window_size=window_size, threshold=threshold, sigma=sigma, kernel_size = kernel_size)
    plot_figures(img,Ix, Iy, r, c) 

    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, 360-45, 1.0)
    img_rot45 = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    gray_img = cv2.cvtColor(img_rot45, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.astype(np.float32)
    H, r, c, Ix, Iy = harris_corner_detector(gray_img, window_size=window_size, threshold=threshold, sigma=sigma, kernel_size = kernel_size)
    plot_figures(img_rot45,Ix, Iy, r, c) 

    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, 360-90, 1.0)
    img_rot90 = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    gray_img = cv2.cvtColor(img_rot90, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.astype(np.float32)
    H, r, c, Ix, Iy = harris_corner_detector(gray_img, window_size=window_size, threshold=threshold, sigma=sigma, kernel_size = kernel_size)
    plot_figures(img_rot90,Ix, Iy, r, c) 


