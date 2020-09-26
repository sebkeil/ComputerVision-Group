import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


sphere1 = cv2.imread('sphere1.ppm', cv2.IMREAD_GRAYSCALE)
sphere2 = cv2.imread('sphere2.ppm', cv2.IMREAD_GRAYSCALE)

cv2.imshow('sphere1', sphere1)


"""
def split_windows(image, winsize_r=15, winsize_c=15):
    imheight = image.shape[0]
    imwidth = image.shape[1]
    crop_size = 5
    windows = []

    for y in range(0, imheight-crop_size, winsize_c):
        for x in range(0, imwidth-crop_size, winsize_r):
            window = image[y:y + winsize_c, x:x + winsize_r]
            windows.append(window)
            #print(window.shape)
            #cv2.imshow("window_{}_{}".format(x,y), window)

    return windows

sphere1_wins = split_windows(sphere1)
sphere2_wins = split_windows(sphere2)

"""


def optical_flow(image1, image2):

    # normalize images
    image1 = image1/255
    image2 = image2/255

    sobel_x = np.asarray([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.asarray([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]],  dtype=np.float32)

    kernel_t = np.ones((3,3), dtype=np.float32)

    imheight = image1.shape[0]
    imwidth = image1.shape[1]
    crop_size = 5
    winsize_r = 15
    winsize_c = 15

    v_matrix = np.zeros((imheight, imwidth, 2))     # -crop_size?

    # take the image derivatives using sobel kernels
    mode = 'same'
    fx = signal.convolve2d(image1, sobel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(image1, sobel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(image2, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(image1, -kernel_t, boundary='symm', mode=mode)

    for y in range(0, imheight-crop_size, winsize_c):
        for x in range(0, imwidth-crop_size, winsize_r):
            Ix = fx[y:y + winsize_c, x:x + winsize_r].flatten()
            Iy = fy[y:y + winsize_c, x:x + winsize_r].flatten()
            It = ft[y:y + winsize_c, x:x + winsize_r].flatten()
            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here
            v = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T, A)),A.T), b)
            v_matrix[x][y][0] = v[0]
            v_matrix[x][y][1] = v[1]
    return v_matrix


v = optical_flow(sphere1, sphere2)


print("Sphere 1 x index:", sphere1.ndindex.shape)
print("Sphere 1 y:", sphere1[0].shape)
print("V", v.shape)
print("V[0, :, 0]", v[0, :, 0].shape)
print("V[:, 0, 1]", v[:, 0, 1].shape)
print("V[:, :, 1]", v[:, :, 1].shape)


def plot_optical_flow(image1, v):
    fig, ax = plt.subplots()

    x_pos = image1[1].index
    y_pos = image1[0]
    x_direct = v[0, :, 0]
    y_direct = v[:, 0, 1]

    ax.quiver(x_pos, y_pos, x_direct, y_direct)
    plt.show()


plot_optical_flow(sphere1, v)


"""
def optical_flow(image1, image2):

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])

    image1 = image1/255     # normalize image
    image2 = image2/255

    # calculate Ix, Iy, It for each point
    mode = 'same'
    fx = signal.convolve2d(image1, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(image1, kernel_x, boundary='symm', mode=mode)
    ft = signal.convolve2d(image2, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(image1, -kernel_t, boundary='symm', mode=mode)

    u = np.zeros(image1.shape)
    v = np.zeros(image2.shape)

    for i in range(w, image1.shape[0]-w):
        for j in range(w, image2.shape[1]-w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

            b = np.reshape(It, (It.shape[0], 1))  # get b here
            A = np.vstack((Ix, Iy)).T  # get A here

            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
                u[i, j] = nu[0]
                v[i, j] = nu[1]

    return v    # v=[Vx, Vy], components of the optical flow


def lucas_kanade_np(im1, im2, win=2):
    assert im1.shape == im2.shape
    I_x = np.zeros(im1.shape)
    I_y = np.zeros(im1.shape)
    I_t = np.zeros(im1.shape)
    I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
    params = np.zeros(im1.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
    params[..., 0] = I_x * I_x # I_x2
    params[..., 1] = I_y * I_y # I_y2
    params[..., 2] = I_x * I_y # I_xy
    params[..., 3] = I_x * I_t # I_xt
    params[..., 4] = I_y * I_t # I_yt
    del I_x, I_y, I_t
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    del params
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])
    del cum_params
    op_flow = np.zeros(im1.shape + (2,))
    det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2
    op_flow_x = np.where(det != 0,
                         (win_params[..., 1] * win_params[..., 3] -
                          win_params[..., 2] * win_params[..., 4]) / det,
                         0)
    op_flow_y = np.where(det != 0,
                         (win_params[..., 0] * win_params[..., 4] -
                          win_params[..., 2] * win_params[..., 3]) / det,
                         0)
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
    return op_flow


v = lucas_kanade_np(sphere1, sphere2)

print(v.shape)

"""










