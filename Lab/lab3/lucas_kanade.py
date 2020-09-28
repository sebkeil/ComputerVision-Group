import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


sphere1 = cv2.imread('sphere1.ppm', cv2.IMREAD_GRAYSCALE)
sphere2 = cv2.imread('sphere2.ppm', cv2.IMREAD_GRAYSCALE)

synth1 = cv2.imread('synth1.ppm', cv2.IMREAD_GRAYSCALE)
synth2 = cv2.imread('synth2.ppm', cv2.IMREAD_GRAYSCALE)

jpeg0000 = cv2.imread('pingpong/0000.jpeg', cv2.IMREAD_GRAYSCALE)
jpeg0001 = cv2.imread('pingpong/0001.jpeg', cv2.IMREAD_GRAYSCALE)


def split_windows(image, winsize_r=15, winsize_c=15):
    imheight = image.shape[0]
    imwidth = image.shape[1]
    crop_size = 5
    windows = []

    x_positions = []      # keep track of central pixel locations for the quiver plot
    y_positions = []

    for y in range(0, imheight-crop_size, winsize_c):
        for x in range(0, imwidth-crop_size, winsize_r):
            window = image[y:y + winsize_c, x:x + winsize_r]
            x_pos = round(x + winsize_r/2, 0)       # round up
            y_pos = round(y + winsize_c / 2, 0)
            x_positions.append(x_pos)
            y_positions.append(y_pos)
            windows.append(window)
            #print(window.shape)
            #cv2.imshow("window_{}_{}".format(x,y), window)

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)

    return windows, x_positions, y_positions


def optical_flow(win1, win2):

    sobel_x = np.asarray([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.asarray([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]],  dtype=np.float32)

    kernel_t = np.ones((3,3), dtype=np.float32)

    # take the image derivatives using sobel kernels
    mode = 'same'
    Ix = signal.convolve2d(win1, sobel_x, boundary='symm', mode=mode)
    Iy = signal.convolve2d(win1, sobel_y, boundary='symm', mode=mode)
    It = signal.convolve2d(win2, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(win1, -kernel_t, boundary='symm', mode=mode)

    b = It.flatten()  # get b here
    A = np.vstack((Ix.flatten(), Iy.flatten())).T  # get A here
    v = np.matmul(np.matmul(np.linalg.pinv(np.matmul(A.T, A)),A.T), b)

    return v


def arrange_vector(win_list1, win_list2):
    # arange and fill up the vector of velocities
    v_vector = []

    for i in range(len(win_list1)):
        v = optical_flow(win_list1[i], win_list2[i])
        v_vector.append(v)

    v_vector = np.array(v_vector)
    return v_vector


def plot_optical_flow(x_pos, y_pos, v_vector):

    fig, ax = plt.subplots()
    plt.quiver(x_pos, y_pos, v_vector[:, 0], v_vector[:, 1], angles='xy', color='green')
    #plt.xlim(0,200)
    #plt.ylim(0,200)
    plt.show()


# for the spheres
sphere1_wins, sphere1_xpos, sphere1_ypos = split_windows(sphere1)
sphere2_wins, sphere2_xpos, sphere2_ypos = split_windows(sphere2)
sphere_v = arrange_vector(sphere1_wins, sphere2_wins)
plot_optical_flow(sphere1_xpos, sphere1_ypos, sphere_v)


# for the synth images
synth1_wins, synth1_xpos, synth1_ypos = split_windows(synth1)
synth2_wins, synth2_xpos, synth2_ypos = split_windows(synth2)
synth_v = arrange_vector(synth1_wins, synth2_wins)
plot_optical_flow(synth1_xpos, synth1_ypos, synth_v)


# for the jpeg ims
jpeg0000_wins, jpeg0000_xpos, jpeg0000_ypos = split_windows(jpeg0000)
jpeg0001_wins, jpeg0001_xpos, jpeg0001_ypos = split_windows(jpeg0001)
jpeg_v = arrange_vector(jpeg0000_wins, jpeg0001_wins)
print(jpeg0000_xpos.shape, jpeg0000_ypos.shape, jpeg_v.shape)
plot_optical_flow(jpeg0000_xpos, jpeg0000_ypos, jpeg_v)






