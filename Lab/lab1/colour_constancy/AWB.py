import cv2 as cv
import numpy as np
import os

def show(final):
    print('display')
    cv.imshow('Temple', final)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Insert any filename with path
cur_dir = os.path.dirname(os.path.realpath(__file__))
print(cur_dir)
target_img = '/awb.jpg'
image_dir = cur_dir+ target_img
img = cv.imread(image_dir) 

def white_balance_loops(img):
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result


final = np.hstack((img, white_balance_loops(img)))
show(final)
cv.imwrite('result.jpg', final)
