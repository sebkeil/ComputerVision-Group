import numpy as np
import cv2
import os
from utils import *
from estimate_alb_nrm import estimate_alb_nrm
from check_integrability import check_integrability
from construct_surface import construct_surface
import matplotlib.pyplot as plt

print('Part 1: Photometric Stereo\n')

def photometric_stereo(image_dir='./SphereGray5/' ):

    # obtain many images in a fixed view under different illumination
    print('Loading images...\n')
    [image_stack, scriptV] = load_syn_images(image_dir, channel=0)
    #[h, w, n] = image_stack.shape
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)

 
    # compute the surface gradient from the stack of imgs and light source mat
    print('Computing surface albedo and normal map...\n')
    [albedo, normals] = estimate_alb_nrm(image_stack, scriptV)


    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking\n')
    [p, q, SE] = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q)

    # show results
    show_results(albedo, normals, height_map, SE)


## Face
def photometric_stereo_face(image_dir='./yaleB02/'):
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)

    threshold = 0.005;
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q , path_type='column')

    # show results
    show_results(albedo, normals, height_map, SE)
    
    
if __name__ == '__main__':
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = '/photometrics_images/SphereGray5'
    image_dir = cur_dir+target_dir
    #photometric_stereo(image_dir)

    target_dir2 = '/photometrics_images/SphereGray25'
    image_dir2 = cur_dir+target_dir2
    #photometric_stereo(image_dir=image_dir2)


    target_dir_mon = '/photometrics_images/MonkeyGray'
    image_dir_mon = cur_dir+target_dir_mon
    #photometric_stereo(image_dir=image_dir_mon)

    target_dir_color = '/photometrics_images/SphereColor'
    image_dir_color = cur_dir+target_dir_color
    #photometric_stereo(image_dir=image_dir_color)


    target_dir3 = '/photometrics_images/yaleB02/'
    image_dir3 = cur_dir+target_dir3
    photometric_stereo_face(image_dir=image_dir3)





 

