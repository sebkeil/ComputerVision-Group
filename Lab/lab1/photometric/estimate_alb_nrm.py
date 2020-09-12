import numpy as np
import os
from utils import *

def estimate_alb_nrm( image_stack, scriptV, shadow_trick=False):
    
    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    h, w, _ = image_stack.shape
    
    # create arrays for 
    # albedo (1 channel)
    # normal (3 channels)
    albedo = np.zeros([h, w])
    normal = np.zeros([h, w, 3])
    

    """
    ================
    Your code here
    ================
    for each point in the image array
        stack image values into a vector i
        construct the diagonal matrix scriptI
        solve scriptI * scriptV * g = scriptI * i to obtain g for this point
        albedo at this point is |g|
        normal at this point is g / |g|
    """
    np.seterr(divide='ignore', invalid='ignore')

    for x in range(h):
        for y in range(w):
            i = image_stack[x][y].T
            scriptI = np.diag(i)
            A = np.matmul(scriptI, scriptV)  # multiply matrices so we can solve the linear system
            B = np.matmul(scriptI, i)
            if shadow_trick:
                g, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)  # solve linear algebra system for a constant
            else:
                g, residuals, rank, s = np.linalg.lstsq(scriptV,i, rcond=None)
            g_norm = np.linalg.norm(g)
            albedo[x][y] = g_norm
            normal[x,y,:] = np.divide(g,g_norm)

    
    return albedo, normal
    
if __name__ == '__main__':
    n = 5
    image_stack = np.zeros([10,10,n])
    scriptV = np.zeros([n,3])
    estimate_alb_nrm( image_stack, scriptV, shadow_trick=False)

'''    cur_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = '/photometrics_images/SphereGray5'
    image_dir = cur_dir+target_dir
    

    [image_stack, scriptV] = load_syn_images(image_dir)
    [h, w, n] = image_stack.shape
    [albedo, normals] = estimate_alb_nrm(image_stack, scriptV)'''

