import numpy as np

def construct_surface(p, q, path_type='column'):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    np.seterr(divide='ignore', invalid='ignore')

    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value
        
        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value
        
        """

        height_map[0][0] = 0
        for x in range(1,h):
            height_map[x][0] = height_map[x - 1][0] + q[x][0]
            
        for y in range(1,w):
            height_map[:, y] = height_map[:,y-1] + p[:,y]


    elif path_type=='row':
        """
        ================
        Your code here
        ================
        """
        height_map[0][0] = 0
        for x in range(1, w):
            height_map[0][x] = height_map[0][x-1] + q[0][x]
        for y in range(1,h):
            height_map[y,:] = height_map[y-1,:] + q[y,:]


    elif path_type=='average':
        """
        ================
        Your code here
        ================
        """

        height_map[0][0] = 0
        for x in range(1,h):
            height_map[x][0] = height_map[x - 1][0] + q[x][0]
            
        for y in range(1,w):
            height_map[:, y] = height_map[:,y-1] + p[:,y]

        height_map_temp = height_map

        height_map = 0 
        for x in range(1, w):
            height_map[0][x] = height_map[0][x-1] + q[0][x]
        for y in range(1,h):
            height_map[y,:] = height_map[y-1,:] + q[y,:]

        height_map = (height_map + height_map_temp)/2

        
    return height_map
        
