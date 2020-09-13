import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

def visualize(input_image):
    # Fill in this function. Remember to remove the pass command
    # Showing of Image 
    if(len(input_image.shape) == 3 and input_image.shape[2]==3):
        #input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        fig = plt.figure(1)
        # set up subplot grid
        gridspec.GridSpec(3,3)

        # Main subplot
        plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
        plt.title('Proccessed Image')
        plt.imshow(input_image)
        # Red Channel subplot
        plt.subplot2grid((3,3), (0,2))
        plt.title('Red Channel')
        plt.imshow(input_image[:,:,0]) # Red channel 
        plt.subplot2grid((3,3), (1,2))
        plt.title('Green Channel')
        plt.imshow(input_image[:,:,1]) # Green channel
        plt.subplot2grid((3,3), (2,2))
        plt.title('Blue Channel')
        plt.imshow(input_image[:,:,2]) # Blue channel
        
        plt.tight_layout()
        figname = 'plot.png'
        plt.savefig(figname)
        plt.show()
    else:
        fig = plt.figure()
        plt.imshow(input_image, cmap='gray')
        plt.savefig('plot.png')
        plt.show()
        




