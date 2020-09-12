import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def visualize(input_image):
    # Fill in this function. Remember to remove the pass command
    # Showing of Image 
    

    if(len(input_image.shape) == 3 and input_image.shape[2]==3):

        fig = plt.figure(1)
        # set up subplot grid
        gridspec.GridSpec(3,3)

        # large subplot
        plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
        plt.title('Proccessed Image')
        plt.imshow(input_image)

        # small subplot 1
        plt.subplot2grid((3,3), (0,2))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.title('Red Channel')
        plt.imshow(input_image[:,:,0]) # Red 
    

        # small subplot 2
        plt.subplot2grid((3,3), (1,2))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.title('Green Channel')
        plt.imshow(input_image[:,:,1]) # Green

        # small subplot 3
        plt.subplot2grid((3,3), (2,2))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.title('Blue Channel')
        plt.imshow(input_image[:,:,2]) # Blue

        # fit subplots and save fig
        fig.tight_layout()
        fig.set_size_inches(w=11,h=7)
        fig_name = 'plot.png'
        fig.savefig(fig_name)

        plt.show()
    else:
        fig = plt.figure()
        plt.imshow(input_image, cmap='gray')
        plt.show()
        fig_name = 'plot.png'
        fig.savefig(fig_name)



