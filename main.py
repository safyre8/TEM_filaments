import orientation as orn
import line_detection as ld

#install orientationpy with PIP


def find_line_orientation():
    """
    Find the largest contoured shape in an image to detect the filaments present.

    Parameters:
    data folder with tiff images

    Returns:
    plots of the orientation for the detected filaments
    """
    # find largest contour and crop the image to find filaments
    ld.main()
    # find the orientation of the dectected filaments
    orn.main()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    find_line_orientation()
