
import orientation as orn
import line_detection as ld

#install PyWavelets and
def find_line_orientation():
    """
    Find the largest contoured shape in an image to detect the filaments present.

    Parameters:
    data folder with tiff images

    Returns:
    plots of the orientation for the detected filaments
    """
    ld.main()
    orn.main()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    find_line_orientation()
