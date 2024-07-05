import os
import sys
import cv2
import tifffile as tiff
import re
import numpy as np
from skimage import exposure
from skimage.filters import meijering, sato, frangi
from skimage import filters
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import img_as_float

def open_image(file_path):
    """
    Open an image file using TIFF format. None of the metadata is saved

    Args:
    - file_name (str): The name of the image file.

    Returns:
    - ndarray: Loaded image data.
    """
    image = tiff.imread(file_path)
    return image

def display_image(image):
    # Ensure the image data is within the valid range for display
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0, 1)
    else:
        image = np.clip(image, 0, 255)
    plt.imshow(image, cmap='gray')
    plt.show()

def adaptive_noise_estimation(image):
    image_float = img_as_float(image)
    sigma_est = estimate_sigma(image_float, average_sigmas=True)
    return sigma_est

def preprocess_image(image):
    """
    Preprocess the image.
    """
    # image = filters.gaussian(image, sigma=0.1)  # Denoise
    sigma_est = adaptive_noise_estimation(image)
    image = denoise_nl_means(image, h=.05 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3)
    # image = exposure.equalize_hist(image)  # Enhance contrast
    # # Apply median filter
    # kernel_size = 3
    # image = cv2.medianBlur(image, kernel_size)
    #
    # # Apply CLAHE
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    # clahe_image = clahe.apply(image)

    thresh = filters.threshold_otsu(image)  # Global thresholding
    binary_image = image > thresh  # Binarize the image
    display_image(binary_image)
    print(f"+ processed")
    return binary_image


def detect_largest_shape(image):
    """
    Detects the largest enclosed shape in an 8-bit image.

    Parameters:
    image (numpy.ndarray): Input 8-bit grayscale image.

    Returns:
    tuple: (center_x, center_y, width, height) of the bounding box of the largest shape.
    """

    # Check if the image is 8-bit
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Step 1: Threshold the image to create a binary image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_shape = None
    largest_area = 0

    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Update if this contour has the largest area found so far
        if area > largest_area:
            largest_area = area
            largest_shape = contour

    # Calculate the bounding box of the largest shape
    x, y, width, height = cv2.boundingRect(largest_shape)
    center_x, center_y = x + width // 2, y + height // 2
    largest_circle = (center_x, center_y, width, height)

    return largest_circle


def crop_and_rescale_image(image, center_x, center_y, radius):
    """
    Take the mask from the largest circle detected to crop the image.
    """

    # Define the region of interest (ROI) based on the detected circle
    top_left_x = max(0, center_x - radius)
    top_left_y = max(0, center_y - radius)
    bottom_right_x = min(image.shape[1], center_x + radius)
    bottom_right_y = min(image.shape[0], center_y + radius)

    # Crop the image to the ROI
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Create a mask for the circle in the cropped image
    circle_mask = np.zeros_like(cropped_image, dtype=np.uint8)
    cv2.circle(circle_mask, (radius, radius), radius, 255, -1)

    # Apply the mask to the cropped image
    masked_cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=circle_mask)

    #convert to an 8 bit image
    size = (768, 768)
    image_8bit = cv2.normalize(masked_cropped_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resized_image = cv2.resize(image_8bit, size, interpolation=cv2.INTER_AREA)

    return resized_image, circle_mask


def detect_filament_orientation_ridge(binary_image, file_name):
    """
    Detect filament orientation by calculating the orientation from the eigenvalues
    of the Hessian matrix. If you're have issues with filament detection do a parameter run
    with the different methods to see which method and sigma is best for your data.

    Parameters:
    binary_image (numpy.ndarray): Binary image containing filaments.

    Returns:
    numpy.ndarray: Array representing the orientation of filaments.
    """
    # Parameters
    # sigmas = range(4)  # Adjust sigma values as necessary

    # Apply the Meijering method
    # detected_result = meijering(binary_image, sigmas=sigmas, black_ridges=False, mode='reflect')

    # # Apply the sato method
    detected_result = sato(binary_image, sigmas=range(1, 4), black_ridges=False, mode='reflect')

    # # Apply the sato method
    detected_combined_result = frangi(detected_result, sigmas=range(5), black_ridges=False, mode='reflect')

    alpha = 175  # Contrast control
    beta = 150  # Brightness control
    adjusted_result = cv2.convertScaleAbs(detected_combined_result, alpha=alpha, beta=beta)

    print(f"+ Finished plotting the line detection for {file_name}")

    return adjusted_result


def main():
    data_folder = os.path.normpath(os.path.join(sys.path[1], "data"))
    # Initialize a counter variable
    iteration_count = 0

    for file_name in os.listdir(data_folder):
        # file_name = '20240514_95pc5pip_hex_50nM_02003.tif'

        iteration_count += 1

        # Prepare save path
        name = file_name[:-4]
        save_path = os.path.normpath(os.path.join(sys.path[1], "detected_filaments"))
        filament_file_path = os.path.join(save_path, f"{name}_detecting_filament_sato.tif")

        # load image
        file_path = os.path.join(data_folder, file_name)
        image_data = open_image(file_path)

        if image_data is None:
            print("Error loading image")
            return

        print(f"+ Opened {name}")

        binary_image = preprocess_image(image_data)

        # Detect the largest shape
        center_x, center_y, width, height = detect_largest_shape(binary_image)
        print(f"+ Detected largest shape with bounding box: center=({center_x}, {center_y}), width={width}, height={height}")

        # Crop and rescale the image based on the detected shape
        rescaled_image = crop_and_rescale_image(image_data, center_x, center_y, max(width, height) // 2)
        cropped_img = rescaled_image[0]
        mask_img = rescaled_image[1]
        print(f"+ Image masked and cropped")

        # rescaled_image[0] is the cropped image and rescaled_image[1] is the mask
        cropped_img = rescaled_image[0]
        mask_img = rescaled_image[1]

        filament_detected = detect_filament_orientation_ridge(cropped_img, file_name)

        print(f"+ filaments detected")

        # plt.imshow(filament_detected, cmap='gray')
        # plt.show()

        tiff.imwrite(filament_file_path, filament_detected, photometric='minisblack')
        print(f"+ Cropped image and mask saved as tif")


        print(f"Line Iteration {iteration_count}: Processed {file_name} for line detection and orientation plotting.")

if __name__ == "__main__":
    main()
