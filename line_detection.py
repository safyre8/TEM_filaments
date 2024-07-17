import os
import sys
import cv2
import tifffile as tiff
import numpy as np
from skimage.filters import meijering, sato, frangi
from skimage import filters, color
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
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
    Preprocess the input image.

    Args:
    - image (ndarray): Input image data.

    Returns:
    - ndarray: Preprocessed image data.
    """
    # Convert to grayscale if necessary
    if image.ndim == 3 and image.shape[-1] == 3:
        image = color.rgb2gray(image)

    # Apply Gaussian blur for denoising
    image = filters.gaussian(image, sigma=0.1)

    # Binarize the image using Otsu's thresholding
    thresh = filters.threshold_otsu(image)
    binary_image = image > thresh

    return binary_image


def create_contour_mask(image, contour):
    """
    Create a binary mask from the largest contour.

    Parameters:
    - image (numpy.ndarray): Input image.
    - contour (numpy.ndarray): Largest contour.

    Returns:
    - mask (numpy.ndarray): Binary mask with the contour.
    """
    # Create a blank mask with the same dimensions as the image
    mask = np.zeros_like(image, dtype=np.uint8)

    # If the image is not grayscale, convert it to grayscale to match the mask
    if len(image.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Draw the contour on the mask
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    return mask


def detect_largest_shape(image):
    """
    Detects the largest enclosed shape in an 8-bit image.

    Parameters:
    image (numpy.ndarray): Input 8-bit grayscale image.

    Returns:
    tuple: (center_x, center_y, width, height) of the bounding box of the largest shape.
    """
    # Convert the image to grayscale if it is RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check if the image is 8-bit
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Step 1: Threshold the image to create a binary image
    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(image)

    # Sharpen the equalized image
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened_image = cv2.filter2D(equalized_image, -1, sharpening_kernel)

    # Step 2: Find contours
    contours, _ = cv2.findContours(sharpened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 3: Find largest contour
    if not contours:
        raise ValueError("No contours found in the image")

    largest_contour = max(contours, key=cv2.contourArea)
    print(largest_contour)

    # Calculate the bounding box of the largest shape
    x, y, width, height = cv2.boundingRect(largest_contour)
    center_x, center_y = x + width // 2, y + height // 2
    largest_circle = (center_x, center_y, width, height)

    return largest_circle


def crop_and_rescale_image(image, center_x, center_y, radius):
    """
    Take the mask from the largest circle detected to crop the image.
    """
    # Ensure the image is grayscale
    if image.ndim == 3 and image.shape[-1] == 3:
        image = color.rgb2gray(image)

    # Define the region of interest (ROI) based on the detected circle
    top_left_x = max(0, center_x - radius)
    top_left_y = max(0, center_y - radius)
    bottom_right_x = min(image.shape[1], center_x + radius)
    bottom_right_y = min(image.shape[0], center_y + radius)

    # Crop the image to the ROI
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Create a mask for the circle in the cropped image
    circle_mask = np.zeros_like(cropped_image, dtype=np.uint8)
    circle = cv2.circle(circle_mask, (radius, radius), radius, 255, -1)
    display_image(circle)
    # Apply the mask to the cropped image
    masked_cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=circle_mask)

    if masked_cropped_image is None:
        print("Error: masked_cropped_image is None")
        return None, None  # Handle the error gracefully

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
    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(binary_image)


    # equalized_image = adaptive_noise_estimation(binary_image)

    # Apply the Meijering filtering method
    # detected_result = meijering(equalized_image, sigmas=range(1, 4), black_ridges=False, mode='reflect')

    # Apply the sato filtering method
    detected_result = sato(equalized_image, sigmas=range(1, 4), black_ridges=False, mode='reflect')

    # # Apply the sato filtering method
    detected_combined_result = frangi(detected_result, sigmas=range(4), black_ridges=False, mode='reflect')

    print(f"+ Finished plotting the line detection for {file_name}")

    return detected_combined_result


def main():
    data_folder = os.path.normpath(os.path.join(sys.path[1], "data"))

    # Initialize a counter variable
    iteration_count = 0

    for file_name in os.listdir(data_folder):
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
        print(f"+ Image masked and cropped")
        display_image(cropped_img)

        filament_detected = detect_filament_orientation_ridge(cropped_img, file_name)
        print(f"+ filaments detected")
        display_image(filament_detected)
        tiff.imwrite(filament_file_path, filament_detected, photometric='minisblack')
        print(f"+ Cropped image and mask saved as tif")

        print(f"Line Iteration {iteration_count}: Processed {file_name} for line detection and orientation plotting.")

if __name__ == "__main__":
    main()
