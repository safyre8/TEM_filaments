import os
import sys
import tifffile as tiff
import cv2
import numpy as np
from skimage import filters, color, img_as_float, exposure
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from skimage.filters import gaussian

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


def display_image(image, title=""):
    """Display an image."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def preprocess_image(image):
    """Preprocess the input image: downsample, denoise, and normalize."""
    # Downsample for faster processing
    image_resized = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

    # Normalize image to range [0, 1]
    image_normalized = img_as_float(image_resized)

    # Apply Gaussian denoising
    image_denoised = filters.gaussian(image_normalized, sigma=0.5)

    return image_denoised


def apply_clahe(image):
    """Apply CLAHE for contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def subtract_background(image):
    """Subtract the background using Gaussian blur."""
    # Estimate background
    background = cv2.GaussianBlur(image, (21, 21), 0)
    # Subtract and normalize
    subtracted = cv2.subtract(image, background)
    return subtracted


def filter_image(image, method, sigma):
    """Apply filament detection using specified method and sigma."""
    methods = {
        'meijering': meijering,
        'sato': sato,
        'frangi': frangi,
        'hessian': hessian,
    }
    if method not in methods:
        raise ValueError(f"Unsupported method: {method}")

    return methods[method](image, sigmas=[sigma], black_ridges=False, mode='reflect')


def process_filaments(image, method, sigma_range):
    """Run filament detection in parallel across sigma values."""
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(filter_image, image, method, sigma): sigma for sigma in sigma_range}
        for future in futures:
            result = future.result()
            results.append(result)
    return results


def isolate_filaments(image, filtered_image, threshold=0.1):
    """Threshold and clean up filament mask."""
    _, binary_mask = cv2.threshold(filtered_image.astype(np.float32), threshold, 1, cv2.THRESH_BINARY)
    binary_mask = (binary_mask * 255).astype(np.uint8)

    # Clean mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply the mask to the original image
    return cv2.bitwise_and(image, image, mask=cleaned_mask)


def detect_filament_orientation_ridge(image):
    """
    Detect filament orientation by calculating the orientation from the eigenvalues
    of the Hessian matrix. If you're have issues with filament detection do a parameter run
    with the different methods to see which method and sigma is best for your data.

    Parameters:
    binary_image (numpy.ndarray): Binary image containing filaments.

    Returns:
    numpy.ndarray: Array representing the orientation of filaments.
    """

    # Apply the Meijering filtering method
    detected_result = meijering(image, sigmas=range(2, 4), black_ridges=False, mode='reflect')

    return detected_result


def run_parameter_sweep(image, filament_file_path, name, methods=('meijering', 'sato', 'frangi', 'hessian'), sigma_range=range(1, 6)):
    """Run a sweep across methods and sigma values, saving results."""
    for method in methods:
        print(f"Processing {method} method...")
        results = process_filaments(image, method, sigma_range)

        for sigma, result in zip(sigma_range, results):
            result_filename = f"{name}_{method}_sigma_{sigma}.tiff"
            save_path = os.path.join(filament_file_path, result_filename)
            tiff.imwrite(save_path, (result * 255).astype(np.uint8), photometric='minisblack')

            print(f"Saved result: {result_filename}")

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

        # Load image
        file_path = os.path.join(data_folder, file_name)
        image_data = open_image(file_path)

        if image_data is None:
            print(f"Error loading {file_name}. Skipping...")
            continue

        image_preprocessed = preprocess_image(image_data)

        filament_detected = detect_filament_orientation_ridge(image_preprocessed)
        smoothed_img = gaussian(filament_detected, sigma=2)

        print(f"Completed processing for {file_name}")

        print(f"+ Filaments detected")

        tiff.imwrite(filament_file_path, smoothed_img, photometric='minisblack')
        print(f"+ Combined image and mask saved as tif")

        print(f"Line Iteration {iteration_count}: Processed {file_name} for line detection and orientation plotting.")


if __name__ == "__main__":
    main()