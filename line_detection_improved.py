import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt

from skimage import filters, img_as_float
import cv2
import tifffile as tiff
import orientationpy
import seaborn as sns
import numpy as np
import seaborn.objects as so

import plotting


def open_image(file_path):
    """
    Open an image file using TIFF format.

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


def preprocess_image_for_orientation(image):
    """
    Preprocess image for orientation analysis.

    Args:
    - image (ndarray): Input image data.

    Returns:
    - ndarray: Preprocessed image data.
    """
    # Apply Gaussian filter for smoothing
    image = filters.gaussian(image, sigma=0.5)


    # Ensure the image is 8-bit
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 1, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return image


def plot_image_gradients(image):
    """
    Plot image gradients.

    Args:
    - image (ndarray): Input image data.

    Returns:
    - dir: processed with 3 different modes: "finite_difference", "gaussian", "splines"
        for Gx, Gy directions
    """
    gradients = []
    modes = ["finite_difference", "gaussian", "splines"]

    for n, mode in enumerate(modes):
        Gy, Gx = orientationpy.computeGradient(image, mode=mode)

        # Store gradients
        gradients_extraction = {
            "mode": mode,
            "Gy" : Gy,
            "Gx" : Gx
        }
        gradients.append(gradients_extraction)
    return gradients


def calculate_orientation(img_x, img_y, sigma=2):
    """
    Calculate orientation from image gradients.

    Args:
    - img_x (ndarray): Gradient in the x direction.
    - img_y (ndarray): Gradient in the y direction.
    - sigma (float): Sigma for computing the structure tensor.

    Returns:
    - dict: Orientations dictionary.
    """
    # Calculate structure tensor
    structure_tensor = orientationpy.computeStructureTensor([img_x, img_y], sigma=sigma)
    print(f"+ calculated structure tensor")

    orientations = orientationpy.computeOrientation(structure_tensor, computeEnergy=True, computeCoherency=True)

    # Retrieve theta, energy, and coherency from orientations
    theta = orientations.get("theta")
    energy = orientations.get("energy")
    coherency = orientations.get("coherency")

    # Handle NaN or Inf values in coherency
    if coherency is not None:
        nan_mask = np.isnan(coherency)
        coherency[nan_mask] = 0
        orientations["coherency"] = coherency

    # Assign other orientation metrics
    orientation_dir = {
        "theta": theta,
        "energy": energy,
        "coherency": coherency
    }
    return orientation_dir


def save_orientation_to_csv(orientation_data, file_name, orientation_type):
    """
    Save orientation data to a CSV file.

    Args:
    - orientation_data (ndarray): Array containing either 'theta', 'energy', or 'coherency'.
    - file_name (str): Name of the original image file.
    - orientation_type (str): Type of orientation data ('Theta', 'Energy', 'Coherency').
    """
    # Flatten the orientation data to create a DataFrame
    df = pd.DataFrame(orientation_data.flatten(), columns=[orientation_type])

    # Create directories if they do not exist
    save_path = os.path.normpath(os.path.join(sys.path[1], "orientations", orientation_type))
    os.makedirs(save_path, exist_ok=True)

    # Get the file name without extension
    name = os.path.splitext(file_name)[0]

    # Save to CSV
    file_path = os.path.join(save_path, f"{name}_{orientation_type}.csv")
    df.to_csv(file_path, index=False)
    print(f"{name} {orientation_type} saved as CSV at {file_path}")


def save_plot(plot_data, file_name, plot_type):
    """
    Save plot data to a png file.

    Args:
    - orientation_data (dict): Dictionary containing 'theta', 'energy', and 'coherency'.
    - file_name (str): Name of the CSV file to save.
    """
    if not isinstance(plot_data, plt.Figure):
        raise TypeError(f"Expected a Matplotlib Figure, got {type(plot_data)}")
    # Create directories if they do not exist
    save_path = os.path.normpath(os.path.join(sys.path[1], "figure", plot_type))
    os.makedirs(save_path, exist_ok=True)

    # Get the file name without extension
    name = os.path.splitext(file_name)[0]

    # Save to CSV
    file_path = os.path.join(save_path, f"{name}_{plot_type}.png")
    plot_data.savefig(file_path, dpi=300, bbox_inches="tight")  # Save the figure
    print(f"{name} for {plot_type} saved")


def process_orientation_data(orientation_summary):
    # Create an empty list to hold rows
    data_rows = []

    # Iterate through each file's data
    for entry in orientation_summary:
        file_name = entry['file_name']
        date = entry['date']
        concentration = entry['concentration']
        lipid_composition = entry['lipid_composition']
        septin_type = entry['septin_type']

        # Get the shape of the arrays
        rows, cols = entry['theta'].shape

        # Flatten and create rows
        for i in range(rows):
            for j in range(cols):
                data_rows.append({
                    "file_name": file_name,
                    "date": date,
                    "concentration": concentration,
                    "lipid_composition": lipid_composition,
                    "septin_type": septin_type,
                    "theta": entry['theta'][i, j],
                    "energy": entry['energy'][i, j],
                    "coherency": entry['coherency'][i, j],
                    "Row": i,
                    "Column": j
                })

    # Create a Pandas DataFrame
    return pd.DataFrame(data_rows)


def main():
    data_folder = os.path.normpath(os.path.join(sys.path[1], "detected_filaments"))

    # Initialize a counter variable
    iteration_count = 0

    # Initialize a dictionary to hold orientation data
    orientation_summary = []

    # goes through each image in the data folder to determine the orientation data
    for file_name in os.listdir(data_folder):
        iteration_count += 1

        file_path = os.path.join(data_folder, file_name)
        image_data = open_image(file_path)
        print(f"+ Opened the processed file: {file_name[:-4]}")

        # Plot image gradients
        gradients = plot_image_gradients(image_data)
        gradient_mode = 'splines'
        print(f"+ Used {gradient_mode} gradient")

        # Initialize variables for Gy and Gx
        gradient_Gy, gradient_Gx = None, None

        # Find the gradient for the desired mode
        for gradient in gradients:
            if gradient['mode'] == gradient_mode:
                gradient_Gy, gradient_Gx = gradient['Gy'], gradient['Gx']
                break

        # Calculate structure tensor and orientations
        orientations = calculate_orientation(gradient_Gy, gradient_Gx)
        print(f"+ Found orientations")

        _, detected_edges = cv2.threshold(orientations, 50, 255, cv2.THRESH_BINARY)



        # Create a dictionary for this file's data
        orientation_file = {
            "file_name": file_name,
            "date": file_name.split("_")[0],
            "concentration": file_name.split("_")[1],
            "lipid_composition": file_name.split("_")[2],
            "septin_type": file_name.split("_")[3],
            "theta": orientations["theta"],
            "energy": orientations["energy"],
            "coherency": orientations["coherency"],
        }

        # Append it to the list
        orientation_summary.append(orientation_file)

        plot_histogram = plotting.plot_orientation_histogram(image_data, orientations)
        save_plot(plot_histogram, file_name, "histogram")
        plot_orientation_layover = plotting.plot_orientation_layover(file_name, image_data, orientations)
        save_plot(plot_orientation_layover, file_name, "layover")
        plot_energy = plotting.plot_norm_energy_and_coherency(image_data, orientations)
        save_plot(plot_energy, file_name, "energy_coherency")
        plot_orientation_box = plotting.plot_orientation_boxes(file_name, image_data, orientations, gradient_Gy, gradient_Gx)
        save_plot(plot_orientation_box, file_name, "orientation_boxes")
        print(f"Orientation Iteration {iteration_count}: Processed {file_name} for orientation gradients.")

    # for orientation in orientation_summary:
    #     print(orientation)

    # Plot circular histograms grouped by lipid and septin type
    # Assuming `orientation_summary` is populated
    # summary = plotting.plot_circular_histograms(orientation_summary)
    # print(summary)
    #
    # # Save and display the figure for a specific lipid and septin type
    # lipid_type = '95pc0ps5pip'
    # septin_type = 'hex'
    #
    # if lipid_type in summary and septin_type in summary[lipid_type]:
    #     fig = summary[lipid_type][septin_type]
    #     fig.show()  # Activate the specific figure
    #     plt.show()  # Display all active figures
        print(f"Line Iteration {iteration_count}: Processed {file_name} for line detection and orientation plotting.")
    df_orientation = process_orientation_data(orientation_summary)

    return df_orientation

if __name__ == "__main__":
    main()