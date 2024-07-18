import os
import sys
import re
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from skimage import filters
import cv2
import tifffile as tiff
import orientationpy
import setup_steerable as steerable
# from matplotlib.gridspec import GridSpec
# from scipy import interpolat


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


def open_original_image(original_file_name):
    """
    Open an image file using TIFF format and extract metadata.

    Args:
    - original_file_name (str): The name of the original image file.

    Returns:
    - tuple: (image_data, metadata)
    """
    original_file_path = os.path.normpath(os.path.join(sys.path[1], "data", original_file_name))

    # Use tifffile to open the image and read metadata
    with tiff.TiffFile(original_file_path) as tif:
        original_image = tif.asarray()
        metadata = tif.pages[0].tags

    print(f"+ Opened the original file: {original_file_name[:-4]}")
    return original_image, metadata


def extract_metadata(original_file_name):
    """
    Extract pixel and nanometer values from the file name for scaling.

    Args:
    - file_name (str): The name of the image file.

    Returns:
    - tuple: width in pixels, width in nanometers, pixels per nanometer
    """

    try:
        widthPX = float(re.findall(r'(\d+)px', original_file_name)[0])
        print(widthPX)
        widthNM = float(re.findall(r'(\d+)nm', original_file_name)[0])
    except IndexError:
        raise ValueError("File name does not contain pixel or nanometer information.")

    px_per_nm = widthPX / widthNM
    return widthPX, widthNM, px_per_nm


def display_image(image):
    # Ensure the image data is within the valid range for display
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0, 1)
    else:
        image = np.clip(image, 0, 255)
    plt.imshow(image, cmap='gray')
    # plt.show()


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
    # cv2.bilateralFilter(image, 9, 75, 75)
    # # Apply median filter
    # kernel_size = 3
    # image = cv2.medianBlur(image, kernel_size)
    #
    # # Apply CLAHE
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    # clahe_image = clahe.apply(image)
    # return clahe_image

    # Adjust contrast and brightness to enhance shadows
    # alpha = 3  # Contrast control (1.0-3.0) - started with 1.5
    # beta = 20  # Brightness control (0-100)
    #
    # adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    #
    # # Combine with original image for better visual quality
    # image = cv2.addWeighted(image, 0.5, adjusted_image, 0.5, 0)

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


def plot_norm_energy_and_coherency(file_name, image, orientations):
    """
    Plot image with overlaid orientations.

    Args:
    - image (ndarray): Input image data.
    - orientations (dict): Orientations dictionary.
    """
    plt.figure(figsize=(10, 4))

    # The energy represents how strong the orientation signal is
    plt.subplot(1, 2, 1)
    plt.imshow(orientations["energy"] / orientations["energy"].max(), vmin=0, vmax=1)
    plt.colorbar(shrink=0.7)
    plt.title("Energy Normalised")

    # The coherency measures how strongly aligned the image is locally
    orientations["coherency"][image == 0] = 0

    plt.subplot(1, 2, 2)
    plt.imshow(orientations["coherency"], vmin=0, vmax=1)
    plt.title("Coherency")
    plt.colorbar(shrink=0.7)
    plt.tight_layout()
    # plt.show()

    print(f"+ plot Energy normalized and coherency")

    # Save the plot as a TIFF image
    name = os.path.splitext(file_name)[0]  # Get the file name without extension
    print(sys.path[1])
    save_path = os.path.normpath(os.path.join(sys.path[1], "figure", "norm_energy_and_coherency"))
    os.makedirs(save_path, exist_ok=True)  # Create directories if they do not exist
    file_path = os.path.join(save_path, f"{name}_norm_energy_and_coherency.png")  # Use .png format for saving
    plt.savefig(file_path, dpi=300)

    print(f"+ Normalized energy and coherency plot saved as {file_path}")


def plot_orientation_layover(file_name, preprocessed_img, orientations):
    # Initialize HSV image with 3 channels
    imDisplayHSV = np.zeros((preprocessed_img.shape[0], preprocessed_img.shape[1], 3), dtype="float32")

    # Hue is the orientation mapped to [0, 1] (where 0 = -90 degrees, 1 = +90 degrees)
    imDisplayHSV[:, :, 0] = (orientations["theta"] + 90) / 180

    # Saturation is coherency, normalized to [0, 1]
    max_coherency = orientations["coherency"].max()
    if max_coherency > 0:
        imDisplayHSV[:, :, 1] = orientations["coherency"] / max_coherency
    else:
        imDisplayHSV[:, :, 1] = 0  # Default to zero if no coherency

    # Value is the preprocessed image intensity, normalized to [0, 1]
    max_intensity = preprocessed_img.max()
    if max_intensity > 0:
        imDisplayHSV[:, :, 2] = preprocessed_img / max_intensity
    else:
        imDisplayHSV[:, :, 2] = 0  # Default to zero if no intensity

    # Convert HSV to RGB for visualization
    imDisplayRGB = hsv_to_rgb(imDisplayHSV)

    # Plot the composite image
    fig, ax = plt.subplots()

    ax.imshow(imDisplayRGB)
    ax.axis('off')  # Hide axis

    # Colorbar for orientation
    # Colorbar for orientation using the 'hsv' colormap
    cmap = "hsv"
    norm = Normalize(vmin=-90, vmax=90)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Required for colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", label="Degrees from Horizontal", shrink=0.7)
    cbar.set_ticks([-90, -45, 0, 45, 90])
    cbar.set_ticklabels(["-90°", "-45°", "0°", "45°", "90°"])

    # Ensure the imDisplayRGB is in the correct format (uint8 for TIFF)
    imDisplayRGB = (imDisplayRGB * 255).astype(np.uint8)

    # Save the plot as a TIFF image
    name = os.path.splitext(file_name)[0]  # Get the file name without extension
    save_path = os.path.normpath(os.path.join(sys.path[1], "figure", "orientation_layover"))
    os.makedirs(save_path, exist_ok=True)  # Create directories if they do not exist
    file_path = os.path.join(save_path, f"{name}_orientation_layover.png")
    plt.savefig(file_path, dpi=300)  # Save the entire figure as a high-resolution PNG
    print(f"+ orientation layover saved as tif")
    plt.show()
    return imDisplayRGB


def plot_orientation_boxes(file_name, preprocessed_img, orientations, gradient_Gy, gradient_Gx):
    boxSizePixels = 7
    structureTensorBoxes = orientationpy.computeStructureTensorBoxes(
        [gradient_Gy, gradient_Gx],
        [boxSizePixels, boxSizePixels],
    )

    # The structure tensor in boxes is passed to the same function to compute
    # The orientation
    orientationsBoxes = orientationpy.computeOrientation(
        structureTensorBoxes,
        mode="fiber",
        computeEnergy=True,
        computeCoherency=True,
    )

    # We normalise the energy, to be able to hide arrows in the subsequent quiver plot
    orientationsBoxes["energy"] /= orientationsBoxes["energy"].max()

    # Compute box centres
    boxCentresY = np.arange(orientationsBoxes["theta"].shape[0]) * boxSizePixels + boxSizePixels // 2
    boxCentresX = np.arange(orientationsBoxes["theta"].shape[1]) * boxSizePixels + boxSizePixels // 2

    # Compute X and Y components of the vector
    boxVectorsYX = orientationpy.anglesToVectors(orientationsBoxes)

    # Vectors with low energy reset
    boxVectorsYX[:, orientationsBoxes["energy"] < 0.05] = 0.0

    plt.title("Local orientation vector in boxes")
    plt.imshow(preprocessed_img, cmap="Greys_r", vmin=0)

    # Warning, matplotlib is XY convention, not YX!
    plt.quiver(
        boxCentresX,
        boxCentresY,
        boxVectorsYX[1],
        boxVectorsYX[0],
        angles="xy",
        scale_units="xy",
        # scale=energyBoxes.ravel(),
        color="r",
        headwidth=0,
        headlength=0,
        headaxislength=1,
    )
    plt.show()
    # Save the plot as a TIFF image


def plot_orientation_histogram(file_name, orientations):
    """
    Plot a linear histogram of orientation values ranging from -90 to 90 degrees.

    Parameters:
    - orientations: Dictionary containing 'theta' key with orientation values in degrees.
    """
    # Extract the orientation values and convert to a 1D array
    orientation_values = orientations['theta'].flatten()

    # Define the number of bins and the range of the histogram
    num_bins = 36  # Number of bins for the histogram
    bins = np.linspace(-90, 90, num_bins + 1)  # Bins from -90 to 90 degrees

    # Create the histogram
    hist, bin_edges = np.histogram(orientation_values, bins=bins)

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bin_edges[:-1], hist, width=(180 / num_bins), align='edge', color='blue', alpha=1, edgecolor='k')

    # Set labels and title
    ax.set_xlabel('Orientation (Degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Orientation Histogram')
    ax.set_xticks(np.arange(-90, 91, 30))  # Set x-ticks to cover the range from -90 to 90 degrees
    # Show the plot
    plt.show()

    # Save the plot as a TIFF image
    name = os.path.splitext(file_name)[0]  # Get the file name without extension
    save_path = os.path.normpath(os.path.join(sys.path[1], "figure", "orientation_histogram"))
    os.makedirs(save_path, exist_ok=True)  # Create directories if they do not exist
    file_path = os.path.join(save_path, f"{name}_orientation_histogram.png")
    plt.savefig(file_path, dpi=300)  # Save the entire figure as a high-resolution PNG
    print(f"+ orientation histogram saved as tif")
    # plt.close()

    # return fig, ax



def main():
    data_folder = os.path.normpath(os.path.join(sys.path[1], "detected_filaments"))

    # Initialize a counter variable
    iteration_count = 0

    for file_name in os.listdir(data_folder):
        iteration_count += 1

        file_path = os.path.join(data_folder, file_name)
        image_data = open_image(file_path)
        print(f"+ Opened the processed file: {file_name[:-4]}")

        # Preprocess image for orientation analysis
        preprocessed_img = preprocess_image_for_orientation(image_data)
        print(f"+ Preprocessed image")

        # Plot image gradients
        gradients = plot_image_gradients(preprocessed_img)
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

        # Plot normalized energy and coherency (assuming this function requires preprocessed_img and orientations)
        plot_norm_energy_and_coherency(file_name, preprocessed_img, orientations)
        print(f"+ Normalized energy and coherency plot saved as tif")

        # Plot orientation layover (assuming this function requires file_name, preprocessed_img, and orientations)
        plot_orientation_layover(file_name, preprocessed_img, orientations)
        print(f"+ Orientation layover saved as tif")

        # Plot orientation histogram (assuming this function requires orientations)
        plot_orientation_histogram(file_name, orientations)
        print(f"+ Orientation histogram saved as tif")

        plot_orientation_boxes(file_name, preprocessed_img, orientations, gradient_Gy, gradient_Gx)
        print(f"+ Orientation boxes vectors saved as tif")


        print(f"Orientation Iteration {iteration_count}: Processed {file_name} for line detection and orientation plotting.")



if __name__ == "__main__":
    main()