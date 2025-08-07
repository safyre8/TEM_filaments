import os
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import tifffile as tiff
import orientationpy
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm


def open_file(file_path):
    """
    Open a file using pandas.

    Args:
    - file_path (str): The full path to the file.

    Returns:
    - ndarray: Loaded file data as a NumPy array.
    """
    try:
        data = pd.read_csv(file_path, header=None).values
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def open_image(image_folder, name):
    """
    Open an image file using TIFF format.

    Args:
    - file_name (str): The name of the image file.

    Returns:
    - ndarray: Loaded image data.
    """
    image_file_path = os.path.join(image_folder, name)
    image = tiff.imread(image_file_path)
    return image


def display_image(image):
    # Ensure the image data is within the valid range for display
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0, 1)
    else:
        image = np.clip(image, 0, 255)
    plt.imshow(image, cmap='gray')
    plt.show()


def plot_norm_energy_and_coherency(image, orientations):
    """
    Plot normalized energy and coherency as side-by-side subplots.

    Args:
    - image (ndarray): The input image data.
    - orientations (dict): Dictionary containing 'theta', 'energy', and 'coherency'.

    Returns:
    - matplotlib.figure.Figure: The generated figure.
    """
    theta = orientations['theta']
    energy = orientations['energy']
    coherency = orientations['coherency']

    # Create subplots (1 row, 2 columns)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Energy subplot
    energy_normalized = energy / energy.max() if energy.max() > 0 else energy
    im1 = axs[0].imshow(energy_normalized, vmin=0, vmax=1, cmap='viridis')
    fig.colorbar(im1, ax=axs[0], shrink=0.7)
    axs[0].set_title("Energy Normalized")

    # Coherency subplot
    coherency = np.clip(coherency, 0, 1)
    coherency[image == 0] = 0  # Mask out zero areas
    im2 = axs[1].imshow(coherency, vmin=0, vmax=1, cmap='viridis')
    fig.colorbar(im2, ax=axs[1], shrink=0.7)
    axs[1].set_title("Coherency")

    # Adjust layout for better spacing
    # fig.tight_layout()

    return fig
def plot_orientation_layover(image, orientations):
    theta = orientations['theta']
    coherency = orientations['coherency']

    # Initialize HSV image with 3 channels
    imDisplayHSV = np.zeros((image.shape[0], image.shape[1], 3), dtype="float32")

    # Hue is the orientation mapped to [0, 1] (where 0 = -90 degrees, 1 = +90 degrees)
    imDisplayHSV[:, :, 0] = (theta + 90) / 180

    # Saturation is coherency, normalized to [0, 1]
    max_coherency = coherency.max()
    if max_coherency > 0:
        imDisplayHSV[:, :, 1] = coherency / max_coherency
    else:
        imDisplayHSV[:, :, 1] = 0  # Default to zero if no coherency

    # Value is the preprocessed image intensity, normalized to [0, 1]
    max_intensity = image.max()
    if max_intensity > 0:
        imDisplayHSV[:, :, 2] = image / max_intensity
    else:
        imDisplayHSV[:, :, 2] = 0  # Default to zero if no intensity

    # Convert HSV to RGB for visualization
    imDisplayRGB = hsv_to_rgb(imDisplayHSV)

    # Plot the composite image
    fig, ax = plt.subplots()

    ax.imshow(imDisplayRGB)
    ax.axis('off')  # Hide axis

    # Colorbar for orientation
    cmap = "hsv"
    norm = Normalize(vmin=-90, vmax=90)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Required for colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", label="Degrees from Horizontal", shrink=0.7)
    cbar.set_ticks([-89.9, -45, 0, 45, 89.9])
    cbar.set_ticklabels(["-90°", "-45°", "0°", "45°", "90°"])

    # Ensure the imDisplayRGB is in the correct format (uint8 for TIFF)
    imDisplayRGB = (imDisplayRGB * 255).astype(np.uint8)

    # Return the RGB image
    plt.show()

    return fig


def plot_orientation_boxes(file_name, image, orientations, gradient_Gy, gradient_Gx):
    """
    Plots local orientation vectors in boxes overlaid on the input image.

    Args:
    - file_name (str): Name of the file.
    - image (ndarray): The input image.
    - orientations (dict): Dictionary containing orientation data.
    - gradient_Gy (ndarray): Gradient in Y direction.
    - gradient_Gx (ndarray): Gradient in X direction.

    Returns:
    - matplotlib.figure.Figure: The generated figure.
    """
    boxSizePixels = 7
    structureTensorBoxes = orientationpy.computeStructureTensorBoxes(
        [gradient_Gy, gradient_Gx],
        [boxSizePixels, boxSizePixels],
    )

    # Compute orientations from the structure tensor
    orientationsBoxes = orientationpy.computeOrientation(
        structureTensorBoxes,
        mode="fiber",
        computeEnergy=True,
        computeCoherency=True,
    )

    # Normalize energy for better visualization
    orientationsBoxes["energy"] /= orientationsBoxes["energy"].max()

    # Compute box centers
    boxCentresY = np.arange(orientationsBoxes["theta"].shape[0]) * boxSizePixels + boxSizePixels // 2
    boxCentresX = np.arange(orientationsBoxes["theta"].shape[1]) * boxSizePixels + boxSizePixels // 2

    # Compute vector components
    boxVectorsYX = orientationpy.anglesToVectors(orientationsBoxes)

    # Reset vectors with low energy
    boxVectorsYX[:, orientationsBoxes["energy"] < 0.05] = 0.0

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))

    # ax.set_title("Local orientation vectors in boxes")
    ax.imshow(image, cmap="Greys_r", vmin=0)

    # Overlay vectors on the image
    ax.quiver(
        boxCentresX,
        boxCentresY,
        boxVectorsYX[1],
        boxVectorsYX[0],
        angles="xy",
        scale_units="xy",
        color="r",
        headwidth=0,
        headlength=0,
        headaxislength=5,
    )
    plt.tight_layout()
    # plt.show()
    return fig, boxVectorsYX


def normalize_histogram(hist):
    # Find the peak of the histogram
    peak_index = np.argmax(hist)

    # Roll the histogram so that the peak is centered at the zero bin
    normalized_hist = np.roll(hist, shift=-peak_index + len(hist) // 2)

    return normalized_hist


def plot_orientation_histogram(image, orientations):
    """
    Plot a linear histogram of orientation values ranging from -90 to 90 degrees and overlay orientation vectors.

    Parameters:
    - image (ndarray): The image data.
    - file_name (str): File name for saving plots.
    - orientations: Dictionary containing 'theta' and 'energy' keys with values in degrees.
    """

    # Extract the orientation values and convert to a 1D array
    orientation_values = orientations['theta']
    energy_values = orientations['energy']

    # Define the number of bins and the range of the histogram
    num_bins = 36  # Number of bins for the histogram
    bins = np.linspace(-90, 90, num_bins + 1)  # Bins from -90 to 90 degrees

    # Create the histogram, weighted by energy
    hist, bin_edges = np.histogram(orientation_values.flatten(), bins=bins, weights=energy_values.flatten())

    # Normalize the histogram so that the peak is centered at 0 degrees
    normalized_hist = normalize_histogram(hist)

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bin_edges[:-1], normalized_hist, width=(180 / num_bins), align='edge', color='blue', alpha=1, edgecolor='k')

    # Set labels and title
    ax.set_xlabel('Orientation (Degrees)')
    ax.set_ylabel('Frequency (Weighted by Energy)')
    ax.set_xticks(np.arange(-90, 91, 30))  # Set x-ticks to cover the range from -90 to 90 degrees

    # Show the plot
    # plt.show()
    return fig

