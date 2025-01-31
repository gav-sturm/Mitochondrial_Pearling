import tifffile as tiff
import numpy as np
from skimage import filters
import napari
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from tqdm import tqdm
import os


# Load the OME-TIFF file
def load_ome_tiff(file_path):
    with tiff.TiffFile(file_path) as tif:
        images = tif.asarray()
    return images


def select_roi(channels, channel=0):
    """Allow the user to select an XY ROI on the max projection and return the coordinates."""
    if len(channels.shape) > 3:
        # max_projection = np.max(channels, axis=1)  # Max projection along the channel axis
        max_projection = channels[:, channel, :, :]
    else:
        max_projection = channels

    viewer = napari.Viewer()
    viewer.add_image(max_projection, name='Max Projection')
    shapes = viewer.add_shapes(name='ROI', shape_type='rectangle', ndim=2)
    roi_coords = None

    @shapes.mouse_drag_callbacks.append
    def get_rectangle(layer, event):
        nonlocal roi_coords
        if len(layer.data) > 0:
            roi = layer.data[-1]
            y_start, x_start = roi[0]
            y_end, x_end = roi[2]
            x_start = max(0, int(x_start))
            y_start = max(0, int(y_start))
            x_end = min(max_projection.shape[-1], int(x_end))
            y_end = min(max_projection.shape[-2], int(y_end))

            roi_coords = (x_start, y_start, x_end, y_end)
            napari.run()
            viewer.close()

    napari.run()

    if roi_coords is None:
        raise ValueError("No ROI selected")

    return roi_coords


def crop_channels(channels, roi_coords):
    """Crop the channels to the selected ROI."""
    x_start, y_start, x_end, y_end = roi_coords
    if len(channels.shape) > 3:
        return channels[:, :, y_start:y_end, x_start:x_end]

    return channels[:, y_start:y_end, x_start:x_end]


# Apply mask from the 4th channel to all channels
def apply_mask(image, channel=0):
    if len(image.shape) == 3:  # Case with 1 channel or multiple channels (C, X, Y)
        mask = image[channel] > filters.threshold_otsu(image[channel])
        masked_channels = image * mask[np.newaxis, :, :]
        return mask, masked_channels

    elif len(image.shape) == 4:  # Case with multiple time points and channels (T, C, X, Y)
        mask = image[:, channel, :, :] > filters.threshold_otsu(image[:, channel, :, :])
        masked_channels = image * mask[:, np.newaxis, :, :]
        return mask, masked_channels

    else:
        raise ValueError("Input array must be 3D (C, X, Y) or 4D (T, C, X, Y).")


def upscale_images(images, scale_factor):
    """
    Upscale a series of images by a given scale factor.

    Parameters:
    images (numpy.ndarray): A 3D or 4D numpy array of images (T, C, X, Y) or (C, X, Y) or (X, Y).
    scale_factor (float): The factor by which to scale the images.

    Returns:
    numpy.ndarray: The upscaled images.
    """
    if len(images.shape) == 4:  # (T, C, X, Y)
        T, C, X, Y = images.shape
        upscaled_images = zoom(images, (1, 1, scale_factor, scale_factor), order=0)
    elif len(images.shape) == 3:  # (C, X, Y) or (T, X, Y)
        upscaled_images = zoom(images, (1, scale_factor, scale_factor), order=0)
    elif len(images.shape) == 2:  # (X, Y)
        upscaled_images = zoom(images, (scale_factor, scale_factor), order=0)
    else:
        raise ValueError("Input images must be 2D, 3D, or 4D.")

    return upscaled_images


def interpret_metadata(metadata, image_data):
    axes = metadata.get('axes', '')
    dimension_info = {}

    for i, axis in enumerate(axes):
        if axis == 'T':
            dimension_info['Time'] = image_data.shape[i]
        elif axis == 'C':
            dimension_info['Channels'] = image_data.shape[i]
        elif axis == 'Z':
            dimension_info['Depth'] = image_data.shape[i]
        elif axis == 'Y':
            dimension_info['Height'] = image_data.shape[i]
        elif axis == 'X':
            dimension_info['Width'] = image_data.shape[i]

    return dimension_info
def plot_spon_curv(x,y,error, save_path=None, visualize=False, title="Spontaneous curvature over time"):
    """Plot the GP signal over time with mean and standard error."""

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y, yerr=error, fmt='-o', capsize=5)
    plt.xlabel('Time Point')
    plt.ylabel('Spontenaous curvature (um^-1)')
    plt.title('Spontenous curvature over time')
    # plt.ylim(0,100)
    # plt.yscale('log')
    plt.grid(True)

    if save_path:
        filename = title + "_curvature.png"
        output_path = os.path.join(save_path, filename)
        plt.savefig(output_path)
    if visualize:
        plt.show()


import csv

def save_structured_array_to_csv(array, filename):
    # Ensure the array is not empty
    if array.size == 0:
        print("Array is empty. No data to save.")
        return

    # Get the field names from the dtype
    fieldnames = array.dtype.names
    try:
        # Open the file in write mode
        with open(filename, 'w', newline='') as csvfile:
            # Create a CSV writer object
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            # Write each row of data
            for row in array:
                writer.writerow({field: row[field] for field in fieldnames})

        print(f"Data successfully saved to {filename}")
    except PermissionError as e:
        print(f"PermissionError: {e}. Please check if the file is open or if you have the correct permissions.")
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_spontaneous_curvature(skeleton, channels, microns_per_pixel=None, visualize=False,save_path=None):

    return all_curvarues, normalized_curvatures, sem

def main(file_path, channel, save_path=None, visualize=False):
    # Load the OME-TIFF file
    image = load_ome_tiff(file_path)
    print("image shape:", image.shape)

    roi_coords = select_roi(image, channel)
    cropped_image = crop_channels(image, roi_coords)

    # Apply mask on selected channel
    mask, masked_image = apply_mask(cropped_image, channel=channel)
    print("Mask shape:", mask.shape)
    print("Masked channels shape:", masked_image.shape)

    # # upscale the image
    # upscale_factor = 5
    # masked_image = upscale_images(masked_image, scale_factor=upscale_factor)
    # print("upscasled image shape:", masked_image.shape)

    # convert mask into skeleton
    skeleton = skeletonize(mask)

    # calculate spontaneous curvature
    microns_per_pixel = 0.087
    # microns_per_pixel = microns_per_pixel / upscale_factor
    all_curvatures, normalized_curvatures, sem = calculate_spontaneous_curvature(skeleton, channels,
                                                                 microns_per_pixel=microns_per_pixel, visualize=visualize,
                                                                 save_path=save_path)
    dtype = [('frame_index', int),  ('normalized_curvature', float), ('sem', float), ('baselined_norm_curvature', float)]
    final_curvatures = np.array([], dtype=dtype)
    time_points = np.arange(image.shape[0])
    baselined_norm_curvature = normalized_curvatures / normalized_curvatures[0]
    # Check if the arrays have the same length
    if len(time_points) == len(normalized_curvatures) == len(sem):
        new_data = np.array(list(zip(time_points, normalized_curvatures, sem, baselined_norm_curvature)), dtype=dtype)
        final_curvatures = np.concatenate((final_curvatures, new_data))
    else:
        print("Error: All input arrays must have the same length.")

    # Initialize an empty structured array

    plot_spon_curv(final_curvatures['frame_index'],final_curvatures['normalized_curvature'], final_curvatures['sem'], save_path, visualize=visualize, title="Spontaneous curvature over time")
    plot_spon_curv(final_curvatures['frame_index'], final_curvatures['baselined_norm_curvature'], final_curvatures['sem'],
                   save_path, visualize=visualize, title="Baselined Spontaneous curvature over time")

    save_structured_array_to_csv(final_curvatures, os.path.join(save_path, "normalized_curvatures.csv"))
    save_structured_array_to_csv(all_curvatures, os.path.join(save_path, "all_curvatures.csv"))


if __name__ == "__main__":
    # files = [r"F:\UCSF\W1_FRAP\2023-08-08_ER-PKMO\RPE1-ER_3-pearling\RPE1-ER_3_MMStack_Default.ome.tif",
    #          r"F:\UCSF\W1_FRAP\2023-10-11_immortalized-cell-lines\CA-100ng_ml_4hr\hFB1m-CellLightMitoGFP-frap-CA-4hr-50mW_3\hFB1m-CellLightMitoGFP-frap-CA-4hr-50mW_MMStack_Pos0.ome.tif"
    #          ]
    # channels = [0, 0]
    # for i, file in enumerate(files):
    #     main(file, channel=channels[i])
    file_path = r"F:\UCSF\W1_FRAP\2023-08-08_ER-PKMO\RPE1-ER_4\RPE1-ER_4_MMStack_Default.ome.tif"
    # r"F:\UCSF\W1_FRAP\2023-08-08_ER-PKMO\RPE1-ER_3-pearling\RPE1-ER_3_MMStack_Default.ome.tif"
    # r"F:\UCSF\W1_FRAP\2023-10-11_immortalized-cell-lines\CA-100ng_ml_4hr\hFB1m-CellLightMitoGFP-frap-CA-4hr-50mW_3\hFB1m-CellLightMitoGFP-frap-CA-4hr-50mW_MMStack_Pos0.ome.tif"
    # r"F:\UCSF\W1_FRAP\2024-01-18_Laurdan_v2\ctrl\hFB13_PKMO_laurdan_ctrl_100lp_2-insane-pearling\hFB13_PKMO_laurdan_ctrl_2-ij.tif"
    # r"F:\UCSF\W1_FRAP\2023-08-15_tmre-sybrgold\TMRE\FCCP\RPE1_CellLight_Mito_GFP_FCCP_4uM_25min_1-perfect\RPE1_CellLight_Mito_GFP_FCCP_4uM_25min_1.tif"
    # r"F:\UCSF\W1_FRAP\2023-08-15_tmre-sybrgold\TMRE\FCCP\RPE1_CellLight_Mito_GFP_FCCP_4uM_25min_1-perfect\RPE1_CellLight_Mito_GFP_FCCP_4uM_25min_1_MMStack_Default.ome.tif"
    # r"F:\UCSF\W1_FRAP\2024-01-18_Laurdan_v2\ctrl\hFB13_PKMO_laurdan_ctrl_100lp_2-insane-pearling\hFB13_PKMO_laurdan_ctrl_2-ij.tif"
    # r"F:\UCSF\W1_FRAP\2024-01-18_Laurdan_v2\ctrl\hFB13_PKMO_laurdan_ctrl_100lp_1-pearling\hFB13_PKMO_laurdan_ctrl_1-ij.tif"
    save_path = os.path.dirname(file_path)
    main(file_path, channel=0, save_path=save_path, visualize=True)