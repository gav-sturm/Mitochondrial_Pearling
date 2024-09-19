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


# Load the OME-TIFF file
def load_ome_tiff(file_path):
    with tiff.TiffFile(file_path) as tif:
        images = tif.asarray()
    return images


def select_roi(channels, channel=0):
    """Allow the user to select an XY ROI on the max projection and return the coordinates."""
    if len(channels.shape) > 3:
        # max_projection = np.max(channels, axis=1)  # Max projection along the channel axis
        max_projection = channels[:,channel,:,:]
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

def calculate_spontaneous_curvature(images, channels=None, microns_per_pixel=0.116, visualize=False):
    """
    Calculate the spontaneous curvature and SEM for each time point and channel in the timeseries.
    Collect skeleton and spline points across all time points and visualize them with the original image.

    Parameters:
    images (numpy.ndarray): A numpy array with dimensions (T, C, X, Y), (T, X, Y), (C, X, Y), or (X, Y).
    channels (list or None): A list of channel indices to analyze. If None, analyze all channels.

    Returns:
    tuple: Two lists containing mean_curvature and SEM values for all analyzed time points and channels.
    """
    mean_curvatures = []
    sem_values = []
    contours = np.zeros_like(images)
    binary_images = np.zeros_like(images)
    curvature_image = np.zeros_like(images)
    sum_curvature_image = np.zeros_like(images)
    # Define the structured dtype
    dtype = [('frame_index', int), ('x_coords', float), ('y_coords', float), ('curvature', float)]
    # Initialize an empty structured array
    all_curvatures = np.array([], dtype=dtype)


    # Determine the dimensions and handle accordingly
    if len(images.shape) == 4:  # (T, C, X, Y)
        T, C, X, Y = images.shape
        print("4d shape found: ", images.shape)
        # If channels are not specified, analyze all channels
        if channels is None:
            channels = range(C)
        for channel in channels:
            # print("Calculating channel: ", channel)
            for t in tqdm(range(T)):
                # Extract the image for the current time point and channel
                image = images[t, channel, :, :]

                # Process the image
                # print("Calculating timepoint: ", t)
                mean_curvature, sem, binary_images[t,channel,:,:], contours[t,channel,:,:], curvature_image[t,channel,:,:], curvatures = process_image(image, microns_per_pixel, t=t)
                mean_curvatures.append(mean_curvature)
                sem_values.append(sem)
                all_curvatures = np.concatenate((all_curvatures, curvatures))
                # all_curvatures.append(curvatures)

    elif len(images.shape) == 3:  # (T, X, Y) or (C, X, Y)
        if images.shape[0] > 1:  # (T, X, Y) - multiple time points, no channels
            T, X, Y = images.shape
            for t in range(T):
                image = images[t, :, :]
                mean_curvature, sem, binary_images[t,:,:], contours[t,:,:], curvature_image[t,:,:], curvatures = process_image(image, microns_per_pixel, t=t)
                mean_curvatures.append(mean_curvature)
                sem_values.append(sem)
                all_curvatures = np.concatenate((all_curvatures, curvatures))
                # all_curvatures.append(curvatures)

        else:  # (C, X, Y) - multiple channels, single time point
            C, X, Y = images.shape

            # If channels are not specified, analyze all channels
            if channels is None:
                channels = range(C)

            for channel in channels:
                image = images[channel, :, :]
                mean_curvature, sem, binary_images[:,:], contours[channel,:,:], curvature_image[channel,:,:], curvatures = process_image(image, microns_per_pixel)
                mean_curvatures.append(mean_curvature)
                sem_values.append(sem)
                all_curvatures = np.concatenate((all_curvatures, curvatures))
                # all_curvatures.append(curvatures)
    elif len(images.shape) == 2:  # (X, Y) - single image, no channels, no time points
        mean_curvature, sem, binary_images, contours, curvature_image, curvatures = process_image(images, microns_per_pixel)
        mean_curvatures.append(mean_curvature)
        sem_values.append(sem)
        all_curvatures = np.concatenate((all_curvatures, curvatures))
        # all_curvatures.append(curvatures)

    else:
        raise ValueError("Input array must be 2D, 3D, or 4D.")

    if visualize:
        print("all curvatures shape: ", len(all_curvatures))
        print(all_curvatures[0])
        from matplotlib.animation import FuncAnimation
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))

        # Initial plot
        scatter = ax.scatter([], [], c=[], cmap='viridis', s=50, edgecolor='k', linewidths=0.1, alpha=0.8)
        ax.set_xlabel('X Coordinates')
        ax.set_ylabel('Y Coordinates')
        ax.set_title('Spline Points with Curvature Heatmap')
        ax.set_aspect('equal', adjustable='box')
        colorbar = plt.colorbar(scatter, ax=ax, label='Spontaneous Curvature')

        # Function to update the scatter plot
        def update(frame_index):
            ax.clear()
            # print("frame_index: ", frame_index)
            frame_data = all_curvatures[all_curvatures['frame_index'] == frame_index]
            x_coords = frame_data['x_coords']
            y_coords = frame_data['y_coords']
            curvatures = frame_data['curvature']
            curvature_normalized = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures))

            scatter = ax.scatter(x_coords, y_coords, c=curvature_normalized, cmap='viridis', s=50, edgecolor='k',
                                 linewidths=0.1, alpha=0.8)
            ax.set_xlabel('X Coordinates')
            ax.set_ylabel('Y Coordinates')
            # ax.set_aspect('equal', adjustable='box')
            # plt.colorbar(scatter, ax=ax, label='Spontaneous Curvature')
            scatter.set_offsets(np.c_[x_coords, y_coords])
            scatter.set_array(curvature_normalized)
            ax.set_title(f'Spline Points with Curvature Heatmap (Frame {frame_index})')
            colorbar.update_normal(scatter)

        # Create the animation
        T = images.shape[0]
        print("T: ", T)

        # Global frame index
        current_frame = [0]

        # Scroll event handler
        def on_scroll(event):
            if event.button == 'up':
                current_frame[0] = (current_frame[0] + 1) % T
            elif event.button == 'down':
                current_frame[0] = (current_frame[0] - 1) % T

            update(current_frame[0])
            fig.canvas.draw_idle()

        # Connect the scroll event to the handler
        fig.canvas.mpl_connect('scroll_event', on_scroll)

        anim = FuncAnimation(fig, update, frames=T, repeat=False)

        # Show the plot
        plt.show()

        # Visualization using Napari
        viewer = napari.Viewer()

        # Overlay the original image

        # Overlay combined skeletons
        viewer.add_image(binary_images, name='Binary_images')
        viewer.add_image(images, name=f'Original Image', colormap='gray')
        viewer.add_image(contours, name='Contours')

        # # Apply a Gaussian blur to smooth and enlarge the points
        # curvature_image = gaussian_filter(curvature_image, sigma=1)

        viewer.add_image(curvature_image, name='Curvatures', colormap='viridis')
        # viewer.add_image(spline_images, name='Spline images')
        viewer.dims.ndisplay = 3

        napari.run()

    return mean_curvatures, sem_values



def upscale_image(image, scale_factor):
    """
    Upscale the image by a given scale factor.

    Parameters:
    image (numpy.ndarray): The input image to upscale.
    scale_factor (float): The factor by which to scale the image.

    Returns:
    numpy.ndarray: The upscaled image.
    """
    return zoom(image, scale_factor, order=0)


def calculate_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    p1, p2 (tuple): The (x, y) coordinates of the two points.

    Returns:
    float: The Euclidean distance.
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def spontaneous_curvature(spline_points, window_size=10, t=0):
    """
    Calculate the spontaneous curvature for an entire spline segment using a local curvature approach.

    Parameters:
    spline_points (tuple): x and y coordinates of the spline points.
    window_size (int): The number of points to consider for local curvature calculation.

    Returns:
    float: The spontaneous curvature for the entire segment.
    """

    # Extract x and y coordinates from the spline points
    x_coords = np.array(spline_points[0])
    y_coords = np.array(spline_points[1])

    curvatures = []
    dtype = [('frame_index', int), ('x_coords', float), ('y_coords', float), ('curvature', float)]
    # Initialize an empty structured array
    all_curvatures = np.array([], dtype=dtype)

    for i in range(len(x_coords)):
        point = (y_coords[i], x_coords[i])  # Current point in (y, x) format to match the `compute_curvature` logic

        # Define local neighborhood
        start = max(0, i - window_size // 2)
        end = min(len(x_coords), i + window_size // 2 + 1)

        # Extract the neighborhood points
        neighborhood_x = x_coords[start:end]
        neighborhood_y = y_coords[start:end]

        # Compute the tangent direction and translate points to the central point
        tangent_direction = np.arctan2(np.gradient(neighborhood_y), np.gradient(neighborhood_x))
        tangent_direction.fill(tangent_direction[len(tangent_direction) // 2])

        translated_x = neighborhood_x - point[1]
        translated_y = neighborhood_y - point[0]

        # Rotate points to align with the tangent direction
        rotated_x = translated_x * np.cos(-tangent_direction) - translated_y * np.sin(-tangent_direction)
        rotated_y = translated_x * np.sin(-tangent_direction) + translated_y * np.cos(-tangent_direction)

        # Fit a polynomial and calculate curvature
        if len(rotated_x) > 2:  # Ensure there are enough points to fit a polynomial
            coeffs = np.polyfit(rotated_x, rotated_y, 2)

            # Calculate the first and second derivatives
            dy_dx = np.polyval(np.polyder(coeffs), rotated_x)
            d2y_dx2 = np.polyval(np.polyder(coeffs, 2), rotated_x)

            # Compute the curvature using the absolute value formula
            curvature = np.abs(d2y_dx2) / np.power(1 + np.power(dy_dx, 2), 1.5)
            # print("curvature: ", curvature)

            # curvature = np.polyval(np.polyder(coeffs, 2), rotated_x)
            mean = np.mean(curvature)
            # Create the data point to add to the structured array
            data = np.array([(t, x_coords[i], y_coords[i], mean)], dtype=dtype)

            # Concatenate with the existing array
            all_curvatures = np.concatenate((all_curvatures, data))

            curvatures.append(mean)

    # Calculate the standard error of the mean (SEM)
    sem = np.std(curvatures) / np.sqrt(len(curvatures))
    mean_curvature = np.mean(curvatures)

    # Integrate curvature over the spline (sum of curvature)
    total_curvature = np.sum(curvatures)

    # Calculate the total length of the spline
    dx_dt = np.gradient(x_coords)
    dy_dt = np.gradient(y_coords)
    length = np.sum(np.sqrt(dx_dt ** 2 + dy_dt ** 2))

    # Calculate the spontaneous curvature as the total curvature normalized by length
    normalized_curvature = total_curvature / length if length != 0 else 0

    return mean_curvature, sem, total_curvature, length, normalized_curvature, all_curvatures



def process_image(image, microns_per_pixel=0.116, sigma=5, max_curvature=100, t=0):
    """
    Process the image to calculate curvature using the edges (contours) of the objects and return the contour image.

    Parameters:
    image (numpy.ndarray): A 2D numpy array representing the image.
    microns_per_pixel (float): Scaling factor to convert pixels to micrometers.

    Returns:
    tuple: The mean spontaneous curvature, SEM, and the contour image.
    """

    # # Smooth the binary image using a Gaussian filter
    smoothed_image = gaussian_filter(image, sigma=sigma)


    # Threshold the image to ensure it's binary
    _, binary_image = cv2.threshold(smoothed_image, 127, 255, cv2.THRESH_BINARY)

    # Convert the binary image to 8-bit if it's not already
    if binary_image.dtype != np.uint8:
        binary_image = cv2.normalize(binary_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Find the contours of the objects
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create an empty image with the same shape as the input to store the contours with curvature values
    contour_image = np.zeros_like(image)  # Use float64 to store curvature values
    curvature_image = np.zeros_like(image, dtype=np.float64)  # Use float64 to store curvature values

    # Define the structured dtype
    dtype = [('frame_index', int), ('x_coords', float), ('y_coords', float), ('curvature', float)]
    # Initialize an empty structured array
    all_curvatures = np.array([], dtype=dtype)
    all_sems = []

    total_curvature = 0.0
    total_sem = 0.0
    total_length = 0.0

    # image parameters
    radius = 2  # Radius of the circle
    thickness = -1  # Filled circle

    # Iterate over each contour
    # print("number of contours: ", len(contours))
    for contour in contours:
        #print("contour lenght: ", len(contour))
        if len(contour) >= 4:  # Ensure there are enough points to calculate curvature

            # Draw the contour on the contour image
            cv2.drawContours(contour_image, [contour.astype(np.int32)], -1, 255, 1)

            contour = contour[:, 0, :] * microns_per_pixel  # Scale the contour points to micrometers
            contour_points = len(contour)
            # print("# of contour points: ", contour_points)
            # Convert the contour to a 2D array

            # tck, u = splprep([contour[:, 0], contour[:, 1]], s=0, per=False)
            # spline_points = splev(np.linspace(0, 1, contour_points), tck)
            # print("spline_points shape: ", len(spline_points[0]))

            x, y = contour[:, 0], contour[:, 1]
            # plot(x, y, '-o')
            # get the cumulative distance along the contour
            dist = np.sqrt((x[:-1] - x[1:]) ** 2 + (y[:-1] - y[1:]) ** 2)
            dist_along = np.concatenate(([0], dist.cumsum()))

            # build a spline representation of the contour
            spline, u = splprep([x, y], u=dist_along, s=0)

            # resample it at smaller distance intervals
            interp_d = np.linspace(dist_along[0], dist_along[-1], 500)
            interp_x, interp_y = splev(interp_d, spline)
            spline_points = np.array([interp_x, interp_y])
            # print("spline_points shape: ", spline_points.shape)

            mean_curvature, sem, total_curvature, length, normalized_curvature, curvatures = spontaneous_curvature(spline_points, window_size=50, t=t)
            curvature = normalized_curvature
            # curvature = np.clip(curvature, 0, max_curvature)
            # print("curvatures: ", curvatures)
            all_curvatures = np.concatenate((all_curvatures, curvatures))
            all_sems.append(sem)

            total_curvature += curvature * length
            total_sem += sem * length
            total_length += length

            # Draw the contour segment with curvature values on the curvature image
            for j, (x, y) in enumerate(zip(spline_points[0], spline_points[1])):
                x = int(round(x / microns_per_pixel))
                y = int(round(y / microns_per_pixel))
                if 0 <= x < curvature_image.shape[1] and 0 <= y < curvature_image.shape[0]:
                #     curvature_image[y, x] = curvature
                    # Draw a filled circle at the specified location with the curvature value
                    cv2.circle(curvature_image, (x, y), radius, curvature, thickness)


    if len(all_curvatures) == 0:
        raise ValueError("No valid contours found to calculate curvature.")

    # Combine all curvature values to calculate mean curvature and SEM
    # all_curvatures = np.array(all_curvatures)
    # all_sems= np.array(all_sems)
    # print("number of curvatures: ", len(all_curvatures))
    # Remove zero values before calculating mean curvature
    # non_zero_curvatures = all_curvatures[all_curvatures > 0]
    # print("number of non-zero curvatures: ", len(non_zero_curvatures))
    # print("all_curvatures: ", np.round(non_zero_curvatures,decimals=2))
    # total_spontaneous_curvature = np.sum(all_curvatures)

    # Normalize the total curvature by the total length of all contours
    normalized_spontaneous_curvature = total_curvature / total_length if total_length != 0 else 0

    normalized_sem = total_sem / total_length if total_length != 0 else 0
    # mean_curvature = np.mean(non_zero_curvatures)
    # print("mean_curvature: ", mean_curvature)
    # sem = np.std(non_zero_curvatures, ddof=1) / np.sqrt(len(all_curvatures))

    return normalized_spontaneous_curvature, normalized_sem, binary_image, contour_image, curvature_image, all_curvatures

def plot_spon_curv(spon_curv, error):
    """Plot the GP signal over time with mean and standard error."""
    time_points = np.arange(len(spon_curv))

    plt.figure(figsize=(10, 6))
    plt.errorbar(time_points, spon_curv, yerr=error, fmt='-o', capsize=5)
    plt.xlabel('Time Point')
    plt.ylabel('Spontenaous curvature (um^-1)')
    plt.title('Spontenous curvature over time')
    # plt.ylim(0,100)
    # plt.yscale('log')
    plt.grid(True)
    plt.show()


def main(file_path, channel):
    # Load the OME-TIFF file
    image = load_ome_tiff(file_path)
    print("image shape:", image.shape)

    roi_coords = select_roi(image, channel)
    cropped_image = crop_channels(image, roi_coords)

    # Apply mask on selected channel
    mask, masked_image = apply_mask(cropped_image, channel=channel)
    print("Mask shape:", mask.shape)
    print("Masked channels shape:", masked_image.shape)

    # upscale the image
    upscale_factor = 5
    masked_image = upscale_images(masked_image, scale_factor=upscale_factor)
    print("upscasled image shape:", masked_image.shape)

    # viewer = napari.Viewer()
    # viewer.add_image(cropped_image, name='Cropped Channels', channel_axis=1)
    # viewer.add_image(masked_channels, name='Cropped Channels', channel_axis=1)
    # viewer.add_labels(mask, name='Mask')

    # calculate spontaneous curvature
    microns_per_pixel = 0.087
    microns_per_pixel = microns_per_pixel / upscale_factor
    spontaneous_curvature, sem = calculate_spontaneous_curvature(masked_image, channels=[channel],
                                                                 microns_per_pixel=microns_per_pixel, visualize=True)
    # print("Spontaneous Curvature: ", spontaneous_curvature)

    # Print the results for each channel and time point
    # for channel in spontaneous_curvature:
    #     for t in spontaneous_curvature[channel]:
    #         mean_curvature, sem = spontaneous_curvature[channel][t]
    #         print(f"Channel {channel}, Time {t}:")
    #         print(f"  Mean Spontaneous Curvature: {mean_curvature}")
    #         print(f"  SEM: {sem}")
    # print("number of timepoints: ", len(spontaneous_curvature))
    # Visualization using napari
    # viewer = napari.Viewer()
    # viewer.add_image(cropped_image, name='Cropped Channels', channel_axis=1)
    # viewer.add_labels(mask, name='Mask')
    # viewer.add_image(masked_image, name='Masked Channels', channel_axis=1,)
    # viewer.add_image(mean_curvature, name='spontaneous_curvature', colormap='inferno')
    # napari.run()

    plot_spon_curv(spontaneous_curvature, sem)


if __name__ == "__main__":
    file_path = r"F:\UCSF\W1_FRAP\2023-08-08_ER-PKMO\RPE1-ER_3-pearling\RPE1-ER_3_MMStack_Default.ome.tif"
    # r"F:\UCSF\W1_FRAP\2023-08-08_ER-PKMO\RPE1-ER_3-pearling\RPE1-ER_3_MMStack_Default.ome.tif"
    # r"F:\UCSF\W1_FRAP\2023-10-11_immortalized-cell-lines\CA-100ng_ml_4hr\hFB1m-CellLightMitoGFP-frap-CA-4hr-50mW_3\hFB1m-CellLightMitoGFP-frap-CA-4hr-50mW_MMStack_Pos0.ome.tif"
    # r"F:\UCSF\W1_FRAP\2024-01-18_Laurdan_v2\ctrl\hFB13_PKMO_laurdan_ctrl_100lp_2-insane-pearling\hFB13_PKMO_laurdan_ctrl_2-ij.tif"
    # r"F:\UCSF\W1_FRAP\2023-08-15_tmre-sybrgold\TMRE\FCCP\RPE1_CellLight_Mito_GFP_FCCP_4uM_25min_1-perfect\RPE1_CellLight_Mito_GFP_FCCP_4uM_25min_1.tif"
    # r"F:\UCSF\W1_FRAP\2023-08-15_tmre-sybrgold\TMRE\FCCP\RPE1_CellLight_Mito_GFP_FCCP_4uM_25min_1-perfect\RPE1_CellLight_Mito_GFP_FCCP_4uM_25min_1_MMStack_Default.ome.tif"
    # r"F:\UCSF\W1_FRAP\2024-01-18_Laurdan_v2\ctrl\hFB13_PKMO_laurdan_ctrl_100lp_2-insane-pearling\hFB13_PKMO_laurdan_ctrl_2-ij.tif"
    # r"F:\UCSF\W1_FRAP\2024-01-18_Laurdan_v2\ctrl\hFB13_PKMO_laurdan_ctrl_100lp_1-pearling\hFB13_PKMO_laurdan_ctrl_1-ij.tif"
    main(file_path, channel=0)
