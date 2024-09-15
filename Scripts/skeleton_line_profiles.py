from dataclasses import dataclass

import numpy as np
from collections import deque
from nellie.im_info import verifier
import napari
from scipy.interpolate import UnivariateSpline, splprep, splev
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import tifffile as tif

# Suppress DEBUG messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('napari').setLevel(logging.WARNING)


# logging.basicConfig(level=logging.WARNING)


class SkeletonCurvature:
    def __init__(self, im_path, nellie_path, save_path=None, smoothing_factors=None, manual_smoothing=False, timepoints=None,
                 show_individual_plots=False, show_summary_plots=True, display_frames=None):

        self.im_path = im_path
        self.nellie_path = nellie_path
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.show_individual_plots = show_individual_plots
        self.show_summary_plots = show_summary_plots
        self.file_info = None
        self.im_info = None
        self.timepoints = timepoints
        self.manual_smoothing = manual_smoothing
        self.smoothing_factor_intensity = smoothing_factors[0] if smoothing_factors else None
        self.smoothing_factor_structural = smoothing_factors[1] if smoothing_factors else None
        self.display_frames = display_frames

        dtype = [('file', str), ('crop_count', int), ('timepoint', int), ('cumulative_distances', int),
                 ('intensity_curvature', int), ('structural_curvature', int)]
        # Initialize an empty structured array
        self.results = np.array([], dtype=dtype)

    def get_neighbors(self, point, label_coords):
        if self.im_info.no_z:
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (point[0], point[1] + dy, point[2] + dx)
                    if neighbor in label_coords:
                        neighbors.append(neighbor)
        else:  # 3d image
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = (point[0], point[1] + dz, point[2] + dy, point[3] + dx)
                        if neighbor in label_coords:
                            neighbors.append(neighbor)
        return neighbors

    def find_tips(self, label_coords):
        tips = []
        for coord in label_coords:
            if len(self.get_neighbors(coord, label_coords)) == 1:
                tips.append(coord)
        return tips

    def traverse_skeleton(self, label_coords):
        tips = self.find_tips(label_coords)
        if len(tips) != 2:
            return []

        start_tip = tips[0]
        end_tip = tips[1]

        visited = set()
        path = []
        stack = deque([(start_tip, [start_tip])])

        while stack:
            current, current_path = stack.pop()
            if current == end_tip:
                return current_path

            if current not in visited:
                visited.add(current)
                path = current_path

                for neighbor in self.get_neighbors(current, label_coords):
                    if neighbor not in visited:
                        stack.append((neighbor, current_path + [neighbor]))

        return path  # If end_tip is not reached, return the longest path found

    def calculate_local_curvature(self, cumulative_distances, values, spline_resolution=25, smoothing_factor=0.0):
        """
        Calculate the local curvature of a fitted spline.

        Parameters:
        - cumulative_distances (array-like): The x-values (distances) along the path.
        - values (array-like): The y-values (intensity or structural values).
        - spline_resolution (int): Number of points to evaluate the spline on. Default is 25.
        - smoothing_factor (float): Smoothing factor for the spline. Default is 0 (no smoothing).

        Returns:
        - curvature (numpy.ndarray): The local curvature along the spline.
        - spline_x (numpy.ndarray): The x-values (distances) used for the spline.
        - spline_y (numpy.ndarray): The y-values (fitted spline values).
        - mean_curvature_per_unit_distance (float): The mean curvature per unit distance.
        """

        # Fit a spline to the data using splprep
        tck, u = splprep([cumulative_distances, values], s=smoothing_factor, per=False)

        # Evaluate the spline at evenly spaced points
        u_new = np.linspace(0, 1, spline_resolution)
        spline_x, spline_y = splev(u_new, tck)

        # Calculate the first and second derivatives of the spline
        spline_y_prime, spline_x_prime = splev(u_new, tck, der=1)  # First derivative
        spline_y_double_prime, spline_x_double_prime = splev(u_new, tck, der=2)  # Second derivative

        # Calculate curvature using the curvature formula for parametric curves
        curvature = np.abs(spline_y_double_prime * spline_x_prime - spline_y_prime * spline_x_double_prime) / \
                    (spline_x_prime ** 2 + spline_y_prime ** 2) ** (3 / 2)

        # Calculate the total length of the path
        total_distance = np.max(cumulative_distances) - np.min(cumulative_distances)

        # Calculate the mean curvature per unit distance
        mean_curvature_per_unit_distance = np.trapz(curvature, spline_x) / total_distance

        return curvature, spline_x, spline_y, mean_curvature_per_unit_distance



    def find_optimal_smoothing_factor_across_timepoints(self, skeleton_im, intensity_or_structural_im,
                                                        data_type='intensity'):
        """
        Finds the optimal smoothing factor by maximizing the difference in weighted mean curvature between the peak and average timepoints,
        ensuring that the peak timepoint has a higher curvature.

        Args:
            skeleton_im (ndarray): The skeleton image array.
            intensity_or_structural_im (ndarray): The intensity or structural image array.
            data_type (str): A label indicating whether the data is 'intensity' or 'structural'.

        Returns:
            float: The optimal smoothing factor.
        """
        if self.display_frames is None:
            peak_timepoint = self.timepoints[-1]  # Assuming the last timepoint is the peak timepoint
            avg_timepoint = self.timepoints[0]  # Assuming the first timepoint as the average
        else:
            peak_timepoint = int(self.display_frames[1])  # Assuming the last timepoint is the peak timepoint
            avg_timepoint = int(self.display_frames[0])  # Assuming the middle timepoint as the average

        print(f"Peak timepoint: {peak_timepoint}")
        print(f"Avg timepoint: {avg_timepoint}")

        # Prepare skeleton and data images for peak and average timepoints
        frame_skeleton_im_peak = skeleton_im[peak_timepoint:peak_timepoint + 1]
        frame_data_im_peak = intensity_or_structural_im[peak_timepoint:peak_timepoint + 1]
        frame_skeleton_im_avg = skeleton_im[avg_timepoint:avg_timepoint + 1]
        frame_data_im_avg = intensity_or_structural_im[avg_timepoint:avg_timepoint + 1]

        # # Normalize intensity values by their median to ensure comparability
        # median_peak_intensity = np.median(frame_data_im_peak)
        # median_avg_intensity = np.median(frame_data_im_avg)
        #
        # frame_data_im_peak = frame_data_im_peak / median_peak_intensity if median_peak_intensity != 0 else frame_data_im_peak
        # frame_data_im_avg = frame_data_im_avg / median_avg_intensity if median_avg_intensity != 0 else frame_data_im_avg

        # Define a range of smoothing factors
        smoothing_factors = np.linspace(1, 0, 50)  # Start from 1 and work down

        best_smoothing_factor = 1.0
        best_score = -np.inf

        # Initialize variables to store the best traversals
        best_traversals_peak = None
        best_traversals_avg = None
        longest_traversal_peak = None
        longest_traversal_avg = None

        # Iterate over smoothing factors
        for sf in smoothing_factors:
            label_traversals_peak = []
            label_traversals_avg = []

            # Process each label for the peak timepoint
            unique_labels = np.unique(frame_skeleton_im_peak)
            for label in unique_labels:
                if label == 0:
                    continue

                label_coords = np.argwhere(frame_skeleton_im_peak == label)
                if len(label_coords) < 5:
                    continue

                label_coords_set = set(map(tuple, label_coords))
                skeleton_path = self.traverse_skeleton(label_coords_set)
                if not skeleton_path:
                    continue

                data_vals_peak = [frame_data_im_peak[point] for point in skeleton_path]

                # get scaling factor for cumulative distances
                if self.im_info.no_z:
                    scaling = (peak_timepoint, self.im_info.dim_res['Y'], self.im_info.dim_res['X'])
                else:
                    # print("3D image")
                    scaling = (peak_timepoint, self.im_info.dim_res['Z'], self.im_info.dim_res['Y'], self.im_info.dim_res['X'])


                cumulative_distances_peak = self.getCumulativeDistances(skeleton_path, scaling=scaling)
                if cumulative_distances_peak is None:
                    continue

                curvature_peak, spline_x_peak, spline_y_peak, mean_curvature_peak = self.calculate_local_curvature(
                    cumulative_distances_peak, data_vals_peak, spline_resolution=len(label_coords), smoothing_factor=sf)

                label_traversals_peak.append(
                    self.LabelTraversal(peak_timepoint, label, label_coords, label_coords_set, skeleton_path,
                                        data_vals_peak, cumulative_distances_peak, spline_x_peak, spline_y_peak,
                                        curvature_peak, mean_curvature_peak)
                )

            # Process each label for the average timepoint
            unique_labels = np.unique(frame_skeleton_im_avg)
            for label in unique_labels:
                if label == 0:
                    continue

                label_coords = np.argwhere(frame_skeleton_im_avg == label)
                if len(label_coords) < 10:
                    continue

                label_coords_set = set(map(tuple, label_coords))
                skeleton_path = self.traverse_skeleton(label_coords_set)
                if not skeleton_path:
                    continue

                data_vals_avg = [frame_data_im_avg[point] for point in skeleton_path]

                # get scaling factor for cumulative distances
                # get scaling factor for cumulative distances
                if self.im_info.no_z:
                    scaling = (avg_timepoint, self.im_info.dim_res['Y'], self.im_info.dim_res['X'])
                else:
                    # print("3D image")
                    scaling = (avg_timepoint, self.im_info.dim_res['Z'], self.im_info.dim_res['Y'], self.im_info.dim_res['X'])

                cumulative_distances_avg = self.getCumulativeDistances(skeleton_path, scaling=scaling)
                if cumulative_distances_avg is None:
                    continue

                curvature_avg, spline_x_avg, spline_y_avg, mean_curvature_avg = self.calculate_local_curvature(
                    cumulative_distances_avg, data_vals_avg, spline_resolution=len(label_coords), smoothing_factor=sf)

                label_traversals_avg.append(
                    self.LabelTraversal(avg_timepoint, label, label_coords, label_coords_set, skeleton_path,
                                        data_vals_avg, cumulative_distances_avg, spline_x_avg, spline_y_avg,
                                        curvature_avg, mean_curvature_avg)
                )

            # Initialize with the first traversal if no best is set yet
            if best_traversals_peak is None and label_traversals_peak:
                best_traversals_peak = label_traversals_peak
                longest_traversal_peak = max(label_traversals_peak, key=lambda t: len(t.cumulative_distances),
                                             default=None)

            if best_traversals_avg is None and label_traversals_avg:
                best_traversals_avg = label_traversals_avg
                longest_traversal_avg = max(label_traversals_avg, key=lambda t: len(t.cumulative_distances),
                                            default=None)

            # Calculate weighted mean curvatures for both timepoints
            weighted_mean_curvature_peak = self.compute_weighted_mean_curvatures(label_traversals_peak)[peak_timepoint]
            # print(f"Weighted mean curvature at peak timepoint: {weighted_mean_curvature_peak}")
            weighted_mean_curvature_avg = self.compute_weighted_mean_curvatures(label_traversals_avg)[avg_timepoint]
            # print(f"Weighted mean curvature at average timepoint: {weighted_mean_curvature_avg}")

            # Calculate the score as the difference between weighted mean curvatures, ensuring peak > avg
            if weighted_mean_curvature_peak > weighted_mean_curvature_avg:
                score = weighted_mean_curvature_peak - weighted_mean_curvature_avg

                # If this score is better, update the best score and smoothing factor
                if score > best_score:
                    best_score = score
                    best_smoothing_factor = sf
                    best_traversals_peak = label_traversals_peak
                    best_traversals_avg = label_traversals_avg

                    # Find the longest traversal for plotting
                    longest_traversal_peak = max(label_traversals_peak, key=lambda t: len(t.cumulative_distances))
                    longest_traversal_avg = max(label_traversals_avg, key=lambda t: len(t.cumulative_distances))




        if best_smoothing_factor is None:
            raise ValueError(
                f"No suitable labels found in the peak or average timepoint to determine the smoothing factor for {data_type} data.")


        # Plot and interactively adjust the smoothing factor
        if self.manual_smoothing:
            # Fallback to use the first traversal if the longest traversal is None
            if longest_traversal_peak is None:
                longest_traversal_peak = best_traversals_peak[0]
            if longest_traversal_avg is None:
                longest_traversal_avg = best_traversals_avg[0]

            # Ensure we have valid traversals before plotting
            if not longest_traversal_peak or not longest_traversal_avg:
                # print(f"longest_traversal_peak: {longest_traversal_peak}")
                # print(f"longest_traversal_avg: {longest_traversal_avg}")
                # print(f"best_traversals_peak: {best_traversals_peak}")
                # print(f"best_traversals_avg: {best_traversals_avg}")
                # print(f"best_smoothing_factor: {best_smoothing_factor}")
                raise ValueError("Could not find valid traversal paths for plotting.")

            title = f"Adjust Smoothing Factor for {data_type} Data"
            print(f'longest traversal peak: {len(longest_traversal_peak.cumulative_distances)}')
            print(f'longest traversal avg: {len(longest_traversal_avg.cumulative_distances)}')
            # best_smoothing_factor *= 2
            best_smoothing_factor = self.interactive_smoothing_plot(longest_traversal_peak, longest_traversal_avg, best_smoothing_factor, title)

        return best_smoothing_factor

    def interactive_smoothing_plot(self, traversals_peak, traversals_avg, best_smoothing_factor, title):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        """
        Creates an interactive plot with a slider to adjust the smoothing factor and see the effect on the fitted splines.

        Args:
        - traversals_peak (list): List of label traversals for the peak timepoint.
        - traversals_avg (list): List of label traversals for the average timepoint.
        - best_smoothing_factor (float): The initial smoothing factor to start the plot with.

        Returns:
        - None: Displays the interactive plot.
        """
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.25)

        # Plot the actual data points
        peak_scatter = ax.scatter(traversals_peak.cumulative_distances, traversals_peak.intensity_values, color='blue',
                                  alpha=0.6, label='Peak Data')
        avg_scatter = ax.scatter(traversals_avg.cumulative_distances, traversals_avg.intensity_values, color='red',
                                 alpha=0.6, label='Avg Data')

        # Plot the initial splines with the best smoothing factor
        peak_line, = ax.plot(traversals_peak.cumulative_distances, traversals_peak.spline_y, color='blue',
                             linestyle='-', label='Peak Spline')
        avg_line, = ax.plot(traversals_avg.cumulative_distances, traversals_avg.spline_y, color='red',
                            linestyle='-', label='Avg Spline')

        ax.set_title(title)
        ax.set_xlabel('Cumulative Distance')
        ax.set_ylabel('Intensity')
        ax.legend()

        # Slider for smoothing factor
        ax_smooth = plt.axes([0.1, 0.1, 0.8, 0.03])
        smooth_slider = Slider(ax_smooth, 'Smoothing Factor', 0, 1, valinit=best_smoothing_factor, valstep=0.01)

        # Initialize the final smoothing factor
        final_smoothing_factor = [best_smoothing_factor]

        def update(val):
            sf = smooth_slider.val
            # Update the fitted splines for the peak and average timepoints with the new smoothing factor
            new_curvature_peak, _, spline_y_peak, _ = self.calculate_local_curvature(
                traversals_peak.cumulative_distances, traversals_peak.intensity_values,
                spline_resolution=len(traversals_peak.cumulative_distances), smoothing_factor=sf)

            new_curvature_avg, _, spline_y_avg, _ = self.calculate_local_curvature(
                traversals_avg.cumulative_distances, traversals_avg.intensity_values,
                spline_resolution=len(traversals_avg.cumulative_distances), smoothing_factor=sf)

            # Update plot data
            peak_line.set_ydata(spline_y_peak)
            avg_line.set_ydata(spline_y_avg)
            fig.canvas.draw_idle()

        # Register update function with the slider
        smooth_slider.on_changed(update)

        # Define an event handler to capture the smoothing factor on figure close or 'Enter' key press
        def on_close(event):
            nonlocal final_smoothing_factor
            plt.close(fig)  # Close the figure
            final_smoothing_factor[0] = smooth_slider.val  # Capture the final value from the slider

        def on_key(event):
            if event.key == 'enter':
                on_close(event)

        fig.canvas.mpl_connect('close_event', on_close)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

        return final_smoothing_factor[0]  # Return the final smoothing factor

    def getCumulativeDistances(self, skeleton_path, scaling):
        if self.im_info.no_z:
            scaled_path = tuple(
                [point[0] * scaling[0], point[1] * scaling[1], point[2] * scaling[2]] for point in skeleton_path)
        else:
            scaled_path = tuple(
                [point[0] * scaling[0], point[1] * scaling[1], point[2] * scaling[2], point[3] * scaling[3]] for point
                in skeleton_path)
        if not scaled_path:
            return None
        distances = np.linalg.norm(np.diff(scaled_path, axis=0), axis=1)
        # prepend a 0 to the distances list
        distances = np.insert(distances, 0, 0)
        cumulative_distances = np.cumsum(distances)

        return cumulative_distances

    def compute_weighted_mean_curvatures(self, label_traversals):
        # Dictionary to hold cumulative curvature and cumulative distances per timepoint
        curvature_data = defaultdict(lambda: {'total_curvature': 0.0, 'total_distance': 0.0})

        for traversal in label_traversals:
            timepoint = traversal.timepoint
            # Accumulate the weighted curvature and total distance
            curvature_data[timepoint]['total_curvature'] += traversal.mean_curvature * np.sum(
                traversal.cumulative_distances)
            curvature_data[timepoint]['total_distance'] += np.sum(traversal.cumulative_distances)

        # Compute the weighted mean curvature per timepoint
        weighted_mean_curvatures = {}
        for timepoint, data in curvature_data.items():
            if data['total_distance'] > 0:
                weighted_mean_curvatures[timepoint] = data['total_curvature'] / data['total_distance']
            else:
                weighted_mean_curvatures[timepoint] = 0

        return weighted_mean_curvatures

    def plot_mean_curvature_over_time(self, weighted_mean_curvatures, title='Weighted Mean Curvature Over Time',
                                      save_folder=None, view_plot=False):
        # Sort the timepoints for consistent plotting
        timepoints = sorted(weighted_mean_curvatures.keys())
        mean_curvatures = [weighted_mean_curvatures[t] for t in timepoints]

        # Plotting the mean curvature over time
        plt.figure(figsize=(10, 6))
        plt.plot(timepoints, mean_curvatures, marker='o', linestyle='-', color='b', label='Mean Curvature')
        plt.xlabel('Timepoints')
        plt.ylabel('Weighted Mean Curvature')
        plt.title(title)
        plt.grid(True)
        plt.legend()

        if save_folder:
            # check if the save folder exists
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, (title + '.png'))
            plt.savefig(save_path)

        if view_plot:
            plt.show()

    def napari_kymograph_view(self, images, IMnames=None, cmap=None, contrast_limit=0.1, layer3d=False):
        import napari
        import numpy as np

        # Get the axes string, e.g., 'TCZYX', 'TCYX', 'TZYX', etc.
        axes = self.file_info.axes

        viewer = napari.Viewer()

        for i, image in enumerate(images):
            # Determine the indices of each axis in the image
            axis_dict = {axis: axes.index(axis) if axis in axes else None for axis in 'TCZYX'}

            # Dynamically adjust image shape and dimensions
            if 'T' in axes and 'Z' in axes and 'C' in axes:  # T, C, Z, Y, X
                T, C, Z, Y, X = [image.shape[axis_dict[ax]] for ax in 'TCZYX']
                image_transposed = np.transpose(image, (
                axis_dict['C'], axis_dict['T'], axis_dict['Z'], axis_dict['Y'], axis_dict['X']))
            elif 'T' in axes and 'C' in axes and 'Z' not in axes:  # T, C, Y, X
                T, C, Y, X = [image.shape[axis_dict[ax]] for ax in 'TCYX']
                image_transposed = np.transpose(image, (axis_dict['C'], axis_dict['T'], axis_dict['Y'], axis_dict['X']))
            elif 'T' in axes and 'Z' in axes and 'C' not in axes:  # T, Z, Y, X (C=1 assumed)
                T, Z, Y, X = [image.shape[axis_dict[ax]] for ax in 'TZYX']
                image_expanded = np.expand_dims(image, axis=1)  # Add a channel dimension
                image_transposed = np.transpose(image_expanded, (1, 0, 2, 3, 4))  # Transpose to C, T, Z, Y, X
            elif 'T' in axes and 'C' not in axes and 'Z' not in axes:  # T, Y, X
                T, Y, X = [image.shape[axis_dict[ax]] for ax in 'TYX']
                image_expanded = np.expand_dims(image, axis=(0, 1))  # Add channel and Z dimensions
                image_transposed = np.transpose(image_expanded, (0, 1, 2, 3, 4))  # Transpose to C, T, Z, Y, X
            else:
                raise ValueError(f"Unsupported image shape with axes: {axes}")

            # Amplify the Z dimension spacing for visualizing in 3D
            z_spacing = 1  # Number of empty slices between timepoints
            if image_transposed.shape[2] > 1:  # Check if there's a Z dimension to amplify
                new_z_size = image_transposed.shape[2] + (image_transposed.shape[2] - 1) * z_spacing
                spaced_image = np.zeros((image_transposed.shape[0], image_transposed.shape[1], new_z_size,
                                         image_transposed.shape[3], image_transposed.shape[4]))

                # Fill the spaced_image with original data, leaving gaps for the spacing
                for j in range(image_transposed.shape[2]):
                    spaced_image[:, :, j * (z_spacing + 1), :, :] = image_transposed[:, :, j, :, :]
            else:
                spaced_image = image_transposed  # If no Z dimension, use the transposed image directly

            print("Final shape with amplified Z spacing:", spaced_image.shape)

            # Display the image in Napari
            minI = np.min(spaced_image)
            maxI = np.max(spaced_image)
            title = IMnames[i] if IMnames else f'Image {i}'
            color = cmap[i] if cmap else 'gray'
            print(f"Image {i}: {title}")
            viewer.add_image(spaced_image, name=title,
                             contrast_limits=[minI + (contrast_limit * maxI), maxI - (contrast_limit * maxI)],
                             colormap=color, interpolation2d='nearest', interpolation3d='nearest')
            if layer3d:  # Only set to 3D mode if the image has a Z dimension
                viewer.dims.ndisplay = 3

        napari.run()

    def napari_view(self, images, IMnames=None, cmap=None, visible=None,contrast_limit=0.1, layer3d=False, save_folder=None,
                    view_napari=False):
        viewer = napari.Viewer()
        added_layers = []

        axes = self.file_info.axes

        for i, image in enumerate(images):
            # print(f"Image {i}: {image.shape}")
            # if axes == 'TCZYX':
            #     # collapse the z dimension
            #     image = np.max(image, axis=2)
            #     axes = 'TCYX'
            # elif axes == 'TZYX':
            #     # collapse the z dimension
            #     image = np.max(image, axis=1)
            #     axes = 'TYX'

            # # Determine the indices of each axis in the image
            # axis_dict = {axis: axes.index(axis) if axis in axes else None for axis in 'TCZYX'}
            #
            # # Dynamically adjust image shape and dimensions
            # if 'T' in axes and 'Z' in axes and 'C' in axes:  # T, C, Z, Y, X
            #     T, C, Z, Y, X = [image.shape[axis_dict[ax]] for ax in 'TCZYX']
            #     image_transposed = np.transpose(image, (
            #         axis_dict['C'], axis_dict['T'], axis_dict['Z'], axis_dict['Y'], axis_dict['X']))
            # elif 'T' in axes and 'C' in axes and 'Z' not in axes:  # T, C, Y, X
            #     T, C, Y, X = [image.shape[axis_dict[ax]] for ax in 'TCYX']
            #     image_transposed = np.transpose(image, (axis_dict['C'], axis_dict['T'], axis_dict['Y'], axis_dict['X']))
            # elif 'T' in axes and 'Z' in axes and 'C' not in axes:  # T, Z, Y, X (C=1 assumed)
            #     T, Z, Y, X = [image.shape[axis_dict[ax]] for ax in 'TZYX']
            #     image_expanded = np.expand_dims(image, axis=1)  # Add a channel dimension
            #     image_transposed = np.transpose(image_expanded, (1, 0, 2, 3, 4))  # Transpose to C, T, Z, Y, X
            # elif 'T' in axes and 'C' not in axes and 'Z' not in axes:  # T, Y, X
            #     T, Y, X = [image.shape[axis_dict[ax]] for ax in 'TYX']
            #     image_expanded = np.expand_dims(image, axis=(0, 1))  # Add channel and Z dimensions
            #     image_transposed = np.transpose(image_expanded, (0, 1, 2, 3, 4))  # Transpose to C, T, Z, Y, X
            # else:
            #     raise ValueError(f"Unsupported image shape with axes: {axes}")
            #
            # # Amplify the Z dimension spacing for visualizing in 3D
            # z_spacing = 1  # Number of empty slices between timepoints
            # if image_transposed.shape[2] > 1:  # Check if there's a Z dimension to amplify
            #     new_z_size = image_transposed.shape[2] + (image_transposed.shape[2] - 1) * z_spacing
            #     spaced_image = np.zeros((image_transposed.shape[0], image_transposed.shape[1], new_z_size,
            #                              image_transposed.shape[3], image_transposed.shape[4]))
            #
            #     # Fill the spaced_image with original data, leaving gaps for the spacing
            #     for j in range(image_transposed.shape[2]):
            #         spaced_image[:, :, j * (z_spacing + 1), :, :] = image_transposed[:, :, j, :, :]
            # else:
            #     spaced_image = image_transposed  # If no Z dimension, use the transposed image directly
            #
            # print("Final shape with amplified Z spacing:", spaced_image.shape)
            # image = spaced_image
            minI = np.min(image)
            maxI = np.max(image)
            title = IMnames[i] if IMnames else f'Image {i}'
            color = cmap[i] if cmap else 'gray'
            layer = viewer.add_image(image, name=title,
                                     contrast_limits=[minI + (contrast_limit * maxI), maxI - (contrast_limit * maxI)],
                                     colormap=color, interpolation2d='nearest', interpolation3d='nearest'
                                     )
            added_layers.append(layer)
            layer.visible = visible[i] if visible else True


            if layer3d and image.ndim > 2:  # Only set to 3D mode if the image has a Z dimension
                viewer.dims.ndisplay = 3

        if save_folder:
            # Ensure the save folder exists
            os.makedirs(save_folder, exist_ok=True)

            # Save individual layers as OME-TIFFs
            for layer in added_layers:
                save_path = os.path.join(save_folder, f"{layer.name}.ome.tiff")
                tif.imwrite(save_path, layer.data, photometric='minisblack')
                print(f"Saved {layer.name} as {save_path}")

            # # Save the entire viewer as a screenshot (optional)
            # screenshot_path = os.path.join(save_folder, "viewer_screenshot.png")
            # viewer.screenshot(screenshot_path)
            # print(f"Saved viewer screenshot as {screenshot_path}")

            # # Save combined layers as a multi-layer OME-TIFF
            # combined_data = np.stack([layer.data for layer in added_layers], axis=0)
            # combined_save_path = os.path.join(save_folder, "combined_curvature_layers.ome.tiff")
            # tif.imwrite(combined_save_path, combined_data, photometric='minisblack')
            # print(f"Saved combined layers as {combined_save_path}")
        if view_napari:
            napari.run()

    def log_normalize(self, image):
        # Add a small value to avoid log(0) if needed
        image_log = np.log1p(image - image.min())
        image_log_normalized = (image_log - image_log.min()) / (image_log.max() - image_log.min())
        return image_log_normalized

    @dataclass
    class LabelTraversal:
        timepoint: int
        label_num: int
        label_coords: np.ndarray
        label_coords_set: set
        skeleton_path: np.ndarray
        intensity_values: list
        cumulative_distances: list
        spline_x: np.ndarray
        spline_y: np.ndarray
        curvature: np.ndarray
        mean_curvature: float

    def getMeanCurvature(self):
        return self.results

    def normalize_weighted_mean_curvatures(self, weighted_mean_curvatures):
        """
        Normalize the weighted mean curvatures to the first timepoint.

        Parameters:
        - weighted_mean_curvatures (dict): A dictionary where keys are timepoints and values are the weighted mean curvatures.

        Returns:
        - normalized_curvatures (dict): A dictionary with the same keys as input, where the first timepoint is normalized to 1.0,
                                        and subsequent timepoints are normalized to the first timepoint.
        """
        # Sort the timepoints to ensure correct order
        sorted_timepoints = sorted(weighted_mean_curvatures.keys())

        # Get the first timepoint's curvature value
        first_timepoint = sorted_timepoints[0]
        first_value = weighted_mean_curvatures[first_timepoint]

        # Normalize the curvatures
        normalized_curvatures = {}
        for timepoint in sorted_timepoints:
            normalized_curvatures[timepoint] = weighted_mean_curvatures[timepoint] / first_value

        return normalized_curvatures

    def run_curvature(self):
        self.file_info = verifier.FileInfo(filepath=self.im_path, output_dir=self.nellie_path)
        self.file_info.find_metadata()
        self.file_info.load_metadata()
        print(f'{self.file_info.output_dir}')
        print(f'{self.file_info.shape}')
        # file_info.select_temporal_range(0, 2)
        if self.timepoints is None:
            # print(f'No timepoints specified, using all timepoints.')
            # last_timepoint = self.file_info.shape[self.file_info.axes.index('T')] - 1
            # print(f'last timepoint: {last_timepoint}')
            # print(f'first timepoint: {self.file_info.t_start}')
            # print(f'first timepoint: {self.file_info.t_start}')
            # print(f'last timepoint: {self.file_info.t_end}')
            self.timepoints = list(range(0, self.file_info.t_end + 1))
        # print(f"Timepoints: {self.timepoints}")

        self.im_info = verifier.ImInfo(self.file_info)

        # Load the skeleton, intensity, and frangi processed structural images
        intensity_im = self.im_info.get_memmap(self.im_info.im_path)
        structural_im = self.im_info.get_memmap(self.im_info.pipeline_paths['im_preprocessed'])
        skeleton_im = self.im_info.get_memmap(self.im_info.pipeline_paths['im_skel'])

        path_im = np.zeros_like(skeleton_im)
        intensity_curvature_im = np.zeros_like(skeleton_im)
        structural_curvature_im = np.zeros_like(skeleton_im)
        label_traversals_intensity = []
        label_traversals_structural = []

        # normalize the intensity and structural images
        intensity_im = (intensity_im - intensity_im.min()) / (intensity_im.max() - intensity_im.min())
        structural_im = (structural_im - structural_im.min()) / (structural_im.max() - structural_im.min())

        # Find the optimal smoothing factor for intensity values
        if self.smoothing_factor_intensity is None:
            print(f"Finding optimal intensity smoothing factor...")
            self.smoothing_factor_intensity = self.find_optimal_smoothing_factor_across_timepoints(skeleton_im,
                                                                                                   intensity_im,
                                                                                                   data_type='intensity')
            # self.smoothing_factor_intensity *= 2
            # if self.smoothing_factor_intensity > 1.0:
            #     self.smoothing_factor_intensity = 1.0
            # if self.smoothing_factor_intensity == 0:
            #     self.smoothing_factor_intensity = 0.5
            #     print(f"intensity smoothing factor is 0, setting to 0.5")
            print(f"Found optimal intensity smoothing factor: {self.smoothing_factor_intensity}")
        else:
            print(f"Using user-specified intensity smoothing factor: {self.smoothing_factor_intensity}")

        # Find the optimal smoothing factor for structural values
        if self.smoothing_factor_structural is None:
            print(f"Finding optimal structural smoothing factor...")
            self.smoothing_factor_structural = self.find_optimal_smoothing_factor_across_timepoints(skeleton_im,
                                                                                                    structural_im,
                                                                                                    data_type='structural')
            # self.smoothing_factor_structural *= 2
            # if self.smoothing_factor_structural > 1.0:
            #     self.smoothing_factor_structural = 1.0
            # if self.smoothing_factor_structural == 0:
            #     self.smoothing_factor_structural = 0.5
            #     print(f"structural smoothing factor is 0, setting to 0.5")
            print(f"Found optimal structural smoothing factor: {self.smoothing_factor_structural}")
        else:
            print(f"Using user-specified structural smoothing factor: {self.smoothing_factor_structural}")
        # Apply the smoothing factors to all timepoints
        for t in tqdm(self.timepoints):
            # print(f"Timepoint: {t}")
            frame_skeleton_im = skeleton_im[t:t + 1]
            frame_intensity_im = intensity_im[t:t + 1]
            frame_structural_im = structural_im[t:t + 1]

            unique_labels = np.unique(skeleton_im)
            # print(f"# of Unique labels: {len(unique_labels)} for timepoint {t}")
            # print(f"Unique labels: {unique_labels}")

            if self.im_info.no_z:
                scaling = (t, self.im_info.dim_res['Y'], self.im_info.dim_res['X'])
            else:
                # print("3D image")
                scaling = (t, self.im_info.dim_res['Z'], self.im_info.dim_res['Y'], self.im_info.dim_res['X'])
            for label in unique_labels:
                if label == 0:
                    continue
                # print(f"Label: {label}")
                label_coords = np.argwhere(frame_skeleton_im == label)
                # print(f"Label coords: {label_coords}")
                if len(label_coords) < 2:
                    continue

                # Convert label_coords to a set of tuples for faster lookup
                label_coords_set = set(map(tuple, label_coords))

                # Traverse the skeleton
                skeleton_path = self.traverse_skeleton(label_coords_set)
                if not skeleton_path and len(label_coords) > 1:
                    # remove a random point from the label_coords and try again
                    label_coords_set.remove(tuple(label_coords[0]))
                    skeleton_path = self.traverse_skeleton(label_coords_set)

                intensity_vals = [frame_intensity_im[point] for point in skeleton_path]
                structural_vals = [frame_structural_im[point] for point in skeleton_path]

                for point_num, point in enumerate(skeleton_path):
                    path_im[t:t + 1][point] = point_num

                if len(label_coords) < 5:
                    continue

                # Calculate the cumulative distances along the path
                cumulative_distances = self.getCumulativeDistances(skeleton_path, scaling)
                if cumulative_distances is None:
                    continue

                # fit spline and calculate curvature for intensity values
                spline_resolution = len(label_coords)
                # print(f"Spline resolution: {spline_resolution}")
                curvature_intensity, spline_x, spline_y_intensity, mean_curvature_intensity = self.calculate_local_curvature(
                    cumulative_distances, intensity_vals, spline_resolution=spline_resolution,
                    smoothing_factor=self.smoothing_factor_intensity)

                # store the label traversal
                label_traversals_intensity.append(
                    self.LabelTraversal(t, label, label_coords, label_coords_set, skeleton_path, intensity_vals,
                                        cumulative_distances, spline_x, spline_y_intensity, curvature_intensity,
                                        mean_curvature_intensity))

                # fit spline and calculate curvature for structural values
                curvature_structural, spline_x, spline_y_structural, mean_curvature_structural = self.calculate_local_curvature(
                    cumulative_distances, structural_vals, spline_resolution=spline_resolution,
                    smoothing_factor=self.smoothing_factor_structural)

                # store the label traversal
                label_traversals_structural.append(
                    self.LabelTraversal(t, label, label_coords, label_coords_set, skeleton_path, structural_vals,
                                        cumulative_distances, spline_x, spline_y_structural, curvature_structural,
                                        mean_curvature_structural))

                # Add curvature values to the curvature image
                for i, point in enumerate(skeleton_path):
                    intensity_curvature_im[t:t + 1][point] = curvature_intensity[i]
                    structural_curvature_im[t:t + 1][point] = curvature_structural[i]

        # Compute weighted mean curvatures across all timepoints
        weighted_mean_curvatures_intensity = self.compute_weighted_mean_curvatures(label_traversals_intensity)

        # Compute weighted mean curvatures across all timepoints
        weighted_mean_curvatures_structural = self.compute_weighted_mean_curvatures(label_traversals_structural)

        normalized_weighted_mean_curvatures_intensity = self.normalize_weighted_mean_curvatures(weighted_mean_curvatures_intensity)
        normalized_weighted_mean_curvatures_structural = self.normalize_weighted_mean_curvatures(weighted_mean_curvatures_structural)
        dtype = [('file', 'U256'), ('treatment', str),('timepoint', int), ('intensity_smoothing_factor', float), ('structural_smoothing_factor', float),
                 ('intensity_curvature', float), ('structural_curvature', float),
                 ('normalized_intensity_curvature', float), ('normalized_structural_curvature', float)]

        # Sort the timepoints for consistent plotting
        sorted_timepoints = sorted(weighted_mean_curvatures_intensity.keys())
        num_timepoints = len(sorted_timepoints)

        # Initialize the structured array with the correct number of rows
        self.results = np.zeros(num_timepoints, dtype=dtype)

        # Assign values to each field in the structured array
        self.results['file'] = [self.im_path] * num_timepoints  # Assign the file path to each row
        self.results['timepoint'] = sorted_timepoints  # Assign the sorted timepoints
        self.results['intensity_smoothing_factor'] = [self.smoothing_factor_intensity] * num_timepoints
        self.results['structural_smoothing_factor'] = [self.smoothing_factor_structural] * num_timepoints
        self.results['intensity_curvature'] = [weighted_mean_curvatures_intensity[t] for t in sorted_timepoints]
        self.results['structural_curvature'] = [weighted_mean_curvatures_structural[t] for t in sorted_timepoints]
        self.results['normalized_intensity_curvature'] = [normalized_weighted_mean_curvatures_intensity[t] for t in sorted_timepoints]
        self.results['normalized_structural_curvature'] = [normalized_weighted_mean_curvatures_structural[t] for t in sorted_timepoints]
        import pandas as pd
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.save_path, 'curvatures.csv'), index=False)  # Save DataFrame to CSV

        # # Append results to the DataFrame
        # self.results = self.results.append({
        #     'file': self.path,
        #     'crop_count': label,
        #     'timepoint': t,
        #     'cumulative_distances': len(cumulative_distances),
        #     'intensity_curvature': mean_curvature_intensity,
        #     'structural_curvature': mean_curvature_structural}, ignore_index=True)

        if self.show_individual_plots:

            n_label_travels = len(label_traversals_intensity)
            print("# of Label Traversals: ", n_label_travels)
            for i in range(n_label_travels):
                traversal_intensity = label_traversals_intensity[i]
                traversal_structural = label_traversals_structural[i]
                # Fit and plot a spline for intensity values
                if len(traversal_intensity.label_coords) < 5:
                    continue
                if traversal_intensity.timepoint not in self.display_frames:
                    continue
                # 2 subplots, one for intensity, one for structural
                fig, axs = plt.subplots(4, figsize=(12, 8))
                fig.suptitle(f"timepoint {traversal_intensity.timepoint}, Label {traversal_intensity.label_num}")

                def plot_line_profile(ax, traversal, title, x_label, y_label, scatter=False):
                    if scatter:
                        ax.scatter(traversal.cumulative_distances, traversal.intensity_values)
                        ax.plot(traversal.spline_x, traversal.spline_y, color='red', label="Skeleton path")
                    else:
                        ax.plot(traversal.spline_x, traversal.curvature, color='blue', label="Curvature")
                    ax.set_title(title)
                    ax.legend()
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)

                    return ax

                axs[0] = plot_line_profile(axs[0], traversal_intensity, "Intensity", "Distance (um)",
                                           "Intensity values", scatter=True)
                axs[1] = plot_line_profile(axs[1], traversal_intensity, "Intensity", "Distance (um)",
                                           "Intensity Curvature", scatter=False)
                axs[2] = plot_line_profile(axs[2], traversal_structural, "Structural", "Distance (um)",
                                           "Structural values", scatter=True)
                axs[3] = plot_line_profile(axs[3], traversal_structural, "Structural", "Distance (um)",
                                           "Structral Curvature", scatter=False)

                plt.tight_layout()
                plt.show()
        # # Normalize curvature values to [0, 1] for visualization
        # intensity_curvature_im = (intensity_curvature_im - intensity_curvature_im.min()) / (intensity_curvature_im.max() - intensity_curvature_im.min())
        # structural_curvature_im = (structural_curvature_im - structural_curvature_im.min()) / (structural_curvature_im.max() - structural_curvature_im.min())
        # Apply log normalization
        intensity_curvature_im = self.log_normalize(intensity_curvature_im)
        structural_curvature_im = self.log_normalize(structural_curvature_im)

        # # Plot the results
        self.plot_mean_curvature_over_time(weighted_mean_curvatures_intensity,
                                           title='Intensity Weighted Mean Curvature Over Time',
                                           save_folder=self.save_path, view_plot=self.show_summary_plots)

        # Plot the results
        self.plot_mean_curvature_over_time(weighted_mean_curvatures_structural,
                                           title='Structural Weighted Mean Curvature Over Time',
                                           save_folder=self.save_path, view_plot=self.show_summary_plots)

        # max project the path_im
        # path_im = np.max(path_im, axis=1)
        if self.show_summary_plots:
            self.napari_view([intensity_im, skeleton_im, path_im, intensity_curvature_im, structural_curvature_im],
                             IMnames=['Intensity', 'Skeleton', 'Paths', 'Intensity Curvature', 'Structural Curvature'],
                             cmap=['gray', 'gray', 'gray', 'inferno', 'viridis'],
                             visible=[True,False,False,False,True,True],
                             contrast_limit=0.1, layer3d=True, save_folder=self.save_path,
                             view_napari=self.show_summary_plots)
        # self.napari_kymograph_view(images=[intensity_im, skeleton_im, path_im, intensity_curvature_im, structural_curvature_im],
        #                            IMnames=['Intensity','Skeleton','Paths', 'Intensity Curvature', 'Structural Curvature'],
        #                            cmap=['gray', 'gray', 'gray', 'inferno', 'viridis'],
        #                            contrast_limit=0.1, layer3d=True)

        return self.results

    def set_timepoints(self, timepoints):
        self.timepoints = timepoints


if __name__ == "__main__":
    # Define a list of dictionaries to store the file information
    filemap = [
        {
            'num': 1,
            'path': r"D:\Snouty_data\2D_mito_data\pearling5-FRAP-T1-UCSF-etbr-nucleoids\cropped_file.ome.tiff",
            'nellie_path': r"D:\Snouty_data\2D_mito_data\pearling5-FRAP-T1-UCSF-etbr-nucleoids\2024-08-19_11-52-57_nellie_out",
            'smoothing_factor': 1.0,
            'timepoints': None,
            'treatment': 'FRAP-laser'
        },
        {
            'num': 2,
            'path': r"F:\W1_FRAP\2023-08-08_ER-PKMO\RPE1-ER_3-pearling\cropped_file.ome.tiff",
            'nellie_path': r"F:\W1_FRAP\2023-08-08_ER-PKMO\RPE1-ER_3-pearling\2024-08-19_10-25-07_nellie_out",
            'smoothing_factor': 0.4,
            'timepoints': None,
            'treatment': 'control'
        },
        {
            'num': 3,
            'path': r"E:\SNOUTY_data\Pearling-examples\austin-pearling-metrics-example-3d-snouty\cropped_file.ome.tiff",
            'nellie_path': r"E:\SNOUTY_data\Pearling-examples\austin-pearling-metrics-example-3d-snouty\2024-08-24_20-52-07_nellie_out",
            'smoothing_factor': 0.3,
            'timepoints': None,
            'treatment': 'control'
        },
        {
            'num': 4,
            'path': r"F:\UCSF\W1_FRAP\2023-08-15_tmre-sybrgold\TMRE\FCCP\RPE1_CellLight_Mito_GFP_FCCP_4uM_3min_1-good\cropped_file.ome.tiff",
            'nellie_path': r"F:\UCSF\W1_FRAP\2023-08-15_tmre-sybrgold\TMRE\FCCP\RPE1_CellLight_Mito_GFP_FCCP_4uM_3min_1-good\2024-08-24_15-48-56_nellie_out",
            'smoothing_factor': 0.5,
            'timepoints': None,
            'treatment': 'FCCP'
        }
    ]

    print("Filemap:", filemap)

    # Loop through each entry in the filemap
    for file_entry in filemap:
        num = file_entry['num']
        if num != 3:
            continue
        path = file_entry['path']
        nellie_path = file_entry['nellie_path']
        smoothing_factor = file_entry['smoothing_factor']
        timepoints = file_entry['timepoints']

        print(f"Processing file {num}: {path}")
        print(f"Nellie directory: {nellie_path}")
        print(f"Smoothing factor: {smoothing_factor}")
        print(f"Timepoints: {timepoints}")
        save_path = os.path.join(nellie_path, "curvature_analysis")
        os.makedirs(save_path, exist_ok=True)
        print(f"Save path: {save_path}")

        # Create a SkeletonCurvature instance with the metadata
        skel_curvature = SkeletonCurvature(path, nellie_path, save_path=save_path, smoothing_factor=smoothing_factor,
                                           show_individual_plots=True, show_summary_plots=True)

        # Set the timepoints range if necessary (assume run_curvature supports this)
        skel_curvature.set_timepoints(timepoints)

        # Run the curvature analysis
        skel_curvature.run_curvature()
