from dataclasses import dataclass

import tifffile
import napari
import cupy as xp
import cupyx.scipy.ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

import thresholding
import numpy as np

from scipy.spatial import cKDTree
import skimage.measure as measure
import os
from nellie.im_info.verifier import FileInfo, ImInfo
import pandas as pd
import re
from tqdm import tqdm
import statistics as stats
import randomforest_pearl_classifier as rfc


class PearlingStats:
    def __init__(self, im_path, nellie_path, mask=None, save_path=None, select_frames=None, visualize=False, train_rfc=False):
        self.pearl_objects = None
        self.im_path = im_path
        self.nellie_path = nellie_path
        self.save_path = save_path
        self.save_path2 = None
        self.select_frames = select_frames
        self.visualize = visualize
        self.file_info = None
        self.im_info = None
        self.mask = mask
        self.mask_frame = None
        self.frangi_mask_frame = None
        self.relabelled_labels = None
        self.scaling = None
        self.raw = None
        self.is3D = False
        self.scale = None
        self.z_ratio = None
        self.frangi = None
        self.all_pearls_filtered = None
        self.all_raw = []
        self.all_labels = []
        self.all_frangi = []
        self.average_bead_radius_um = 0.25
        self.train_rfc = train_rfc
        if self.select_frames is not None:
            self.pearl_frame = select_frames[1]
            self.prepearl_frame = select_frames[0]
            self.postpearl_frame = select_frames[2]

    def find_file_in_subdirectories(self, directory, text):
        matching_files = []

        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(directory):
            for file in files:
                if text in file:
                    matching_files.append(os.path.join(root, file))

        return matching_files

    # Integration within the existing run_pearling_stats method
    def run_pearling_stats(self):
        RFC = rfc.PearlClassifier(csv_path='pearl_classifier.csv')
        self.file_info = FileInfo(filepath=self.im_path, output_dir=self.nellie_path)
        self.file_info.find_metadata()
        self.file_info.load_metadata()
        self.im_info = ImInfo(self.file_info)
        ome_path = self.im_info.im_path

        # Get total number of frames (time points)
        total_frames = self.file_info.t_end + 1
        # print(f'Total frames: {total_frames}')


        all_pearl_metrics = []
        all_centroids = []
        all_metrics = []
        label_metrics = []
        training_label_metrics = pd.DataFrame()

        frangi_path = self.find_file_in_subdirectories(self.nellie_path, 'im_preprocessed')[0]
        self.average_bead_radius_um = 0.25

        for frame in tqdm(range(total_frames), desc="Processing frames"):
            # print(f'Processing frame {frame + 1} of {total_frames}')

            # Load raw image and frangi filter for the current frame
            self.raw = xp.array(self.im_info.get_memmap(ome_path)[frame])
            self.frangi = xp.array(self.im_info.get_memmap(frangi_path)[frame])

            # Apply mask to frangi if present
            if self.mask is not None:
                self.frangi *= self.mask
                self.raw *= self.mask

            # Determine if 3D or 2D
            if 'Z' in self.file_info.axes:
                self.is3D = True
                self.scale = xp.array(
                    [self.file_info.dim_res['Z'], self.im_info.dim_res['Y'], self.im_info.dim_res['X']])
                self.z_ratio = self.scale[1] / self.scale[0]
            else:
                self.is3D = False
                self.scale = xp.array([self.im_info.dim_res['Y'], self.im_info.dim_res['X']])
            self.scaling = self.scale.get()
            # Thresholding and edge detection
            frangi_minotri_thresh = thresholding.minotri_threshold(self.frangi[self.frangi > 0])
            self.frangi_mask_frame = self.frangi > frangi_minotri_thresh
            mint_thresh = thresholding.minotri_threshold(self.raw[self.raw > 0])
            self.mask_frame = self.raw > mint_thresh


            smoothed_distance = self.get_distances(self.frangi_mask_frame)
            footprint = self.getFootprint()
            max_filtered = ndi.maximum_filter(smoothed_distance, footprint=footprint)
            peaks = (max_filtered == smoothed_distance) * self.frangi_mask_frame
            peak_labels, num_peaks = ndi.label(peaks)

            # # Apply watershed segmentation
            # self.segment_pearls_watershed(mask_frame)
            self.reassign_labels(peak_labels, peaks, self.frangi_mask_frame)

            # # Remove pearls touching the edge of the mask
            # self.remove_edge_pearls()

            # Remove small pearls
            self.remove_small_pearls()

            # # merge overlapping regions
            # self.merge_by_hierarchical_clustering(distance_threshold=1.0)

            # Store the raw image and labels
            # Collect data for each frame
            self.all_raw.append(self.raw)  # Keep as CuPy array
            self.all_frangi.append(self.frangi)  # Keep as CuPy array
            self.all_labels.append(self.relabelled_labels.copy())

        # # Step 2: Perform parameter optimization using stored regions
        # select_params = ['rectangularity_threshold']  # Specify which parameters to optimize
        # optimized_params = self.optimize_filter_parameters(
        #     frames_to_optimize=[self.prepearl_frame, self.pearl_frame],
        #     select_params=select_params
        # )
        #
        # # Step 3: Apply optimized filters to all frames and compute metrics
        # for frame in tqdm(range(total_frames), desc="Processing frames with optimized filters"):
        #     self.raw = self.all_raw[frame]
        #     self.relabelled_labels = self.all_labels[frame]
        #
        #     # Apply optimized filters
        #     self.apply_filters(
        #         raw=self.raw,
        #         **optimized_params
        #     )

            # Compute pearling statistics
            pearl_metrics, pearl_label_metrics, centroids, metrics = self.compute_pearlings_stats(timepoint=frame)
            # label a subset of labels for random forest training
            if self.train_rfc:
                train_frames = [self.prepearl_frame, self.pearl_frame] # self.pearl_frame + 1, self.postpearl_frame
                if frame in train_frames:
                    # convert to pd.DataFrame
                    pearl_label_metrics = pd.DataFrame(pearl_label_metrics)
                    # check if annotations are pre-selected
                    train_path = os.path.join(self.nellie_path, 'pearling_stats', 'training_label_metrics.csv')
                    # if os.path.exists(train_path):
                    #     trained_annotations = pd.read_csv(train_path)
                    #     # Update the 'is_pearl' and 'pearl_type' columns based on 'timepoint' and 'label_id'
                    #     pearl_label_metrics.update(
                    #         trained_annotations.set_index(['timepoint', 'label_id'])[['is_pearl', 'pearl_type']],
                    #         overwrite=True
                    #     )
                    # else: # if not already selected then select an annotation
                    pearl_order = False
                    if frame == self.pearl_frame or frame == self.pearl_frame+1:
                        pearl_order = True
                    pearl_label_metrics = RFC.annotate_labels(pearl_label_metrics, image=self.raw.get(),
                                                          labels=self.relabelled_labels.get(),
                                                          label_column='label_id', pearl_order=pearl_order)
                    training_label_metrics = pd.concat([training_label_metrics, pearl_label_metrics], ignore_index=True)

            all_pearl_metrics.append(pearl_metrics.__dict__)
            all_centroids.append(centroids)
            label_metrics.append(pearl_label_metrics)
            all_metrics.append(metrics)

            # Update labels after filtering
            self.all_labels[frame] = self.relabelled_labels.copy()


        # Visualize all collected data at the end
        if self.visualize:
            viewer = napari.Viewer() if self.visualize else None
            viewer.add_image(xp.stack(self.all_raw).get(), name='raw images')
            viewer.add_image(xp.stack(self.all_frangi).get(), name='frangi images', visible=False)
            # viewer.add_image(np.stack(all_smoothed_distances), name='smoothed distances', visible=False)
            # viewer.add_image(np.stack(all_peaks), name='peaks', visible=False)
            # viewer.add_labels(np.stack(all_peak_labels), name='peak labels', visible=False)
            viewer.add_labels(xp.stack(self.all_labels).get(), name='pearls')
            viewer.dims.set_point(0, self.pearl_frame)  # 0 is the axis for the T dimension (time)

            # Initialize lists to store all points and labels
            all_points = []
            all_labs = []

            # Loop through each frame in the timeseries
            for frame_index, (frame_centroids, frame_labels) in enumerate(zip(all_centroids, all_metrics)):
                if self.is3D:
                    # Convert centroids to NumPy array and add the time (T) and Z dimensions for 3D
                    points = np.array(
                        [[frame_index, z, y, x] for (z, y, x) in frame_centroids]
                    )  # Original centroid points in (T, Z, Y, X)

                else:
                    # Convert centroids to NumPy array and add the time (T) dimension for 2D
                    points = np.array(
                        [[frame_index, y, x] for (y, x) in frame_centroids]
                    )  # Original centroid points in (T, Y, X)

                # Separate each parameter with newline characters in labels
                formatted_labels = [label.replace(', ', '\n') for label in frame_labels]

                # Append current frame points and labels to the full list
                all_points.extend(points)
                all_labs.extend(formatted_labels)

            # Convert all points and text positions to a single NumPy array
            all_points = np.array(all_points)

            # Add all points as a single layer with labels
            viewer.add_points(
                all_points,
                size=1,  # Adjust point size as needed
                face_color='red',  # Adjust point color as needed
                text={'string': all_labs, 'anchor': 'lower_right', 'translation': (0, 1, 1), 'size': 6},
                name='metric information',
                visible=False
            )

            napari.run()

        # Save the metrics as CSV files
        pearl_metrics_df = pd.DataFrame(all_pearl_metrics)
        save_path = os.path.join(self.save_path, "pearling_metrics.csv")
        pearl_metrics_df.to_csv(save_path, index=False)

        # Save the training label metrics as CSV files
        if self.train_rfc:
            save_path2 = os.path.join(self.save_path, "training_label_metrics.csv")
            training_label_metrics['file_path'] = self.save_path
            training_label_metrics.to_csv(save_path2, index=False)

        print(f'Pearling metrics saved to {save_path}')

        # Plot the number of pearls over time
        # self.plot_pearls_over_time(pearl_metrics_df, y_column='volume_mean', sd='volume_sd')
        # self.plot_pearls_over_time(pearl_metrics_df, y_column='num_pearls')
        # self.plot_pearls_over_time(pearl_metrics_df, y_column='rectangularity_mean', sd='rectangularity_sd')
        # self.plot_pearls_over_time(pearl_metrics_df, y_column='eccentricity_mean', sd='eccentricity_sd')
        # self.plot_pearls_over_time(pearl_metrics_df, y_column='solidity_mean', sd='solidity_sd')
        # self.plot_pearls_over_time(pearl_metrics_df, y_column='sphericity_mean', sd='sphericity_sd')
        # self.plot_pearls_over_time(pearl_metrics_df, y_column='major_axis_mean', sd='major_axis_sd')
        # self.plot_pearls_over_time(pearl_metrics_df, y_column='minor_axis_mean', sd='minor_axis_sd')
        # self.plot_pearls_over_time(pearl_metrics_df, y_column='medio_axis_mean', sd='medio_axis_sd')

        return pearl_metrics_df, training_label_metrics
    def getFootprint(self):
        if self.is3D:
            footprint = xp.ones((int(2 * self.average_bead_radius_um / self.scale[1]),
                                 int(2 * self.average_bead_radius_um / self.scale[1]),
                                 int(2 * self.average_bead_radius_um / self.scale[1])))
        else:
            footprint = xp.ones((int(2 * self.average_bead_radius_um / self.scale[1]),
                                 int(2 * self.average_bead_radius_um / self.scale[1])))
        return footprint

    def apply_filters(self, raw, **kwargs):
        """
        Applies the optimized filters to self.relabelled_labels.

        Parameters:
            raw (ndarray): The raw intensity image.
            **kwargs: Filtering parameters.
        """
        labels_np = self.relabelled_labels.get() if isinstance(self.relabelled_labels,
                                                               xp.ndarray) else self.relabelled_labels

        # Apply filters based on the parameters provided
        if 'axis_ratio_threshold' in kwargs:
            labels_np = self.filter_by_axis_ratio_labels(labels_np, raw, kwargs['axis_ratio_threshold'])
        if 'eccentricity_threshold' in kwargs:
            labels_np = self.filter_by_eccentricity_labels(labels_np, raw, kwargs['eccentricity_threshold'])
        if 'solidity_threshold' in kwargs:
            labels_np = self.filter_by_solidity_labels(labels_np, raw, kwargs['solidity_threshold'])
        if 'hu_moment_threshold' in kwargs:
            labels_np = self.filter_by_hu_moments_labels(labels_np, raw, kwargs['hu_moment_threshold'])
        if 'rectangularity_threshold' in kwargs:
            labels_np = self.filter_by_rectangularity_labels(labels_np, raw, kwargs['rectangularity_threshold'])

        # Update relabelled_labels
        self.relabelled_labels = xp.array(labels_np)

    def optimize_filter_parameters(self, frames_to_optimize, select_params=[]):
        """
        Optimizes the pearl detection parameters to maximize the difference between
        the number of pearls in self.pearl_frame and self.prepearl_frame.

        Parameters:
            frames_to_optimize (list): List containing indices of frames to use for optimization.
            select_params (list): List of parameter names to optimize.

        Returns:
            dict: A dictionary containing the optimized parameters.
        """
        import numpy as np
        from itertools import product

        # Define parameter ranges for all possible parameters
        param_ranges = {
            'axis_ratio_threshold': np.linspace(1.0, 3.0, 10),
            'eccentricity_threshold': np.linspace(0.1, 1.0, 5),
            'solidity_threshold': np.linspace(0.1, 1.0, 15),
            'hu_moment_threshold': np.linspace(-2, 2, 10),
            'rectangularity_threshold': np.linspace(0.1, 1.0, 10)
        }

        # Initialize variables to store the best parameters
        best_params = {}
        best_objective_value = -np.inf
        parameter_combinations_tried = 0

        # Build parameter ranges only for selected parameters
        selected_param_ranges = [param_ranges[param] for param in select_params]

        # Generate all combinations of selected parameters
        parameter_grid = list(product(*selected_param_ranges))

        # Get labels and raw data for the frames to optimize
        labels_prepearl = self.all_labels[frames_to_optimize[0]]
        raw_prepearl = self.all_raw[frames_to_optimize[0]]
        labels_pearl = self.all_labels[frames_to_optimize[1]]
        raw_pearl = self.all_raw[frames_to_optimize[1]]

        # Loop over parameter combinations
        for param_values in parameter_grid:
            parameter_combinations_tried += 1

            # Build a dictionary of parameter values
            params_dict = dict(zip(select_params, param_values))

            # Apply filters to prepearl_frame
            num_pearls_pre = self.apply_filters_to_labels(
                labels=labels_prepearl.copy(),
                raw=raw_prepearl,
                **params_dict
            )

            # Apply filters to pearl_frame
            num_pearls_pearl = self.apply_filters_to_labels(
                labels=labels_pearl.copy(),
                raw=raw_pearl,
                **params_dict
            )

            # Compute objective function: Maximize the difference in the number of pearls
            objective_value = num_pearls_pearl - num_pearls_pre

            # Update the best parameters if the objective value is higher
            if objective_value > best_objective_value:
                print(
                    f'Objective value: {objective_value} for parameters, num_pearls_pre: {num_pearls_pre}, num_pearls_pearl: {num_pearls_pearl}')
                best_objective_value = objective_value
                best_params = params_dict

        print(f"Optimization completed after trying {parameter_combinations_tried} combinations.")
        print(f"Best objective value: {best_objective_value}")
        print(f"Optimized parameters: {best_params}")

        return best_params

    def apply_filters_to_labels(self, labels, raw, **kwargs):
        """
        Applies filters to the given labels using the specified parameters.

        Parameters:
            labels (ndarray): The label image to apply filters to.
            raw (ndarray): The raw intensity image.
            **kwargs: Filtering parameters.

        Returns:
            int: The number of pearls detected after filtering.
        """
        # Convert labels to NumPy array if necessary
        labels_np = labels.get() if isinstance(labels, xp.ndarray) else labels

        # Recompute pearl objects based on the labels
        self.pearl_objects = measure.regionprops(labels_np, intensity_image=raw, spacing=self.scale.get())

        # Apply filters based on the parameters provided
        if 'axis_ratio_threshold' in kwargs:
            labels_np = self.filter_by_axis_ratio_labels(labels_np, raw, kwargs['axis_ratio_threshold'])
        if 'eccentricity_threshold' in kwargs:
            labels_np = self.filter_by_eccentricity_labels(labels_np, raw, kwargs['eccentricity_threshold'])
        if 'solidity_threshold' in kwargs:
            labels_np = self.filter_by_solidity_labels(labels_np, raw, kwargs['solidity_threshold'])
        if 'hu_moment_threshold' in kwargs:
            labels_np = self.filter_by_hu_moments_labels(labels_np, raw, kwargs['hu_moment_threshold'])
        if 'rectangularity_threshold' in kwargs:
            labels_np = self.filter_by_rectangularity_labels(labels_np, raw, kwargs['rectangularity_threshold'])

        # Recompute pearl_objects after filtering
        pearl_objects = measure.regionprops(labels_np, intensity_image=raw, spacing=self.scaling)
        num_pearls = len(pearl_objects)

        return num_pearls

    def filter_by_axis_ratio_labels(self, labels_np, raw, max_ratio_threshold):
        """
        Filters labels based on axis ratio threshold.

        Returns:
            ndarray: Updated labels after filtering.
        """

        pearl_objects = measure.regionprops(labels_np, intensity_image=raw, spacing=self.scale.get())
        for region in pearl_objects:
            major_axis_length = region.major_axis_length
            minor_axis_length = region.minor_axis_length

            # Avoid division by zero
            if minor_axis_length == 0:
                axis_ratio = np.inf
            else:
                axis_ratio = major_axis_length / minor_axis_length

            if axis_ratio > max_ratio_threshold:
                labels_np[labels_np == region.label] = 0

        # Re-label the connected components
        # labels_np = measure.label(labels_np)

        return labels_np

    def filter_by_rectangularity_labels(self, labels_np, raw, max_rectangularity):
        import numpy as np
        from skimage import measure

        # Pass 'spacing' to measure.regionprops to get 'area' in physical units
        spacing = self.scale.get()
        pearl_objects = measure.regionprops(labels_np, intensity_image=raw, spacing=spacing)
        for region in pearl_objects:
            area = region.area  # In physical units (e.g., micrometers squared)

            # Compute the area of the bounding box in physical units
            if self.is3D:
                minz, minr, minc, maxz, maxr, maxc = region.bbox
                bounding_height = (maxr - minr) * spacing[1]
                bounding_width = (maxc - minc) * spacing[2]
                bounding_depth = (maxz - minz) * spacing[0]
                bounding_volume = bounding_height * bounding_width * bounding_depth
                if bounding_volume == 0:
                    rectangularity = 0
                else:
                    rectangularity = area / bounding_volume
            else:
                minr, minc, maxr, maxc = region.bbox
                spacing = self.scale.get()
                bounding_height = (maxr - minr) * spacing[0]
                bounding_width = (maxc - minc) * spacing[1]
                bounding_area = bounding_height * bounding_width
                if bounding_area == 0:
                    rectangularity = 0
                else:
                    rectangularity = area / bounding_area

            # print(
            #     f"Region {region.label}: Area = {area:.3f}, Bounding Area = {bounding_area:.3f}, Rectangularity = {rectangularity:.3f}")
            if rectangularity > max_rectangularity:
                labels_np[labels_np == region.label] = 0

        # Re-label the connected components
        # labels_np = measure.label(labels_np)

        return labels_np

    def filter_by_eccentricity_labels(self, labels_np, raw, max_eccentricity):
        import numpy as np
        from skimage import measure

        pearl_objects = measure.regionprops(labels_np, intensity_image=raw, spacing=self.scale.get())
        for region in pearl_objects:
            eccentricity = region.eccentricity
            if eccentricity > max_eccentricity:
                labels_np[labels_np == region.label] = 0

        # labels_np = measure.label(labels_np)

        return labels_np

    def filter_by_solidity_labels(self, labels_np, raw, min_solidity):
        import numpy as np
        from skimage import measure

        pearl_objects = measure.regionprops(labels_np, intensity_image=raw, spacing=self.scale.get())
        for region in pearl_objects:
            solidity = region.solidity
            if solidity < min_solidity:
                labels_np[labels_np == region.label] = 0

         #labels_np = measure.label(labels_np)

        return labels_np

    def filter_by_hu_moments_labels(self, labels_np, raw, hu_moment_threshold, hu_moment_index=0):
        import numpy as np
        from skimage.measure import moments_central, moments_normalized, moments_hu, regionprops

        pearl_objects = measure.regionprops(labels_np, intensity_image=raw, spacing=self.scale.get())
        for region in pearl_objects:
            # Extract region image
            coords = region.coords
            minr, minc, maxr, maxc = region.bbox
            if self.is3D:
                minz, minr, minc, maxz, maxr, maxc = region.bbox
                region_slice = (slice(minz, maxz), slice(minr, maxr), slice(minc, maxc))
            else:
                region_slice = (slice(minr, maxr), slice(minc, maxc))

            region_image = labels_np[region_slice] == region.label

            # Compute moments
            m = moments_central(region_image.astype(np.float64))
            nu = moments_normalized(m)
            hu = moments_hu(nu)

            # Get the selected Hu moment
            hu_moment_value = hu[hu_moment_index]
            hu_moment_log_abs = -np.sign(hu_moment_value) * np.log10(abs(hu_moment_value) + 1e-10)

            if hu_moment_log_abs > hu_moment_threshold:
                labels_np[labels_np == region.label] = 0

        # labels_np = measure.label(labels_np)

        return labels_np

    def segment_pearls_watershed(self, binary):
        from skimage import segmentation, morphology, measure
        from scipy.ndimage import gaussian_filter, distance_transform_edt
        from skimage.feature import peak_local_max

        """
        Segments pearls in the current frame using the watershed algorithm from scikit-image.
        If the object is an intact tubule, it identifies it as one large region.
        """

        # Remove small objects (noise)
        binary = morphology.remove_small_objects(binary.get(), min_size=64)

        # Compute the distance transform
        distance = distance_transform_edt(binary)

        # Calculate min_distance parameter
        min_distance = max(1, int(2 * self.average_bead_radius_um / self.scale[1].get()))

        # Find local maxima
        if self.is3D:
            # For 3D data
            coordinates = peak_local_max(
                distance,
                min_distance=min_distance,
                labels=binary,
                footprint=np.ones((3, 3, 3)),
                exclude_border=False
            )
            # Create a mask of the peaks
            local_maxi = np.zeros_like(distance, dtype=bool)
            local_maxi[tuple(coordinates.T)] = True
        else:
            # For 2D data
            coordinates = peak_local_max(
                distance,
                min_distance=min_distance,
                labels=binary,
                exclude_border=False
            )
            # Create a mask of the peaks
            local_maxi = np.zeros_like(distance, dtype=bool)
            local_maxi[tuple(coordinates.T)] = True

        # Label markers
        markers = measure.label(local_maxi)

        # Apply watershed
        labels = segmentation.watershed(-distance, markers, mask=binary)

        # Convert labels back to CuPy array
        self.relabelled_labels = xp.array(labels)

    def get_distances(self, mask_frame):
        border_image = ndi.binary_dilation(mask_frame) ^ mask_frame
        border_coords = xp.argwhere(border_image) * self.scale
        mask_coords = xp.argwhere(mask_frame) * self.scale

        border_kdtree = cKDTree(border_coords.get())
        mask_nn = border_kdtree.query(mask_coords.get())

        distance_image = xp.zeros_like(self.frangi)
        distance_image[mask_frame] = mask_nn[0]
        # smoothed_distance = ndi.gaussian_filter(distance_image, sigma=[z_ratio, 1, 1]) * mask_frame
        if self.is3D:
            smoothed_distance = ndi.gaussian_filter(self.frangi, sigma=[self.z_ratio, 1, 1]) * mask_frame
        else:
            smoothed_distance = ndi.gaussian_filter(self.frangi, sigma=[1, 1]) * mask_frame

        return smoothed_distance

    def reassign_labels(self, peak_labels, peaks, mask_frame):
        """
        Refines and merges labeled regions (peaks) by expanding the labeled areas into adjacent regions that are not yet labeled.
        Also removes regions that touch the edges of the image.

        Parameters:
        - peak_labels (ndarray): Initial labeled regions of interest.
        - peaks (ndarray): Binary mask indicating peak regions.
        - mask_frame (ndarray): Binary mask indicating the valid region where labeling should be performed.
        """

        # Define a structuring element for dilation
        structure = xp.ones((3, 3, 3)) if self.is3D else xp.ones((3, 3))  # Use a 3x3x3 or 3x3 structuring element

        # Initialize relabeled labels and branch skeleton labels
        self.relabelled_labels = peak_labels.copy()  # Copy initial peak labels
        branch_skel_labels = self.relabelled_labels.copy()  # Another copy for use in updating labels

        # Convert peaks and mask_frame to integer format
        peak_mask_int = peaks.astype('uint16')
        label_mask_int = mask_frame.astype('uint16')

        # Create a "border" around peaks by dilating and subtracting the original peaks
        peak_border = (ndi.binary_dilation(peak_mask_int, iterations=1,
                                           structure=structure) ^ peak_mask_int) * label_mask_int

        # Create a binary mask indicating where peak labels are present
        peak_label_mask = (peak_labels > 0).get()

        # Find the coordinates of all labeled voxels (peaks)
        vox_matched = np.argwhere(peak_label_mask)
        # Find the coordinates of all potential unlabeled border voxels
        vox_next_unmatched = np.argwhere(peak_border.get())

        # Initialize a variable to track the difference in unmatched voxels after each iteration
        unmatched_diff = np.inf
        self.scaling = self.scale.get()  # Retrieve scaling factor

        # Iteratively match unlabeled voxels to the nearest labeled voxels
        while True:
            num_unmatched = len(vox_next_unmatched)  # Number of unmatched voxels
            if num_unmatched == 0:
                break  # Exit if no unmatched voxels remain

            # Build a KD-tree for efficient nearest-neighbor search
            tree = cKDTree(vox_matched * self.scaling)
            # Find the nearest labeled voxel for each unmatched voxel
            dists, idxs = tree.query(vox_next_unmatched * self.scaling, k=1, workers=-1)

            # Define a maximum distance for matching
            max_dist = 2 * np.min(self.scaling)  # Maximum distance is twice the minimum scaling factor

            # Filter out matches that are too far away
            unmatched_matches = np.array(
                [[vox_matched[idx], vox_next_unmatched[i]] for i, idx in enumerate(idxs) if dists[i] < max_dist]
            )

            if len(unmatched_matches) == 0:
                break  # Exit if no valid matches are found

            # Get the labels of the matched voxels
            matched_labels = branch_skel_labels[tuple(np.transpose(unmatched_matches[:, 0]))]
            # Assign these labels to the corresponding unmatched voxels
            self.relabelled_labels[tuple(np.transpose(unmatched_matches[:, 1]))] = matched_labels

            # Update the branch skeleton labels and relabeled mask
            branch_skel_labels = self.relabelled_labels.copy()
            self.relabelled_labels_mask = self.relabelled_labels > 0
            self.relabelled_labels_mask_cpu = self.relabelled_labels_mask.get()

            # Update the list of matched voxel coordinates
            vox_matched = np.argwhere(self.relabelled_labels_mask_cpu)
            relabelled_mask = self.relabelled_labels_mask.astype('uint8')

            # Update the peak border for the next iteration
            peak_border = (ndi.binary_dilation(relabelled_mask, iterations=1,
                                               structure=structure) ^ relabelled_mask) * label_mask_int
            vox_next_unmatched = np.argwhere(peak_border.get())

            # Check if the number of unmatched voxels has changed
            new_num_unmatched = len(vox_next_unmatched)
            unmatched_diff_temp = abs(num_unmatched - new_num_unmatched)
            if unmatched_diff_temp == unmatched_diff:
                break  # Exit if no improvement in unmatched voxels
            unmatched_diff = unmatched_diff_temp  # Update the unmatched difference

        # Remove objects that touch the edges of the image
        pearl_objects_not_spaced = measure.regionprops(self.relabelled_labels.get())
        if self.is3D:
            # For 3D data
            max_z, max_y, max_x = self.relabelled_labels.shape
            for region in pearl_objects_not_spaced:
                bbox = region.bbox
                # Check if any part of the bounding box is on the edge of the 3D volume
                if bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] <= 0 or \
                        bbox[3] + 1 >= max_z or bbox[4] + 1 >= max_y or bbox[5] + 1 >= max_x:
                    # Remove objects touching the edge
                    self.relabelled_labels[self.relabelled_labels == region.label] = 0
        else:
            # For 2D data
            max_y, max_x = self.relabelled_labels.shape
            for region in pearl_objects_not_spaced:
                bbox = region.bbox
                # Check if any part of the bounding box is on the edge of the 2D image
                if bbox[0] <= 0 or bbox[1] <= 0 or \
                        bbox[2] + 5 >= max_y or bbox[3] + 5 >= max_x:
                    # Remove objects touching the edge
                    self.relabelled_labels[self.relabelled_labels == region.label] = 0

    def remove_edge_pearls(self):
        """
        Removes pearls that touch the edge of the mask.
        """
        from skimage.segmentation import find_boundaries
        import numpy as np

        # Ensure that the mask is available
        if self.mask is None:
            raise ValueError("Mask is not defined.")

        # Convert mask to NumPy array if it's a CuPy array
        if isinstance(self.mask, xp.ndarray):
            mask_np = self.mask.get()
        else:
            mask_np = self.mask

        # Find the boundary of the mask
        mask_boundary = find_boundaries(mask_np, mode='outer')
        # mask_boundary is a boolean array where True indicates boundary pixels

        # Convert relabelled_labels to NumPy array if necessary
        if isinstance(self.relabelled_labels, xp.ndarray):
            labels_np = self.relabelled_labels.get()
        else:
            labels_np = self.relabelled_labels

        # For each pearl (region), check if it touches the boundary
        pearl_objects = measure.regionprops(labels_np)

        for region in pearl_objects:
            print(f"Checking pearl {region.label} for boundary touch.")
            # Get the coordinates of the region's pixels
            coords = region.coords  # ndarray of (N, ndim)

            # Check if any of the coords are in the boundary
            # Get the mask_boundary values at the region's coordinates
            boundary_values = mask_boundary[tuple(coords.T)]
            if np.any(boundary_values):
                # The pearl touches the boundary, remove it
                print(f"Removing pearl {region.label} touching the mask boundary.")
                labels_np[labels_np == region.label] = 0

        # # Optionally, re-label the connected components to ensure labels are consecutive
        # labels_np = measure.label(labels_np)

        # Convert labels back to CuPy array
        self.relabelled_labels = xp.array(labels_np)

    def remove_small_pearls(self):
        """
        Removes pearls (regions) where the average bead radius is below half the defined average bead radius.

        The function computes the bead radius for each pearl using its minor axis length or equivalent diameter.
        Regions that do not meet the criteria are removed by setting their labels to 0.
        """
        # Ensure relabelled_labels and raw images are available
        if self.relabelled_labels is None or self.raw is None:
            raise ValueError("Labelled image or raw image is not defined.")

        self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.frangi.get(),
                                                 spacing=self.scale.get())
        # print('len of pearl objects: ', len(self.pearl_objects))

        # Iterate over each pearl object to check its size
        for region in self.pearl_objects:
            # Calculate the average bead radius; you can use the equivalent diameter or another suitable measure
            bead_radius = region.equivalent_diameter / 2.0  # Using equivalent diameter as a proxy for bead radius
            # print(f"Pearl {region.label} has bead radius: {bead_radius:.2f} (um)")

            # Check if the bead radius is below the threshold
            if bead_radius < self.average_bead_radius_um / 2:
                # Set the label to 0 to remove the region
                self.relabelled_labels[self.relabelled_labels == region.label] = 0
                # print(f"Removed pearl {region.label} with bead radius {bead_radius:.2f} below threshold of {half_avg_bead_radius:.2f}")

        # Optional: Refresh the list of regions if needed
        self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.frangi.get(),
                                                 spacing=self.scaling)
        # print('len of pearl objects after removal: ', len(self.pearl_objects))

        # print("Completed removal of small pearls.")


    def blob_detection_filter(self, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.001,
                              overlap=0.1, rectangularity_threshold=0.7, visualize=True):
        from skimage import feature, filters, measure, draw, morphology
        import numpy as np
        import napari

        """
        Use blob detection to filter out regions that are not pearls, merge overlapping regions, and update relabelled_labels and pearl_objects.

        Parameters:
        - min_sigma (float): The minimum standard deviation for the Gaussian kernel. Smaller values detect smaller blobs.
        - max_sigma (float): The maximum standard deviation for the Gaussian kernel. Larger values detect larger blobs.
        - num_sigma (int): The number of standard deviations to test between min_sigma and max_sigma.
        - threshold (float): The absolute lower bound for scale space maxima. Local maxima smaller than threshold are ignored.
        - overlap (float): A value between 0 and 1. If the area of two blobs overlaps by a fraction greater than this value, the smaller blob is eliminated.
        - rectangularity_threshold (float): Maximum rectangularity allowed to keep a pearl.
        - visualize (bool): Whether to visualize the results using Napari.

        Returns:
        - filtered_labels (ndarray): The relabelled labels after filtering and merging.
        """
        # Ensure relabelled_labels and raw images are available
        if self.relabelled_labels is None or self.raw is None:
            raise ValueError("Labelled image or raw image is not defined.")

        # Convert CuPy arrays to NumPy arrays for processing with skimage
        relabelled_labels_np = self.relabelled_labels.get()

        # Apply mask to raw image if present
        masked_raw = self.raw
        if self.mask is not None:
            masked_raw *= self.mask

        # Detect blobs in the raw image using Laplacian of Gaussian (LoG)
        image_smooth = filters.gaussian(masked_raw.get(), sigma=1)
        blobs = feature.blob_log(
            image_smooth,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap
        )
        print(f'Number of detected blobs: {len(blobs)}')

        # Compute the radius of each blob
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

        # Create a binary mask for detected blobs
        blobs_mask = np.zeros_like(masked_raw.get(), dtype=bool)
        for blob in blobs:
            if len(blob) == 3:  # 2D
                y, x, r = blob
                rr, cc = np.ogrid[:masked_raw.get().shape[0], :masked_raw.get().shape[1]]
                mask = (rr - y) ** 2 + (cc - x) ** 2 <= r ** 2
                blobs_mask[mask] = True
            else:  # 3D
                z, y, x, r = blob
                zz, yy, xx = np.ogrid[:masked_raw.get().shape[0], :masked_raw.get().shape[1],
                             :masked_raw.get().shape[2]]
                mask = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2
                blobs_mask[mask] = True

        # Create a new labels array initialized to 0 (background)
        filtered_labels = np.zeros_like(relabelled_labels_np, dtype=np.int32)

        # Recompute pearl objects based on the relabelled labels
        self.pearl_objects = measure.regionprops(relabelled_labels_np, intensity_image=masked_raw.get(),
                                                 spacing=self.scale.get())

        # Step 1: Find pearls that overlap with detected blobs
        overlapping_pearl_objects = []
        for region in self.pearl_objects:
            # Get the centroid of the region
            centroid = region.centroid / self.scale.get()

            # Check if the region's centroid overlaps with any detected blob
            valid_pearl = False
            for blob in blobs:
                if len(blob) == 3:  # 2D
                    y, x, r = blob
                    if np.sqrt((centroid[0] - y) ** 2 + (centroid[1] - x) ** 2) <= r:
                        valid_pearl = True
                        break
                else:  # 3D
                    z, y, x, r = blob
                    if np.sqrt((centroid[0] - z) ** 2 + (centroid[1] - y) ** 2 + (centroid[2] - x) ** 2) <= r:
                        valid_pearl = True
                        break

            if valid_pearl:
                overlapping_pearl_objects.append(region)

        # # Step 2: Filter overlapping pearls using rectangularity
        # filtered_pearl_objects = []
        # for region in overlapping_pearl_objects:
        #     # Calculate rectangularity: Area / Bounding Box Area
        #     bbox_area = region.bbox_area
        #     if bbox_area > 0:
        #         rectangularity = region.area / bbox_area
        #     else:
        #         rectangularity = 0
        #
        #     # Filter based on rectangularity
        #     if rectangularity <= rectangularity_threshold:
        #         filtered_pearl_objects.append(region)

        # Step 3: Merge pearls that are part of the same blob
        merged_labels = np.zeros_like(relabelled_labels_np, dtype=np.int32)

        # Create a mask for merging overlapping regions
        for region in overlapping_pearl_objects:
            # Fill the region with a new label in the mask
            merged_labels[relabelled_labels_np == region.label] = 1  # Use 1 for merging mask

        # Relabel merged blobs to create unique labels for each merged region
        # `measure.label` will create new labels for each connected component (merged blob)
        filtered_labels, num_features = measure.label(merged_labels, return_num=True)

        print(f"Number of merged blobs: {num_features}")

        # # Step 3: Update the relabelled_labels with the filtered pearls
        # new_label = 1  # Start labeling from 1
        # for region in overlapping_pearl_objects:
        #     # Keep the region in the new labels array with a new label
        #     filtered_labels[relabelled_labels_np == region.label] = new_label
        #     new_label += 1  # Increment the label for the next valid region

        # Update the relabelled_labels with the merged and filtered labels
        self.relabelled_labels = xp.array(filtered_labels)

        # Recompute pearl objects with the updated relabelled labels
        self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.raw.get(),
                                                 spacing=self.scale.get())

        # Print the number of pearls retained after filtering
        print(f'Number of pearls retained after Blob detection: {len(self.pearl_objects)}')

        if visualize:
            # Visualize using Napari
            viewer = napari.Viewer()
            # Add raw image
            viewer.add_image(self.raw.get(), name='Raw Image', colormap='gray')
            # Add blobs mask as a layer
            viewer.add_labels(blobs_mask.astype(int), name='Detected Blobs')
            # Add filtered relabelled_labels (pearls) as a layer
            viewer.add_labels(filtered_labels, name='Filtered Pearls')
            napari.run()

        # Return the blob frame (filtered labels) for later use
        blobs_mask = xp.array(blobs_mask, dtype=np.int32)
        return blobs_mask

    def merge_by_hierarchical_clustering(self, distance_threshold):
        from scipy.cluster.hierarchy import fcluster, linkage
        import numpy as np
        from skimage.measure import regionprops

        """
        Merges pearl objects using hierarchical clustering based on their centroids.

        Args:
            pearl_labels (ndarray): Labeled image array where each pearl/object is labeled with a unique integer.
            distance_threshold (float): The distance threshold for merging objects.
            min_size (int): Minimum number of pixels to consider a pearl. Smaller objects will be ignored.

        Returns:
            ndarray: Updated labeled image array with merged pearls.
        """
        self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.frangi.get(),
                                                 spacing=self.scaling)
        # print('len of pearl objects: ', len(self.pearl_objects))
        # Step 1: Calculate centroids for each pearl/object
        centroids = xp.array([region.centroid for region in self.pearl_objects])
        labels = xp.array([region.label for region in self.pearl_objects])

        if len(centroids.get()) == 0 or len(labels.get()) == 1:
            print("Not enough pearls found. Skipping hierarchical clustering.")
            # No pearls found, return the original labels
            return
        #print(f"Number of pearls before merging: {len(centroids)}")
        # print(f'centroids: {centroids}')
        # Step 2: Perform hierarchical clustering on centroids
        Z = linkage(centroids.get(), method='ward')  # Use 'ward' method for hierarchical clustering

        # Step 3: Assign clusters based on a distance threshold
        clusters = fcluster(Z, distance_threshold, criterion='distance')

        # Step 4: Merge the labels based on cluster assignments
        merged_labels = xp.zeros_like(self.relabelled_labels.get())
        new_label = 1  # Start labeling from 1

        for cluster_id in np.unique(clusters):
            # Find labels that belong to the current cluster
            cluster_labels = labels[clusters == cluster_id].get()
            if len(cluster_labels) > 1:  # If there is more than one label in the cluster, merge them
                # print(f"Merging labels: {cluster_labels} into new label: {new_label}")
                # Create a mask for the merged region
                merge_mask = np.isin(self.relabelled_labels.get(), cluster_labels)
                merged_labels[merge_mask] = new_label
                new_label += 1
            else:
                # print(f"Retaining label: {cluster_labels[0]}")
                # Retain the original label for single objects
                merged_labels[self.relabelled_labels.get() == cluster_labels[0]] = new_label
                new_label += 1

        self.relabelled_labels = merged_labels

        self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.frangi.get(),
                                                 spacing=self.scaling)

    # def calculate_volume(self, region, scaling):
    #     """Calculate the volume of a region scaled by voxel size."""
    #     return region.area * np.prod(scaling)

    # def calculate_volume(self, minor_axis_length, major_axis_length, medior_axis_length):
    #     """
    #     Calculate the volume of a region using an ellipsoid approximation.
    #
    #     Parameters:
    #     - minor_axis_length: Length of the minor axis of the region.
    #     - major_axis_length: Length of the major axis of the region.
    #     - medior_axis_length: Length of the medio axis (equivalent diameter) of the region.
    #
    #     Returns:
    #     - volume: The volume of the region assuming an ellipsoidal shape.
    #     """
    #     if self.is3D:
    #         # Convert full lengths to semi-axis lengths
    #         semi_minor = minor_axis_length / 2
    #         semi_major = major_axis_length / 2
    #         semi_medior = medior_axis_length / 2
    #
    #         # Calculate the volume of the ellipsoid
    #         volume = (4 / 3) * np.pi * semi_minor * semi_major * semi_medior
    #         return volume
    #     else:
    #         # Convert full lengths to semi-axis lengths
    #         semi_minor = minor_axis_length / 2
    #         semi_major = major_axis_length / 2
    #         # Calculate the area of the ellipse
    #         area = np.pi * semi_minor * semi_major
    #         return area

    def calculate_volume(self, region):
        """
        Calculate the volume of a given region.

        For 2D regions, volume is equivalent to the area.
        For 3D regions, volume is calculated as the number of voxels multiplied by the voxel size.

        Parameters:
        - region (RegionProperties): Region properties from skimage.measure.regionprops.

        Returns:
        - volume (float): Volume of the region (area in 2D or volume in 3D).
        """
        # Get spacing (physical pixel/voxel size) from self.scale.get()
        #spacing = self.scale.get()  # e.g., [z_spacing, y_spacing, x_spacing] for 3D

        # Check if the region is 2D or 3D
        if not self.is3D:  # 2D case
            # In 2D, the "volume" is simply the area of the region
            area = region.area  # This is the number of pixels in the region
            volume = area * self.scaling[0] * self.scaling[1]  # Scale area by pixel dimensions

        elif self.is3D:  # 3D case
            # In 3D, volume is the number of voxels multiplied by the voxel size
            voxel_count = region.area  # In 3D, region.area gives the number of voxels
            volume = voxel_count * self.scaling[0] * self.scaling[1] * self.scaling[2]  # Scale by voxel dimensions

        else:
            raise ValueError("Unsupported region dimensionality. Only 2D and 3D regions are supported.")

        return volume

    def calculate_axis_lengths(self, region):
        """Calculate minor and major axis lengths using eigenvalues."""
        try:
            eigenvalues = np.sort(region.inertia_tensor_eigvals)

            # Ensure eigenvalues are non-negative
            if eigenvalues[0] >= 0 and eigenvalues[1] >= 0:
                # Minor axis length
                if self.is3D:
                    minor_axis_length = np.sqrt(10 * (-eigenvalues[0] + eigenvalues[1] + eigenvalues[2]))
                else:
                    minor_axis_length = np.sqrt(10 * (-eigenvalues[0] + eigenvalues[1]))
            else:
                minor_axis_length = np.nan

            # Major axis length: square root of the largest eigenvalue
            if self.is3D and eigenvalues[2] >= 0:
                major_axis_length = np.sqrt(10 * eigenvalues[2])
            else:
                major_axis_length = np.nan

        except ValueError as e:
            print(f"Skipping region {region.label} due to ValueError: {e}")
            minor_axis_length, major_axis_length = np.nan, np.nan

        return minor_axis_length, major_axis_length

    def calculate_medio_axis_length(self, region):
        """Calculate the equivalent diameter (medio axis length)."""
        return region.equivalent_diameter

    # def calculate_surface_area_3d(self, mask, scaling):
    #     from scipy.ndimage import convolve
    #     """Calculate the surface area of a 3D region using convolution."""
    #     kernel = np.array([[[0, 0, 0],
    #                         [0, 1, 0],
    #                         [0, 0, 0]],
    #                        [[0, 1, 0],
    #                         [1, -6, 1],
    #                         [0, 1, 0]],
    #                        [[0, 0, 0],
    #                         [0, 1, 0],
    #                         [0, 0, 0]]])
    #     face_count = convolve(mask.astype(int), kernel, mode='constant', cval=0)
    #     exposed_faces = np.sum(face_count[mask] > 0)
    #     # print(f"Exposed faces: {exposed_faces}")
    #
    #     # Calculate surface area with voxel size scaling
    #     surface_area = exposed_faces * (scaling[0] * scaling[1] +
    #                                     scaling[1] * scaling[2] +
    #                                     scaling[0] * scaling[2])
    #     return surface_area

    def calculate_surface_area_marching_cubes(self, mask, voxel_size=(1, 1, 1)):
        from skimage.measure import marching_cubes, mesh_surface_area
        """
        Calculate the surface area of a 3D region using the marching cubes algorithm.

        Parameters:
        - mask: A binary 3D numpy array where the region is True and the background is False.
        - voxel_size: A tuple representing the size of each voxel in the x, y, and z dimensions.

        Returns:
        - surface_area: The surface area of the region in physical units.
        """
        # Compute the surface mesh using marching cubes
        verts, faces, _, _ = marching_cubes(mask, level=0, spacing=voxel_size)

        # Calculate the surface area using skimage's mesh_surface_area
        surface_area = mesh_surface_area(verts, faces)

        return surface_area
    def calculate_sphericity(self, volume, surface_area, perimeter):
        """Calculate the sphericity for both 2D and 3D regions."""
        if self.is3D:
            if surface_area > 0:  # Avoid division by zero
                sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area
                # print(f"Final Volume: {volume:.2f}, Surface area: {surface_area:.2f}, Sphericity: {sphericity:.2f}")
            else:
                sphericity = 0
        else:
            if perimeter > 0:  # Avoid division by zero
                sphericity = (4 * np.pi * volume) / (perimeter ** 2)  # Circularity in 2D
            else:
                sphericity = 0

        return sphericity

    def calculate_rectangularity(self, region):
        """
        Calculate the rectangularity of a given region.

        Rectangularity is defined as the ratio of the region's area (2D) or volume (3D)
        to the area or volume of its bounding box.

        Parameters:
        - region (RegionProperties): Region properties from skimage.measure.regionprops.

        Returns:
        - rectangularity (float): Rectangularity value (Area / Bounding Box Area for 2D,
                                  Volume / Bounding Box Volume for 3D).
        """
        # Get spacing (physical pixel size) from self.scale.get()
        spacing = self.scale.get()  # e.g., [z_spacing, y_spacing, x_spacing] for 3D

        # Check if the region is 2D or 3D
        if not self.is3D:  # 2D case
            # Calculate area of the region (in physical units)
            area = region.area  # Already in physical units because of spacing

            # Extract bounding box coordinates
            min_row, min_col, max_row, max_col = region.bbox

            # Calculate bounding box dimensions in physical units
            height = (max_row - min_row) * spacing[0]  # Assuming spacing[0] is y_spacing
            width = (max_col - min_col) * spacing[1]  # Assuming spacing[1] is x_spacing

            # Calculate area of the bounding box in physical units
            bbox_area = height * width

            # Calculate rectangularity
            if bbox_area > 0:
                rectangularity = area / bbox_area
            else:
                rectangularity = 0  # Avoid division by zero

        elif self.is3D:  # 3D case
            # Calculate volume of the region (in physical units)
            volume = region.area  # In 3D, region.area gives the volume in physical units

            # Extract bounding box coordinates
            min_depth, min_row, min_col, max_depth, max_row, max_col = region.bbox

            # Calculate bounding box dimensions in physical units
            depth = (max_depth - min_depth) * spacing[0]  # Assuming spacing[0] is z_spacing
            height = (max_row - min_row) * spacing[1]  # Assuming spacing[1] is y_spacing
            width = (max_col - min_col) * spacing[2]  # Assuming spacing[2] is x_spacing

            # Calculate volume of the bounding box in physical units
            bbox_volume = depth * height * width

            # Calculate rectangularity
            if bbox_volume > 0:
                rectangularity = volume / bbox_volume
            else:
                rectangularity = 0  # Avoid division by zero
        else:
            raise ValueError("Unsupported region dimensionality. Only 2D and 3D regions are supported.")

        return rectangularity

    import numpy as np

    def calculate_3d_eccentricity(self, region):
        """
        Calculate an approximate eccentricity for a 3D region.

        Eccentricity in 3D is approximated using the principal axes of the region,
        which are derived from the covariance matrix of the region's coordinates.

        Parameters:
        - region (RegionProperties): Region properties from skimage.measure.regionprops.

        Returns:
        - eccentricity (float): An approximation of the eccentricity of the 3D region.
                                Values closer to 0 indicate spherical regions, while
                                values closer to 1 indicate more elongated regions.
        """
        # Extract the coordinates of the region's voxels
        coords = region.coords

        # Calculate the covariance matrix of the voxel coordinates
        covariance_matrix = np.cov(coords, rowvar=False)

        # Perform eigenvalue decomposition of the covariance matrix
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)

        # Sort the eigenvalues (principal axes lengths)
        eigenvalues = np.sort(eigenvalues)

        # Calculate eccentricity as the ratio of the smallest and largest principal axes
        if eigenvalues[-1] > 0:
            eccentricity = np.sqrt(1 - (eigenvalues[0] / eigenvalues[-1]))
        else:
            eccentricity = 0  # In case of degenerate regions

        return eccentricity

    def calculate_eccentricity(self, region):
        """
        Calculate the eccentricity of a given region.

        Eccentricity is defined as the ratio of the distance between the foci of the ellipse (2D)
        or ellipsoid (3D) that has the same second-moments as the region to the length of its major axis.

        Parameters:
        - region (RegionProperties): Region properties from skimage.measure.regionprops.

        Returns:
        - eccentricity (float): Eccentricity value (0 for circular or spherical regions, closer to 1 for elongated regions).
        """
        # Check if the region is 2D or 3D
        if not self.is3D:  # 2D case
            # Eccentricity is a built-in property for 2D regions
            return region.eccentricity
        elif self.is3D:  # 3D case
            # Use custom eccentricity calculation for 3D regions
            return self.calculate_3d_eccentricity(region)
        else:
            raise ValueError("Unsupported region dimensionality. Only 2D and 3D regions are supported.")

    def calculate_solidity(self, label_image, region):
        from scipy.spatial import ConvexHull

        """
            Calculate the solidity of the max projection of a labeled 3D region along the z-dimension.
    
            Solidity is defined as the ratio of the area of the region to the area of its convex hull in 2D.
    
            Parameters:
            - label_image (ndarray): 3D labeled image where each label represents a region.
            - region (RegionProperties): Region properties from skimage.measure.regionprops for 3D regions.
    
            Returns:
            - solidity (float): Solidity value of the max projection (2D) of the region.
            """
        if not self.is3D:
            return region.solidity

        try:
            # Extract the label of the current region
            label = region.label

            # Project the label into 2D by taking the maximum across the z-dimension
            label_projection = np.max(label_image == label, axis=0)

            # Label the 2D projected region
            labeled_projection = measure.label(label_projection)

            # Get region properties from the 2D projection
            projection_props = measure.regionprops(labeled_projection)

            # Assuming we have one large connected component (projection of the region), get its solidity
            if len(projection_props) > 0:
                return projection_props[0].solidity

            return 0  # Return 0 if no region found

        except Exception as e:
            print(f"Error calculating solidity of max projection for region {region.label}: {e}")
            return 0

    def format_value(self, value, precision=2):
        """
        Formats the value to a string with the specified precision.
        If the value is not a valid number, it returns 'N/A'.

        Parameters:
        - value: The value to format.
        - precision: The number of decimal places.

        Returns:
        - Formatted string.
        """
        try:
            # Check if the value is a valid number
            formatted_value = f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            # Return 'N/A' if the value is not a number
            formatted_value = "N/A"
        return formatted_value

    def visualize_with_napari(self, pearl_labels, frangi_labels, matched_regions):
        """
        Visualize the pearl regions and frangi regions using Napari.

        Parameters:
        - pearl_labels (ndarray): Labeled regions from the pearl image.
        - frangi_labels (ndarray): Labeled regions from the frangi image.
        - matched_regions (list): List of matched frangi region properties.
        """

        # Convert frangi_labels to integer type (necessary for Napari Labels layer)
        frangi_labels = frangi_labels.astype(np.int32)

        # Create a Napari viewer
        viewer = napari.Viewer()

        # Add the raw image and labeled regions to Napari
        viewer.add_labels(pearl_labels, name="Pearl Labels", opacity=0.5)
        viewer.add_labels(frangi_labels, name="Frangi Labels", opacity=0.5)

        # Optional: Add centroids of matched regions as points for better visualization
        matched_centroids = np.array([region.centroid for region in matched_regions])
        viewer.add_points(matched_centroids, name="Matched Centroids", size=10, face_color="red")

        # Start the Napari event loop
        napari.run()

    def match_labels_in_frangi_image(self, show=False):
        """
        Label regions in the Frangi-filtered image based on the bounding boxes
        of the pearl objects without performing any matching based on centroids.

        Each bounding box from the pearl objects is assigned a unique label
        in the Frangi labels image. If no Frangi region is found for a pearl object,
        a blank value is assigned to maintain the mapping.

        Parameters:
        - show (bool): If True, visualize the raw image, Frangi image, Frangi mask,
                       and labeled regions using Napari.

        Returns:
        - frangi_labels (numpy.ndarray): Labeled Frangi image where each label corresponds
                                         to a bounding box from the pearl objects.
        """
        # Extract region properties from the relabelled pearl objects
        self.pearl_objects = measure.regionprops(
            self.relabelled_labels.get(),
            intensity_image=self.frangi.get(),
            spacing=self.scaling
        )

        # Initialize an empty label image for Frangi regions
        frangi_labels = np.zeros_like(self.raw.get(), dtype=np.int32)

        # Initialize a label counter
        current_label = 1
        missing_labels = []

        # Keep track of labels assigned to each pearl object
        pearl_to_frangi_mapping = []

        # Iterate over each pearl object to assign labels based on bounding boxes
        for i, region in enumerate(self.pearl_objects):
            # Extract the bounding box coordinates
            if self.is3D:
                min_depth, min_row, min_col, max_depth, max_row, max_col = region.bbox
                # Define the slice for the current bounding box
                frangi_slice = (
                    slice(min_depth, max_depth),
                    slice(min_row, max_row),
                    slice(min_col, max_col)
                )
            else:
                min_row, min_col, max_row, max_col = region.bbox
                # add  factor to size of label region
                # factor = 0.1
                # min_row = max(0, min_row - int((max_row - min_row) * factor))
                # max_row = min(self.raw.get().shape[0], max_row + int((max_row - min_row) * factor))
                # min_col = max(0, min_col - int((max_col - min_col) * factor))
                # max_col = min(self.raw.get().shape[1], max_col + int((max_col - min_col) * factor))
                frangi_slice = (
                    slice(min_row, max_row),
                    slice(min_col, max_col)
                )

            # Extract the Frangi mask within the bounding box
            frangi_mask_region = self.mask_frame.get()[frangi_slice]

            # Check if there are any non-zero values in the Frangi mask region
            if np.any(frangi_mask_region):
                # Assign the current label to the Frangi labels where the mask is True
                frangi_labels[frangi_slice] = np.where(
                    frangi_mask_region,
                    current_label,
                    frangi_labels[frangi_slice]
                )
                pearl_to_frangi_mapping.append(current_label)
            else:
                # No Frangi region found for this pearl region, assign blank value (e.g., -1)
                missing_labels.append(i+1)
                frangi_labels[frangi_slice] = -1
                pearl_to_frangi_mapping.append(-1)

            # Increment the label counter for the next region
            current_label += 1

        if show:
            # Visualization using Napari
            viewer = napari.Viewer()
            viewer.add_image(self.raw.get(), name='Raw Image', blending='additive', visible=True)
            viewer.add_image(self.frangi.get(), name='Frangi Image', blending='additive', visible=False)
            viewer.add_image(self.mask_frame.get(), name='Frangi Mask', blending='additive', visible=False)
            viewer.add_labels(frangi_labels, name='Labeled Raw Regions', opacity=0.5, visible=True)
            viewer.add_labels(self.relabelled_labels.get(), name='Pearl Labels', opacity=0.5, visible=False)
            napari.run()

        # Calculate regions in the Frangi labels
        frangi_regions = measure.regionprops(frangi_labels, intensity_image=self.raw.get(), spacing=self.scaling)
        # Create a set of valid labels that were identified by measure.regionprops
        valid_frangi_labels = {region.label for region in frangi_regions}

        # Check for missing regions in the frangi_labels
        for idx, label in enumerate(pearl_to_frangi_mapping):
            # If the label is -1 or the label was not found in valid_frangi_labels, insert None
            if label == -1 or label not in valid_frangi_labels:
                frangi_regions.insert(idx, None)
        # Error check: number of Frangi regions should match the number of pearl objects
        if len(frangi_regions) != len(self.pearl_objects):
            raise ValueError(
                f"Number of pearls in Frangi image ({len(frangi_regions)}) does not match the number of pearls in the raw image ({len(self.pearl_objects)})")

        # if len(missing_labels) > 0:
        #     print(f"Missing {missing_labels} Frangi labels of {len(self.pearl_objects)} pearls.")
        #     print(f'frangi region: {frangi_regions}')

        return frangi_labels, frangi_regions

    @dataclass
    class PearlMetrics:
        filename: str
        timepoint: int
        n_timepoints: int
        pre_pearl_timepoint: int
        pearl_timepoint: int
        post_pearl_timepoint: int
        num_pearls: int
        peak_peak_distance_mean: float
        peak_peak_distance_sd: float
        minor_axis_mean: float
        minor_axis_sd: float
        major_axis_mean: float
        major_axis_sd: float
        medio_axis_mean: float
        medio_axis_sd: float
        volume_mean: float
        volume_sd: float
        volume_sum: float
        sphericity_mean: float
        sphericity_sd: float
        rectangularity_mean: float
        rectangularity_sd: float
        eccentricity_mean: float
        eccentricity_sd: float
        solidity_mean: float
        solidity_sd: float
        t0_tortuosity_mean: float = None
        t2_tortuosity_mean: float = None
        t0_total_length: float = None
        t2_total_length: float = None
        t0_tubule_width_mean: float = None
        t2_tubule_width_mean: float = None
        t0_pearls_per_micron: float = None
        t2_pearls_per_micron: float = None
        time_to_peak: float = None
        time_to_recovery: float = None
        duration_of_event: float = None
        intensity_mean: float = None
        intensity_stdv: float = None
        intenstiy_max: float = None
        intensity_min: float = None
        intensity_sum: float = None
        tortuosity_mean: float = None
        tortuosity_stdv: float = None
        tubule_volume_mean: float = None
        tubule_volume_stdv: float = None
        tubule_volume_sum: float = None
        tubule_width_mean: float = None
        tubule_width_stdv: float = None
        tubule_width_sum: float = None
        tubule_length_mean: float = None
        tubule_length_stdv: float = None
        tubule_length_sum: float = None
        aspect_ratio_mean: float = None
        node_width_mean: float = None
        node_width_stdv: float = None
        node_width_sum: float = None
        lin_vel_mag_mean: float = None
        lin_vel_mag_stdv: float = None
        lin_acc_mag_mean: float = None
        lin_acc_mag_stdv: float = None
        ang_vel_mag_mean: float = None
        ang_vel_mag_stdv: float = None
        ang_acc_mag_mean: float = None
        ang_acc_mag_stdv: float = None
        label_count: int = None
        # lin_vel_mag_mean	lin_vel_mag_std_dev	lin_vel_mag_min	lin_vel_mag_max	lin_vel_mag_sum	lin_vel_mag_rel_mean
        # lin_vel_mag_rel_std_dev	lin_vel_mag_rel_min	lin_vel_mag_rel_max	lin_vel_mag_rel_sum	ang_vel_mag_mean
        # ang_vel_mag_std_dev	ang_vel_mag_min	ang_vel_mag_max	ang_vel_mag_sum	ang_vel_mag_rel_mean
        # ang_vel_mag_rel_std_dev	ang_vel_mag_rel_min	ang_vel_mag_rel_max	ang_vel_mag_rel_sum	lin_acc_mag_mean
        # lin_acc_mag_std_dev	lin_acc_mag_min	lin_acc_mag_max	lin_acc_mag_sum	lin_acc_mag_rel_mean
        # lin_acc_mag_rel_std_dev	lin_acc_mag_rel_min	lin_acc_mag_rel_max	lin_acc_mag_rel_sum	ang_acc_mag_mean
        # ang_acc_mag_std_dev	ang_acc_mag_min	ang_acc_mag_max	ang_acc_mag_sum	ang_acc_mag_rel_mean
        # ang_acc_mag_rel_std_dev	ang_acc_mag_rel_min	ang_acc_mag_rel_max	ang_acc_mag_rel_sum
        # directionality_rel_mean	directionality_rel_std_dev	directionality_rel_min	directionality_rel_max	directionality_rel_sum

    @dataclass
    class PearlMetricsList:
        filename: str
        timepoint: int
        label_id: int
        location_pixel: tuple[int, int, int]
        location_micron: tuple[float, float, float]
        peak_peak_distance: float
        minor_axis: float
        major_axis: float
        medio_axis: float
        volume: float
        perimeter: float
        surface_area: float
        sphericity: float
        rectangularity: float
        eccentricity: float
        solidity: float
        intensity_mean: float
        intensity_sd: float
        raw_mean: float
        raw_sd: float
        raw_eccentricity: float
        raw_rectangularity: float
        raw_solidity: float
        raw_volume: float
        raw_minor_axis_length: float
        raw_major_axis_length: float
        raw_medior_axis_length: float
        is_pearl: bool = False
        pearl_type: str = None

    def compute_pearlings_stats(self, timepoint=0):
        from skimage import measure, feature
        from scipy.ndimage import gaussian_filter
        from scipy.spatial import cKDTree
        """
        Computes pearling statistics for the given regions and peak labels using the eigenvalue approach,
        and calculates the sphericity for each pearl.

        Args:
            peaks (ndarray): Boolean array indicating peak locations.
            peak_labels (ndarray): Labeled array of peaks.
            timepoint (int): The timepoint being analyzed.

        Returns:
            dict: A dictionary containing computed pearling statistics.
        """

        # Ensure labels and intensity images are 3D and properly spaced
        self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.frangi.get(),
                                         spacing=self.scaling)

        # Collect centroids of pearls
        centroids = np.array([region.centroid for region in self.pearl_objects])
        show = False
        # if timepoint in self.select_frames:
        #     show = True
        frangi_labels, frangi_regions = self.match_labels_in_frangi_image(show=show)

        # Check if there are any pearls
        if len(centroids) > 0:
            # Scale centroids to physical units
            scaled_centroids = centroids #it is already in microns.... otherwise use * self.scale.get()
            # print(f"Centroids: {scaled_centroids}")

            # Compute nearest neighbor distances
            if len(scaled_centroids) > 1:
                centroid_tree = cKDTree(scaled_centroids)
                distances, indices = centroid_tree.query(scaled_centroids, k=2)
                nn_distances = distances[:, 1]  # Exclude self-distance at index 0
                nn_distance_mean = np.nanmean(nn_distances)
                nn_distance_sd = np.nanstd(nn_distances)
            else:
                nn_distances = np.array([0])
                nn_distance_mean = 0
                nn_distance_sd = 0
        else:
            # No pearls detected
            nn_distances = np.array([])
            nn_distance_mean = np.nan
            nn_distance_sd = np.nan
            scaled_centroids = np.array([])

        # Calculate properties for each region
        pearl_volumes = []
        pearl_minor_lengths = []
        pearl_major_lengths = []
        pearl_medior_lengths = []
        pearl_sphericities = []
        pearl_rectangularities = []
        pearl_eccentricities = []
        pearl_solidities = []
        points = []  # Store points for text annotation
        labels = []  # Store labels for text annotation
        pearl_label_metrics = []  # Store metrics for each pearl

        for i, region in enumerate(self.pearl_objects):
            # Calculate axis lengths
            if self.is3D:
                minor_axis_length, major_axis_length = self.calculate_axis_lengths(region)
                medior_axis_length = region.equivalent_diameter # self.calculate_medio_axis_length(region)
            else:
                minor_axis_length = region.minor_axis_length
                major_axis_length = region.major_axis_length
                medior_axis_length = region.equivalent_diameter
            rectangularity = self.calculate_rectangularity(region)
            eccentricity = self.calculate_eccentricity(region)
            solidity = self.calculate_solidity(self.relabelled_labels.get(), region)
            pearl_minor_lengths.append(minor_axis_length)
            pearl_major_lengths.append(major_axis_length)
            pearl_medior_lengths.append(medior_axis_length)
            pearl_rectangularities.append(rectangularity)
            pearl_eccentricities.append(eccentricity)
            pearl_solidities.append(solidity)

            # Calculate volume
            # volume = self.calculate_volume(minor_axis_length, major_axis_length, medior_axis_length)
            volume = self.calculate_volume(region)# region.area
            pearl_volumes.append(volume)

            # Attempt to get the corresponding Frangi region
            frangi_region = frangi_regions[i]
            if frangi_region is not None:
                # Extract parameters from the Frangi region
                frangi_intensity_mean = frangi_region.mean_intensity
                frangi_intensity_stdv = frangi_region.intensity_image.std()
                frangi_eccentricity = self.calculate_eccentricity(frangi_region)
                frangi_rectangularity = self.calculate_rectangularity(frangi_region)
                frangi_solidity = self.calculate_solidity(frangi_labels, frangi_region)
                frangi_minor_axis_length = frangi_region.minor_axis_length
                frangi_major_axis_length = frangi_region.major_axis_length
                frangi_medior_axis_length = frangi_region.equivalent_diameter
                frangi_volume = self.calculate_volume(frangi_region)
            else:  # Handle the case where there is no corresponding Frangi region
                frangi_intensity_mean = None
                frangi_intensity_stdv = None
                frangi_eccentricity = None
                frangi_rectangularity = None
                frangi_solidity = None
                frangi_minor_axis_length = None
                frangi_major_axis_length = None
                frangi_medior_axis_length = None
                frangi_volume = None
                # # Optional: Log or print a message
                # print(f"No Frangi region found for pearl region {i + 1} with centroid at {region.centroid}")

            # Calculate surface area or perimeter and sphericity
            if self.is3D:
                mask = (self.relabelled_labels.get() == region.label)  # Binary mask for the current region
                surface_area = self.calculate_surface_area_marching_cubes(mask)
                perimeter = surface_area
                # print(f"Pearl {region.label} surface area: {surface_area:.2f}")
                sphericity = self.calculate_sphericity(volume, surface_area, None)
            else:
                # print(f"Pearl {region.label} perimeter: {perimeter:.2f}")
                perimeter = region.perimeter
                surface_area = perimeter
                # print(f"Pearl {region.label} volume: {volume:.2f}")
                sphericity = self.calculate_sphericity(volume, None, perimeter)
            # print(f"Pearl {region.label} sphericity: {sphericity:.3f}")
            pearl_sphericities.append(sphericity)


            # Create label text with surface area, volume, and sphericity
            label_text = (
                f"SA: {self.format_value(surface_area)}, "
                f"Vol: {self.format_value(volume)}, "
                f"Sph: {self.format_value(sphericity)}, "
                f"Rect: {self.format_value(rectangularity)}, "
                f"Ecc: {self.format_value(eccentricity)}, "
                f"Sol: {self.format_value(solidity)}, "
                f"MinorL: {self.format_value(minor_axis_length)}, "
                f"MajorL: {self.format_value(major_axis_length)}, "
                f"MediorL: {self.format_value(medior_axis_length)}"
            )
            labels.append(label_text)
            scaled_centroid = region.centroid / self.scaling
            # if self.is3D:
            #     point_pixels = (region.centroid[0], region.centroid[1], region.centroid[2])
            #     point_microns = (scaled_centroid[0], scaled_centroid[1], scaled_centroid[2])
            # else:
            #     point_pixels = (region.centroid[0], region.centroid[1])
            #     point_microns = (scaled_centroid[0], scaled_centroid[1])
            points.append(scaled_centroid)
            intensity_mean = region.mean_intensity
            intensity_stdv = region.intensity_image.std()
            pearl_frames = self.pearl_frame,  self.pearl_frame + 1
            label_metrics = self.PearlMetricsList(
                filename=os.path.basename(self.im_path),
                timepoint=timepoint,
                label_id=region.label,
                location_pixel=scaled_centroid,
                location_micron=region.centroid,
                peak_peak_distance=nn_distances[i],
                minor_axis=minor_axis_length,
                major_axis=major_axis_length,
                medio_axis=medior_axis_length,
                volume=volume,
                sphericity=sphericity,
                rectangularity=rectangularity,
                eccentricity=eccentricity,
                solidity=solidity,
                surface_area=surface_area,
                perimeter=perimeter,
                intensity_mean=intensity_mean,
                intensity_sd=intensity_stdv,
                raw_mean=frangi_intensity_mean,
                raw_sd=frangi_intensity_stdv,
                raw_eccentricity=frangi_eccentricity,
                raw_rectangularity=frangi_rectangularity,
                raw_solidity=frangi_solidity,
                raw_minor_axis_length=frangi_minor_axis_length,
                raw_major_axis_length=frangi_major_axis_length,
                raw_medior_axis_length=frangi_medior_axis_length,
                raw_volume=frangi_volume,
                is_pearl=False,
                pearl_type='non_pearl' if timepoint not in pearl_frames else 'pearl'
            )
            pearl_label_metrics.append(label_metrics)


        # Calculate pearling statistics
        num_pearls = len(self.pearl_objects)
        # print(f"Number of pearls: {num_pearls}")
        nn_distance_mean = np.nanmean(nn_distances)
        pearl_volume_mean = np.nanmean(pearl_volumes)
        pearl_volume_sum = np.nansum(pearl_volumes)
        pearl_minor_axis_mean = np.nanmean(pearl_minor_lengths)
        pearl_major_axis_mean = np.nanmean(pearl_major_lengths)
        pearl_medio_axis_mean = np.nanmean(pearl_medior_lengths)
        pearl_sphericity_mean = np.nanmean(pearl_sphericities)  # Mean sphericity
        pearl_rectangularity_mean = np.nanmean(pearl_rectangularities)  # Mean rectangularity


        branch_features_path = self.im_info.pipeline_paths['features_branches']
        image_features_path = self.im_info.pipeline_paths['features_image']
        organelle_features_path = self.im_info.pipeline_paths['features_organelles']
        # open df of branch features path
        branch_feature_csv = pd.read_csv(branch_features_path)
        image_feature_csv = pd.read_csv(image_features_path)
        organelle_feature_csv = pd.read_csv(organelle_features_path)

        t0_branch_features = branch_feature_csv[branch_feature_csv['t'] == self.prepearl_frame]
        t2_branch_features = branch_feature_csv[branch_feature_csv['t'] == self.postpearl_frame]
        t0_image_features = image_feature_csv[image_feature_csv['t'] == self.prepearl_frame]
        t2_image_features = image_feature_csv[image_feature_csv['t'] == self.postpearl_frame]

        t0_tortuosity_mean = t0_image_features['branch_tortuosity_mean'].values[0]
        t2_tortuosity_mean = t2_image_features['branch_tortuosity_mean'].values[0]
        t0_total_length = t0_image_features['branch_length_sum'].values[0]
        t2_total_length = t2_image_features['branch_length_sum'].values[0]
        t0_tubule_width = t0_image_features['node_thickness_mean'].values[0]
        t2_tubule_width = t2_image_features['node_thickness_mean'].values[0]
        t0_pearls_per_micron = num_pearls / t0_total_length
        t2_pearls_per_micron = num_pearls / t2_total_length

        # Initialize tortuosity and other metrics dynamically based on the number of timepoints
        tortuosity_mean = image_feature_csv['branch_tortuosity_mean'].values
        tortuosity_stdv = image_feature_csv['branch_tortuosity_std_dev'].values
        # tortuosity_stdv = ', '.join(map(str, image_feature_csv['branch_tortuosity_std_dev'].values))

        intensity_mean = image_feature_csv['intensity_mean'].values
        intensity_stdv = image_feature_csv['intensity_std_dev'].values
        intensity_max = image_feature_csv['intensity_max'].values
        intensity_min = image_feature_csv['intensity_min'].values
        intensity_sum = image_feature_csv['intensity_sum'].values


        tubule_volume_mean = image_feature_csv['branch_area_mean'].values
        tubule_volume_stdv = image_feature_csv['branch_area_std_dev'].values
        tubule_volume_sum = image_feature_csv['branch_area_sum'].values

        tubule_length_mean = image_feature_csv['branch_length_mean'].values
        tubule_length_stdv = image_feature_csv['branch_length_std_dev'].values
        tubule_length_sum = image_feature_csv['branch_length_sum'].values

        asp_ratio_mean = image_feature_csv['branch_aspect_ratio_mean'].values

        node_width_mean = image_feature_csv['node_thickness_mean'].values
        node_width_stdv = image_feature_csv['node_thickness_std_dev'].values
        node_width_sum = image_feature_csv['node_thickness_sum'].values
        tubule_width_mean = image_feature_csv['branch_thickness_mean'].values
        tubule_width_stdv = image_feature_csv['branch_thickness_std_dev'].values
        tubule_width_sum = image_feature_csv['branch_thickness_sum'].values


        # velocity values
        lin_vel_mag_mean = image_feature_csv['lin_vel_mag_mean'].values
        lin_vel_mag_stdv = image_feature_csv['lin_vel_mag_std_dev'].values
        lin_acc_mag_mean = image_feature_csv['lin_acc_mag_mean'].values
        lin_acc_mag_stdv = image_feature_csv['lin_acc_mag_std_dev'].values
        ang_vel_mag_mean = image_feature_csv['ang_vel_mag_mean'].values
        ang_vel_mag_stdv = image_feature_csv['ang_vel_mag_std_dev'].values
        ang_acc_mag_mean = image_feature_csv['ang_acc_mag_mean'].values
        ang_acc_mag_stdv = image_feature_csv['ang_acc_mag_std_dev'].values

        # get number of mitochondria in the image
        # loop through each timepoint and get the number of mitochondria
        organelle_feature_csv = organelle_feature_csv[organelle_feature_csv['t'] == timepoint]
        label_count = organelle_feature_csv['label'].nunique()
        # print(f"Number of mitochondria in the image at t={timepoint}: {label_count}")


        if self.select_frames is None:
            self.select_frames = [0, 1, 2]
        time_interval = self.file_info.dim_res['T']
        initial_time = self.prepearl_frame * time_interval # - time_interval
        peak_time = self.pearl_frame * time_interval # - time_interval
        final_time = self.postpearl_frame * time_interval # - time_interval

        time_to_peak = peak_time - initial_time
        time_to_recovery = final_time - peak_time
        duration_of_event = final_time - initial_time

        n_timepoints = self.file_info.t_end + 1

        pearl_metrics = self.PearlMetrics(
            filename=os.path.basename(self.im_path), timepoint=timepoint, n_timepoints=n_timepoints,
            pre_pearl_timepoint=self.prepearl_frame, pearl_timepoint=self.pearl_frame, post_pearl_timepoint=self.postpearl_frame,
            num_pearls=num_pearls,
            peak_peak_distance_mean=nn_distance_mean,
            peak_peak_distance_sd=stats.stdev(nn_distances) if len(pearl_minor_lengths) > 1 else 0,
            minor_axis_mean=pearl_minor_axis_mean,
            minor_axis_sd=stats.stdev(pearl_minor_lengths) if len(pearl_minor_lengths) > 1 else 0,
            major_axis_mean=pearl_major_axis_mean,
            major_axis_sd=stats.stdev(pearl_major_lengths) if len(pearl_minor_lengths) > 1 else 0,
            medio_axis_mean=pearl_medio_axis_mean,
            medio_axis_sd=stats.stdev(pearl_medior_lengths) if len(pearl_minor_lengths) > 1 else 0,
            sphericity_mean=pearl_sphericity_mean,
            sphericity_sd=stats.stdev(pearl_sphericities) if len(pearl_minor_lengths) > 1 else 0,
            rectangularity_mean=pearl_rectangularity_mean,
            rectangularity_sd=stats.stdev(pearl_rectangularities) if len(pearl_rectangularities) > 1 else 0,
            eccentricity_mean=np.nanmean(pearl_eccentricities),
            eccentricity_sd=stats.stdev(pearl_eccentricities) if len(pearl_eccentricities) > 1 else 0,
            solidity_mean=np.nanmean(pearl_solidities),
            solidity_sd=stats.stdev(pearl_solidities) if len(pearl_solidities) > 1 else 0,
            volume_mean=pearl_volume_mean,
            volume_sd=stats.stdev(pearl_volumes) if len(pearl_minor_lengths) > 1 else 0,
            volume_sum=pearl_volume_sum,
            t0_tortuosity_mean=t0_tortuosity_mean, t2_tortuosity_mean=t2_tortuosity_mean,
            t0_total_length=t0_total_length, t2_total_length=t2_total_length,
            t0_tubule_width_mean=t0_tubule_width, t2_tubule_width_mean=t2_tubule_width,
            t0_pearls_per_micron=t0_pearls_per_micron, t2_pearls_per_micron=t2_pearls_per_micron,
            time_to_peak=time_to_peak, time_to_recovery=time_to_recovery, duration_of_event=duration_of_event,
            intensity_mean=intensity_mean[timepoint], intensity_stdv=intensity_stdv[timepoint],
            intenstiy_max=intensity_max[timepoint], intensity_min=intensity_min[timepoint], intensity_sum=intensity_sum[timepoint],
            tortuosity_mean=tortuosity_mean[timepoint], tortuosity_stdv=tortuosity_stdv[timepoint],
            tubule_volume_mean=tubule_volume_mean[timepoint], tubule_volume_stdv=tubule_volume_stdv[timepoint], tubule_volume_sum=tubule_volume_sum[timepoint],
            tubule_width_mean=tubule_width_mean[timepoint], tubule_width_stdv=tubule_width_stdv[timepoint], tubule_width_sum=tubule_width_sum[timepoint],
            node_width_mean=node_width_mean[timepoint], node_width_stdv=node_width_stdv[timepoint], node_width_sum=node_width_sum[timepoint],
            tubule_length_mean=tubule_length_mean[timepoint], tubule_length_stdv=tubule_length_stdv[timepoint], tubule_length_sum=tubule_length_sum[timepoint],
            aspect_ratio_mean=asp_ratio_mean[timepoint],
            lin_vel_mag_mean=lin_vel_mag_mean[timepoint], lin_vel_mag_stdv=lin_vel_mag_stdv[timepoint],
            lin_acc_mag_mean=lin_acc_mag_mean[timepoint], lin_acc_mag_stdv=lin_acc_mag_stdv[timepoint],
            ang_vel_mag_mean=ang_vel_mag_mean[timepoint], ang_vel_mag_stdv=ang_vel_mag_stdv[timepoint],
            ang_acc_mag_mean=ang_acc_mag_mean[timepoint], ang_acc_mag_stdv=ang_acc_mag_stdv[timepoint],
            label_count=label_count
        )
        return pearl_metrics, pearl_label_metrics, points, labels

    def plot_pearls_over_time(self, pearl_metrics, y_column='num_pearls', sd=None, show_plot=False, verbose=False):
        import matplotlib.pyplot as plt
        import os  # Ensure os is imported for os.path.join

        """
        Plots the number of pearls over time using the provided pearl metrics data and saves the plot as JPEG and SVG.

        Parameters:
        - pearl_metrics: A dictionary containing pearl metrics for each frame. 
                         Each dictionary should have a 'frame' key and a 'num_pearls' key.
        - y_column: The column in pearl_metrics to plot on the y-axis (default is 'num_pearls').
        - sd: The column name in pearl_metrics containing standard deviation values for error bars.
        - show_plot: Whether to display the plot interactively (default is False).
        - verbose: Whether to print information messages (default is True).
        """
        # Extract frame numbers and number of pearls from the pearl metrics
        frames = pearl_metrics['timepoint']
        num_pearls = pearl_metrics[y_column]

        # Plot the number of pearls over time
        plt.figure(figsize=(10, 6))
        plt.plot(frames, num_pearls, marker='o', linestyle='-', color='b')
        if sd is not None:
            sd_values = pearl_metrics[sd]
            plt.errorbar(frames, num_pearls, yerr=sd_values, fmt='o', color='b', ecolor='r', capsize=5)

        plt.xlabel('Frame Number (Time)')
        plt.ylabel(y_column)
        plt.title(f'{y_column} Over Time')
        plt.grid(True)

        # Save the plot as JPEG and SVG
        save_path = os.path.join(self.save_path, f"{y_column}_over_time")
        plt.savefig(f"{save_path}.jpeg", format='jpeg', dpi=300)
        plt.savefig(f"{save_path}.svg", format='svg')

        if verbose:
            print(f"Plots saved as {save_path}.jpeg and {save_path}.svg")

        # Show the plot if requested
        if show_plot:
            plt.show()

        # Close the plot to free up memory
        plt.close()


if __name__ == "__main__":
    im_path = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-09-03_demo\event4_2022-12-01_16-48-40_000_COX8A_CellLigt-ER-RFP_300volumes_beading\crop1_snout\crop1.ome.tif"
    nellie_path = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-09-03_demo\event4_2022-12-01_16-48-40_000_COX8A_CellLigt-ER-RFP_300volumes_beading\crop1_snout\crop1_nellie_out"
    pearling_stats_path = os.path.join(nellie_path, "pearling_stats")
    if not os.path.exists(pearling_stats_path):
        os.makedirs(pearling_stats_path)
    # im_path = r"E:\SNOUTY_data\Pearling-examples\austin-pearling-metrics-example-3d-snouty\crop2.ome.tif"
    # nellie_path = r"E:\SNOUTY_data\Pearling-examples\austin-pearling-metrics-example-3d-snouty\nellie-crop2"
    mask = xp.load(os.path.join(nellie_path, 'mask.npy'))
    pearl_stats = PearlingStats(im_path, nellie_path, save_path=pearling_stats_path,mask=mask, select_frames=[33,37,46], visualize=True)
    pearl_stats.run_pearling_stats()