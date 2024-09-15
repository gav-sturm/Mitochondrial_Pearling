from dataclasses import dataclass

import tifffile
import napari
import cupy as xp
import cupyx.scipy.ndimage as ndi
import thresholding
import numpy as np

from scipy.spatial import cKDTree
import skimage.measure as measure
import os
from nellie.im_info.verifier import FileInfo, ImInfo
import pandas as pd
import re
from tqdm import tqdm


class PearlingStats:
    def __init__(self, im_path, nellie_path, mask=None, save_path=None, select_frames=None, visualize=False):
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
        self.relabelled_labels = None
        self.scaling = None
        self.raw = None
        self.is3D = False
        self.scale = None
        self.z_ratio = None
        self.average_bead_radius_um = None
        self.frangi = None
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

    # def find_files_with_last_char_number(self, directory):
    #     matching_files = []
    #     # Regular expression to check if the last character before the extension is a number
    #     pattern = re.compile(r'.*\d\.ome.tif$')
    #
    #     # Walk through the directory and its subdirectories
    #     for root, _, files in os.walk(directory):
    #         for file in files:
    #             # Check if the last character before the extension is a digit
    #             if pattern.match(file):
    #                 matching_files.append(os.path.join(root, file))
    #     if len(matching_files) == 0:
    #         raise FileNotFoundError(f'No files found in {directory} with a number before the extension.')
    #     return matching_files


    # def run_pearling_stats(self):
    #     viewer = napari.Viewer() if self.visualize else None
    #     self.file_info = FileInfo(filepath=self.im_path, output_dir=self.nellie_path)
    #     self.file_info.find_metadata()
    #     self.file_info.load_metadata()
    #     self.im_info = ImInfo(self.file_info)
    #     # print(f'image pipeline paths: {self.im_info.pipeline_paths}')
    #     # print(f'image pipeline paths keys: {self.im_info.pipeline_paths.keys()}')
    #     #ome_path = self.find_files_with_last_char_number(self.nellie_path)[0]
    #     ome_path =  self.im_info.im_path
    #     # print(f'ome path: {ome_path}')
    #     self.raw = xp.array(self.im_info.get_memmap(ome_path)[self.pearl_frame]) # self.im_info.pipeline_paths['im_preprocessed']
    #     viewer.add_image(self.raw.get(), name='raw image') if self.visualize else None
    #     frangi_path = self.find_file_in_subdirectories(self.nellie_path, 'im_preprocessed')[0]
    #     # print(f'frangi path: {frangi_path}')
    #     self.frangi = xp.array(self.im_info.get_memmap(frangi_path)[self.pearl_frame])
    #     # crop off edges of frangi image
    #     # from nellie.segmentation.filtering import crop_edges
    #     # determine if 3D or 2D
    #     if 'Z' in self.file_info.axes:
    #         self.is3D = True
    #         self.scale = xp.array([self.file_info.dim_res['Z'], self.im_info.dim_res['Y'], self.im_info.dim_res['X']])
    #         self.z_ratio = self.scale[1] / self.scale[0]
    #     else:
    #         self.is3D = False
    #         self.scale = xp.array([self.im_info.dim_res['Y'], self.im_info.dim_res['X']])
    #     self.average_bead_radius_um = 0.25
    #
    #
    #     # apply mask to frangi
    #     if self.mask is not None:
    #         self.frangi *= self.mask
    #
    #     # Thresholding and edge detection
    #     minotri_thresh = thresholding.minotri_threshold(self.frangi[self.frangi > 0])
    #     mask_frame = self.frangi > minotri_thresh
    #
    #     smoothed_distance = self.get_distances(mask_frame)
    #     if self.visualize:
    #         layer = viewer.add_image(self.frangi.get(), name='frangi image')
    #         layer.visible = False
    #         layer = viewer.add_image(smoothed_distance.get(), name='smoothed distances')
    #         layer.visible = False
    #
    #
    #     # footprint should be 3D, with length/width being 2*average_bead_radius_um / self.scale[1] and height being 2*average_bead_radius_um / self.scale[0]
    #     # footprint = xp.ones((int(2 * average_bead_radius_um / self.scale[0]), int(2 * average_bead_radius_um / self.scale[1]), int(2 * average_bead_radius_um / self.scale[1])))
    #     if self.is3D:
    #         footprint = xp.ones((int(2 * self.average_bead_radius_um / self.scale[1]), int(2 * self.average_bead_radius_um / self.scale[1]),
    #                          int(2 * self.average_bead_radius_um / self.scale[1])))
    #     else:
    #         footprint = xp.ones((int(2 * self.average_bead_radius_um / self.scale[1]), int(2 * self.average_bead_radius_um / self.scale[1])))
    #
    #     max_filtered = ndi.maximum_filter(smoothed_distance, footprint=footprint)
    #     peaks = (max_filtered == smoothed_distance) * mask_frame
    #     peak_labels, num_peaks = ndi.label(peaks)
    #     if self.visualize:
    #         layer = viewer.add_image(max_filtered.get(), name='max filter')
    #         layer.visible = False
    #         layer = viewer.add_image(peaks.get(), name="peaks")
    #         layer.visible = False
    #         layer = viewer.add_labels(peak_labels.get(), name='peak label')
    #         layer.visible = False
    #
    #     # reassign labels
    #     self.reassign_labels(peak_labels, peaks, mask_frame)
    #     #self.relabelled_labels = peak_labels.get()
    #
    #     self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.raw.get(),
    #                                         spacing=self.scaling)
    #
    #     # remove small pearls
    #     self.remove_small_pearls()
    #
    #     # merge close pearls
    #     #self.merge_close_pearls(neighbor_threshold=50, max_distance=1.5)
    #     distance_threshold = 0.5  # Adjust this based on your data, higher mean more merging
    #     # self.merge_by_hierarchical_clustering(distance_threshold)
    #
    #     print(f'# of pearl objects: {len(self.pearl_objects)}')
    #
    #     # add relabelled labels to viewer
    #     viewer.add_labels(self.relabelled_labels.get(), name='relabelled labels') if self.visualize else None
    #     viewer.layers['relabelled labels'].rendering = 'translucent'
    #
    #     # Set to 3D mode
    #     if self.visualize:
    #         viewer.dims.ndisplay = 3
    #         napari.run()
    #
    #     pm_data_arranged, pearl_metrics = self.compute_pearlings_stats(peaks, peak_labels)
    #     # save as a csv in the same directory
    #     pearl_metrics_df = pd.DataFrame(pearl_metrics.__dict__, index=[0])
    #     self.save_path = os.path.join(self.nellie_path, "pearling_metrics.csv")
    #     pearl_metrics_df.to_csv(self.save_path, index=False)
    #
    #     pm_da_df = pd.DataFrame(pm_data_arranged)
    #     self.save_path2 = os.path.join(self.nellie_path, "pearling_metrics_time_arranged.csv")
    #     pm_da_df.to_csv(self.save_path2, index=False)
    #
    #     print(f'Pearling metrics saved to {self.save_path} and {self.save_path2}')
    #     return pm_da_df, pearl_metrics_df

    def run_pearling_stats(self):
        viewer = napari.Viewer() if self.visualize else None
        self.file_info = FileInfo(filepath=self.im_path, output_dir=self.nellie_path)
        self.file_info.find_metadata()
        self.file_info.load_metadata()
        self.im_info = ImInfo(self.file_info)
        ome_path = self.im_info.im_path

        # Get total number of frames (time points)
        total_frames = self.file_info.t_end + 1
        print(f'Total frames: {total_frames}')

        # Prepare to store results across all frames
        all_raw = []
        all_frangi = []
        all_smoothed_distances = []
        all_peaks = []
        all_peak_labels = []
        all_pearls = []
        all_labels = []
        all_centroids = []

        # Store pearling metrics for each frame
        all_pm_data_arranged = []
        all_pearl_metrics = []

        frangi_path = self.find_file_in_subdirectories(self.nellie_path, 'im_preprocessed')[0]
        self.average_bead_radius_um = 0.25

        for frame in tqdm(range(total_frames)):
            # print(f'Processing frame {frame + 1} of {total_frames}')

            # Load raw image and frangi filter for the current frame
            self.raw = xp.array(self.im_info.get_memmap(ome_path)[frame])
            self.frangi = xp.array(self.im_info.get_memmap(frangi_path)[frame])

            # Apply mask to frangi if present
            if self.mask is not None:
                self.frangi *= self.mask

            # Determine if 3D or 2D
            if 'Z' in self.file_info.axes:
                self.is3D = True
                print(f'3D data: {self.file_info.axes}')
                self.scale = xp.array(
                    [self.file_info.dim_res['Z'], self.im_info.dim_res['Y'], self.im_info.dim_res['X']])
                self.z_ratio = self.scale[1] / self.scale[0]
            else:
                self.is3D = False
                print(f'2D data: {self.file_info.axes}')
                self.scale = xp.array([self.im_info.dim_res['Y'], self.im_info.dim_res['X']])

            # Thresholding and edge detection
            minotri_thresh = thresholding.minotri_threshold(self.frangi[self.frangi > 0])
            mask_frame = self.frangi > minotri_thresh

            # Distance transform
            smoothed_distance = self.get_distances(mask_frame)

            # Footprint for maximum filter
            if self.is3D:
                footprint = xp.ones((int(2 * self.average_bead_radius_um / self.scale[1]),
                                     int(2 * self.average_bead_radius_um / self.scale[1]),
                                     int(2 * self.average_bead_radius_um / self.scale[1])))
            else:
                footprint = xp.ones((int(2 * self.average_bead_radius_um / self.scale[1]),
                                     int(2 * self.average_bead_radius_um / self.scale[1])))

            # Peak detection
            max_filtered = ndi.maximum_filter(smoothed_distance, footprint=footprint)
            peaks = (max_filtered == smoothed_distance) * mask_frame
            peak_labels, num_peaks = ndi.label(peaks)

            # Reassign labels
            self.reassign_labels(peak_labels, peaks, mask_frame)

            # Collect data for each frame
            all_raw.append(self.raw)  # Keep as CuPy array
            all_frangi.append(self.frangi)  # Keep as CuPy array
            all_smoothed_distances.append(smoothed_distance)  # Keep as CuPy array
            all_peaks.append(peaks)  # Keep as CuPy array
            all_peak_labels.append(peak_labels)  # Keep as CuPy array

            # Store pearl objects and remove small pearls
            self.remove_small_pearls()

            # merge close pearls
            #self.merge_close_pearls(neighbor_threshold=50, max_distance=1.5)
            distance_threshold = 1  # Adjust this based on your data, higher mean more merging
            self.merge_by_hierarchical_clustering(distance_threshold)

            all_pearls.append(self.relabelled_labels)  # Keep as CuPy array

            pm_data_arranged, pearl_metrics, centroids, labels = self.compute_pearlings_stats(peaks, peak_labels, timepoint=frame)
            all_pm_data_arranged.append(pm_data_arranged)
            all_pearl_metrics.append(pearl_metrics.__dict__)
            all_labels.append(labels)
            all_centroids.append(centroids)


        # Visualize all collected data at the end
        if self.visualize:
            viewer.add_image(xp.stack(all_raw).get(), name='raw images')
            viewer.add_image(xp.stack(all_frangi).get(), name='frangi images', visible=False)
            viewer.add_image(xp.stack(all_smoothed_distances).get(), name='smoothed distances', visible=False)
            viewer.add_image(xp.stack(all_peaks).get(), name='peaks', visible=False)
            viewer.add_labels(xp.stack(all_peak_labels).get(), name='peak labels', visible=False)
            viewer.add_labels(xp.stack(all_pearls).get(), name='pearls')

            # print(f'shape of centroids: {all_centroids}')
            # print(f'shape of labels: {all_labels}')

            # Initialize lists to store all points and labels
            all_points = []
            all_labs = []

            # Loop through each frame in the timeseries
            for frame_index, (frame_centroids, frame_labels) in enumerate(zip(all_centroids, all_labels)):
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
                # print(f'frame {frame_index} points: {points}, labels: {formatted_labels}')
                # print(f'len of points: {len(points)}, len of labels: {len(formatted_labels)}')
                all_points.extend(points)
                all_labs.extend(formatted_labels)

            # Convert all points and text positions to a single NumPy array
            all_points = np.array(all_points)

            # Add all points as a single layer with labels
            viewer.add_points(
                all_points,
                size=1,  # Adjust point size as needed
                face_color='red',  # Adjust point color as needed
                text={'string': all_labs, 'anchor': 'lower_right', 'translation': (0, 1, 1), 'size': 5},
                name='metric information'
            )
            #viewer.dims.ndisplay = 3
            napari.run()

        # Save the metrics as CSV files
        pearl_metrics_df = pd.DataFrame(all_pearl_metrics)
        save_path = os.path.join(self.save_path, "pearling_metrics_v2.csv")
        pearl_metrics_df.to_csv(save_path, index=False)

        pm_da_df = pd.DataFrame(all_pm_data_arranged)
        save_path2 = os.path.join(self.save_path, "pearling_metrics_time_arranged_v2.csv")
        pm_da_df.to_csv(save_path2, index=False)

        print(f'Pearling metrics saved to {save_path} and {save_path2}')

        # plot the number of pearls over time

        self.plot_pearls_over_time(pearl_metrics_df, y_column='num_pearls')
        self.plot_pearls_over_time(pearl_metrics_df, y_column='pearl_sphericity_mean')
        self.plot_pearls_over_time(pearl_metrics_df, y_column='pearl_major_axis_mean')
        self.plot_pearls_over_time(pearl_metrics_df, y_column='pearl_minor_axis_mean')
        self.plot_pearls_over_time(pearl_metrics_df, y_column='pearl_medio_axis_mean')

        return pm_da_df, pearl_metrics_df

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
        structure = xp.ones((3, 3, 3))
        if not self.is3D:
            structure = xp.ones((3, 3))

        self.relabelled_labels = peak_labels.copy()
        branch_skel_labels = self.relabelled_labels.copy()
        peak_mask_int = peaks.astype('uint16')
        label_mask_int = mask_frame.astype('uint16')
        peak_border = (ndi.binary_dilation(peak_mask_int, iterations=1,
                                           structure=structure) ^ peak_mask_int) * label_mask_int
        peak_label_mask = (peak_labels > 0)
        peak_label_mask = peak_label_mask.get()

        vox_matched = np.argwhere(peak_label_mask)
        vox_next_unmatched = np.argwhere(peak_border.get())

        unmatched_diff = np.inf
        self.scaling = self.scale.get()
        while True:
            num_unmatched = len(vox_next_unmatched)
            if num_unmatched == 0:
                break
            tree = cKDTree(vox_matched * self.scaling)
            dists, idxs = tree.query(vox_next_unmatched * self.scaling, k=1, workers=-1)
            # remove any matches that are too far away
            max_dist = 2 * np.min(self.scaling)  # sqrt 3 * max self.scaling
            unmatched_matches = np.array(
                [[vox_matched[idx], vox_next_unmatched[i]] for i, idx in enumerate(idxs) if dists[i] < max_dist]
            )
            if len(unmatched_matches) == 0:
                break
            matched_labels = branch_skel_labels[tuple(np.transpose(unmatched_matches[:, 0]))]
            self.relabelled_labels[tuple(np.transpose(unmatched_matches[:, 1]))] = matched_labels
            branch_skel_labels = self.relabelled_labels.copy()
            self.relabelled_labels_mask = self.relabelled_labels > 0

            self.relabelled_labels_mask_cpu = self.relabelled_labels_mask.get()

            vox_matched = np.argwhere(self.relabelled_labels_mask_cpu)
            relabelled_mask = self.relabelled_labels_mask.astype('uint8')
            # add unmatched matches to coords_matched
            peak_border = (ndi.binary_dilation(relabelled_mask, iterations=1,
                                               structure=structure) ^ relabelled_mask) * label_mask_int

            vox_next_unmatched = np.argwhere(peak_border.get())

            new_num_unmatched = len(vox_next_unmatched)
            unmatched_diff_temp = abs(num_unmatched - new_num_unmatched)
            if unmatched_diff_temp == unmatched_diff:
                break
            unmatched_diff = unmatched_diff_temp
            # print(f'Reassigned {unmatched_diff}/{num_unmatched} unassigned voxels. '
            #       f'{new_num_unmatched} remain.')

        # remove any objects with labels touching the edges of the image
        pearl_objects_not_spaced = measure.regionprops(self.relabelled_labels.get())
        if self.is3D:
            # For 3D data
            max_z, max_y, max_x = self.relabelled_labels.shape
            for region in pearl_objects_not_spaced:
                bbox = region.bbox
                # Check if any part of the bounding box is on the edge of the 3D volume
                if bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] <= 0 or \
                        bbox[3] + 1 >= max_z or bbox[4] + 1 >= max_y or bbox[5] + 1 >= max_x:
                    self.relabelled_labels[self.relabelled_labels == region.label] = 0
                    # print(f'Removed object {region.label} due to touching edge of image.')
        else:
            # For 2D data
            max_y, max_x = self.relabelled_labels.shape
            for region in pearl_objects_not_spaced:
                bbox = region.bbox
                # Check if any part of the bounding box is on the edge of the 2D image
                if bbox[0] <= 0 or bbox[1] <= 0 or \
                        bbox[2] + 1 >= max_y or bbox[3] + 1 >= max_x:
                    self.relabelled_labels[self.relabelled_labels == region.label] = 0
                    # print(f'Removed object {region.label} due to touching edge of image.')

    def remove_small_pearls(self):
        """
        Removes pearls (regions) where the average bead radius is below half the defined average bead radius.

        The function computes the bead radius for each pearl using its minor axis length or equivalent diameter.
        Regions that do not meet the criteria are removed by setting their labels to 0.
        """
        # Ensure relabelled_labels and raw images are available
        if self.relabelled_labels is None or self.raw is None:
            raise ValueError("Labelled image or raw image is not defined.")

        self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.raw.get(),
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
        self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.raw.get(),
                                                 spacing=self.scaling)
        # print('len of pearl objects after removal: ', len(self.pearl_objects))

        # print("Completed removal of small pearls.")


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
        # print('len of pearl objects: ', len(self.pearl_objects))
        # Step 1: Calculate centroids for each pearl/object
        centroids = xp.array([region.centroid for region in self.pearl_objects])
        labels = xp.array([region.label for region in self.pearl_objects])

        if len(centroids) == 0:
            print("No pearls found. Skipping hierarchical clustering.")
            # No pearls found, return the original labels
            return

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

        self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.raw.get(),
                                            spacing=self.scaling)
        # print('len of pearl objects after merging: ', len(self.pearl_objects))



    # def calculate_volume(self, region, scaling):
    #     """Calculate the volume of a region scaled by voxel size."""
    #     return region.area * np.prod(scaling)

    def calculate_volume(self, minor_axis_length, major_axis_length, medior_axis_length):
        """
        Calculate the volume of a region using an ellipsoid approximation.

        Parameters:
        - minor_axis_length: Length of the minor axis of the region.
        - major_axis_length: Length of the major axis of the region.
        - medior_axis_length: Length of the medio axis (equivalent diameter) of the region.

        Returns:
        - volume: The volume of the region assuming an ellipsoidal shape.
        """
        if self.is3D:
            # Convert full lengths to semi-axis lengths
            semi_minor = minor_axis_length / 2
            semi_major = major_axis_length / 2
            semi_medior = medior_axis_length / 2

            # Calculate the volume of the ellipsoid
            volume = (4 / 3) * np.pi * semi_minor * semi_major * semi_medior

            return volume
        else:
            # Convert full lengths to semi-axis lengths
            semi_minor = minor_axis_length / 2
            semi_medior = medior_axis_length / 2

            # Calculate the area of the ellipse
            area = np.pi * semi_minor * semi_medior
            return area


    def calculate_axis_lengths(self, region, is3D):
        """Calculate minor and major axis lengths using eigenvalues."""
        try:
            eigenvalues = np.sort(region.inertia_tensor_eigvals)

            # Ensure eigenvalues are non-negative
            if eigenvalues[0] >= 0 and eigenvalues[1] >= 0:
                # Minor axis length
                if is3D:
                    minor_axis_length = np.sqrt(10 * (-eigenvalues[0] + eigenvalues[1] + eigenvalues[2]))
                else:
                    minor_axis_length = np.sqrt(10 * (-eigenvalues[0] + eigenvalues[1]))
            else:
                minor_axis_length = np.nan

            # Major axis length: square root of the largest eigenvalue
            if is3D and eigenvalues[2] >= 0:
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

    @dataclass
    class PearlMetrics:
        filename: str
        timepoint: int
        n_timepoints: int
        peak_peak_distance_mean: float
        pearl_volume_mean: float
        pearl_minor_axis_mean: float
        pearl_major_axis_mean: float
        pearl_medio_axis_mean: float
        num_pearls: int
        pearl_sphericity_mean: float
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



    def compute_pearlings_stats(self, peaks, peak_labels, timepoint=0):
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
        self.pearl_objects = measure.regionprops(self.relabelled_labels.get(), intensity_image=self.raw.get(),
                                         spacing=self.scaling)

        # Find nearest neighboring peaks
        peak_coords = xp.argwhere(peaks)
        if self.is3D:
            peak_labels_at_coords = peak_labels[peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2]]
        else:
            peak_labels_at_coords = peak_labels[peak_coords[:, 0], peak_coords[:, 1]]
        peak_tree = cKDTree(peak_coords.get() * self.scaling)
        distances, indices = peak_tree.query(peak_coords.get() * self.scaling, k=2, workers=-1)
        nn_distances = distances[:, 1]

        # Calculate properties for each region
        pearl_volumes = []
        pearl_minor_lengths = []
        pearl_major_lengths = []
        pearl_medior_lengths = []
        pearl_sphericities = []
        points = []  # Store points for text annotation
        labels = []  # Store labels for text annotation

        for region in self.pearl_objects:
            # Calculate axis lengths
            minor_axis_length, major_axis_length = self.calculate_axis_lengths(region, self.is3D)
            # print(f"Pearl {region.label} minor axis: {minor_axis_length:.2f}, major axis: {major_axis_length:.2f}")
            pearl_minor_lengths.append(minor_axis_length)
            pearl_major_lengths.append(major_axis_length)

            # Calculate medio axis length (equivalent diameter)
            medior_axis_length = self.calculate_medio_axis_length(region)
            # print(f"Pearl {region.label} medior axis: {medior_axis_length:.2f}")
            pearl_medior_lengths.append(medior_axis_length)

            # Calculate volume
            volume = self.calculate_volume(minor_axis_length, major_axis_length, medior_axis_length)
            pearl_volumes.append(volume)


            # Calculate surface area or perimeter and sphericity
            if self.is3D:
                mask = (self.relabelled_labels.get() == region.label)  # Binary mask for the current region
                surface_area = self.calculate_surface_area_marching_cubes(mask)
                # print(f"Pearl {region.label} surface area: {surface_area:.2f}")
                sphericity = self.calculate_sphericity(volume, surface_area, None)
            else:
                perimeter = region.perimeter
                # print(f"Pearl {region.label} perimeter: {perimeter:.2f}")
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
                f"MinorL: {self.format_value(minor_axis_length)}, "
                f"MajorL: {self.format_value(major_axis_length)}, "
                f"MediorL: {self.format_value(medior_axis_length)}"
            )
            labels.append(label_text)
            #print(f'centroid: {region.centroid}')
            scaled_centroid = region.centroid / self.scaling
            print(f'scaled centroid: {scaled_centroid}')
            points.append(scaled_centroid)
            # print(f"Region {region.label} processed.")

        # print(f'len centroids: {len(points)}')
        # print(f'len labels: {len(labels)}')

        # Calculate pearling statistics
        num_pearls = len(self.pearl_objects)
        print(f"Number of pearls: {num_pearls}")
        nn_distance_mean = np.nanmean(nn_distances)
        pearl_volume_mean = np.nanmean(pearl_volumes)
        pearl_minor_axis_mean = np.nanmean(pearl_minor_lengths)
        pearl_major_axis_mean = np.nanmean(pearl_major_lengths)
        pearl_medio_axis_mean = np.nanmean(pearl_medior_lengths)
        pearl_sphericity_mean = np.nanmean(pearl_sphericities)  # Mean sphericity

        branch_features_path = self.im_info.pipeline_paths['features_branches']
        image_features_path = self.im_info.pipeline_paths['features_image']
        # open df of branch features path
        branch_feature_csv = pd.read_csv(branch_features_path)
        image_feature_csv = pd.read_csv(image_features_path)

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

        tubule_length = image_feature_csv['branch_length_sum'].values
        tubule_length_stdv = image_feature_csv['branch_length_std_dev'].values

        tubule_width_mean = image_feature_csv['node_thickness_mean'].values
        tubule_width_stdv = image_feature_csv['node_thickness_std_dev'].values


        if self.select_frames is None:
            self.select_frames = [0, 1, 2]
        time_interval = self.file_info.dim_res['T']
        initial_time = self.prepearl_frame * time_interval - time_interval
        peak_time = self.pearl_frame * time_interval - time_interval
        final_time = self.postpearl_frame * time_interval - time_interval

        time_to_peak = peak_time - initial_time
        time_to_recovery = final_time - peak_time
        duration_of_event = final_time - initial_time

        n_timepoints = self.file_info.t_end + 1

        pearl_metrics = self.PearlMetrics(
            filename=os.path.basename(self.im_path), timepoint=timepoint, n_timepoints=n_timepoints,
            peak_peak_distance_mean=nn_distance_mean, pearl_volume_mean=pearl_volume_mean,
            pearl_minor_axis_mean=pearl_minor_axis_mean, pearl_major_axis_mean=pearl_major_axis_mean,
            pearl_medio_axis_mean=pearl_medio_axis_mean, pearl_sphericity_mean=pearl_sphericity_mean,
            num_pearls=num_pearls,
            t0_tortuosity_mean=t0_tortuosity_mean, t2_tortuosity_mean=t2_tortuosity_mean,
            t0_total_length=t0_total_length, t2_total_length=t2_total_length,
            t0_tubule_width_mean=t0_tubule_width, t2_tubule_width_mean=t2_tubule_width,
            t0_pearls_per_micron=t0_pearls_per_micron, t2_pearls_per_micron=t2_pearls_per_micron,
            # intensity_mean=intensity_mean, intensity_stdv=intensity_stdv,
            # tortuosity_mean=tortuosity_mean, tortuosity_stdv=tortuosity_stdv,
            # tubule_length=tubule_length, tubule_lenth_stdv=tubule_length_stdv,
            # tubule_width_mean=tubule_width_mean, tubule_width_stdv=tubule_width_stdv,
            time_to_peak=time_to_peak, time_to_recovery=time_to_recovery, duration_of_event=duration_of_event
        )

        dtype = [('file', 'U256'), ('treatment', str), ('timepoint', int),
                 ('peak_peak_distance_mean', float), ('pearl_volume_mean', float),
                 ('pearl_minor_axis_mean', float),('pearl_major_axis_mean', float),
                 ('pearl_medio_axis_mean', float), ('num_peaks', int),
                 ('t0_tortuosity_mean', float),('t2_tortuosity_mean', float),
                 ('t0_total_length', float),('t2_total_length', float),
                 ('t0_tubule_width_mean', float),('t2_tubule_width_mean', float),
                 ('t0_pearls_per_micron', float),('t2_pearls_per_micron', float),
                 ('time_to_peak', float),('time_to_recovery', float),
                 ('duration_of_event', float),
                 ('intensity_mean', float), ('intensity_stdv', float),
                 ('tortuosity_mean', float), ('tortuosity_stdv', float),
                 ('tubule_width_mean', float), ('tubule_width_stdv', float),
                 ('tubule_length', float), ('tubule_length_stdv', float)]

        # Initialize the structured array with the correct number of rows
        pearl_metrics_time_arranged = np.zeros(n_timepoints, dtype=dtype)

        timepoints = list(range(n_timepoints))

        # Assign values to each field in the structured array
        pearl_metrics_time_arranged['file'] = [self.im_path] * n_timepoints  # Assign the file path to each row
        pearl_metrics_time_arranged['timepoint'] = timepoints  # Assign the sorted timepoints
        pearl_metrics_time_arranged['peak_peak_distance_mean'] = [nn_distance_mean] * n_timepoints
        pearl_metrics_time_arranged['pearl_volume_mean'] = [pearl_volume_mean] * n_timepoints
        pearl_metrics_time_arranged['pearl_minor_axis_mean'] = [pearl_minor_axis_mean] * n_timepoints
        pearl_metrics_time_arranged['pearl_major_axis_mean'] = [pearl_major_axis_mean] * n_timepoints
        pearl_metrics_time_arranged['pearl_medio_axis_mean'] = [pearl_medio_axis_mean] * n_timepoints
        pearl_metrics_time_arranged['num_peaks'] = [num_pearls] * n_timepoints
        pearl_metrics_time_arranged['t0_tortuosity_mean'] = [t0_tortuosity_mean] * n_timepoints
        pearl_metrics_time_arranged['t2_tortuosity_mean'] = [t2_tortuosity_mean] * n_timepoints
        pearl_metrics_time_arranged['t0_total_length'] = [t0_total_length] * n_timepoints
        pearl_metrics_time_arranged['t2_total_length'] = [t2_total_length] * n_timepoints
        pearl_metrics_time_arranged['t0_tubule_width_mean'] = [t0_tubule_width] * n_timepoints
        pearl_metrics_time_arranged['t2_tubule_width_mean'] = [t2_tubule_width] * n_timepoints
        pearl_metrics_time_arranged['t0_pearls_per_micron'] = [t0_pearls_per_micron] * n_timepoints
        pearl_metrics_time_arranged['t2_pearls_per_micron'] = [t2_pearls_per_micron] * n_timepoints
        pearl_metrics_time_arranged['time_to_peak'] = [time_to_peak] * n_timepoints
        pearl_metrics_time_arranged['time_to_recovery'] = [time_to_recovery] * n_timepoints
        pearl_metrics_time_arranged['duration_of_event'] = [duration_of_event] * n_timepoints
        pearl_metrics_time_arranged['intensity_mean'] = [intensity_mean[t] for t in timepoints]
        pearl_metrics_time_arranged['intensity_stdv'] = [intensity_stdv[t] for t in timepoints]
        pearl_metrics_time_arranged['tortuosity_mean'] = [tortuosity_mean[t] for t in timepoints]
        pearl_metrics_time_arranged['tortuosity_stdv'] = [tortuosity_stdv[t] for t in timepoints]
        pearl_metrics_time_arranged['tubule_width_mean'] = [tubule_width_mean[t] for t in timepoints]
        pearl_metrics_time_arranged['tubule_width_stdv'] = [tubule_width_stdv[t] for t in timepoints]
        pearl_metrics_time_arranged['tubule_length'] = [tubule_length[t] for t in timepoints]
        pearl_metrics_time_arranged['tubule_length_stdv'] = [tubule_length_stdv[t] for t in timepoints]
        # pearl_metrics_time_arranged['num_pearls'] = [num_pearls_over_time[t] for t in timepoints]

        return pearl_metrics_time_arranged, pearl_metrics, points, labels

    def plot_pearls_over_time(self, pearl_metrics, y_column='num_pearls'):
        import matplotlib.pyplot as plt
        """
        Plots the number of pearls over time using the provided pearl metrics data and saves the plot as JPEG and SVG.

        Parameters:
        - pearl_metrics: A list of dictionaries containing pearl metrics for each frame.
                         Each dictionary should have a 'frame' key and a 'num_pearls' key.
        - save_path: The base path where the plots will be saved (without file extension).
        """
        # Extract frame numbers and number of pearls from the pearl metrics
        frames = pearl_metrics['timepoint']
        num_pearls = pearl_metrics[y_column]

        # Plot the number of pearls over time
        plt.figure(figsize=(10, 6))
        plt.plot(frames, num_pearls, marker='o', linestyle='-', color='b')
        plt.xlabel('Frame Number (Time)')
        plt.ylabel('Number of Pearls')
        plt.title(y_column + ' Over Time')
        plt.grid(True)

        # Save the plot as JPEG and SVG
        save_path = os.path.join(self.save_path, (y_column+"_over_time"))
        plt.savefig(f"{save_path}.jpeg", format='jpeg', dpi=300)
        plt.savefig(f"{save_path}.svg", format='svg')

        # Show the plot
        plt.show()

        print(f"Plots saved as {save_path}.jpeg and {save_path}.svg")


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