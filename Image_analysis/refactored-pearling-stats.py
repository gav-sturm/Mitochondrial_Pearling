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


class PearlingStats:
    def __init__(self, im_path, nellie_path, mask=None, save_path=None, select_frames=None, visualize=False):
        """
        Initializes the PearlingStats class with necessary parameters.

        Args:
            im_path (str): Path to the input image file.
            nellie_path (str): Path to the output directory.
            mask (array, optional): Mask to apply to the image.
            save_path (str, optional): Path to save the output metrics.
            select_frames (list, optional): Frames to select for processing.
            visualize (bool, optional): Whether to visualize using Napari.
        """
        self.im_path = im_path
        self.nellie_path = nellie_path
        self.save_path = save_path
        self.select_frames = select_frames
        self.visualize = visualize
        self.file_info = None
        self.im_info = None
        self.mask = mask
        self.relabelled_labels = None

    @dataclass
    class PearlMetrics:
        """
        Data class to store pearling metrics.
        """
        filename: str
        peak_peak_distance_mean: float
        pearl_volume_mean: float
        pearl_minor_axis_mean: float
        pearl_major_axis_mean: float
        pearl_medio_axis_mean: float
        num_peaks: int

        # Optional metrics initialized as None
        t0_tortuosity_mean: float = None
        t2_tortuosity_mean: float = None
        t0_total_length: float = None
        t2_total_length: float = None
        t0_tubule_width_mean: float = None
        t2_tubule_width_mean: float = None
        t0_pearls_per_micron: float = None
        t2_pearls_per_micron: float = None

    def find_file_in_subdirectories(self, directory, text):
        """
        Finds files in subdirectories that contain a specific text.

        Args:
            directory (str): The directory to search.
            text (str): The text to search for in filenames.

        Returns:
            List[str]: A list of matching file paths.
        """
        matching_files = []
        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(directory):
            for file in files:
                if text in file:
                    matching_files.append(os.path.join(root, file))
        return matching_files

    def find_files_with_last_char_number(self, directory):
        """
        Finds files in subdirectories where the last character before '.ome.tif' is a number.

        Args:
            directory (str): The directory to search.

        Returns:
            List[str]: A list of matching file paths.
        """
        matching_files = []
        pattern = re.compile(r'.*\d\.ome.tif$')  # Regular expression pattern

        for root, _, files in os.walk(directory):
            for file in files:
                if pattern.match(file):
                    matching_files.append(os.path.join(root, file))
        if len(matching_files) == 0:
            raise FileNotFoundError(f'No files found in {directory} with a number before the extension.')
        return matching_files

    def run_pearling_stats(self):
        """
        Runs the pearling statistics calculation and visualization, and saves the results to a CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the computed pearling metrics.
        """
        # Set up Napari viewer if visualization is enabled
        viewer = napari.Viewer() if self.visualize else None

        # Initialize file information and metadata
        self.file_info = FileInfo(filepath=self.im_path, output_dir=self.nellie_path)
        self.file_info.find_metadata()
        self.file_info.load_metadata()
        self.im_info = ImInfo(self.file_info)

        # Load necessary image data and paths
        ome_path = self.im_info.im_path
        raw = xp.array(self.im_info.get_memmap(ome_path)[1])
        frangi_path = self.find_file_in_subdirectories(self.nellie_path, 'im_preprocessed')[0]
        frangi = xp.array(self.im_info.get_memmap(frangi_path)[1])
        scale = xp.array([self.file_info.dim_res['Z'], self.im_info.dim_res['Y'], self.im_info.dim_res['X']])
        z_ratio = scale[1] / scale[0]
        average_bead_radius_um = 0.25

        # Visualization in Napari
        if self.visualize:
            viewer.add_image(frangi.get())

        # Apply mask if provided
        if self.mask is not None:
            frangi *= self.mask

        # Thresholding and edge detection
        minotri_thresh = thresholding.minotri_threshold(frangi[frangi > 0])
        mask_frame = frangi > minotri_thresh
        border_image = ndi.binary_dilation(mask_frame) ^ mask_frame
        border_coords = xp.argwhere(border_image) * scale
        mask_coords = xp.argwhere(mask_frame) * scale

        # Calculate distances using KDTree
        border_kdtree = cKDTree(border_coords.get())
        mask_nn = border_kdtree.query(mask_coords.get())
        distance_image = xp.zeros_like(frangi)
        distance_image[mask_frame] = mask_nn[0]

        # Apply Gaussian smoothing
        smoothed_distance = ndi.gaussian_filter(frangi, sigma=[z_ratio, 1, 1]) * mask_frame

        # Visualization in Napari
        if self.visualize:
            viewer.add_image(smoothed_distance.get())

        # Maximum filtering to find peaks
        footprint = xp.ones((int(2 * average_bead_radius_um / scale[1]), int(2 * average_bead_radius_um / scale[1]),
                             int(2 * average_bead_radius_um / scale[1])))
        max_filtered = ndi.maximum_filter(smoothed_distance, footprint=footprint)
        if self.visualize:
            viewer.add_image(max_filtered.get())
        peaks = (max_filtered == smoothed_distance) * mask_frame

        # Label peaks
        peak_labels, num_peaks = ndi.label(peaks)
        if self.visualize:
            viewer.add_labels(peak_labels.get())

        # Further processing for pearling metrics
        structure = xp.ones((3, 3, 3))
        self.relabelled_labels = self._reassign_labels(peak_labels, peaks, mask_frame, structure, scale, viewer)

        # Remove objects touching the edges
        pearl_objects = self._remove_edge_objects(self.relabelled_labels, viewer)

        # Compute pearling metrics
        pearl_metrics_df = self._compute_and_save_metrics(pearl_objects, raw, scale)
        return pearl_metrics_df

    def _reassign_labels(self, peak_labels, peaks, mask_frame, structure, scale, viewer):
        """
        Reassigns labels to peaks to ensure all objects are correctly labeled.

        Args:
            peak_labels (ndarray): The labeled peak array.
            peaks (ndarray): The binary array of peaks.
            mask_frame (ndarray): The binary mask of the image frame.
            structure (ndarray): The structure element for dilation.
            scale (ndarray): The scaling factors for each dimension.
            viewer (napari.Viewer): The Napari viewer object for visualization.

        Returns:
            ndarray: The relabelled peak array.
        """
        self.relabelled_labels = peak_labels.copy()
        branch_skel_labels = self.relabelled_labels.copy()
        peak_mask_int = peaks.astype('uint16')
        label_mask_int = mask_frame.astype('uint16')
        peak_border = (ndi.binary_dilation(peak_mask_int, iterations=1,
                                           structure=structure) ^ peak_mask_int) * label_mask_int

        peak_label_mask = (peak_labels > 0).get()
        vox_matched = np.argwhere(peak_label_mask)
        vox_next_unmatched = np.argwhere(peak_border.get())

        unmatched_diff = np.inf
        scaling = scale.get()
        while True:
            num_unmatched = len(vox_next_unmatched)
            if num_unmatched == 0:
                break
            tree = cKDTree(vox_matched * scaling)
            dists, idxs = tree.query(vox_next_unmatched * scaling, k=1, workers=-1)
            max_dist = 2 * np.min(scaling)  # sqrt 3 * max scaling
            unmatched_matches = np.array(
                [[vox_matched[idx], vox_next_unmatched[i]] for i, idx in enumerate(idxs) if dists[i] < max_dist]
            )
            if len(unmatched_matches) == 0:
                break
            matched_labels = branch_skel_labels[tuple(np.transpose(unmatched_matches[:, 0]))]
            self.relabelled_labels[tuple(np.transpose(unmatched_matches[:, 1]))] = matched_labels
            branch_skel_labels = self.relabelled_labels.copy()
            self.relabelled_labels_mask = self.relabelled_labels > 0

            vox_matched = np.argwhere(self.relabelled_labels_mask.get())
            relabelled_mask = self.relabelled_labels_mask.astype('uint8')
            peak_border = (ndi.binary_dilation(relabelled_mask, iterations=1,
                                               structure=structure) ^ relabelled_mask) * label_mask_int
            vox_next_unmatched = np.argwhere(peak_border.get())

            new_num_unmatched = len(vox_next_unmatched)
            unmatched_diff_temp = abs(num_unmatched - new_num_unmatched)
            if unmatched_diff_temp == unmatched_diff:
                break
            unmatched_diff = unmatched_diff_temp
            print(f'Reassigned {unmatched_diff}/{num_unmatched} unassigned voxels. '
                  f'{new_num_unmatched} remain.')

        if self.visualize:
            viewer.add_labels(self.relabelled_labels.get())
        return self.relabelled_labels

    def _remove_edge_objects(self, relabelled_labels, viewer):
        """
        Removes objects with labels touching the edges of the image.

        Args:
            self.relabelled_labels (ndarray): The relabelled peak array.
            viewer (napari.Viewer): The Napari viewer object for visualization.

        Returns:
            list: List of pearl objects after edge removal.
        """
        pearl_objects_not_spaced = measure.regionprops(self.relabelled_labels.get())
        max_z, max_y, max_x = self.relabelled_labels.shape
        for region in pearl_objects_not_spaced:
            bbox = region.bbox
            if bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] <= 0 or bbox[3] + 1 >= max_z or bbox[4] + 1 >= max_y or bbox[
                5] + 1 >= max_x:
                self.relabelled_labels[self.relabelled_labels == region.label] = 0
                print(f'Removed object {region.label} due to touching edge of image.')

        if self.visualize:
            viewer.add_labels(self.relabelled_labels.get())
        return measure.regionprops(self.relabelled_labels.get())

    def _compute_and_save_metrics(self, pearl_objects, raw, scale):
        """
        Computes pearling metrics and saves them to a CSV file.

        Args:
            pearl_objects (list): List of pearl objects after edge removal.
            raw (ndarray): The raw image array.
            scale (ndarray): The scaling factors for each dimension.

        Returns:
            pd.DataFrame: A DataFrame containing the computed pearling metrics.
        """
        # Extract metrics for each pearl object
        peak_coords = xp.argwhere(self.relabelled_labels > 0)
        peak_labels_at_coords = self.relabelled_labels[peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2]]
        peak_tree = cKDTree(peak_coords.get() * scale.get())
        distances, indices = peak_tree.query(peak_coords.get() * scale.get(), k=2, workers=-1)
        nn_distances = distances[:, 1]  # Nearest neighbor distances

        # Calculate additional metrics for each pearl object
        pearl_volumes = [region.area for region in pearl_objects]
        pearl_minor_lengths = [region.axis_minor_length for region in pearl_objects]
        pearl_major_lengths = [region.axis_major_length for region in pearl_objects]
        pearl_medior_lengths = [region.equivalent_diameter for region in pearl_objects]
        num_pearls = len(pearl_objects)

        # Compute mean values
        nn_distance_mean = np.nanmean(nn_distances)
        pearl_volume_mean = np.nanmean(pearl_volumes)
        pearl_minor_axis_mean = np.nanmean(pearl_minor_lengths)
        pearl_major_axis_mean = np.nanmean(pearl_major_lengths)
        pearl_medio_axis_mean = np.nanmean(pearl_medior_lengths)

        # Load branch and image feature CSV files
        branch_features_path = self.im_info.pipeline_paths['features_branches']
        image_features_path = self.im_info.pipeline_paths['features_image']
        branch_feature_csv = pd.read_csv(branch_features_path)
        image_feature_csv = pd.read_csv(image_features_path)

        # Extract features for specific time points
        t0_branch_features = branch_feature_csv[branch_feature_csv['t'] == 0]
        t2_branch_features = branch_feature_csv[branch_feature_csv['t'] == 2]
        t0_image_features = image_feature_csv[image_feature_csv['t'] == 0]
        t2_image_features = image_feature_csv[image_feature_csv['t'] == 2]

        # Extract relevant metrics from CSV data
        t0_tortuosity_mean = t0_image_features['branch_tortuosity_mean'].values[0]
        t2_tortuosity_mean = t2_image_features['branch_tortuosity_mean'].values[0]
        t0_total_length = t0_image_features['branch_length_sum'].values[0]
        t2_total_length = t2_image_features['branch_length_sum'].values[0]
        t0_tubule_width = t0_image_features['node_thickness_mean'].values[0]
        t2_tubule_width = t2_image_features['node_thickness_mean'].values[0]
        t0_pearls_per_micron = num_pearls / t0_total_length
        t2_pearls_per_micron = num_pearls / t2_total_length

        # Create a dataclass instance to store all metrics
        pearl_metrics = self.PearlMetrics(
            filename=os.path.basename(self.im_path),
            peak_peak_distance_mean=nn_distance_mean,
            pearl_volume_mean=pearl_volume_mean,
            pearl_minor_axis_mean=pearl_minor_axis_mean,
            pearl_major_axis_mean=pearl_major_axis_mean,
            pearl_medio_axis_mean=pearl_medio_axis_mean,
            num_peaks=num_pearls,
            t0_tortuosity_mean=t0_tortuosity_mean,
            t2_tortuosity_mean=t2_tortuosity_mean,
            t0_total_length=t0_total_length,
            t2_total_length=t2_total_length,
            t0_tubule_width_mean=t0_tubule_width,
            t2_tubule_width_mean=t2_tubule_width,
            t0_pearls_per_micron=t0_pearls_per_micron,
            t2_pearls_per_micron=t2_pearls_per_micron
        )

        # Convert the dataclass instance to a DataFrame
        pearl_metrics_df = pd.DataFrame(pearl_metrics.__dict__, index=[0])

        # Save the DataFrame to a CSV file
        self.save_path = os.path.join(self.nellie_path, f"{pearl_metrics.filename}-pearl_metrics.csv")
        pearl_metrics_df.to_csv(self.save_path, index=False)
        print(f'Pearling metrics saved to {self.save_path}')

        return pearl_metrics_df

if __name__ == "__main__":
    # Define input paths and run pearling stats
    im_path = r"E:\SNOUTY_data\Pearling-examples\austin-pearling-metrics-example-3d-snouty\crop2.ome.tif"
    nellie_path = r"E:\SNOUTY_data\Pearling-examples\austin-pearling-metrics-example-3d-snouty\nellie-crop2"
    mask = xp.load(os.path.join(nellie_path, 'mask.npy'))
    pearl_stats = PearlingStats(im_path, nellie_path, mask=mask, visualize=True)
    pearl_stats.run_pearling_stats()
