from napari.utils import notifications
from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.im_info.verifier import FileInfo, ImInfo
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner
from crop_snouty import CropSnout
from skeleton_line_profiles import SkeletonCurvature
from pearling_stats import PearlingStats
from graphs import Graph_timecourse
import os
import datetime as dt
import numpy as np
import pandas as pd
import cupy as xp
import pathlib
import tifffile as tif
import logging

# Set logging level for Napari and other related modules to WARNING
logging.getLogger('napari').setLevel(logging.WARNING)
logging.getLogger('in_n_out').setLevel(logging.WARNING)
os.environ['NAPARI_LOG_LEVEL'] = 'WARNING'
logging.getLogger('matplotlib').setLevel(logging.WARNING)




pd.options.mode.copy_on_write = True  # Enable copy-on-write for pandas DataFrame


def run(file_info, remove_edges=False, otsu_thresh_intensity=False, threshold=None, mask=None):
    im_info = ImInfo(file_info)
    preprocessing = Filter(im_info, remove_edges=remove_edges)
    preprocessing.run()

    segmenting = Label(im_info, otsu_thresh_intensity=otsu_thresh_intensity, threshold=threshold, mask=mask)
    segmenting.run()

    networking = Network(im_info)
    networking.run()

    mocap_marking = Markers(im_info)
    mocap_marking.run()

    hu_tracking = HuMomentTracking(im_info)
    hu_tracking.run()

    vox_reassign = VoxelReassigner(im_info)
    vox_reassign.run()

    hierarchy = Hierarchy(im_info, skip_nodes=False)
    hierarchy.run()

    return im_info


# Convert the string representation of the array to an actual list of integers, handling NaN values
def convert_to_list(cell):
    if pd.isna(cell):  # Check if the cell is NaN
        return np.nan  # You could also return [] for an empty list
    return [int(i.strip()) for i in cell.split(',')]


def save_completion_file(path, filename):
    # Construct the filename for the completion file
    completion_filename = os.path.join(path, filename)

    # Open the file in write mode and add some content
    with open(completion_filename, 'w') as f:
        f.write("Process completed successfully.")

    print(f"Completion file '{completion_filename}' has been saved.")


def generate_folder_structure(final_save_folder, index, im_type, folder, basename, crop_count):
    if not os.path.exists(final_save_folder):
        os.makedirs(final_save_folder)
    parent_dir = os.path.join(final_save_folder, f'event{index + 1}_{os.path.basename(os.path.normpath(folder))}')
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if im_type == 'unskewed':
        crop_folder = os.path.join(parent_dir, f'crop{crop_count}_snout')
        if not os.path.exists(crop_folder):
            os.makedirs(crop_folder)
        basename = f"deskewed-crop{crop_count}.ome.tif"
        im_path = folder
        crop_path = os.path.join(crop_folder, (f'crop{crop_count}.ome.tif'))
        nellie_path = os.path.join(crop_folder, (f'crop{crop_count}_nellie_out'))
    else:
        im_path = os.path.join(folder, basename)
        crop_path = os.path.join(parent_dir, (f'crop{crop_count}.ome.tif'))
        nellie_path = os.path.join(parent_dir, (f'crop{crop_count}_nellie_out'))

    if not os.path.exists(nellie_path):
        os.makedirs(nellie_path)

    return basename, im_path, nellie_path, crop_path





if __name__ == "__main__":
    current_time_str = dt.datetime.now().strftime("%Y-%m-%d_%H-%M_")
    results_df = pd.DataFrame()
    # import csv with all metadata and filepaths
    final_save_folder = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-09-14"
    csv_path = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-09-14\Pearling_metadata_for_quantification.csv"
    pearls_df = pd.read_csv(csv_path, dtype={'im_path': 'object', 'nellie_path': 'object', 'crop_path': 'object'})
    selected_timepoints = pearls_df['selected_timepoints'].apply(convert_to_list)
    # pearls_df['nellie_path'] = np.nan
    print(pearls_df.head())

    crop_override = False
    nellie_overide = False
    curvature_override = False
    pearling_stats_override = True
    all_training_data = pd.DataFrame()
    for index, row in pearls_df.iterrows():
        # if index != 3:
        #     continue
        # print(f'processing file {index+1} of {len(pearls_df)}')
        folder = pathlib.Path(pearls_df['home_folder'][index].strip('"'))
        basename = None if pd.isna(pearls_df['basename'][index]) or pearls_df['basename'][index].strip() == '' else \
        pearls_df['basename'][index].strip('"')
        crop = pearls_df['select_crop'][index]
        crop_count = pearls_df['crop_count'][index]
        image_type = pearls_df['Image_type'][index]
        microscope = pearls_df['Microscope'][index]
        include = pearls_df['Include'][index]
        select_channel = pearls_df['select_channel'][index]
        time_interval = pearls_df['time_interval_seconds'][index]
        center_coords = (pearls_df['x_pos'][index], pearls_df['y_pos'][index])
        _, im_path, nellie_path, crop_path = generate_folder_structure(final_save_folder, index, image_type,
                                                                       folder, basename, crop_count)
        pearls_df.loc[index, 'nellie_path'] = nellie_path
        pearls_df.loc[index, 'crop_path'] = crop_path
        pearls_df.loc[index, 'im_path'] = im_path
        print(f'processing {im_path}')
        # dt = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # check if the selected timepoints is a list


        frame1 = int(pearls_df['pre-pearl-frame'][index] - 1)
        frame2 = int(pearls_df['max-pearl-frame'][index] - 1)
        frame3 = int(pearls_df['post-pearl-frame'][index] - 1)

        timepoints = None
        if isinstance(selected_timepoints[index], list):
            timepoints = [x - 1 for x in selected_timepoints[index]]
            frame1 = frame1 - timepoints[0]
            frame2 = frame2 - timepoints[0]
            frame3 = frame3 - timepoints[0]
        frame_list = [frame1, frame2, frame3]

        mask = None
        crop_z = False
        if microscope == 'SNOUTY':
            crop_z = True

        pearls_df.loc[index, 'crop_path'] = crop_path
        ### RUN CROP ###
        if crop:
            if not os.path.exists(os.path.join(nellie_path, "completed_crop.txt")) or crop_override:
                # User inputs
                print(f'cropping to {crop_path}')
                print(f'cropping from {im_path}')
                crop_snout = CropSnout(im_path=im_path, crop_path=crop_path, nellie_path=nellie_path,
                                       image_type=image_type, crop_count=crop_count,
                                       select_timepoints=timepoints, select_channel=select_channel, select_z=crop_z,
                                       time_interval=time_interval, center_coords=center_coords, show_crop=False,
                                       pearl_frame=frame2)
                crop_snout.crop_image()  # Allow the user to select the ROI
                print(f'{pearls_df["select_mask"][index]=}')
                if pearls_df['select_mask'][index]:
                    crop_snout.select_mask()
                save_completion_file(nellie_path, "completed_crop.txt")
        ### RUN NELLIE ###
        if not os.path.exists(os.path.join(nellie_path, "completed_nellie.txt")) or nellie_overide:
            remove_edges = False
            if microscope == 'SNOUTY':
                remove_edges = True
            print(f'running nellie on {crop_path}')
            print(f'saving nellie outputs to {nellie_path=}')
            import shutil

            if os.path.exists(os.path.join(nellie_path, "nellie_necessities")):
                shutil.rmtree(
                    os.path.join(nellie_path, "nellie_necessities"))  # Deletes the old folder and all its contents

            file_info = FileInfo(filepath=crop_path, output_dir=nellie_path)
            file_info.find_metadata()
            file_info.load_metadata()
            if select_channel > 0:
                file_info.change_selected_channel(select_channel)
            print(f'selected channel {file_info.ch=}')
            print(f'image shape: {file_info.shape=}')
            if pearls_df['select_mask'][index]:
                mask = xp.load(os.path.join(nellie_path, 'mask.npy'))
                print(f'mask shape: {mask.shape}')
            run(file_info, mask=mask, remove_edges=remove_edges)
            save_completion_file(nellie_path, "completed_nellie.txt")
        ### RUN CURVATURE ###
        if not os.path.exists(os.path.join(nellie_path, "completed_curvature.txt")) or curvature_override:
            curvature_path = os.path.join(nellie_path, "curvature_analysis")
            sm = pearls_df['smoothing_factor'][index]
            # Create a SkeletonCurvature instance with the metadata
            skel_curvature = SkeletonCurvature(crop_path, nellie_path=nellie_path, save_path=curvature_path,
                                               smoothing_factors=None,
                                               manual_smoothing=True,
                                               show_individual_plots=False, show_summary_plots=True,
                                               display_frames=frame_list)
            # Run the curvature analysis
            curvatures = skel_curvature.run_curvature()
            save_completion_file(nellie_path, "completed_curvature.txt")
        ### RUN PEARLING STATS ###
        if not os.path.exists(os.path.join(nellie_path, "completed_pearling-stats.txt")) or pearling_stats_override:
            if pearls_df['select_mask'][index]:
                mask = xp.load(os.path.join(nellie_path, 'mask.npy'))
            pearling_stats_path = os.path.join(nellie_path, "pearling_stats")
            if not os.path.exists(pearling_stats_path):
                os.makedirs(pearling_stats_path)
            pearl_stats = PearlingStats(crop_path, nellie_path=nellie_path, save_path=pearling_stats_path, mask=mask,
                                        visualize=True, select_frames=frame_list, train_rfc=True)
            pm_df, training_data = pearl_stats.run_pearling_stats()
            # all_training_data = pd.concat([all_training_data, training_data], ignore_index=True)
            # pm_df['treatment'] = pearls_df['Treatment'][index]
            # results_df = pd.concat([results_df, pm_df],
            #                        ignore_index=True)  # append the results to the results array
            save_completion_file(nellie_path, "completed_pearling-stats.txt")
        #### get the data from the saved csv files ###
        # open the curvature csv
        curvature_path = os.path.join(nellie_path, "curvature_analysis")
        curvature_csv = os.path.join(curvature_path, 'curvatures.csv')
        curvatures = pd.DataFrame()
        if os.path.exists(curvature_csv):
            curvatures = pd.read_csv(curvature_csv)
        # open the pearling stats csv
        pearling_stats_csv = os.path.join(nellie_path, "pearling_stats",'pearling_metrics.csv')
        pm_df = pd.DataFrame()
        if os.path.exists(pearling_stats_csv):
            pm_df = pd.read_csv(pearling_stats_csv)
        # merge the two dataframes
        merged_rows = pd.merge(curvatures, pm_df, on='timepoint', how='inner')
        merged_rows['treatment'] = pearls_df['Treatment'][index]
        merged_rows['timepoint_seconds'] = merged_rows['timepoint'] * pearls_df['time_interval_seconds'][index]
        time = merged_rows['timepoint_seconds']
        merged_rows['time_interval'] = pearls_df['time_interval_seconds'][index]
        merged_rows['induction_method'] = pearls_df['Induction_method'][index]
        merged_rows['microscope'] = pearls_df['Microscope'][index]
        merged_rows['pearl_frame'] = frame2
        merged_rows['pre_pearl_frame'] = frame1
        merged_rows['post_pearl_frame'] = frame3
        # Combine to `df1` by adding new rows
        if include:
            results_df = pd.concat([results_df, merged_rows], ignore_index=True)

        # save the results to a csv
        if not os.path.exists(final_save_folder):
            os.makedirs(final_save_folder)
        results_df.to_csv(os.path.join(final_save_folder, (current_time_str + 'all_pearling_results.csv')), index=False)

        # open the training data csv
        training_data_csv = os.path.join(nellie_path, "pearling_stats", 'training_label_metrics.csv')
        # concat the training data
        all_training_data = pd.concat([all_training_data, training_data], ignore_index=True)
        all_training_data.to_csv(os.path.join(final_save_folder, 'rfc_training_data.csv'), index=False)

        print(f'completed processing {im_path}')

    # plot the results
    plot_folder = os.path.join(final_save_folder, 'treatment_plots')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    print(results_df.head())
    norm_methods = ['first_point'] # 'min-max', 'mean', 'median', 'pre-pearl', 'post-pearl', 'max_pearl'
    y_columns = ['intensity_curvature', 'structural_curvature', 'tortuosity_mean','tubule_width_mean', 'tubule_length_mean',
                 'volume_mean', 'rectangularity_mean', 'peak_peak_distance_mean', 'sphericity_mean', 'solidity_mean']
    graphs = Graph_timecourse()
    for y_col in y_columns:
        for norm_method in norm_methods:
            graphs.plot_treatment_graph(results_df, y_col, select_treatments=None, select_method=None,
                                 save_folder=plot_folder, time=current_time_str,
                                 pre_window_size=5, post_window_size=30,
                                 normalization='first_point')
