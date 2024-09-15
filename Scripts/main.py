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
    final_save_folder = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-09-09"
    csv_path = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-09-09\Pearling_metadata_for_quantification.csv"
    pearls_df = pd.read_csv(csv_path, dtype={'im_path': 'object', 'nellie_path': 'object', 'crop_path': 'object'})
    selected_timepoints = pearls_df['selected_timepoints'].apply(convert_to_list)
    # pearls_df['nellie_path'] = np.nan
    print(pearls_df.head())

    crop_override = False
    nellie_overide = False
    curvature_override = False
    pearling_stats_override = False
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
        print(f'{frame_list=}')

        # # check if nellie was already run on this crop
        # print(f'crop{crop_count} already cropped, skipping')
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
            # mask = None
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
            # Set the timepoints range if necessary (assume run_curvature supports this)
            # skel_curvature.set_timepoints(timepoints)

            # Run the curvature analysis
            curvatures = skel_curvature.run_curvature()
            # df = pd.DataFrame(curvatures)
            # df.to_csv(os.path.join(curvature_path, 'curvatures.csv'), index=False) # Save DataFrame to CSV
            # curvatures_df['treatment'] = pearls_df['Treatment'][index]
            # results_df = pd.concat([results_df, curvatures_df],
            #                             ignore_index=True)  # append the results to the results array
            save_completion_file(nellie_path, "completed_curvature.txt")
        ### RUN PEARLING STATS ###
        if not os.path.exists(os.path.join(nellie_path, "completed_pearling-stats.txt")) or pearling_stats_override:
            if pearls_df['select_mask'][index]:
                mask = xp.load(os.path.join(nellie_path, 'mask.npy'))
            pearling_stats_path = os.path.join(nellie_path, "pearling_stats")
            pearl_stats = PearlingStats(crop_path, nellie_path=nellie_path, save_path=pearling_stats_path, mask=mask,
                                        visualize=True, select_frames=frame_list)
            pm_df, _ = pearl_stats.run_pearling_stats()
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
            # print(f'length of curvatures found: {len(curvatures)}')
            # results_df = pd.concat([results_df, curvatures], ignore_index=True)    # append the results to the results array
            # print('curvature data already exists, adding to results array')
        # open the pearling stats csv
        pearling_stats_csv = os.path.join(nellie_path, 'pearling_metrics_time_arranged.csv')
        pm_df = pd.DataFrame()
        if os.path.exists(pearling_stats_csv):
            pm_df = pd.read_csv(pearling_stats_csv)
            pm_df['treatment'] = pearls_df['Treatment'][index]
            # print(f'length of pearling stats found: {len(pm_df)}')
            # print('pearling stats data already exists, adding to results array')
            # matching_columns = pm_df.columns.intersection(curvatures.columns)
            # row combine curvatures and pearling stats

        # Align columns of df2 and df3 to match df1's columns with NaNs for missing columns
        # curvatures = curvatures.reindex(columns=results_df.columns)
        # pm_df = pm_df.reindex(columns=results_df.columns, fill_value=np.nan)

        merged_rows = pd.merge(curvatures, pm_df, on='timepoint', how='inner')
        merged_rows['treatment'] = pearls_df['Treatment'][index]
        merged_rows['induction_method'] = pearls_df['Induction_method'][index]
        merged_rows['microscope'] = pearls_df['Microscope'][index]
        merged_rows['pearl_frame'] = frame2
        merged_rows['pre_pearl_frame'] = frame1
        merged_rows['post_pearl_frame'] = frame3
        # Combine to `df1` by adding new rows
        results_df = pd.concat([results_df, merged_rows], ignore_index=True)
        # results_df[matching_columns] = pm_df[matching_columns]
        # results_df = pd.concat([results_df, pm_df], ignore_index=False)

        # save the results to a csv
        if not os.path.exists(final_save_folder):
            os.makedirs(final_save_folder)
        results_df.to_csv(os.path.join(final_save_folder, (current_time_str + 'all_pearling_results.csv')), index=False)
        # print(f'results df length: {len(results_df)}')
        print(f'completed processing {im_path}')

    # plot the results
    plot_folder = os.path.join(final_save_folder, 'treatment_plots')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    norm_methods = ['first_point'] # 'min-max', 'mean', 'median', 'pre-pearl', 'post-pearl', 'max_pearl'
    y_columns = ['intensity_curvature', 'structural_curvature', 'tortuosity_mean','tubule_width_mean', 'tubule_length'] #
    graphs = Graph_timecourse()
    for y_col in y_columns:
        for norm_method in norm_methods:
            graphs.plot_treatment_graph(results_df, y_col, select_treatments=None, select_method=None,
                                 save_folder=plot_folder, time=current_time_str, window_size=20,
                                 normalization=norm_method)


# folder = r"E:\SNOUTY_data\Pearling-examples\austin-pearling-metrics-example-3d-snouty"
#     # FCCP: r"F:\UCSF\W1_FRAP\2023-08-15_tmre-sybrgold\TMRE\FCCP\RPE1_CellLight_Mito_GFP_FCCP_4uM_3min_1-good"
#     # 2D: r"F:\UCSF\W1_FRAP\2023-08-08_ER-PKMO\RPE1-ER_3-pearling"
#     # 3D: r"E:\SNOUTY_data\Pearling-examples\austin-pearling-metrics-example-3d-snouty"
# basename =  "deskewed-austin-pearling-metrics-example-3d-snouty.ome.tif"
#     # FCCP: "RPE1_CellLight_Mito_GFP_FCCP_4uM_3min_1_MMStack_Default.ome.tif"
#     # 2D: "RPE1-ER_3_MMStack_Default.ome.tif"
#     # 3D: "deskewed-austin-pearling-metrics-example-3d-snouty.ome.tif"
