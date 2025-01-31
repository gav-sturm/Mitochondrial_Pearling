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

# Convert the string representation of the array to an actual list of integers, handling NaN values
def convert_to_list_float(cell):
    if pd.isna(cell):  # Check if the cell is NaN
        return np.nan  # You could also return [] for an empty list
    return [float(i.strip()) for i in cell.split(',')]


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
    # check if the folder name is more then 10 characters and shorten it
    folder_name = os.path.basename(os.path.normpath(folder))
    cutoff = 20 # 50 for daria data, 20 for all others
    if len(os.path.basename(os.path.normpath(folder))) > cutoff:
        folder_name = folder_name[:cutoff]
    parent_dir = os.path.join(final_save_folder, f'event{index + 1}_{folder_name}') #
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if im_type == 'unskewed' or im_type == 'unskewed-multifile':
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

    results_list = []
    all_single_volume_data = pd.DataFrame()
    # import csv with all metadata and filepaths
    final_save_folder = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-11-04_osmotic_shock_calciumnapa"
    csv_path = os.path.join(final_save_folder, "Pearling_metadata_for_quantification.csv")
    # csv_path = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-09-29_microneedle\Pearling_metadata_for_quantification.csv"
    pearls_df = pd.read_csv(csv_path, dtype={'im_path': 'object', 'nellie_path': 'object', 'crop_path': 'object'})
    selected_timepoints = pearls_df['selected_timepoints'].apply(convert_to_list)
    skip_frames = pearls_df['skip_frames'].apply(convert_to_list) if 'skip_frames' in pearls_df.columns else None
    manual_dim_res = pearls_df['dim_res'].apply(convert_to_list_float) if 'dim_res' in pearls_df.columns else [None] * len(pearls_df)
    # pearls_df['nellie_path'] = np.nan
    # print(pearls_df.head())

    crop_override = False
    nellie_override = False
    curvature_override = False
    pearling_stats_override = False
    train_rfc = False
    all_training_data = pd.DataFrame()
    training_data_list = []
    for index, row in pearls_df.iterrows():
        print(f'processing file {index + 1} of {len(pearls_df)}')
        # if index != 3:
        #     continue
        # print(f'processing file {index+1} of {len(pearls_df)}')
        folder = pathlib.Path(pearls_df['home_folder'][index].strip('"'))
        basename = None if pd.isna(pearls_df['basename'][index]) or pearls_df['basename'][index].strip() == '' else \
        pearls_df['basename'][index].strip('"')
        crop = pearls_df['select_crop'][index]
        crop_count = int(pearls_df['crop_count'][index])
        image_type = pearls_df['Image_type'][index]
        microscope = pearls_df['Microscope'][index]
        analysis_type = pearls_df['analysis_type'][index] if 'analysis_type' in pearls_df.columns else False
        project_z = pearls_df['project_z'][index] if 'project_z' in pearls_df.columns else False
        include = pearls_df['Include'][index] if 'Include' in pearls_df.columns else True
        if not include:
            print(f'Skipping file {index + 1} of {len(pearls_df)}')
            continue
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
        if analysis_type == 'single_volume':
            timepoints = [0,1,2]
            frame1 = 0
            frame2 = 1
            frame3 = 2
        elif isinstance(selected_timepoints[index], list):
            timepoints = [x - 1 for x in selected_timepoints[index]]
            frame1 = frame1 - timepoints[0]
            frame2 = frame2 - timepoints[0]
            frame3 = frame3 - timepoints[0]
            if frame3 > len(timepoints):
                frame3 = timepoints[-1] - timepoints[0]
        frame_list = [frame1, frame2, frame3]
        if skip_frames[index] is not None:
            if isinstance(skip_frames[index], list):
                frames_to_skip = [x - 1 for x in skip_frames[index]]
                # duplicate the timepoint before the skipped frames for each skipped frame
                duplicate_frame = timepoints[frames_to_skip[0]-1]
                for skip_frame in frames_to_skip:
                    timepoints[skip_frame] = duplicate_frame
                # timepoints = [x for x in timepoints if x not in skip_frames]
                print(f'skipping frames {frames_to_skip}')
                # ggprint(f'updated timepoints {timepoints}')


        mask = None
        crop_z = False

        if microscope == 'SNOUTY':
            crop_z = True

        pearls_df.loc[index, 'crop_path'] = crop_path
        ### RUN CROP ###
        if crop:
            if not os.path.exists(os.path.join(nellie_path, "completed_crop.txt")) or crop_override:
                # User inputs
                # print(f'cropping to {crop_path}')
                # print(f'cropping from {im_path}')
                print(f'selected timepoints: {timepoints}')
                crop_snout = CropSnout(im_path=im_path, crop_path=crop_path, nellie_path=nellie_path,
                                       image_type=image_type, crop_count=crop_count,
                                       select_timepoints=timepoints, select_channel=select_channel, select_z=crop_z,
                                       project_z=project_z,
                                       time_interval=time_interval, center_coords=center_coords, show_crop=False,
                                       pearl_frame=frame2, dim_res=manual_dim_res[index])
                crop_snout.crop_image()  # Allow the user to select the ROI
                # print(f'{pearls_df["select_mask"][index]=}')
                if pearls_df['select_mask'][index]:
                    crop_snout.select_mask()
                save_completion_file(nellie_path, "completed_crop.txt")
        ### RUN NELLIE ###
        if not os.path.exists(os.path.join(nellie_path, "completed_nellie.txt")) or nellie_override:
            remove_edges = False
            if microscope == 'SNOUTY' and not project_z:
                remove_edges = True
            print(f'running nellie on {crop_path}')
            # print(f'saving nellie outputs to {nellie_path=}')
            import shutil

            if os.path.exists(os.path.join(nellie_path, "nellie_necessities")):
                shutil.rmtree(
                    os.path.join(nellie_path, "nellie_necessities"))  # Deletes the old folder and all its contents

            file_info = FileInfo(filepath=crop_path, output_dir=nellie_path)
            file_info.find_metadata()
            file_info.load_metadata()
            if select_channel > 0:

                file_info.change_selected_channel(select_channel)
            # print(f'selected channel {file_info.ch=}')
            # print(f'image shape: {file_info.shape=}')
            if pearls_df['select_mask'][index]:
                mask = xp.load(os.path.join(nellie_path, 'mask.npy'))
                # print(f'mask shape: {mask.shape}')


            # print(f'remove edges: {remove_edges=}')
            run(file_info, mask=mask, remove_edges=remove_edges)
            save_completion_file(nellie_path, "completed_nellie.txt")
        ### RUN CURVATURE ###
        if not os.path.exists(os.path.join(nellie_path, "completed_curvature.txt")) or curvature_override:
            curvature_path = os.path.join(nellie_path, "curvature_analysis")


            # Create a SkeletonCurvature instance with the metadata
            skel_curvature = SkeletonCurvature(crop_path, nellie_path=nellie_path, save_path=curvature_path,
                                               smoothing_factors=[1.0,1.0],
                                               manual_smoothing=True,
                                               show_individual_plots=False, show_summary_plots=False,
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
                                        visualize=False, select_frames=frame_list, train_rfc=train_rfc)
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
        single_volume_csv = os.path.join(nellie_path, "pearling_stats", 'single_volume_data.csv')
        pm_df = pd.DataFrame()
        if os.path.exists(pearling_stats_csv):
            pm_df = pd.read_csv(pearling_stats_csv)
            # get single_volume data
            if analysis_type == 'single_volume':
                avg_volume = np.mean(pm_df['tubule_volume_sum'])
                CoV_volume = np.std(pm_df['tubule_volume_sum']) / avg_volume * 100
                avg_length = np.mean(pm_df['tubule_length_sum'])
                CoV_length = np.std(pm_df['tubule_length_sum']) / avg_length * 100
                label_count = np.mean(pm_df['label_count'])
                CoV_count = np.std(pm_df['label_count']) / label_count * 100

                event_length = pearls_df['MitoLength'][index]
                pearled_length = pearls_df['pearled_length'][index]
                percent_length = pearled_length / avg_length * 100

                event_area = pearls_df['MitoArea'][index]
                pearled_area = pearls_df['pearled_area'][index]
                percent_area = pearled_area / avg_volume * 100

                total_time = pearls_df['n-timepoints'][index] * time_interval
                events_per_cell = pearls_df['# of events'][index] / pearls_df['# of cells'][index]
                events_per_second = pearls_df['# of events'][index] / total_time
                events_per_10min = events_per_second * 60 * 10
                events_per_mito_per10min = events_per_10min / label_count
                event_probability = (1 - np.exp(-events_per_mito_per10min)) * 100

                # save the single volume data to a csv
                single_volume_data = pd.DataFrame(
                    {'file_count': [index + 1], 'file': folder,
                     'crop_path': crop_path, 'Treatment': [pearls_df['Treatment'][index]],
                     'replicate': [pearls_df['notes'][index]], 'crop_count': [crop_count],
                     'avg_volume': [avg_volume], 'CoV_volume': [CoV_volume],
                     'avg_length': [avg_length], 'CoV_length': [CoV_length],
                     'label_count': [label_count], 'CoV_count': [CoV_count],
                     'event_length': event_length, 'pearled_length': [pearled_length],
                     'percent_length': [percent_length],
                     'event_area': event_area, 'pearled_area': [pearled_area], 'percent_area': [percent_area],
                     'n_timepoints': [pearls_df['n-timepoints'][index]], 'time_interval': [time_interval],
                     'total_time': [total_time], 'events_per_second': [events_per_second],
                     '# of events': [pearls_df['# of events'][index]], '# of mitos': [label_count],
                     '# of cells': [pearls_df['# of cells'][index]], 'events_per_cell': [events_per_cell],
                     'events_per_10min': [events_per_10min], 'events_per_mito_per10min': [events_per_mito_per10min],
                     'event_probability': [event_probability]
                     },
                )
                single_volume_data.to_csv(single_volume_csv, index=False)
        if os.path.exists(single_volume_csv):
            single_volume_data = pd.read_csv(single_volume_csv)
            all_single_volume_data = pd.concat([all_single_volume_data, single_volume_data], ignore_index=True)
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
            results_list.append(merged_rows)
            # results_df = pd.concat([results_df, merged_rows], ignore_index=True)

        # # save the results to a csv
        # if not os.path.exists(final_save_folder):
        #     os.makedirs(final_save_folder)
        # results_df.to_csv(os.path.join(final_save_folder, (current_time_str + 'all_pearling_results.csv')), index=False)

        # open the training data csv
        if train_rfc:
            training_data_csv = os.path.join(nellie_path, "pearling_stats", 'training_label_metrics.csv')
            training_data = pd.DataFrame()
            if os.path.exists(training_data_csv):
                training_data = pd.read_csv(training_data_csv)
                # training_data.insert(0, 'folder', folder)  # Insert 'folder' as the first column
                training_data.insert(0, 'basename', basename)  # Insert 'basename' as the second column
                training_data.insert(1, 'crop_path', crop_path)

            # concat the training data
            # all_training_data = pd.concat([all_training_data, training_data], ignore_index=True)
            # all_training_data.to_csv(os.path.join(final_save_folder, 'rfc_training_data.csv'), index=False)
            training_data_list.append(training_data)

        print(f'completed processing {im_path}')

    # save the results to a csv
    if len(results_list) > 0:
        results_df = pd.concat(results_list, ignore_index=True)
        results_df.to_csv(os.path.join(final_save_folder, (current_time_str + 'all_pearling_results.csv')), index=False)
    if len(training_data_list) > 0:
        all_training_data = pd.concat(training_data_list, ignore_index=True)
        all_training_data.to_csv(os.path.join(final_save_folder, 'rfc_training_data.csv'), index=False)
    if len(all_single_volume_data) > 0:
        all_single_volume_data.to_csv(os.path.join(final_save_folder, 'single_volume_data.csv'), index=False)



    # plot the results
    outer_folder = os.path.join(final_save_folder, 'treatment_plots')
    if not os.path.exists(outer_folder):
        os.makedirs(outer_folder)
    plot_folder = os.path.join(outer_folder, current_time_str)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    norm_methods = ['first_point'] # 'min-max', 'mean', 'median', 'pre-pearl', 'post-pearl', 'max_pearl'
    y_columns = ['intensity_curvature', 'intensity_mean', 'intenstiy_max',
                 'aspect_ratio_mean', 'tortuosity_mean',
                 'lin_vel_mag_mean', 'lin_acc_mag_mean', 'ang_vel_mag_mean', 'ang_acc_mag_mean',
                 'tubule_length_mean', 'tubule_length_sum',
                 'tubule_volume_mean','tubule_volume_sum',
                 'tubule_width_mean','tubule_width_sum',]
         # 'lin_vel_mag_mean', 'lin_acc_mag_mean', 'ang_vel_mag_mean', 'ang_acc_mag_mean',
          #        'intensity_mean', 'intenstiy_max']
            # , 'lin_vel_mag_mean', 'lin_acc_mag_mean', 'ang_vel_mag_mean', 'ang_acc_mag_mean',
            # 'volume_mean', 'volume_sum', 'tubule_volume_sum', 'tubule_volume_mean',
            # 'tubule_length_sum', 'tubule_length_mean',
            # 'tubule_width_sum', 'tubule_width_mean',
            # 'node_width_mean', 'aspect_ratio_mean', 'tortuosity_mean']
            #        #['intensity_curvature',  'intensity_mean', 'tortuosity_mean','tubule_width_mean','tubule_length_mean'] # 'tortuosity_mean','tubule_width_mean', 'tubule_length_mean',
             # 'volume_mean', 'rectangularity_mean', 'peak_peak_distance_mean', 'sphericity_mean', 'solidity_mean'
    graphs = Graph_timecourse()
    # treatments = ['control']
    # results_df = results_df[results_df['treatment'].isin(treatments_to_graph)]
    # treatment_order = ['spontaneous','ionomycin','thapsigargin','valinomycin','NS1619', 'BKA','PXA','digitonin']
    treatments = results_df['treatment'].unique()
    # treatments= ['ionomycin_hypertonic_recovery','hypertonic_ionomycin_recovery']
    # treatments = ['control', 'latrunculinB', 'nocodazole', 'taxol']

    # treatment_order = ['U2OS', 'HEK293', 'RPE1', 'COS7', 'Jurkat_Tcells', 'primary_fibroblasts', 'iPSC_neurons', 'Budding_yeast']
    # select_treatments = ['U2OS', 'primary_fibroblasts']
    #treatment_order = ['elongated','shrunk']
    # select_treatments = ['repeated_v1', 'repeated_v2','hypotonic_recovery','hypertonic','hypotonic_FCCP',]
    select_treatments = ['hypotonic','hypotonic_FCCP']
    # treatment_order = ['spontaneous', 'ionomycin', 'hypotonic', 'FRAP', 'microneedle']
    # treatments = ['control', 'NAC','MitoTempo']
    print("PLOTTING TIMECOURSE GRAPHS")
    # 10, 30 for spon control
    # 3, 20 for Daria data, with 2, first_points
    # 10, 30 for atovostatin, first_points
    # 20, 40 for ETC data, with 5, first_points
    # 25, 250, for osmotic recovery, with 10, control_first
    # 20, 60 for ionophores, with 5, first_points, select_treatment=treatment, interpolate_time=2, viridis, treatment_order, select_treatments=treatment
    pre_window_size = 20
    post_window_size = 127
    for y_col in y_columns:
        # for treatment in treatments:
            # if treatment == 'spontaneous':
            #     pre_window_size = 10
            #     post_window_size = 20
        graphs.plot_treatment_graph(results_df, y_col,
                                    select_treatments=select_treatments, #treatments_to_graph,
                                    select_method=None,
                                    save_folder=plot_folder,
                                    time=current_time_str,
                                    color_scheme=None, # 'viridis',
                                    treatment_order=None, # treatment_order, # 'viridis'
                                    use_median=False,
                                    pre_window_size=pre_window_size,
                                    post_window_size=post_window_size,
                                    normalization='control_first', # 'first_points', 'control_first'
                                    control_first_x_timepoints=5,
                                    transparent=False,
                                    plot_individual_events=False,
                                    show=False,
                                    show_legend=False,
                                    show_titles=False,
                                    interpolate_time=True,
                                    interpolation_time_step=2)
    y_columns = ['intensity_curvature','lin_vel_mag_mean', 'lin_acc_mag_mean', 'ang_vel_mag_mean', 'ang_acc_mag_mean',
                 'peak_peak_distance_mean', 'volume_mean', 't0_pearls_per_micron', 'time_to_peak',
                 'time_to_recovery', 'duration_of_event', 'tubule_length_mean',
                 'medio_axis_mean', 'major_axis_mean', 'minor_axis_mean',
                 'volume_mean', 'volume_sum', 'tubule_volume_sum', 'tubule_volume_mean',
                 'tubule_length_sum', 'tubule_length_mean',
                 'tubule_width_sum', 'tubule_width_mean',
                 'node_width_mean', 'aspect_ratio_mean', 'tortuosity_mean']
    print("PLOTTING BAR PLOTS")
    # for y_col in y_columns:
    #     graphs.plot_treatment_barplot(results_df, y_col,
    #                                   select_treatments=None,
    #                                   select_method=None,
    #                                   save_folder=plot_folder,
    #                                   time=current_time_str,
    #                                   pre_window_size=pre_window_size,
    #                                   post_window_size=post_window_size,
    #                                   normalization=None,
    #                                   transparent=False,
    #                                   color_scheme='viridis',
    #                                   show=False,
    #                                   show_legend=False,
    #                                   show_sigs=False,
    #                                   treatment_order=None, # treatment_order, # treatment_order,
    #                                   compare_window_mean=None)
    #
    # # Initialize the plotter
    # from graphs_v2 import TreatmentGraphPlotter
    # plotter = TreatmentGraphPlotter()

    # Example DataFrame (replace with your actual data)
    # Ensure your DataFrame has the required columns:
    # 'treatment', 'induction_method', 'file', 'time_interval',
    # 'pre_pearl_frame', 'post_pearl_frame', 'pearl_frame',
    # 'timepoint', 'timepoint_seconds', and the y_column.

    # # Call the plot_treatment_graph method
    # for y_col in y_columns:
    #     plotter.plot_treatment_graph(
    #         input_data=results_df,
    #         y_column=y_col,  # Replace with your y_column
    #         select_treatments=None,  # Replace with your treatments
    #         select_method=None,  # Replace with induction methods if any or set to None
    #         save_folder=plot_folder,  # Replace with your save directory
    #         time=current_time_str,  # Replace with your time identifier
    #         pre_window_size=pre_window_size,  # Number of frames before the event
    #         post_window_size=post_window_size,  # Number of frames after the event
    #         color_scheme=None,  # Choose 'viridis' or 'custom'
    #         treatment_order=None,  # Specify if you have a preferred order
    #         plot_individual_events=False,  # Set to True to plot individual files
    #         transparent=False,  # Set to True for transparent background
    #         show=False,  # Set to False to not display the plot
    #         show_legend=True,  # Set to False to hide the legend
    #         show_titles=False,  # Set to False to hide the titles
    #         control_first_x_timepoints=3  # Number of initial timepoints for normalization
    #     )
