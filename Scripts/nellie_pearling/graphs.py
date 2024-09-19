import os
import datetime as dt
import numpy as np
import pandas as pd
import cupy as xp
import pathlib
import tifffile as tif
import logging

from napari.utils import notifications

logging.getLogger('matplotlib').setLevel(logging.WARNING)



class Graph_timecourse:
    def __init__(self):
        pass

    def plot_treatment_graph(self, input_data, y_column, select_treatments, select_method, save_folder, time=None,
                             pre_window_size=5, post_window_size=30,
                             normalization='min-max', plot_individual_events=False):
        import matplotlib.pyplot as plt
        from itertools import cycle
        import numpy as np
        import pandas as pd
        import os
        from scipy.stats import sem  # Import SEM calculation

        """
        Plots a graph with time in seconds on the x-axis and the specified y_column on the y-axis,
        colored by the 'treatment' column. The graph shows data within the specified frame windows
        before and after the 'pearl_frame' event for each file.

        Args:
        - input_data (pd.DataFrame): The input DataFrame containing data to be plotted.
        - y_column (str): The name of the column to plot on the y-axis.
        - save_folder (str): Folder where the plot will be saved.
        - time (str, optional): Time identifier to be used in the saved file name.
        - window_size (int, optional): Number of frames before and after the pearl event to include.
        - normalization (str, optional): The normalization method to apply to y_column ('min-max', 'mean', 'median').
        - plot_individual_events (bool, optional): Whether to plot individual events.

        Returns:
        - None: Displays and saves the plot.
        """
        plt.figure(figsize=(10, 6))

        # Filter dataset to chosen treatments and/or induction methods
        if select_treatments:
            input_data = input_data[input_data['treatment'].isin(select_treatments)]
        if select_method:
            input_data = input_data[input_data['induction_method'].isin(select_method)]

        # Initialize the 'aligned_frame' and 'aligned_time_seconds' columns in the main DataFrame
        input_data['aligned_frame'] = None
        input_data['aligned_time_seconds'] = None

        # Get unique treatments and induction methods
        treatments = input_data['treatment'].unique()
        induction_methods = input_data['induction_method'].unique()

        # Define specific colors for each treatment
        treatment_colors = {
            'control': 'gray',
            'fccp': 'yellow',
            'nocodazole': 'blue',
            'taxol': 'green',
            'latrunculinb': 'red'
            # Add more treatments and their colors as needed
        }

        # Generate a color cycle for treatments that are not predefined
        color_cycle = cycle(plt.get_cmap('tab10').colors)
        for treatment in treatments:
            if treatment.lower() not in treatment_colors:
                treatment_colors[treatment.lower()] = next(color_cycle)

        print(f"treatment_colors: {treatment_colors}")

        # Define line styles for each induction method
        line_styles = {
            'FRAP': '-',
            'spontaneous': '--',
            'method_3': '-.'
            # Add more methods and their styles as needed
        }

        # Assign line styles to missing induction methods
        for method in induction_methods:
            if method not in line_styles:
                line_styles[method] = next(cycle(['-', '--', '-.', ':']))

        # To keep track of unique treatment-induction combinations for the legend
        legend_handles = {}

        # Loop through each file and calculate the aligned_frame and aligned_time_seconds for each
        for file in input_data['file'].unique():
            # Filter data by file
            file_data = input_data[input_data['file'] == file]

            time_interval = file_data['time_interval'].diff().mean()

            # Get the 'pearl_frame' for the current file group
            pearl_frame = file_data['pre_pearl_frame'].values[0]

            # Calculate 'aligned_frame' numbers
            aligned_frame = file_data['timepoint'] - pearl_frame

            # Update the 'aligned_frame' in the main DataFrame
            input_data.loc[file_data.index, 'aligned_frame'] = aligned_frame

            # Get the time in seconds corresponding to the pearl_frame
            pearl_time_seconds_array = file_data.loc[file_data['timepoint'] == pearl_frame, 'timepoint_seconds'].values
            if len(pearl_time_seconds_array) == 0:
                print(f"Pearl time in seconds not found for file {file}. Skipping.")
                continue
            pearl_time_seconds = pearl_time_seconds_array[0]
            # print(f"Pearl time in seconds: {pearl_time_seconds}")

            # Calculate aligned_time_seconds
            aligned_time_seconds = file_data['timepoint_seconds'] - pearl_time_seconds
            # print(f"Aligned time in seconds: {aligned_time_seconds}")

            # Update the 'aligned_time_seconds' in the main DataFrame
            input_data.loc[file_data.index, 'aligned_time_seconds'] = aligned_time_seconds
            file_data['aligned_time_seconds'] = aligned_time_seconds

            # Normalize the y_column data
            # Ensure that the y_column is numeric
            input_data.loc[file_data.index, y_column] = pd.to_numeric(input_data.loc[file_data.index, y_column],
                                                                      errors='coerce')

            # Normalize the y_column data
            if normalization == 'none':
                pass
            elif normalization == 'min-max':
                min_val = file_data[y_column].min()
                max_val = file_data[y_column].max()
                if max_val != min_val:  # Check if not zero range
                    input_data.loc[file_data.index, y_column] = (file_data[y_column] - min_val) / (max_val - min_val)
                else:
                    notifications.show_error("Zero range for min-max normalization.")
                    input_data.loc[file_data.index, y_column] = np.nan  # or handle differently

            elif normalization == 'mean':
                mean_val = file_data[y_column].mean()
                if mean_val != 0:
                    input_data.loc[file_data.index, y_column] = file_data[y_column] / mean_val
                else:
                    notifications.show_error("Zero range mean normalization.")
                    input_data.loc[file_data.index, y_column] = np.nan
            elif normalization == 'first_point':
                # Filter data within the specified frame windows
                first_frame_index = file_data[file_data['aligned_time_seconds'] >= -pre_window_size].index[0]
                # # Get the pearl frame's time in seconds
                # index = file_data[file_data['timepoint'] == pearl_frame].index[0]
                # pearl_time_seconds = input_data.loc[index, 'aligned_time_seconds']
                # print(f'Pearl time in seconds: {pearl_time_seconds}')
                # # Calculate the time for the first point (pre_window_size seconds before pearl_time_seconds)
                # target_time = pre_window_size
                # print(f'Target time: {target_time}')
                # # Find the frame that is closest to this target_time
                # first_frame_index = (file_data['aligned_time_seconds'] - target_time).abs().idxmin()
                # print(f'First frame index: {first_frame_index}')
                # Now, normalize based on the values around the first frame
                first_points = file_data.loc[first_frame_index:first_frame_index+1, y_column]
                # print(f'First points: {first_points}')

                if len(first_points) == 2:
                    first_avg = first_points.mean()
                    input_data.loc[file_data.index, y_column] = file_data[y_column] / first_avg if first_avg != 0 else np.nan

            elif normalization == 'median':
                median_val = file_data[y_column].median()
                if median_val != 0:
                    input_data.loc[file_data.index, y_column] = file_data[y_column] / median_val
                else:
                    notifications.show_error("Zero range median normalization.")
                    input_data.loc[file_data.index, y_column] = np.nan
            elif normalization == 'pre-pearl':
                pre_pearl_timepoint = file_data['pre_pearl_frame'].values[0]
                pre_pearl_indices = file_data[(file_data['timepoint'] >= pre_pearl_timepoint - 3) &
                                              (file_data['timepoint'] < pre_pearl_timepoint)].index

                if len(pre_pearl_indices) == 3:
                    pre_pearl_avg = file_data.loc[pre_pearl_indices, y_column].mean()
                    input_data.loc[file_data.index, y_column] = file_data[
                                                                    y_column] / pre_pearl_avg if pre_pearl_avg != 0 else np.nan

            elif normalization == 'post-pearl':
                post_pearl_timepoint = file_data['post_pearl_frame'].values[0]
                post_pearl_indices = file_data[(file_data['timepoint'] > post_pearl_timepoint) &
                                               (file_data['timepoint'] <= post_pearl_timepoint + 3)].index

                if len(post_pearl_indices) == 3:
                    post_pearl_avg = file_data.loc[post_pearl_indices, y_column].mean()
                    input_data.loc[file_data.index, y_column] = file_data[
                                                                    y_column] / post_pearl_avg if post_pearl_avg != 0 else np.nan

            elif normalization == 'max_pearl':
                max_pearl_timepoint = file_data['pearl_frame'].values[0]
                max_pearl_indices = file_data[(file_data['timepoint'] >= max_pearl_timepoint - 1) &
                                              (file_data['timepoint'] <= max_pearl_timepoint + 1)].index

                if len(max_pearl_indices) == 3:
                    max_pearl_avg = file_data.loc[max_pearl_indices, y_column].mean()
                    input_data.loc[file_data.index, y_column] = file_data[y_column] / max_pearl_avg if max_pearl_avg != 0 else np.nan

            else:
                raise ValueError("Invalid normalization method. Choose from 'min-max', 'mean', 'median', 'post-pearl', 'max-pearl', 'or 'pre-pearl'.")

        # Convert 'aligned_frame' and 'aligned_time_seconds' to numeric and drop NaNs
        input_data['aligned_frame'] = pd.to_numeric(input_data['aligned_frame'], errors='coerce')
        input_data['aligned_time_seconds'] = pd.to_numeric(input_data['aligned_time_seconds'], errors='coerce')
        input_data = input_data.dropna(subset=['aligned_frame', 'aligned_time_seconds', y_column])

        # Now all aligned_frame and aligned_time_seconds are set in input_data; proceed to plotting
        for treatment in treatments:
            # Filter data by Treatment
            treatment_data = input_data[input_data['treatment'] == treatment]

            # If plot_individual_events is True, plot individual lines per file
            if plot_individual_events:
                for file in treatment_data['file'].unique():
                    file_data = treatment_data[treatment_data['file'] == file]

                    # Filter data within the specified frame windows
                    aligned_data = file_data[(file_data['aligned_time_seconds'] >= -pre_window_size) &
                                             (file_data['aligned_time_seconds'] <= post_window_size)]

                    # Set the color and line style
                    color = treatment_colors.get(treatment.lower(), 'black')
                    induction_method = file_data['induction_method'].iloc[0]
                    line_style = line_styles.get(induction_method, '-')

                    # Plot individual events
                    plt.plot(aligned_data['aligned_time_seconds'], aligned_data[y_column],
                             alpha=0.1, color=color, linestyle=line_style)

            # Filter data within the specified frame windows
            treatment_data = treatment_data[(treatment_data['aligned_time_seconds'] >= -pre_window_size) &
                                            (treatment_data['aligned_time_seconds'] <= post_window_size)]

            # Ensure there are multiple files
            num_files = treatment_data['file'].nunique()
            if num_files < 2:
                print(f"Not enough files for {treatment} to compute error bars.")
                continue

            # === Interpolation and Aggregation ===

            # Collect all aligned_time_seconds within the window
            all_aligned_times = []

            for file in treatment_data['file'].unique():
                file_data = treatment_data[treatment_data['file'] == file]
                # Filter data within the frame window
                aligned_data = file_data[(file_data['aligned_time_seconds'] >= -pre_window_size) &
                                         (file_data['aligned_time_seconds'] <= post_window_size)]
                all_aligned_times.extend(aligned_data['aligned_time_seconds'].dropna().values)

            # Convert to numpy array
            all_aligned_times = np.array(all_aligned_times, dtype=float)
            # Remove NaNs
            all_aligned_times = all_aligned_times[~np.isnan(all_aligned_times)]

            # Determine min and max
            min_time = np.floor(np.min(all_aligned_times))
            max_time = np.ceil(np.max(all_aligned_times))

            # Define the common time grid
            time_step = 2  # Adjust the time_step as needed (e.g., 1 second)
            common_timepoints = np.arange(min_time, max_time + time_step, time_step)
            common_timepoints = common_timepoints.astype(float)  # Ensure it's float

            # Create a DataFrame to collect interpolated values
            interpolated_values = pd.DataFrame({'aligned_time_seconds': common_timepoints})

            # For each file, interpolate the y_column values onto common_timepoints
            for file in treatment_data['file'].unique():
                file_data = treatment_data[treatment_data['file'] == file]

                # Filter data within the frame window
                file_data = file_data[(file_data['aligned_time_seconds'] >= -pre_window_size) &
                                      (file_data['aligned_time_seconds'] <= post_window_size)]

                # Get the aligned_time_seconds and y_column for this file
                times = file_data['aligned_time_seconds'].astype(float).values
                values = file_data[y_column].astype(float).values

                # Remove NaNs from times and values
                valid_indices = ~np.isnan(times) & ~np.isnan(values)
                times = times[valid_indices]
                values = values[valid_indices]

                # Check if there are enough data points to interpolate
                if len(times) < 2:
                    print(f"Not enough data points to interpolate for file {file}")
                    continue

                # Sort times and values
                sorted_indices = np.argsort(times)
                times = times[sorted_indices]
                values = values[sorted_indices]

                # Remove duplicate times (if any)
                unique_times, unique_indices = np.unique(times, return_index=True)
                times = times[unique_indices]
                values = values[unique_indices]

                # Ensure times, values, and common_timepoints are float64
                times = times.astype(np.float64)
                values = values.astype(np.float64)
                common_timepoints = common_timepoints.astype(np.float64)

                # Debugging: Print dtypes and values
                # print(f"Interpolating {file}...")
                # print(f"Times dtype: {times.dtype}")
                # print(f"Values dtype: {values.dtype}")
                # print(f"Common timepoints dtype: {common_timepoints.dtype}")
                # print(f"Times: {times}")
                # print(f"Values: {values}")
                # print(f"Common timepoints: {common_timepoints}")

                # Use np.interp to interpolate
                # Values outside the range will be set to np.nan
                interpolated = np.interp(common_timepoints, times, values, left=np.nan, right=np.nan)

                # Add the interpolated values to the DataFrame
                interpolated_values[file] = interpolated

            # Compute mean and SEM across files at each common timepoint
            # Exclude NaN values when computing mean and SEM
            mean_values = interpolated_values.drop(columns=['aligned_time_seconds']).mean(axis=1, skipna=True)
            sem_values = interpolated_values.drop(columns=['aligned_time_seconds']).apply(sem, axis=1,
                                                                                          nan_policy='omit')

            timepoints = interpolated_values['aligned_time_seconds']

            # Plot the average line with SEM as shaded areas
            plt.errorbar(
                timepoints,
                mean_values,
                # yerr=sem_values,  # Uncomment if you prefer error bars instead of shaded areas
                label=f'{treatment}',
                linewidth=2,
                color=treatment_colors.get(treatment.lower(), 'black'),
                alpha=0.8,
                capsize=2,
                elinewidth=0.8
            )

            # Fill the area between the upper and lower SEM boundaries
            plt.fill_between(
                timepoints,
                mean_values - sem_values,
                mean_values + sem_values,
                color=treatment_colors.get(treatment.lower(), 'black'),
                alpha=0.2
            )

            # Add a legend entry if not already added
            if treatment not in legend_handles:
                legend_handles[treatment] = plt.Line2D(
                    [], [], color=treatment_colors.get(treatment.lower(), 'black'),
                    label=f'{treatment}'
                )

        plt.axvline(0, color='black', linestyle='--', linewidth=1)  # Line at the pearl event
        plt.xlabel('Aligned Time (seconds relative to start of pearling event)')
        y_column_title = ' '.join([word.capitalize() for word in y_column.split('_')])
        plt.ylabel(f'{normalization} {y_column_title}')
        # capitalize the first letter of y_column for the title and break up by '_'
        plt.title(f'{y_column_title} Over Time')
        # Add legend with only treatment colors
        plt.legend(handles=list(legend_handles.values()), loc='best', title='Treatment')
        plt.grid(True)

        # Save the plot as PNG and SVG
        save_path = os.path.join(save_folder, f'{time}treatment_effect_{y_column}_over_time_{normalization}')
        plt.savefig(f'{save_path}.png', format='png', dpi=300)
        plt.savefig(f'{save_path}.svg', format='svg')

        plt.show()


