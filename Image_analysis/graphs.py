# Description: This script contains functions to plot graphs for timecourse data.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
from napari.utils import notifications
from itertools import cycle, combinations
from scipy.stats import sem, mannwhitneyu
logging.getLogger('matplotlib').setLevel(logging.WARNING)




class Graph_timecourse:
    def __init__(self):
        """
        Initializes the TreatmentGraphPlotter with a configured logger.
        """
        # Configure logging to output to both console and file
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler("treatment_graph_plotter.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def plot_treatment_graph(self, input_data, y_column, select_treatments, select_method, save_folder, time=None,
                             pre_window_size=5, post_window_size=30, color_scheme='viridis', treatment_order=None, use_median=False,
                             normalization='control_first', control_first_x_timepoints=5, plot_individual_events=False, transparent=False,
                             show=False, show_legend=True, show_titles=True, interpolate_time=False, interpolation_time_step=2 ):
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
        # plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots()

        # Filter dataset to chosen treatments and/or induction methods
        all_treatments = input_data['treatment'].unique()
        if select_treatments:
            # check if select_treatments is a string or a list
            if isinstance(select_treatments, str):
                select_treatments = [select_treatments]
            input_data = input_data[input_data['treatment'].isin(select_treatments)]
        if select_method:
            input_data = input_data[input_data['induction_method'].isin(select_method)]

        # Initialize the 'aligned_frame' and 'aligned_time_seconds' columns in the main DataFrame
        input_data['aligned_frame'] = None
        input_data['aligned_time_seconds'] = None

        # Get unique treatments and induction methods
        treatments = input_data['treatment'].unique()
        # Reorder so that 'control' is first if present
        if treatment_order:
            treatments = treatment_order
        else:
            if 'control' in treatments:
                treatments = np.array(['control'] + [treatment for treatment in treatments if treatment != 'control'])
            else:
                treatments = np.array(treatments)

        induction_methods = input_data['induction_method'].unique()

        # Assign colors based on the selected color scheme
        if color_scheme == 'viridis':
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i / max(1, len(treatments) - 1)) for i in range(len(treatments))]
            # print(f"colors: {colors}")
            # print(f"treatments: {treatments}")
        else:
            # Existing custom color mapping
            gray = '#696969' if not transparent else '#D3D3D3'
            treatment_colors = {
                'control': gray,
                'fccp': 'orange',
                'oligomycin': 'darkblue', # SteelBlue, 'darkblue', '#4682B4'
                'rotenone': 'tomato', # 'tomato', 'darkred', '#FF6347'
                'antimycina': 'purple', # 'plum', 'purple', '#DDA0DD'
                'nocodazole':  'darkorange',
                'taxol': 'lightgreen',
                'latrunculinb': '#4682B4',
                'mitoflippertr': 'darkgreen',
                'atorvastatin': 'orange',
                'kcl': 'darkorange',
                'elongated': 'darkblue',
                'shrunk': 'darkred',
                'ionomycin_hypertonic_recovery': '#4682B4', # SteelBlue, 'darkblue'
                'hypertonic_ionomycin_recovery': 'darkorange',
                'nac': 'lightgreen',
                'mitotempo': 'darkblue',
                'hypertonic': '#FFA500',
                'hypotonic': '#696969', # '#696969' (gray) '#8B0000', (red)
                'hypotonic_fccp': '#B59A19',
                # Add more treatments and their colors as needed
            }
            # Generate a color cycle for treatments that are not predefined
            color_cycle = cycle(plt.get_cmap('tab10').colors)
            for treatment in treatments:
                if treatment.lower() not in treatment_colors:
                    print(f"Color not found for treatment: {treatment}. Assigning a new color.")
                    treatment_colors[treatment.lower()] = next(color_cycle)

            colors = [treatment_colors.get(t.lower(), 'black') for t in treatments]


        # print(f"treatment_colors: {treatment_colors}")

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

        # print color treatment combinations
        # for i, (treatment, color) in enumerate(zip(treatments, colors)):
        #      print(f"{treatment}: {color}")

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
            elif normalization == 'first_points':
                # Filter data within the specified frame windows
                first_frame_index = file_data[file_data['aligned_time_seconds'] >= -pre_window_size].index[0]
                first_points = file_data.loc[first_frame_index:first_frame_index+control_first_x_timepoints-1, y_column]
                # print(f'First points: {first_points}')

                if len(first_points) == control_first_x_timepoints:
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

        if normalization == 'control_first':
            # normalize to first x points in of control
            control_baseline = self.compute_control_baseline(input_data, y_column, pre_window_size, control_first_x_timepoints)
            # input_data[y_column] = pd.to_numeric(input_data[y_column], errors='coerce')
            input_data[y_column] = input_data[y_column] / control_baseline
            self.logger.info(f"Normalized '{y_column}' based on control baseline.")

        # Convert 'aligned_frame' and 'aligned_time_seconds' to numeric and drop NaNs
        input_data['aligned_frame'] = pd.to_numeric(input_data['aligned_frame'], errors='coerce')
        input_data['aligned_time_seconds'] = pd.to_numeric(input_data['aligned_time_seconds'], errors='coerce')
        input_data = input_data.dropna(subset=['aligned_frame', 'aligned_time_seconds', y_column])

        save_path = os.path.join(save_folder, f'{time}_treatment_effect_over_time_{normalization}.csv')
        input_data.to_csv(save_path.replace('.csv', '_raw.csv'), index=False)

        graph_csv = pd.DataFrame()

        # Now all aligned_frame and aligned_time_seconds are set in input_data; proceed to plotting
        for i, treatment in enumerate(treatments):

            if select_treatments is not None:
                if treatment not in select_treatments:
                    continue
            # Filter data by Treatment
            treatment_data = input_data[input_data['treatment'] == treatment]
            # print out the number of files for each treatment
            # print(f"Number of files for {treatment}: {treatment_data['file'].nunique()}")

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

            # === Interpolation and Aggregation ===
            if interpolate_time:

                # Ensure there are multiple files
                num_files = treatment_data['file'].nunique()
                if num_files < 2:
                    print(f"Not enough files for {treatment} to compute error bars.")
                    continue

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
                # print(f'treatment: {treatment}')
                # print(f"Min time: {min_time}, Max time: {max_time}")

                # Define the common time grid
                time_step = interpolation_time_step  # Adjust the time_step as needed (e.g., 1 second)
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
                    if use_median:
                        mean_values = interpolated_values.drop(columns=['aligned_time_seconds']).median(axis=1, skipna=True)
                    sem_values = interpolated_values.drop(columns=['aligned_time_seconds']).apply(sem, axis=1,
                                                                                                  nan_policy='omit')
                    n_values = interpolated_values.drop(columns=['aligned_time_seconds']).count(axis=1)

                    timepoints = interpolated_values['aligned_time_seconds']
            else:
                # Compute mean and SEM across files at each timepoint
                mean_values = treatment_data.groupby('aligned_time_seconds')[y_column].mean()
                if use_median:
                    mean_values = treatment_data.groupby('aligned_time_seconds')[y_column].median()
                sem_values = treatment_data.groupby('aligned_time_seconds')[y_column].sem()
                n_values = treatment_data.groupby('aligned_time_seconds')[y_column].count()
                timepoints = mean_values.index




            graph_csv = pd.concat([graph_csv, pd.DataFrame({'treatment': treatment,'time': timepoints, 'mean': mean_values, 'sem': sem_values, 'n': n_values})])


            # Plot the average line with SEM as shaded areas
            plt.errorbar(
                timepoints,
                mean_values,
                # yerr=sem_values,  # Uncomment if you prefer error bars instead of shaded areas
                label=f'{treatment}',
                linewidth=3,
                color= colors[i], #treatment_colors.get(treatment.lower(), 'black'),
                alpha=1.0,
                capsize=2,
                elinewidth=0.8
            )

            # Fill the area between the upper and lower SEM boundaries
            plt.fill_between(
                timepoints,
                mean_values - sem_values,
                mean_values + sem_values,
                color=colors[i], # treatment_colors.get(treatment.lower(), 'black'),
                alpha=0.3
            )

            # Add a legend entry if not already added
            if treatment not in legend_handles:
                legend_handles[treatment] = plt.Line2D(
                    [], [], color= colors[i], #treatment_colors.get(treatment.lower(), 'black'),
                    label=f'{treatment}'
                )
        # Example of custom font sizes
        title_fontsize = 18
        label_fontsize = 15
        tick_fontsize = 15
        legend_fontsize = 11
        legend_title_fontsize = 14
        color = 'black'
        if transparent:
            color = 'white'
        # plt.axvline(0, color=color, linestyle='--', linewidth=2, alpha=1.0)  # Line at the pearl event
        # plt.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7) # laser stimulation line
        plt.axhline(1, color=color, linestyle='--', linewidth=1)  # Line at the horizontal baseline
        # plt.axvline(100, color="blue", linestyle='--', linewidth=2)  # Line at the hyperosmotic recovery event
        # plt.axvline(135, color="darkorange", linestyle='--', linewidth=2)  # Line at the DMEM recovery event
        y_column_title = ' '.join([word.capitalize() for word in y_column.split('_')])
        normalization_title = ' '.join([word.lower() for word in normalization.split('_')])
        if show_titles:
            plt.xlabel('Aligned Time (sec)', color=color, fontsize=label_fontsize)
            plt.ylabel(f'{y_column_title} ({normalization_title})', color=color, fontsize=label_fontsize)
            # capitalize the first letter of y_column for the title and break up by '_'
            plt.title(f'{y_column_title} Over Time', color=color, fontsize=title_fontsize)
        # Add legend with only treatment colors
        if show_legend:
            legend = plt.legend(handles=list(legend_handles.values()), loc='upper right',
                                # bbox_to_anchor=(0.99, 0.99),
                                # title='Treatment',
                                fontsize=legend_fontsize,  # Legend text font size
                                # title_fontsize=legend_title_fontsize  # Legend title font size
                                ) # loc='best',
            legend.get_frame().set_alpha(0.2) # Set legend frame to transparent
            plt.setp(legend.get_texts(), color=color)  # Legend text
        # plt.setp(legend.get_title(), color=color)  # Legend title

        # plt.grid(True)

        # Set white tick labels
        plt.tick_params(axis='x', colors=color, labelsize=tick_fontsize)
        plt.tick_params(axis='y', colors=color, labelsize=tick_fontsize)

        # Set the y-axis to a logarithmic scale
        # plt.yscale('log')
        # ax.set_ylim(0, 8)  # Set the visual limits

        # Set white borders (spines) around the graph
        ax.spines['top'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['right'].set_color(color)

        # Set white grid lines
        plt.grid(True, color=color, alpha=0.3)

        # Set the background color to transparent
        if transparent:
            ax.patch.set_alpha(0)

        # selected_treatments string
        select = 'all_treatments'
        if select_treatments:
            select = '_'.join(select_treatments)
        lim = 20
        if len(select) > lim:
            # just take first 25 characters
            select = select[:lim]


        # Save the plot as PNG and SVG
        save_path = os.path.join(save_folder, f'{time}_{y_column}_over_time_{normalization}_{select}')
        plt.savefig(f'{save_path}.png', format='png', dpi=300, transparent=True)
        plt.savefig(f'{save_path}.svg', format='svg', transparent=True)
        save_path = os.path.join(save_folder, f'{time}_{y_column}_over_time_{normalization}_{select}.csv')
        graph_csv.to_csv(save_path, index=False)

        if show:
            plt.show()
        plt.close(fig)

    def compute_control_baseline(self, input_data, y_column, pre_window_size, control_first_x_timepoints):
        """
        Computes the control baseline by averaging the first X timepoints starting at the pre-window period.

        Args:
            input_data (pd.DataFrame): The input DataFrame with aligned time.
            y_column (str): The column to normalize.
            pre_window_size (int): Number of frames before the event to define the pre-window.
            control_first_x_timepoints (int): Number of initial timepoints for averaging.

        Returns:
            float: Control baseline for normalization.
        """
        control_data = input_data[input_data['treatment'] == 'control']
        if control_data.empty:
            if 'hypotonic' in input_data['treatment'].unique():
                control_data = input_data[input_data['treatment'] == 'hypotonic']
            else:
                self.logger.error("No control group data found for 'control_avg' normalization.")
                raise ValueError("No control group data found for 'control_avg' normalization.")

        control_averages = []

        for file in control_data['file'].unique():
            file_data = control_data[control_data['file'] == file]

            # Select data starting from the pre-window period
            pre_window_data = file_data[file_data['aligned_time_seconds'] >= -pre_window_size]
            pre_window_data_sorted = pre_window_data.sort_values('aligned_time_seconds')

            # Select the first X timepoints
            first_x = pre_window_data_sorted.head(control_first_x_timepoints)

            if len(first_x) < control_first_x_timepoints:
                self.logger.warning(
                    f"Not enough timepoints in control file {file}. Expected {control_first_x_timepoints}, got {len(first_x)}. Skipping this file.")
                continue

            # Calculate the average of y_column
            avg = first_x[y_column].mean()

            if pd.isna(avg):
                self.logger.warning(f"Average is NaN for control file {file}. Skipping this file.")
                continue

            control_averages.append(avg)

        if not control_averages:
            self.logger.error("No valid control averages computed for 'control_avg' normalization.")
            raise ValueError("No valid control averages computed for 'control_avg' normalization.")

        control_baseline = np.mean(control_averages)
        if control_baseline == 0:
            self.logger.error("Control baseline average is zero, cannot normalize.")
            raise ValueError("Control baseline average is zero, cannot normalize.")

        self.logger.info(
            f"Control baseline (average of first {control_first_x_timepoints} timepoints starting at pre-window): {control_baseline}")
        return control_baseline

    def plot_treatment_barplot(
            self,
            input_data,
            y_column,
            select_treatments,
            select_method,
            save_folder,
            time=None,
            normalization=None,
            color_scheme='viridis',
            plot_individual_events=False,
            transparent=False,
            show=False,
            show_legend=True,
            show_sigs=True,
            pre_window_size=5,
            post_window_size=30,
            treatment_order=None,
            compare_window_mean=None,

    ):
        """
        Plots a barplot comparing treatment groups using the y_column value at the pearl_timepoint for each file.
        The barplot includes individual datapoints, error bars, and p-value annotations
        using a non-parametric multiple comparisons statistical method.

        Args:
        - input_data (pd.DataFrame): The input DataFrame containing data to be plotted.
        - y_column (str): The name of the column to plot.
        - select_treatments (list): List of treatments to include.
        - select_method (list): List of induction methods to include.
        - save_folder (str): Folder where the plot will be saved.
        - time (str, optional): Time identifier to be used in the saved file name.
        - normalization (str, optional): Normalization method ('none', 'min-max', 'mean', etc.).
        - plot_individual_events (bool, optional): Whether to plot individual events.
        - transparent (bool, optional): Whether the plot background is transparent.
        - show (bool, optional): Whether to display the plot.
        - show_legend (bool, optional): Whether to show the legend.
        - pre_window_size (int, optional): Pre-window size for normalization.
        - post_window_size (int, optional): Post-window size for normalization.

        Returns:
        - None: Displays and saves the plot.
        """
        # Check if 'pearl_timepoint' column exists
        if 'pearl_timepoint' not in input_data.columns:
            raise ValueError("The input_data must contain a 'pearl_timepoint' column.")

        # Filter dataset to chosen treatments and/or induction methods
        if select_treatments:
            input_data = input_data[input_data['treatment'].isin(select_treatments)]
        if select_method:
            input_data = input_data[input_data['induction_method'].isin(select_method)]

        # Apply normalization if needed
        if normalization is not None:
            input_data = self._normalize_data(input_data, y_column, normalization, pre_window_size, post_window_size)

        # Extract the value at the pearl_timepoint for each file
        if compare_window_mean is None:
            pearl_values = self._extract_pearl_values_from_column(input_data, y_column)
        else:
            pearl_values = self._extract_mean_values_around_pearl(input_data, y_column, pre_window_size=compare_window_mean/2, post_window_size=compare_window_mean/2)

        # Drop NaN values in y_column
        pearl_values = pearl_values.dropna(subset=[y_column])



        # Get unique treatments
        treatments = pearl_values['treatment'].unique()
        # Reorder so that 'control' is first if present
        if treatment_order:
            treatments = treatment_order
        else:
            if 'control' in treatments:
                treatments = np.array(['control'] + [treatment for treatment in treatments if treatment != 'control'])
            else:
                treatments = np.array(treatments)

        # Assign colors using the Viridis colormap
        if color_scheme == 'viridis':
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i / max(1, len(treatments) - 1)) for i in range(len(treatments))]
        else:
            # Existing custom color mapping
            gray = '#696969' if not transparent else '#D3D3D3'
            treatment_colors = {
                'control': gray,
                'fccp': 'orange',
                'nocodazole': 'blue',
                'taxol': 'green',
                'latrunculinb': 'red',
                'mitoflippertr': 'darkgreen',
                'atorvastatin': 'orange',
                'kcl': 'darkorange',
                'elongated': 'darkblue',
                'shrunk': 'darkred',

                # Add more treatments and their colors as needed
            }

            # Generate a color cycle for treatments that are not predefined
            color_cycle = cycle(plt.get_cmap('tab10').colors)
            for treatment in treatments:
                if treatment.lower() not in treatment_colors:
                    treatment_colors[treatment.lower()] = next(color_cycle)

            colors = [treatment_colors.get(t.lower(), 'black') for t in treatments]

        # Prepare data for plotting
        plot_data = []
        for treatment in treatments:
            treatment_data = pearl_values[pearl_values['treatment'] == treatment][y_column].dropna().values
            plot_data.append(treatment_data)

        # Calculate means and SEM
        means = [np.mean(data) for data in plot_data]
        sems = [sem(data) if len(data) > 1 else 0 for data in plot_data]

        # Create bar plot
        bar_positions = np.arange(len(treatments))
        bar_width = 0.6

        # Initialize the plot with a single figure and axes
        fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size for better readability

        bars = ax.bar(
            bar_positions, means, yerr=sems, align='center',
            alpha=0.8, ecolor='black', capsize=10,
            color=colors,  # Use Viridis colors
            width=bar_width
        )

        # Set x-axis ticks and labels with 45-degree rotation
        # Set x-axis ticks and labels with 45-degree rotation and formatted labels
        formatted_labels = [label.replace('_', ' ').title() for label in treatments]
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(formatted_labels, fontsize=38, rotation=25, ha='right')  # Increased font size and angled labels

        # Overlay individual data points with matching semi-transparent colors and thin borders
        for i, data in enumerate(plot_data):
            if len(data) == 0:
                continue  # Skip if no data points
            # Scatter with matching color, semi-transparent, larger size, thin semi-transparent border
            ax.scatter(
                np.random.normal(bar_positions[i], 0.04, size=len(data)),
                data,
                color=colors[i],
                edgecolor='gray',
                alpha=0.6,
                s=300,  # Larger size
                linewidth=0.5,
                zorder=10
            )

        # Perform pairwise Mann-Whitney U tests with Bonferroni correction
        p_values = {}
        comparisons = list(combinations(range(len(treatments)), 2))
        for (i, j) in comparisons:
            data1 = plot_data[i]
            data2 = plot_data[j]
            if len(data1) < 2 or len(data2) < 2:
                p = np.nan  # Not enough data for statistical test
            else:
                stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
            p_values[(i, j)] = p

        # Apply Bonferroni correction
        num_comparisons = len(comparisons)
        corrected_p_values = {k: (v * num_comparisons) for k, v in p_values.items()}
        corrected_p_values = {k: min(v, 1.0) for k, v in corrected_p_values.items()}

        # Determine significance levels
        def get_significance(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return 'ns'

        significance = {k: get_significance(v) for k, v in corrected_p_values.items()}

        # Annotate significance on the plot
        if show_sigs:
            self._add_significance_annotations(ax, bar_positions, means, sems, comparisons, significance, colors)

        # Customize plot
        label_fontsize = 40
        tick_fontsize = 36
        legend_fontsize = 36
        color = 'black'
        if transparent:
            color = 'white'

        # Remove plot title
        # plt.title(...)  # Commented out to remove the title

        # Set axis labels with increased font size
        # plt.xlabel('Treatment', color=color, fontsize=label_fontsize)
        y_column_title = ' '.join([word.capitalize() for word in y_column.split('_')])
        ylabel = f'{y_column_title}'
        if normalization is not None:
            normalization_title = ' '.join([word.lower() for word in normalization.split('_')])
            ylabel += f' ({normalization_title})'
        plt.ylabel(ylabel, color=color, fontsize=label_fontsize)

        # if y_column.lower() == 'duration_of_event':
        #     from matplotlib.ticker import MultipleLocator
        #     ax.yaxis.set_major_locator(MultipleLocator(10))  # Sets ticks every 5 units

        # Add legend if required (optional since treatments are on x-axis)
        if show_legend:
            # Create custom legend handles
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=t,
                       markerfacecolor=colors[i], markersize=10, alpha=0.6,
                       markeredgecolor='gray', markeredgewidth=0.5)
                for i, t in enumerate(treatments)
            ]
            legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=legend_fontsize)
            legend.get_frame().set_alpha(0.2)  # Set legend frame to semi-transparent
            plt.setp(legend.get_texts(), color=color)  # Legend text color

        # Set tick parameters with increased font size
        plt.tick_params(axis='x', colors=color, labelsize=tick_fontsize)
        plt.tick_params(axis='y', colors=color, labelsize=tick_fontsize)

        # Set spine colors
        ax.spines['top'].set_color(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['right'].set_color(color)

        # Set grid lines
        plt.grid(True, color=color, alpha=0.3, axis='y')

        # Set background color to transparent if needed
        if transparent:
            ax.patch.set_alpha(0)

        # Ensure save_folder exists
        os.makedirs(save_folder, exist_ok=True)

        # Save the plot as PNG and SVG with tight layout
        save_path = os.path.join(save_folder, f'{time}_treatment_barplot_{y_column}_{normalization}')
        plt.savefig(f'{save_path}.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        plt.savefig(f'{save_path}.svg', format='svg', transparent=True, bbox_inches='tight')

        # Optionally save the data
        pearl_values.to_csv(
            os.path.join(save_folder, f'{time}_treatment_barplot_{y_column}_{normalization}.csv'),
            index=False
        )

        if show:
            plt.show()
        plt.close()

    def _normalize_data(self, df, y_column, normalization, pre_window_size, post_window_size):
        """
        Normalizes the y_column in the DataFrame based on the specified method.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - y_column (str): Column to normalize.
        - normalization (str): Normalization method.
        - pre_window_size (int): Pre-window size for certain normalization methods.
        - post_window_size (int): Post-window size for certain normalization methods.

        Returns:
        - pd.DataFrame: Normalized DataFrame.
        """
        normalized_df = df.copy()
        for file in normalized_df['file'].unique():
            file_data = normalized_df[normalized_df['file'] == file]
            # Ensure y_column is numeric
            normalized_df.loc[file_data.index, y_column] = pd.to_numeric(
                normalized_df.loc[file_data.index, y_column], errors='coerce'
            )

            if normalization == 'min-max':
                min_val = file_data[y_column].min()
                max_val = file_data[y_column].max()
                if max_val != min_val:
                    normalized_df.loc[file_data.index, y_column] = (
                                                                           file_data[y_column] - min_val
                                                                   ) / (max_val - min_val)
                else:
                    notifications.show_error(f"Zero range for min-max normalization in file {file}.")
                    normalized_df.loc[file_data.index, y_column] = np.nan

            elif normalization == 'mean':
                mean_val = file_data[y_column].mean()
                if mean_val != 0:
                    normalized_df.loc[file_data.index, y_column] = file_data[y_column] / mean_val
                else:
                    notifications.show_error(f"Zero mean for mean normalization in file {file}.")
                    normalized_df.loc[file_data.index, y_column] = np.nan

            elif normalization == 'median':
                median_val = file_data[y_column].median()
                if median_val != 0:
                    normalized_df.loc[file_data.index, y_column] = file_data[y_column] / median_val
                else:
                    notifications.show_error(f"Zero median for median normalization in file {file}.")
                    normalized_df.loc[file_data.index, y_column] = np.nan

            elif normalization == 'first_point':
                first_value = file_data[y_column].iloc[0]
                if first_value != 0:
                    normalized_df.loc[file_data.index, y_column] = file_data[y_column] / first_value
                else:
                    notifications.show_error(f"First point is zero for first_point normalization in file {file}.")
                    normalized_df.loc[file_data.index, y_column] = np.nan

            # Add more normalization methods as needed

            else:
                raise ValueError(f"Invalid normalization method: {normalization}")

        return normalized_df

    import pandas as pd
    import logging

    def _extract_mean_values_around_pearl(self, df, y_column, pre_window_size=5, post_window_size=30):
        """
        Extracts the mean of y_column values within a specified range around the 'pearl_timepoint' for each file.

        Args:
        - df (pd.DataFrame): Input DataFrame containing the data.
        - y_column (str): The name of the column to extract and average values from.
        - pre_window_size (int): Number of timepoints before the 'pearl_timepoint' to include.
        - post_window_size (int): Number of timepoints after the 'pearl_timepoint' to include.

        Returns:
        - pd.DataFrame: DataFrame containing the mean y_column values within the specified window for each file.
        """
        # Ensure 'pearl_timepoint' and 'timepoint' are numeric
        df['pearl_timepoint'] = pd.to_numeric(df['pearl_timepoint'], errors='coerce')
        df['timepoint'] = pd.to_numeric(df['timepoint'], errors='coerce')

        # Initialize list to collect mean values
        mean_list = []

        # Iterate over each file
        for file, group in df.groupby('file'):
            # Get the unique pearl_timepoint for the file (assumed to be the same across the file)
            pearl_timepoints = group['pearl_timepoint'].dropna().unique()
            if len(pearl_timepoints) == 0:
                logging.warning(f"No 'pearl_timepoint' found for file {file}. Skipping.")
                continue
            elif len(pearl_timepoints) > 1:
                logging.warning(f"Multiple 'pearl_timepoint' values found for file {file}. Using the first one.")
            pearl_timepoint = pearl_timepoints[0]

            # Define the window range
            start_time = pearl_timepoint - pre_window_size
            end_time = pearl_timepoint + post_window_size

            # Extract the rows within the window
            window_data = group[(group['timepoint'] >= start_time) & (group['timepoint'] <= end_time)]

            if window_data.empty:
                logging.warning(
                    f"No data found within the window ({start_time} to {end_time}) for file {file}. Skipping."
                )
                continue

            # Ensure y_column is numeric
            window_data[y_column] = pd.to_numeric(window_data[y_column], errors='coerce')

            # Calculate the mean, excluding NaN values
            mean_value = window_data[y_column].mean()

            if pd.isna(mean_value):
                logging.warning(
                    f"Mean value is NaN for file {file} within the window ({start_time} to {end_time}). Skipping."
                )
                continue

            # Append to the list with treatment information
            mean_list.append({
                'file': file,
                'treatment': window_data.iloc[0]['treatment'],  # Assumes treatment is consistent within the file
                'pearl_timepoint': pearl_timepoint,
                f'{y_column}': mean_value
            })

        # Convert list to DataFrame
        mean_values = pd.DataFrame(mean_list)

        return mean_values

    def _extract_pearl_values_from_column(self, df, y_column):
        """
        Extracts the y_column value at the 'pearl_timepoint' for each file.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - y_column (str): Column to extract values from.

        Returns:
        - pd.DataFrame: DataFrame containing the y_column value at the 'pearl_timepoint' for each file.
        """
        # Ensure 'pearl_timepoint' and 'timepoint' are numeric
        df['pearl_timepoint'] = pd.to_numeric(df['pearl_timepoint'], errors='coerce')
        df['timepoint'] = pd.to_numeric(df['timepoint'], errors='coerce')

        # Initialize list to collect pearl values
        pearl_list = []

        # Iterate over each file
        for file, group in df.groupby('file'):
            # Get the unique pearl_timepoint for the file (assumed to be the same across the file)
            pearl_timepoints = group['pearl_timepoint'].dropna().unique()
            if len(pearl_timepoints) == 0:
                logging.warning(f"No 'pearl_timepoint' found for file {file}. Skipping.")
                continue
            elif len(pearl_timepoints) > 1:
                logging.warning(f"Multiple 'pearl_timepoint' values found for file {file}. Using the first one.")
            pearl_timepoint = pearl_timepoints[0]

            # Extract the row where 'timepoint' == 'pearl_timepoint'
            pearl_row = group[group['timepoint'] == pearl_timepoint]

            if pearl_row.empty:
                logging.warning(
                    f"No data found at 'pearl_timepoint' {pearl_timepoint} for file {file}. Skipping."
                )
                continue

            # If multiple rows match, take the first one
            pearl_value = pearl_row.iloc[0][y_column]

            # Append to the list with treatment information
            pearl_list.append({
                'file': file,
                'treatment': pearl_row.iloc[0]['treatment'],
                'pearl_timepoint': pearl_timepoint,
                y_column: pearl_value
            })

        # Convert list to DataFrame
        pearl_values = pd.DataFrame(pearl_list)

        return pearl_values

    def _add_significance_annotations(self, ax, bar_positions, means, sems, comparisons, significance, colors):
        """
        Adds significance annotations (asterisks) between bars.

        Args:
        - ax (matplotlib.axes.Axes): The axes to annotate.
        - bar_positions (list or np.array): Positions of the bars on the x-axis.
        - means (list): Mean values of each bar.
        - sems (list): SEM values of each bar.
        - comparisons (list of tuples): List of index pairs to compare.
        - significance (dict): Mapping from comparison tuple to significance string.
        - colors (list): List of colors corresponding to each treatment.

        Returns:
        - None
        """
        # Determine the maximum y-value to set the starting point for annotations
        max_y = max([mean + sem for mean, sem in zip(means, sems)])

        # Set an initial y-offset above the highest bar
        y_offset = max_y * 1.05  # 5% above the highest bar

        # Define the height of the annotation lines
        h = max(sems) * 0.5 if sems else 0.05 * max_y

        # Define the vertical spacing between multiple annotations
        step = h * 1.2

        for idx, ((i, j), sig) in enumerate(significance.items()):
            if sig == 'ns':
                continue  # Skip non-significant comparisons

            # Coordinates for the annotation lines
            x1, x2 = bar_positions[i], bar_positions[j]
            y = y_offset + step * idx

            # Draw the lines
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2.5, c='black')

            # Add the significance text
            ax.text((x1 + x2) * 0.5, y + h, sig, ha='center', va='bottom', color='black', fontsize=32)
