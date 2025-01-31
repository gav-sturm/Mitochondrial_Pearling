import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.stats import sem
import seaborn as sns

class TreatmentGraphPlotter:
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

    def filter_data(self, input_data, select_treatments, select_method):
        """
        Filters the input DataFrame based on selected treatments and induction methods.

        Args:
            input_data (pd.DataFrame): The input DataFrame.
            select_treatments (list or None): Treatments to include.
            select_method (list or None): Induction methods to include.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        if select_treatments:
            input_data = input_data[input_data['treatment'].isin(select_treatments)]
            self.logger.info(f"Filtered data to treatments: {select_treatments}")
        if select_method:
            input_data = input_data[input_data['induction_method'].isin(select_method)]
            self.logger.info(f"Filtered data to induction methods: {select_method}")
        return input_data

    def assign_colors(self, treatments, color_scheme, transparent):
        """
        Assigns colors to treatments based on the selected color scheme.

        Args:
            treatments (list or np.ndarray): List of unique treatments.
            color_scheme (str): 'viridis' or 'custom'.
            transparent (bool): Whether to use a transparent color for control.

        Returns:
            list: List of colors corresponding to treatments.
        """
        if color_scheme == 'viridis':
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i / max(1, len(treatments) - 1)) for i in range(len(treatments))]
            self.logger.info("Assigned colors using 'viridis' colormap.")
        else:
            # Custom color mapping
            gray = '#696969' if not transparent else '#D3D3D3'
            treatment_colors = {
                'control': gray,
                'fccp': 'orange',
                'oligomycin': 'darkblue',
                'rotenone': 'darkred',
                'antimycina': 'purple',
                'nocodazole': 'blue',
                'taxol': 'green',
                'latrunculinb': 'red',
                'mitoflippertr': 'darkgreen',
                'atorvastatin': 'orange',
                'kcl': 'darkorange',
                'elongated': 'darkblue',
                'shrunk': 'darkred',
                'ionomycin_hypertonic_recovery': 'darkblue',
                'hypertonic_ionomycin_recovery': 'darkorange',
                # Add more treatments and their colors as needed
            }

            # Generate a color cycle for treatments that are not predefined
            color_cycle = cycle(plt.get_cmap('tab10').colors)
            for treatment in treatments:
                if treatment.lower() not in treatment_colors:
                    treatment_colors[treatment.lower()] = next(color_cycle)

            colors = [treatment_colors.get(t.lower(), 'black') for t in treatments]
            self.logger.info("Assigned colors using custom color mapping.")
        return colors

    def align_frames_and_time(self, input_data):
        """
        Aligns frames and time based on the 'pearl_frame' event.

        Args:
            input_data (pd.DataFrame): The input DataFrame with 'pearl_frame' information.

        Returns:
            pd.DataFrame: DataFrame with aligned frames and time.
        """
        # Initialize alignment columns
        input_data['aligned_frame'] = None
        input_data['aligned_time_seconds'] = None

        for file in input_data['file'].unique():
            file_data = input_data[input_data['file'] == file]

            # Get the 'pearl_frame' for the current file
            pearl_frame = file_data['pre_pearl_frame'].values[0]

            # Calculate 'aligned_frame'
            aligned_frame = file_data['timepoint'] - pearl_frame
            input_data.loc[file_data.index, 'aligned_frame'] = aligned_frame

            # Get 'pearl_time_seconds'
            pearl_time_seconds_array = file_data.loc[file_data['timepoint'] == pearl_frame, 'timepoint_seconds'].values
            if len(pearl_time_seconds_array) == 0:
                self.logger.warning(f"Pearl time in seconds not found for file {file}. Skipping.")
                continue
            pearl_time_seconds = pearl_time_seconds_array[0]

            # Calculate 'aligned_time_seconds'
            aligned_time_seconds = file_data['timepoint_seconds'] - pearl_time_seconds
            input_data.loc[file_data.index, 'aligned_time_seconds'] = aligned_time_seconds

        # Convert to numeric and drop NaNs
        input_data['aligned_frame'] = pd.to_numeric(input_data['aligned_frame'], errors='coerce')
        input_data['aligned_time_seconds'] = pd.to_numeric(input_data['aligned_time_seconds'], errors='coerce')
        input_data = input_data.dropna(subset=['aligned_frame', 'aligned_time_seconds'])

        self.logger.info("Aligned frames and time based on 'pearl_frame' event.")
        return input_data

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
                self.logger.warning(f"Not enough timepoints in control file {file}. Expected {control_first_x_timepoints}, got {len(first_x)}. Skipping this file.")
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

        self.logger.info(f"Control baseline (average of first {control_first_x_timepoints} timepoints starting at pre-window): {control_baseline}")
        return control_baseline

    def compute_independent_baselines(self, input_data, y_column, pre_window_size, independent_first_x_timepoints):
        """
        Computes independent baselines for each treatment by averaging the first X timepoints starting at the pre-window period.

        Args:
            input_data (pd.DataFrame): The input DataFrame with aligned time.
            y_column (str): The column to normalize.
            pre_window_size (int): Number of frames before the event to define the pre-window.
            independent_first_x_timepoints (int): Number of initial timepoints for averaging per treatment.

        Returns:
            dict: Dictionary mapping treatments to their independent baselines.
        """
        independent_baselines = {}
        treatments = input_data['treatment'].unique()

        for treatment in treatments:
            treatment_data = input_data[input_data['treatment'] == treatment]

            baselines = []

            for file in treatment_data['file'].unique():
                file_data = treatment_data[treatment_data['file'] == file]

                # Select data starting from the pre-window period
                pre_window_data = file_data[file_data['aligned_time_seconds'] >= -pre_window_size]
                pre_window_data_sorted = pre_window_data.sort_values('aligned_time_seconds')

                # Select the first X timepoints
                first_x = pre_window_data_sorted.head(independent_first_x_timepoints)

                if len(first_x) < independent_first_x_timepoints:
                    self.logger.warning(f"Not enough timepoints in treatment '{treatment}' file {file}. Expected {independent_first_x_timepoints}, got {len(first_x)}. Skipping this file.")
                    continue

                # Calculate the average of y_column
                avg = first_x[y_column].mean()

                if pd.isna(avg):
                    self.logger.warning(f"Average is NaN for treatment '{treatment}' file {file}. Skipping this file.")
                    continue

                baselines.append(avg)

            if baselines:
                independent_baselines[treatment] = np.mean(baselines)
                self.logger.info(f"Independent baseline for treatment '{treatment}': {independent_baselines[treatment]}")
            else:
                self.logger.warning(f"No valid baselines found for treatment '{treatment}'. It will not be normalized independently.")

        return independent_baselines

    def normalize_data(self, input_data, y_column, control_baseline, independent_baselines):
        """
        Normalizes the y_column using both control-based and independent normalization methods.

        Args:
            input_data (pd.DataFrame): The input DataFrame with aligned time.
            y_column (str): The column to normalize.
            control_baseline (float): The control baseline for normalization.
            independent_baselines (dict): Dictionary of independent baselines per treatment.

        Returns:
            pd.DataFrame: DataFrame with normalized y_columns.
        """
        input_data[y_column] = pd.to_numeric(input_data[y_column], errors='coerce')

        # Control-Based Normalization
        input_data['y_control_normalized'] = input_data[y_column] / control_baseline
        self.logger.info(f"Normalized '{y_column}' based on control baseline.")

        # Independent Normalization
        input_data['y_independent_normalized'] = np.nan  # Initialize column

        for treatment, baseline in independent_baselines.items():
            input_data.loc[input_data['treatment'] == treatment, 'y_independent_normalized'] = input_data.loc[input_data['treatment'] == treatment, y_column] / baseline

        self.logger.info(f"Normalized '{y_column}' independently based on each treatment's baseline.")
        return input_data

    def interpolate_and_aggregate(self, treatment_data, y_normalized_column, pre_window_size, post_window_size, time_step=2):
        """
        Interpolates individual files and aggregates data for plotting.

        Args:
            treatment_data (pd.DataFrame): DataFrame for a specific treatment.
            y_normalized_column (str): The normalized y_column to plot.
            pre_window_size (int): Pre-window period size.
            post_window_size (int): Post-window period size.
            time_step (int): Step size for common timepoints.

        Returns:
            pd.DataFrame: DataFrame with mean, SEM, and timepoints.
        """
        # Collect all aligned_time_seconds within the window
        all_aligned_times = []

        for file in treatment_data['file'].unique():
            file_subset = treatment_data[treatment_data['file'] == file]
            aligned_data = file_subset[
                (file_subset['aligned_time_seconds'] >= -pre_window_size) &
                (file_subset['aligned_time_seconds'] <= post_window_size)
            ]
            all_aligned_times.extend(aligned_data['aligned_time_seconds'].dropna().values)

        if not all_aligned_times:
            self.logger.warning("No aligned time points found within the specified window.")
            return pd.DataFrame()

        all_aligned_times = np.array(all_aligned_times, dtype=float)
        all_aligned_times = all_aligned_times[~np.isnan(all_aligned_times)]

        min_time = np.floor(np.min(all_aligned_times))
        max_time = np.ceil(np.max(all_aligned_times))

        common_timepoints = np.arange(min_time, max_time + time_step, time_step).astype(float)

        interpolated_values = pd.DataFrame({'aligned_time_seconds': common_timepoints})

        for file in treatment_data['file'].unique():
            file_subset = treatment_data[treatment_data['file'] == file]
            aligned_data = file_subset[
                (file_subset['aligned_time_seconds'] >= -pre_window_size) &
                (file_subset['aligned_time_seconds'] <= post_window_size)
            ]

            times = file_subset['aligned_time_seconds'].astype(float).values
            values = file_subset[y_normalized_column].astype(float).values

            valid_indices = ~np.isnan(times) & ~np.isnan(values)
            times = times[valid_indices]
            values = values[valid_indices]

            if len(times) < 2:
                self.logger.warning(f"Not enough data points to interpolate for file {file}. Skipping.")
                continue

            sorted_indices = np.argsort(times)
            times = times[sorted_indices]
            values = values[sorted_indices]

            unique_times, unique_indices = np.unique(times, return_index=True)
            times = times[unique_indices]
            values = values[unique_indices]

            interpolated = np.interp(common_timepoints, times, values, left=np.nan, right=np.nan)
            interpolated_values[file] = interpolated

        # Compute mean and SEM across files
        mean_values = interpolated_values.drop(columns=['aligned_time_seconds']).mean(axis=1, skipna=True)
        sem_values = interpolated_values.drop(columns=['aligned_time_seconds']).apply(sem, axis=1, nan_policy='omit')
        n_values = interpolated_values.drop(columns=['aligned_time_seconds']).count(axis=1)

        aggregated_data = pd.DataFrame({
            'aligned_time_seconds': common_timepoints,
            'mean': mean_values,
            'sem': sem_values,
            'n': n_values
        })

        self.logger.info("Interpolated and aggregated data for treatment.")
        return aggregated_data

    def calculate_percent_change(self, aggregated_data_dict, normalization_type):
        """
        Calculates the percent change from control for the peak curvature of each treatment group.

        Args:
            aggregated_data_dict (dict): Dictionary with treatment as keys and aggregated data as values.
            normalization_type (str): Type of normalization ('control' or 'independent').

        Returns:
            pd.DataFrame: DataFrame containing percent change from control for each treatment.
        """
        percent_change_results = []
        control_peak = None

        # First, identify the control peak curvature
        if 'control' in aggregated_data_dict:
            control_peak = aggregated_data_dict['control']['mean'].max()
            self.logger.info(f"Control peak curvature ({normalization_type} normalized): {control_peak}")
        else:
            self.logger.error(f"Control data not found in aggregated_data_dict for normalization '{normalization_type}'. Cannot compute percent change.")
            raise ValueError(f"Control data not found for normalization '{normalization_type}'.")

        # Iterate over treatments and calculate percent change
        for treatment, data in aggregated_data_dict.items():
            if treatment == 'control':
                continue  # Skip control

            treatment_peak = data['mean'].max()
            if control_peak == 0:
                self.logger.warning(f"Control peak curvature is zero. Cannot compute percent change for treatment '{treatment}'.")
                percent_change = np.nan
            else:
                percent_change = ((treatment_peak - control_peak) / control_peak) * 100

            percent_change_results.append({
                'treatment': treatment,
                'normalization': normalization_type,
                'control_peak_curvature': control_peak,
                'treatment_peak_curvature': treatment_peak,
                'percent_change_from_control': percent_change
            })
            self.logger.info(f"Calculated percent change for treatment '{treatment}': {percent_change:.2f}%")

        percent_change_df = pd.DataFrame(percent_change_results)
        return percent_change_df

    def plot_data(self, aggregated_data, treatment, color, normalization_type):
        """
        Plots the mean and SEM for a specific treatment and normalization type.

        Args:
            aggregated_data (pd.DataFrame): Aggregated data with mean and SEM.
            treatment (str): Treatment name.
            color (str): Color for the plot.
            normalization_type (str): Type of normalization ('control' or 'independent').

        Returns:
            None
        """
        line_styles = {
            'control': '-',
            'independent': '--'
        }

        plt.errorbar(
            aggregated_data['aligned_time_seconds'],
            aggregated_data['mean'],
            label=f'{treatment} ({normalization_type.capitalize()} Normalized)',
            linewidth=2,
            color=color,
            linestyle='-', # line_styles.get(normalization_type, '-')
            alpha=1.0,
            capsize=2,
            elinewidth=0.8
        )

        plt.fill_between(
            aggregated_data['aligned_time_seconds'],
            aggregated_data['mean'] - aggregated_data['sem'],
            aggregated_data['mean'] + aggregated_data['sem'],
            color=color,
            alpha=0.3
        )

        self.logger.info(f"Plotted data for treatment: {treatment} with {normalization_type} normalization.")

    def plot_treatment_graph(
        self, input_data, y_column, select_treatments, select_method, save_folder, time=None,
        pre_window_size=5, post_window_size=30, color_scheme='viridis', treatment_order=None,
        plot_individual_events=False, transparent=False, show=False, show_legend=True,
        show_titles=True, control_first_x_timepoints=5, independent_first_x_timepoints=5
    ):
        """
        Plots graphs with time in seconds on the x-axis and the specified y_column on the y-axis,
        colored by the 'treatment' column. Generates separate graphs for control-based and
        independent normalization methods. The graphs show data within the specified frame windows
        before and after the 'pearl_frame' event for each file. Data is normalized based on both
        the control group's average and each treatment's own average.

        Additionally, calculates the percent change from control for the peak curvature of each
        treatment group for both normalization methods and aggregates the results into a CSV file.

        Args:
            input_data (pd.DataFrame): The input DataFrame containing data to be plotted.
            y_column (str): The name of the column to plot on the y-axis.
            select_treatments (list or None): List of treatments to include. If None, all treatments are included.
            select_method (list or None): List of induction methods to include. If None, all methods are included.
            save_folder (str): Folder where the plots and CSVs will be saved.
            time (str, optional): Time identifier to be used in the saved file name.
            pre_window_size (int, optional): Number of frames before the event to include.
            post_window_size (int, optional): Number of frames after the event to include.
            color_scheme (str, optional): Color scheme for the plots ('viridis' or 'custom').
            treatment_order (list or None, optional): Specific order of treatments for plotting.
            plot_individual_events (bool, optional): Whether to plot individual events.
            transparent (bool, optional): Whether to save the plots with a transparent background.
            show (bool, optional): Whether to display the plots.
            show_legend (bool, optional): Whether to display the legends.
            show_titles (bool, optional): Whether to display the plot titles.
            control_first_x_timepoints (int, optional): Number of initial timepoints from control to average for control normalization.
            independent_first_x_timepoints (int, optional): Number of initial timepoints from each treatment to average for independent normalization.

        Returns:
            None: Displays and saves the plots and CSV files.
        """
        # Step 1: Filter data
        input_data = self.filter_data(input_data, select_treatments, select_method)

        # Step 2: Align frames and time
        input_data = self.align_frames_and_time(input_data)

        # Step 3: Compute control baseline
        control_baseline = self.compute_control_baseline(
            input_data, y_column, pre_window_size, control_first_x_timepoints
        )

        # Step 4: Compute independent baselines
        independent_baselines = self.compute_independent_baselines(
            input_data, y_column, pre_window_size, independent_first_x_timepoints
        )

        # Step 5: Normalize data
        input_data = self.normalize_data(input_data, y_column, control_baseline, independent_baselines)

        # Step 6: Get unique treatments and reorder
        treatments = input_data['treatment'].unique()
        if treatment_order:
            treatments = treatment_order
        else:
            if 'control' in treatments:
                treatments = ['control'] + [t for t in treatments if t != 'control']
            else:
                treatments = list(treatments)

        # Step 7: Assign colors
        colors = self.assign_colors(treatments, color_scheme, transparent)

        # Step 8: Iterate over normalization types
        normalization_types = ['control', 'independent']

        # Initialize a DataFrame to store all percent changes
        all_percent_changes = pd.DataFrame()

        for normalization_type in normalization_types:
            # Initialize DataFrame to store graph data for this normalization
            graph_csv = pd.DataFrame()

            # Initialize legend handles
            legend_handles = {}

            # Initialize plot
            plt.figure(figsize=(12, 8))
            sns.set(style="whitegrid")

            # Dictionary to hold aggregated data for percent change calculation
            aggregated_data_dict = {}

            # Step 9: Iterate over treatments and plot
            for i, treatment in enumerate(treatments):
                if select_treatments and treatment not in select_treatments:
                    continue

                treatment_data = input_data[input_data['treatment'] == treatment]

                # Select the appropriate normalized column
                if normalization_type == 'control':
                    y_normalized_column = 'y_control_normalized'
                elif normalization_type == 'independent':
                    y_normalized_column = 'y_independent_normalized'
                else:
                    self.logger.error(f"Unknown normalization type: {normalization_type}")
                    continue

                # Plot individual events if required
                if plot_individual_events:
                    for file in treatment_data['file'].unique():
                        file_data = treatment_data[treatment_data['file'] == file]
                        aligned_data = file_data[
                            (file_data['aligned_time_seconds'] >= -pre_window_size) &
                            (file_data['aligned_time_seconds'] <= post_window_size)
                        ]

                        if aligned_data.empty:
                            self.logger.warning(f"No data to plot for file {file} under treatment {treatment}.")
                            continue

                        plt.plot(
                            aligned_data['aligned_time_seconds'],
                            aligned_data[y_normalized_column],
                            alpha=0.1,
                            color=colors[i],
                            linestyle='-' if normalization_type == 'control' else '--',
                            label=f'{treatment} ({normalization_type.capitalize()} Normalized)' if file == treatment_data['file'].unique()[0] else ""
                        )

                # Interpolate and aggregate data
                aggregated_data = self.interpolate_and_aggregate(
                    treatment_data, y_normalized_column, pre_window_size, post_window_size
                )

                if aggregated_data.empty:
                    self.logger.warning(f"No aggregated data for treatment '{treatment}' with {normalization_type} normalization. Skipping.")
                    continue

                # Store aggregated data for percent change calculation
                aggregated_data_dict[treatment] = aggregated_data

                # Plot mean and SEM
                self.plot_data(aggregated_data, treatment, colors[i], normalization_type)

                # Append to graph_csv
                graph_csv = pd.concat([
                    graph_csv,
                    pd.DataFrame({
                        'treatment': treatment,
                        'normalization': normalization_type,
                        'time': aggregated_data['aligned_time_seconds'],
                        'mean': aggregated_data['mean'],
                        'sem': aggregated_data['sem'],
                        'n': aggregated_data['n']
                    })
                ], ignore_index=True)

                # Add to legend handles
                if treatment not in legend_handles:
                    legend_handles[treatment] = plt.Line2D([], [], color=colors[i],
                                                           linestyle='-' if normalization_type == 'control' else '--',
                                                           label=f'{treatment} ({normalization_type.capitalize()} Normalized)')

            # Step 10: Customize plot
            color_text = 'white' if transparent else 'black'
            plt.axhline(1, color=color_text, linestyle='--', linewidth=1)  # Baseline at 1

            # Set labels and title
            y_column_title = y_column.replace('_', ' ').capitalize()
            normalization_title = 'Control Avg' if normalization_type == 'control' else 'Independent'
            if show_titles:
                plt.xlabel('Aligned Time (seconds)', fontsize=15, color=color_text)
                plt.ylabel(f'{y_column_title} (Normalized)', fontsize=15, color=color_text)
                plt.title(f'{y_column_title} Over Time (Normalized to {normalization_title} Average)', fontsize=18, color=color_text)

            # Configure legend
            if show_legend:
                legend = plt.legend(handles=list(legend_handles.values()), loc='upper right', fontsize=11)
                legend.get_frame().set_alpha(0.2)  # Transparent legend frame
                plt.setp(legend.get_texts(), color=color_text)  # Legend text color

            # Customize tick parameters
            plt.tick_params(axis='x', colors=color_text, labelsize=15)
            plt.tick_params(axis='y', colors=color_text, labelsize=15)

            # Customize spines
            ax = plt.gca()
            ax.spines['top'].set_color(color_text)
            ax.spines['bottom'].set_color(color_text)
            ax.spines['left'].set_color(color_text)
            ax.spines['right'].set_color(color_text)

            # Customize grid
            plt.grid(True, color=color_text, alpha=0.3)

            # Handle transparent background
            if transparent:
                ax.patch.set_alpha(0)

            # Step 11: Save the plot and CSV
            select = 'all_treatments'
            if select_treatments:
                select = '_'.join(select_treatments)

            plot_filename = f'{time}_treatment_effect_{y_column}_over_time_{normalization_type}_normalized_{select}'
            save_path_png = os.path.join(save_folder, f'{plot_filename}.png')
            save_path_svg = os.path.join(save_folder, f'{plot_filename}.svg')
            plt.savefig(save_path_png, format='png', dpi=300, transparent=transparent)
            plt.savefig(save_path_svg, format='svg', transparent=transparent)

            # Save graph data to CSV
            graph_csv_save_path = os.path.join(save_folder, f'{time}_{y_column}_over_time_{normalization_type}_normalized_{select}.csv')
            graph_csv.to_csv(graph_csv_save_path, index=False)

            self.logger.info(f"Saved plot to {save_path_png} and {save_path_svg}")
            self.logger.info(f"Saved aggregated data to {graph_csv_save_path}")

            # Calculate percent change from control
            try:
                percent_change_df = self.calculate_percent_change(aggregated_data_dict, normalization_type)
                all_percent_changes = pd.concat([all_percent_changes, percent_change_df], ignore_index=True)
            except ValueError as ve:
                self.logger.error(str(ve))

            # Display or close the plot
            if show:
                plt.show()
            else:
                plt.close()

        def calculate_percent_change(self, aggregated_data_dict, normalization_type):
            """
            Calculates the percent change from control for the peak curvature of each treatment group.

            Args:
                aggregated_data_dict (dict): Dictionary with treatment as keys and aggregated data as values.
                normalization_type (str): Type of normalization ('control' or 'independent').

            Returns:
                pd.DataFrame: DataFrame containing percent change from control for each treatment.
            """
            percent_change_results = []
            control_peak = None

            # First, identify the control peak curvature
            if 'control' in aggregated_data_dict:
                control_peak = aggregated_data_dict['control']['mean'].max()
                self.logger.info(f"Control peak curvature ({normalization_type} normalized): {control_peak}")
            else:
                self.logger.error(f"Control data not found in aggregated_data_dict for normalization '{normalization_type}'. Cannot compute percent change.")
                raise ValueError(f"Control data not found for normalization '{normalization_type}'.")

            # Iterate over treatments and calculate percent change
            for treatment, data in aggregated_data_dict.items():
                if treatment == 'control':
                    continue  # Skip control

                treatment_peak = data['mean'].max()
                if control_peak == 0:
                    self.logger.warning(f"Control peak curvature is zero. Cannot compute percent change for treatment '{treatment}'.")
                    percent_change = np.nan
                else:
                    percent_change = ((treatment_peak - control_peak) / control_peak) * 100

                percent_change_results.append({
                    'treatment': treatment,
                    'normalization': normalization_type,
                    'control_peak_curvature': control_peak,
                    'treatment_peak_curvature': treatment_peak,
                    'percent_change_from_control': percent_change
                })
                self.logger.info(f"Calculated percent change for treatment '{treatment}': {percent_change:.2f}%")

            percent_change_df = pd.DataFrame(percent_change_results)
            return percent_change_df
