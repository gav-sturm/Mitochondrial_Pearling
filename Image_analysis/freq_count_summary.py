import pandas as pd
import numpy as np
import sys


def calculate_sem(x):
    """Calculate the Standard Error of the Mean."""
    return x.std(ddof=1) / np.sqrt(x.count())


def compute_statistics(input_csv, output_csv=None, replicate_output_csv=None, treatment_order=None):
    """
    Compute mean, standard deviation, SEM, and count for each numerical column
    grouped by the 'Treatment' column after averaging over replicates.
    Optionally, save the replicate-averaged data to a separate CSV.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str, optional): Path to the output summary CSV file.
                                   If not provided, results will be printed to the console.
    - replicate_output_csv (str, optional): Path to save the replicate-averaged data.
                                            If not provided, replicate-averaged data won't be saved.
    - treatment_order (list, optional): List specifying the desired order of treatments.
                                         Treatments not in this list will be appended at the end.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: The input file is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print("Error: The input file is not a valid CSV.")
        sys.exit(1)

    # **Standardize column names to lowercase and strip spaces**
    df.columns = df.columns.str.strip().str.lower()

    # **Identify 'treatment' and 'replicate' columns (case-insensitive)**
    required_columns = ['treatment', 'replicate']
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        print(f"Error: The input CSV is missing required columns: {missing_required}")
        sys.exit(1)

    # **Preserve the original 'Treatment' column values for accurate output**
    # Assuming 'treatment' column contains categorical data with original casing
    # We'll keep a separate Series for treatment names with original casing
    # First, identify the original 'Treatment' column name with any case
    original_treatment_col = None
    for col in df.columns:
        if col.lower() == 'treatment':
            original_treatment_col = col
            break
    if not original_treatment_col:
        print("Error: Unable to locate the 'Treatment' column.")
        sys.exit(1)

    # Similarly, identify the original 'Replicate' column name with any case
    original_replicate_col = None
    for col in df.columns:
        if col.lower() == 'replicate':
            original_replicate_col = col
            break
    if not original_replicate_col:
        print("Error: Unable to locate the 'Replicate' column.")
        sys.exit(1)

    # **Identify numerical columns excluding 'treatment' and 'replicate'**
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove 'replicate' if it's numerical
    if 'replicate' in numerical_cols:
        numerical_cols.remove('replicate')

    if not numerical_cols:
        print("Error: No numerical columns found to compute statistics.")
        sys.exit(1)

    # **Step 1: Average over Replicates within each Treatment**
    try:
        # Group by 'treatment' and 'replicate', then compute mean for numerical columns
        replicate_averaged = df.groupby(['treatment', 'replicate'])[numerical_cols].mean().reset_index()
    except KeyError as e:
        print(f"Error during replicate averaging: {e}")
        sys.exit(1)

    # **Optionally save replicate-averaged data to a separate CSV**
    if replicate_output_csv:
        try:
            replicate_averaged.to_csv(replicate_output_csv, index=False)
            print(f"Replicate-averaged data successfully written to '{replicate_output_csv}'.")
        except Exception as e:
            print(f"Error writing replicate-averaged data to file: {e}")
            sys.exit(1)

    # **Step 2: Compute summary statistics based on replicate-averaged data**
    grouped = replicate_averaged.groupby('treatment')[numerical_cols].agg(['mean', 'std', 'count'])

    # **Flatten the MultiIndex columns**
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # **Calculate SEM separately**
    for col in numerical_cols:
        std_col = f"{col}_std"
        count_col = f"{col}_count"
        sem_col = f"{col}_sem"
        if std_col in grouped.columns and count_col in grouped.columns:
            grouped[sem_col] = grouped[std_col] / np.sqrt(grouped[count_col])
        else:
            grouped[sem_col] = np.nan  # Assign NaN if necessary columns are missing

    # **Reset index to have 'Treatment' as a column**
    grouped = grouped.reset_index()

    # **Reorder columns to have mean, sd, sem, n for each numerical column**
    ordered_columns = ['treatment']
    for col in numerical_cols:
        ordered_columns.extend([
            f"{col}_mean",
            f"{col}_std",
            f"{col}_sem",
            f"{col}_count"
        ])

    # **Ensure that ordered_columns exist in grouped before selecting**
    missing_cols = [col for col in ordered_columns if col not in grouped.columns]
    if missing_cols:
        print(f"Error: The following expected columns are missing in the grouped data: {missing_cols}")
        print("Available columns:", grouped.columns.tolist())
        sys.exit(1)

    grouped = grouped[ordered_columns]

    # **Rename columns to more readable format (optional)**
    # For example: 'pearled_length_mean' -> 'pearled_length Mean'
    def rename_column(col):
        if col == 'treatment':
            return 'Treatment'
        else:
            parts = col.split('_')
            if len(parts) >= 2:
                # Join all parts except the last one for the base name
                base = '_'.join(parts[:-1])
                stat = parts[-1]
                stat_map = {
                    'mean': 'Mean',
                    'std': 'SD',
                    'sem': 'SEM',
                    'count': 'n'
                }
                stat_readable = stat_map.get(stat, stat)
                return f"{base} {stat_readable}"
            else:
                return col  # If unexpected format, return as is

    grouped.columns = [rename_column(col) for col in grouped.columns]

    # **Reorder the treatments as per treatment_order**
    if treatment_order:
        # Verify that treatment_order is a list
        if not isinstance(treatment_order, list):
            print("Error: 'treatment_order' should be a list of treatment names.")
            sys.exit(1)

        # Treatments present in the data
        data_treatments = grouped['Treatment'].tolist()

        # Treatments specified in treatment_order
        specified_treatments = treatment_order

        # Treatments in data but not specified
        unspecified_treatments = [t for t in data_treatments if t not in specified_treatments]

        # Treatments specified but not in data
        missing_in_data = [t for t in specified_treatments if t not in data_treatments]
        if missing_in_data:
            print(
                f"Warning: The following treatments specified in 'treatment_order' are not present in the data: {missing_in_data}")

        # Define the final order
        final_order = specified_treatments + unspecified_treatments

        # Reorder the DataFrame
        grouped['Treatment'] = pd.Categorical(grouped['Treatment'], categories=final_order, ordered=True)
        grouped = grouped.sort_values('Treatment').reset_index(drop=True)

    # **Output the summary statistics**
    if output_csv:
        try:
            grouped.to_csv(output_csv, index=False)
            print(f"Summary statistics successfully written to '{output_csv}'.")
        except Exception as e:
            print(f"Error writing summary statistics to file: {e}")
            sys.exit(1)
    else:
        print(grouped.to_string(index=False))


def main():
    """
    Main function to execute the computation.
    Edit the 'input_csv', 'output_csv', 'replicate_output_csv', and 'treatment_order' variables below as needed.
    """
    # === User Parameters ===
    input_csv = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-11-21_freq_counts\single_volume_data.csv"  # <-- Set your input CSV file path here
    output_csv = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-11-21_freq_counts\single_volume_data_summary.csv"  # <-- Set your desired output summary CSV file path here
    replicate_output_csv = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-11-21_freq_counts\single_volume_data_replicate_averaged.csv"  # <-- Set your desired replicate-averaged data CSV file path here
    #     or set to None to skip saving replicate-averaged data

    # Define the desired order of treatments
    treatment_order = [  # <-- Specify the order of treatments here
        'U2OS',
        'HEK293',
        'COS7',
        'RPE1',
        'Jurkat_Tcells',
        'primary_fibroblasts',
        'iPSC_neurons',
        'MIC_wildtype',
        # Add more treatments as needed
    ]
    # If you don't want to specify an order, set treatment_order to None
    # treatment_order = None
    # ========================

    compute_statistics(input_csv, output_csv, replicate_output_csv, treatment_order)


if __name__ == "__main__":
    main()
