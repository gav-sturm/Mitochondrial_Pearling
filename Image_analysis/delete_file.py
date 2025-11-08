import os


def delete_files_by_name(root_folder, target_filename):
    """
    Walk through a folder structure and delete all files with the specified filename.

    Parameters:
    - root_folder (str): The root directory to start the search.
    - target_filename (str): The name of the file to search and delete.
    """
    # Walk through the directory structure
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Check if the target file is in the current directory
        for filename in filenames:
            if filename == target_filename:
                # Construct the full file path
                file_path = os.path.join(dirpath, filename)

                try:
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    print(f"Total files deleted: {count}")


# Example usage
root_folder = r"scripts\2024-11-04_osmotic_shock_calcium"
target_filename = "completed_curvature.txt" # "training_label_metrics.csv"

delete_files_by_name(root_folder, target_filename)
