import os


def rename_event_folders(base_path):
    # List all directories in the base_path
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Filter directories that start with 'event'
    event_dirs = sorted([d for d in directories if d.startswith('event') and '_' in d])

    # Sort based on the numeric part of the folder name (assuming it follows 'event')
    event_dirs_sorted = sorted(event_dirs, key=lambda x: int(x.split('_')[0][5:]))

    # Rename the directories
    for i, dirname in enumerate(event_dirs_sorted, start=1):
        new_name = f"event{i}_{dirname.split('_', 1)[1]}"
        old_path = os.path.join(base_path, dirname)
        new_path = os.path.join(base_path, new_name)

        # Rename the folder
        os.rename(old_path, new_path)
        print(f"Renamed '{dirname}' to '{new_name}'")


# Example usage
base_path = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-10-09_ctrls"
rename_event_folders(base_path)
