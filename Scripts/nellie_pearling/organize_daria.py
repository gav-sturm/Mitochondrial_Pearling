import os
import shutil


def organize_files_by_name(base_folder):
    # Get a list of all subfolders in the base folder
    subfolders = [os.path.join(base_folder, subfolder) for subfolder in os.listdir(base_folder) if
                  os.path.isdir(os.path.join(base_folder, subfolder))]

    # Iterate over each subfolder
    for subfolder in subfolders:
        # List all files in the current subfolder
        for file_name in os.listdir(subfolder):
            file_path = os.path.join(subfolder, file_name)

            # Only proceed if it's a file
            if os.path.isfile(file_path):
                if '_cut' in file_name:
                    # If the file name contains '_cut', strip it and find the target folder
                    base_name_no_ext = os.path.splitext(file_name)[0].replace('_cut', '')
                    target_folder_path = os.path.join(subfolder,base_name_no_ext)

                    # Ensure the target folder exists
                    os.makedirs(target_folder_path, exist_ok=True)

                    # Move the file to the existing folder without '_cut' in its name
                    shutil.move(file_path, os.path.join(target_folder_path, file_name))
                else:
                    # Get the file name without extension
                    file_name_no_ext = os.path.splitext(file_name)[0]

                    # Create a new folder in the base folder named after the file
                    new_folder_path = os.path.join(subfolder, file_name_no_ext)
                    os.makedirs(new_folder_path, exist_ok=True)

                    # Move the file to the new folder
                    shutil.move(file_path, os.path.join(new_folder_path, file_name))

    print(f"Files have been organized into folders by their names in '{base_folder}'.")

def convert_tif_to_ome(base_folder):
    import tifffile
    # Traverse the newly organized folder structure
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)

        # Ensure it is a directory
        if os.path.isdir(folder_path):
            sub_list = os.listdir(folder_path)
            for sub in sub_list:
                sub_path = os.path.join(folder_path, sub)
                if os.path.isdir(sub_path):
                    # Look for .tif files that do not have '_cut' in their name
                    for file_name in os.listdir(sub_path):
                        if file_name.endswith('.tif') and '_cut' not in file_name:
                            file_path = os.path.join(sub_path, file_name)

                            # Read the .tif file
                            with tifffile.TiffFile(file_path) as tif:
                                image_data = tif.asarray()
                                metadata = tif.pages[0].tags  # Example of capturing metadata if needed

                            # Define the new .ome.tif file path
                            new_file_path = os.path.splitext(file_path)[0] + '.ome.tif'

                            # Save the image data as a .ome.tif file
                            tifffile.imwrite(new_file_path, image_data, photometric='minisblack', metadata={})

                            print(f"Converted '{file_path}' to '{new_file_path}'.")

# Example usage
base_folder = r"F:\Daria_v2"  # Replace with your base folder path
organize_files_by_name(base_folder)  # Step 1: Organize files into folders
# convert_tif_to_ome(base_folder)  # Step 2: Convert .tif to .ome.tif


# # Example usage
# base_folder = r"F:\Daria_v2"
# organize_files_by_name(base_folder)
