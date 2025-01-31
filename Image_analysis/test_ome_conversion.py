# import tifffile
# import numpy as np
# import os
#
#
# def convert_to_bigtiff(input_path, output_path):
#     """
#     Converts a Bio-Formats OME-TIFF to a BigTIFF format while handling missing pages.
#
#     Parameters:
#     - input_path: str, path to the input OME-TIFF file.
#     - output_path: str, path to save the BigTIFF OME file.
#     """
#
#     # Delete the output file if it already exists
#     if os.path.exists(output_path):
#         print(f"Output file {output_path} already exists. Deleting it.")
#         os.remove(output_path)
#
#     # Open the input OME-TIFF file
#     with tifffile.TiffFile(input_path) as tif:
#         # Check if the file is a multi-page TIFF
#         if len(tif.series) > 0:
#             series = tif.series[0]
#             with tifffile.TiffWriter(output_path, bigtiff=False, ome=True) as tif_writer:
#                 # Iterate through each page in the series and save it in BigTIFF format
#                 for page_index, page in enumerate(series.pages):
#                     print(f"Converting page {page_index + 1}/{len(series.pages)}")
#                     try:
#                         # Extract the image data from the page, handle missing pages
#                         if page is not None:
#                             image_data = page.asarray()
#                         else:
#                             print(f"Warning: Page {page_index} is missing, filling with zeros.")
#                             # Fill missing page with zeros based on the shape of the series (fall back to first page shape)
#                             image_data = np.zeros_like(series.pages[0].asarray())
#
#                         # Write each page to the BigTIFF file
#                         tif_writer.write(
#                             image_data,
#                             description=tif.ome_metadata if page_index == 0 else None,
#                             # Add OME metadata on the first page
#                             photometric='minisblack',  # Adjust if your images are RGB or other formats
#                         )
#                         print(f"Converting page {page_index + 1}/{len(series.pages)}")
#
#                     except Exception as e:
#                         print(f"Error processing page {page_index}: {e}")
#         else:
#             raise ValueError("The input file does not contain any series.")
#
#     print(f"Successfully converted {input_path} to BigTIFF format and saved as {output_path}")
#
#
# input_path = r"F:\UCSF\Micro_needle\20240814_Gav_Mito_Needle_U2OS\DIsh1_cell_1\DIsh1_cell_1_MMStack_Default.ome.tif"
# output_path = r"F:\UCSF\Micro_needle\20240814_Gav_Mito_Needle_U2OS\DIsh1_cell_1\DIsh1_cell_1_MMStack_Default_bigtiff.ome.tif"
# convert_to_bigtiff(input_path, output_path)


#
#
#
# def explore_tiff_pages_with_metadata(input_path):
#     import tifffile
#     import xml.etree.ElementTree as ET
#     """
#     Explores the contents of each page in a TIFF file, using OME metadata to determine
#     timepoint and channel for each page.
#
#     Parameters:
#     - input_path: str, path to the input OME-TIFF file.
#     """
#
#     with tifffile.TiffFile(input_path) as tif:
#         # Parse OME metadata
#         ome_metadata = tif.ome_metadata
#         if ome_metadata is None:
#             print("No OME metadata found in the TIFF file.")
#             return
#
#         # Parse the XML metadata to extract timepoints and channels
#         ome_xml = ET.fromstring(ome_metadata)
#         namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
#
#         # Extract Image metadata (dimensions and ID)
#         image_elements = ome_xml.findall('ome:Image', namespaces)
#         for image_element in image_elements:
#             image_id = image_element.get('ID')
#             pixels_element = image_element.find('ome:Pixels', namespaces)
#
#             # Determine number of T, C, Z dimensions from metadata
#             size_t = int(pixels_element.get('SizeT'))
#             size_c = int(pixels_element.get('SizeC'))
#             size_z = int(pixels_element.get('SizeZ'))
#             size_y = int(pixels_element.get('SizeY'))
#             size_x = int(pixels_element.get('SizeX'))
#
#             print(f"Image ID: {image_id}, SizeT: {size_t}, SizeC: {size_c}, SizeZ: {size_z}, "
#                   f"SizeY: {size_y}, SizeX: {size_x}")
#
#             # Loop through all planes to get the timepoint and channel
#             planes = pixels_element.findall('ome:Plane', namespaces)
#             for plane in planes:
#                 z = plane.get('TheZ')
#                 c = plane.get('TheC')
#                 t = plane.get('TheT')
#                 print(f"Found Plane - Z: {z}, C: {c}, T: {t}")
#
#         # Now, let's match the timepoints and channels to each page in the TIFF file
#         series = tif.series[0]
#         for page_index, page in enumerate(series.pages):
#             try:
#                 if page is None:
#                     print(f"Page {page_index}: Missing")
#                 else:
#                     # Extract metadata based on page index
#                     image_data = page.asarray()
#                     print(f"Page {page_index}: Exists, shape={image_data.shape}, dtype={image_data.dtype}")
#
#                     # Since Bio-Formats orders pages by T, C, Z, we can calculate which timepoint and channel each page belongs to
#                     z_index = page_index % size_z
#                     c_index = (page_index // size_z) % size_c
#                     t_index = (page_index // (size_z * size_c)) % size_t
#
#                     print(f"  Page belongs to Timepoint {t_index}, Channel {c_index}, Z {z_index}")
#
#             except Exception as e:
#                 print(f"Error reading page {page_index}: {e}")
#
#
# # Example usage
# input_tif = r"F:\UCSF\Micro_needle\20240814_Gav_Mito_Needle_U2OS\DIsh1_cell_1\DIsh1_cell_1_MMStack_Default.ome.tif"
# explore_tiff_pages_with_metadata(input_tif)


import tifffile


import tifffile

import numpy as np
import tifffile

import numpy as np
import tifffile
import ome_types


def diagnose_tiff_file(tiff_path):
    """
    Diagnoses potential issues with a TIFF file by checking image dimensions, metadata, and data consistency.
    Additionally, returns the min, max, mean, and median intensity values for each image in the TIFF series
    and identifies the timepoint, channel, and Z-slice of each page.

    Parameters:
    - tiff_path: str, path to the TIFF file to diagnose.

    Returns:
    - stats: dict containing min, max, mean, and median intensity for each page in the TIFF series.
    """
    missing_pages = []  # List to store indices of missing or NoneType pages
    intensity_stats = {}  # Dictionary to store intensity statistics for each page

    try:
        with tifffile.TiffFile(tiff_path) as tif:
            print(f"Opening TIFF file: {tiff_path}")
            print(f"Number of series: {len(tif.series)}")

            for series_index, series in enumerate(tif.series):
                print(f"Series {series_index}:")

                # Print the shape and axes of the image series
                print(f"  Shape: {series.shape}")
                print(f"  Axes: {series.axes}")

                # Check the OME metadata if available
                ome_metadata = tif.ome_metadata
                if ome_metadata:
                    print(f"  OME Metadata found.")
                    ome_xml = ome_types.from_xml(ome_metadata)
                    pixels = ome_xml.images[0].pixels
                    size_t = pixels.size_t
                    size_c = pixels.size_c
                    size_z = pixels.size_z
                    # Update the dimension order extraction to remove any unwanted prefix
                    dimension_order = pixels.dimension_order.name if hasattr(pixels.dimension_order,
                                                                             'name') else pixels.dimension_order

                    # Ensure the dimension order is in the correct format
                    if dimension_order.startswith('Pixels_DimensionOrder.'):
                        dimension_order = dimension_order.replace('Pixels_DimensionOrder.', '')

                    # Continue as before
                    if dimension_order not in ['XYZCT', 'XYCZT', 'XYZTC', 'XYCTZ', 'XYTZC', 'XYTCZ']:
                        print(f"  Warning: Unrecognized dimension order {dimension_order}.")
                        continue

                    print(f"  Dimension order: {dimension_order}")
                    print(f"  SizeT: {size_t}, SizeC: {size_c}, SizeZ: {size_z}")

                    # Iterate through each page in the series and map to T, C, Z
                    print(f'  Number of pages: {len(series.pages)}')
                    for page_index, page in enumerate(series.pages):
                        try:
                            # Check if the page exists and is not None
                            if page is None:
                                missing_pages.append(page_index)  # Store missing page index
                                continue

                            # Extract the image data
                            image_data = page.asarray()

                            # Check for anomalies in the image data
                            if image_data.sum() == 0:
                                print(f"    Warning: Page {page_index} contains only zero values.")
                            elif not image_data.any():
                                print(f"    Warning: Page {page_index} seems empty.")

                            # Compute the timepoint, channel, and z-slice for the page
                            if dimension_order == 'XYZCT':
                                z = page_index % size_z
                                c = (page_index // size_z) % size_c
                                t = (page_index // (size_z * size_c)) % size_t
                            elif dimension_order == 'XYCZT':
                                c = page_index % size_c
                                z = (page_index // size_c) % size_z
                                t = (page_index // (size_z * size_c)) % size_t
                            elif dimension_order == 'XYZTC':
                                z = page_index % size_z
                                t = (page_index // size_z) % size_t
                                c = (page_index // (size_z * size_t)) % size_c
                            elif dimension_order == 'XYCTZ':
                                c = page_index % size_c
                                t = (page_index // size_c) % size_t
                                z = (page_index // (size_t * size_c)) % size_z
                            elif dimension_order == 'XYTZC':
                                t = page_index % size_t
                                z = (page_index // size_t) % size_z
                                c = (page_index // (size_z * size_t)) % size_c
                            elif dimension_order == 'XYTCZ':
                                t = page_index % size_t
                                c = (page_index // size_t) % size_c
                                z = (page_index // (size_c * size_t)) % size_z

                            # Print the identified timepoint, channel, and Z-slice
                            print(f"    Page {page_index}: Timepoint (T) = {t}, Channel (C) = {c}, Z-slice (Z) = {z}")

                            # Calculate intensity statistics
                            min_intensity = np.min(image_data)
                            max_intensity = np.max(image_data)
                            mean_intensity = np.mean(image_data)
                            median_intensity = np.median(image_data)

                            intensity_stats[page_index] = {
                                "timepoint": t,
                                "channel": c,
                                "z_slice": z,
                                "min_intensity": min_intensity,
                                "max_intensity": max_intensity,
                                "mean_intensity": mean_intensity,
                                "median_intensity": median_intensity
                            }

                        except Exception as e:
                            print(f"    Error reading page {page_index}: {e}")

            # Print a summary of missing pages
            if missing_pages:
                print(f"\nNumber of missing pages: {len(missing_pages)}")
                print(f"Missing page indices: {missing_pages}")
            else:
                print("\nNo missing pages detected.")

        return intensity_stats

    except Exception as e:
        print(f"Error opening TIFF file: {e}")


# file = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2024-09-29_microneedle\event1_DIsh1_cell_1\crop1.ome.tif"
# file = r"F:\UCSF\Micro_needle\20240814_Gav_Mito_Needle_U2OS\DIsh1_cell_1\DIsh1_cell_1_MMStack_Default_test.ome.tif"
# # file = r"F:\UCSF\Micro_needle\20240814_Gav_Mito_Needle_U2OS\DIsh1_cell_1\cleaned_DIsh1_cell_1_MMStack_Default.ome.tif"

file = r"F:\UCSF\Micro_needle\20240814_Gav_Mito_Needle_U2OS\DIsh1_cell_1\extra\DIsh1_cell_1_MMStack_Default_imageJ.ome.tif"
diagnose_tiff_file(file)

def load_memmap_tiff(filepath):
    """
    Function to load a multi-dimensional OME-TIFF file using memory mapping.

    Args:
        filepath (str): Path to the OME-TIFF file.

    Returns:
        memmap: The memory-mapped file object.
    """
    # Memory map the TIFF file
    memmap_tiff = tifffile.memmap(filepath)

    # You can access individual parts of the image stack lazily
    # For example, access the first timepoint, first channel, and first Z slice
    first_image = memmap_tiff[0, 0, 0]  # (T, C, Z, Y, X)

    print(f"Shape of memory-mapped TIFF: {memmap_tiff.shape}")
    print(f"First image slice shape: {first_image.shape}")

    return memmap_tiff


# Example usage
# memmap_tiff = load_memmap_tiff(file)

import tifffile as tiff
import numpy as np
from xml.etree import ElementTree as ET

#
# def swap_cz_dimensions(filepath, output_filepath):
#     from ome_types import from_xml
#     # Open the TIFF file and read its metadata
#     with tifffile.TiffFile(filepath) as tif:
#         # Ensure the image is in the TZCYX format
#         if tif.series[0].axes != 'TZCYX':
#             raise ValueError(f"Expected 'TZCYX' format, but got {tif.series[0].axes}")
#
#         # Extract the image data
#         img_data = tif.asarray()
#
#         # Swap the C and Z axes: TZCYX -> TCZYX
#         swapped_data = np.swapaxes(img_data, 1, 2)
#
#         # Update OME metadata if available
#         if tif.is_ome or tif.ome_metadata is not None:
#             ome_xml = tifffile.tiffcomment(filepath)
#             metadata = from_xml(ome_xml)
#
#             # Update the SizeC and SizeZ
#             size_z = metadata.images[0].pixels.size_c
#             size_c = metadata.images[0].pixels.size_z
#             metadata.images[0].pixels.size_c = size_c
#             metadata.images[0].pixels.size_z = size_z
#
#             # Check if 'tiff_data' exists and update it if present
#             if hasattr(metadata.images[0].pixels, 'tiff_data'):
#                 for tiffdata in metadata.images[0].pixels.tiff_data:
#                     first_c = getattr(tiffdata, 'first_c', None)
#                     first_z = getattr(tiffdata, 'first_z', None)
#                     if first_c is not None and first_z is not None:
#                         tiffdata.first_c, tiffdata.first_z = first_z, first_c
#
#             # Use 'XYCZT' as the valid OME-XML dimension order, but actual data is in 'TCZYX'
#             metadata.images[0].pixels.dimension_order = 'XYCZT'
#
#             # Save the swapped image data with OME metadata
#             tifffile.imwrite(output_filepath, swapped_data, description=metadata.to_xml())
#         else:
#             # If not OME, write without metadata updates
#             tifffile.imwrite(output_filepath, swapped_data)
#
#         print(f"File saved with swapped axes at {output_filepath}")
#
#
#
# # Usage
# input_file = r"F:\UCSF\Micro_needle\20240814_Gav_Mito_Needle_U2OS\DIsh1_cell_1\DIsh1_cell_1_MMStack_Default_imageJ.ome.tif"
# output_file = r"F:\UCSF\Micro_needle\20240814_Gav_Mito_Needle_U2OS\DIsh1_cell_1\DIsh1_cell_1_MMStack_Default_swapped.ome.tif"
# swap_cz_dimensions(input_file, output_file)
