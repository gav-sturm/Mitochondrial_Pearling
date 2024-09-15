import os

import tifffile
import numpy as np
import napari
from tqdm import tqdm
from napari.utils import notifications
from ome_types import from_xml, to_xml
from tifffile import imwrite, tiffcomment
from skimage.draw import polygon
import unicodedata
import warnings
import logging
from nellie.im_info.verifier import FileInfo, ImInfo
import cupy as xp
import tifffile as tif

# Set the logging level for the 'napari' logger to WARNING or ERROR
logging.getLogger('napari').setLevel(logging.WARNING)


class CropSnout:
    def __init__(self, im_path: str,
                 crop_path: str,
                 nellie_path: str,
                 image_type='ome',
                 crop_count=1,
                 select_timepoints=None,
                 select_z=False,
                 select_channel=0,
                 time_interval=1,
                 center_coords=None,
                 show_crop=False,
                 pearl_frame=None
                 ):
        self.crop_path = crop_path
        self.im_path = im_path
        self.selected_timepoints = select_timepoints
        self.select_z = select_z
        self.selected_z = None
        self.selected_channel = select_channel
        self.z_stack = []
        self.roi_Zstack = []
        self.timepoints = []
        self.time_interval = time_interval
        self.n_timepoints = None
        self.x_start, self.x_end, self.y_start, self.y_end, self.z_start, self.z_end = [None] * 6
        self.x, self.y, self.dx, self.dy = [None] * 4
        self.shape = None
        self.old_metadata = None
        self.new_metadata = None
        self.is3D = False
        self.isSnouty = False
        self.coords = None
        self.channels = None
        self.n_channels = None
        self.axes = None
        self.nellie_path = nellie_path
        self.mask_coords = None
        self.mask = None
        self.show_crop = show_crop
        self.cropped_images = None
        self.max_z = None
        self.center_coords = center_coords
        self.image_type = image_type
        self.data_folder = None
        self.crop_folder = None
        self.crop_count = crop_count
        self.pearl_frame = pearl_frame

    def napari_view(self, image, IMname='Image', contrast_limit=0.1, layer3d=False):
        minI = np.min(image)
        maxI = np.max(image)

        viewer = napari.Viewer()
        viewer.add_image(image, name=IMname,
                         contrast_limits=[minI + (contrast_limit * maxI), maxI - (contrast_limit * maxI)])
        if layer3d:  # Only set to 3D mode if the image has a Z dimension
            viewer.dims.ndisplay = 3
        napari.run()

    def getMask(self):
        return self.mask

    def _open_ome(self, t=None, c=None):
        with tifffile.TiffFile(self.im_path) as tif:
            series = tif.series[0]
            metadata = tif.ome_metadata
            shape = series.shape
            if self.axes == 'TCZYX':  # Only load Z-stack if Z dimension exists
                for z_index in range(shape[2]):
                    page_index = (t * self.n_channels * self.z_end) + (c * self.z_end) + z_index
                    array = tif.series[0].pages[page_index].asarray()
                    self.z_stack.append(array)
            # elif self.axes == 'TCYX':
            #         # tags = tif.series[0].pages[0].tags
            #         # array = tif.series[0].pages[page_index].asarray()
            #         # print("Array shape:", array.shape)
            #         # for tag in tags.values():
            #         #     print(tag.name, tag.value)
            #         array = tif.series[0].pages.asarray()
            #         array = array[t, c]
            #         self.z_stack.append(array)
            elif self.axes == 'TCYX':
                # T - Time, C - Channel, Y - Height, X - Width (No Z dimension)
                page_index = (t * self.n_channels) + c
                array = tif.series[0].pages[page_index].asarray()
                self.z_stack.append(array)
            elif self.axes == 'TZYX':
                # T - Time, Z - Z-stack, Y - Height, X - Width (No Z dimension)
                for z_index in range(shape[1]):
                    page_index = (t * self.z_end) + z_index
                    array = tif.series[0].pages[page_index].asarray()
                    self.z_stack.append(array)

            elif self.axes == 'CZYX':
                # C - Channel, Z - Z-stack, Y - Height, X - Width
                for z_index in range(shape[1]):
                    page_index = (c * self.z_end) + z_index
                    array = tif.series[0].pages[page_index].asarray()
                    self.z_stack.append(array)

            elif self.axes == 'CYX':
                # C - Channel, Y - Height, X - Width (No Z dimension)
                page_index = c
                array = tif.series[0].pages[page_index].asarray()
                self.z_stack.append(array)

            elif self.axes == 'TYX':
                # T - Time, Y - Height, X - Width (No Z, No Channel)
                page_index = t
                array = tif.series[0].pages[page_index].asarray()
                self.z_stack.append(array)

            elif self.axes == 'YX':
                # Y - Height, X - Width (No Time, No Z, No Channel)
                page_index = 0  # Only one page since there's no T, Z, or C dimensions
                array = tif.series[0].pages[page_index].asarray()
                self.z_stack.append(array)

        max_projection = self.max_project_array()

        return max_projection

    def max_project_array(self, projection_type='max'):
        array = np.stack(self.z_stack, axis=0)
        if self.is3D:  # 3D Z-stack
            max_projection = np.max(array, axis=0)
        else:  # 2D image
            max_projection = array[0]
        return max_projection

    def select_roi(self, input_image=None, center_coords=None, zoom_factor=None):
        """Allow the user to select an XY ROI on the 3D Z-stack or 2D image and return the coordinates."""
        if input_image is None:
            error_message = "No image data provided"
            raise ValueError(error_message)
        max_projection = input_image

        viewer = napari.Viewer()
        viewer.add_image(max_projection, name='max projection volume1')
        shapes = viewer.add_shapes(name='ROI', ndim=2)
        roi_coords = None

        # Center the view on the given coordinates and adjust zoom
        if center_coords and center_coords != (0, 0):
            y, x = center_coords
            viewer.camera.center = (x, y)

            # Adjust zoom if specified
            if zoom_factor:
                viewer.camera.zoom = zoom_factor

        @shapes.mouse_drag_callbacks.append
        def get_shape(layer, event):
            nonlocal roi_coords
            yield  # Start the callback
            while event.type == 'mouse_move':
                yield
            if len(layer.data) > 0:
                shape = layer.data[-1]
                if layer.shape_type[-1] == 'rectangle':
                    self.y_start, self.x_start = shape[0]
                    self.y_end, self.x_end = shape[2]
                    notifications.show_info(f"Pre-adjused Rectangle coordinates: ({self.x_start}, {self.y_start}) to ({self.x_end}, {self.y_end})")
                    # Adjust the coordinates to fit within the image boundaries
                    # if self.image_type == 'ome':
                    self.x_start = max(0, int(self.x_start))
                    self.y_start = max(0, int(self.y_start))
                    self.x_end = min(max_projection.shape[-1], int(self.x_end))
                    self.y_end = min(max_projection.shape[-2], int(self.y_end))
                    roi_coords = {
                        'type': 'rectangle',
                        'coords': (self.x_start, self.y_start, self.x_end, self.y_end)
                    }
                    notifications.show_info(f"Selected XY ROI coordinates: {roi_coords}")
                    viewer.close()

        @viewer.bind_key('Enter', overwrite=True)
        def finalize_polygon(event):
            nonlocal roi_coords
            if len(shapes.data) > 0 and shapes.shape_type[-1] == 'polygon':
                shape = shapes.data[-1]
                if len(shape) > 2:
                    print("shape:", shape)
                    # Adjust the coordinates to fit within the image boundaries
                    polygon_coords = [(min(max_projection.shape[-1], int(x)), min(max_projection.shape[-2], int(y))) for
                                      (y, x) in shape]

                    print("polygon_coords:", polygon_coords)
                    roi_coords = {
                        'type': 'polygon',
                        'coords': polygon_coords
                    }
                    notifications.show_info(f"Selected XY ROI coordinates: {roi_coords}")
                    viewer.close()

        napari.run()

        if roi_coords is None:
            raise ValueError("No ROI selected")

        return roi_coords

    def crop_array(self, array):
        """Crop the array based on the provided coordinates."""
        if 'Z' in self.axes and self.is3D:
            array = array[self.z_start:self.z_end, self.y_start:self.y_end, self.x_start:self.x_end]
        else:
            array = array[self.y_start:self.y_end, self.x_start:self.x_end]
        return array

    def getMetadata(self, input_image_path=None):
        if input_image_path is None:
            input_image_path = self.im_path
        file_info = FileInfo(filepath=input_image_path, output_dir=self.nellie_path)
        file_info.find_metadata()
        file_info.load_metadata()
        return file_info

    def save_file(self, cropped_images):
        file_info = self.getMetadata()

        print(f'original {file_info.shape=}')
        # file_info.shape = self.coords
        print(f'crop path {self.crop_path}')

        file_info.shape = cropped_images.shape
        if self.time_interval is None:
            self.time_interval = 1
        file_info.change_dim_res(dim='T', new_size=self.time_interval)
        if self.selected_timepoints is not None:
            file_info.select_temporal_range(start=0, end=self.n_timepoints - 1)

        # file_info.output_dir = self.nellie_path

        print(f'{file_info.metadata_type=}')
        print(f'{file_info.axes=}')
        print(f'updated {file_info.shape=}')
        print(f'updated {file_info.dim_res=}')
        print(f'{file_info.good_axes=}')
        print(f'{file_info.good_dims=}')
        if self.selected_channel > 0:
            file_info.change_selected_channel(self.selected_channel)
        print(f'{file_info.ch=}')
        print(f'{file_info.ch=}')
        file_info.ome_output_path = self.crop_path
        print(f'{file_info.ome_output_path=}')
        # file_info.output_dir = self.nellie_path
        print(f'{file_info.output_dir=}')
        print(f'updated {file_info.shape=}')
        print('\n')

        # delete the cropped file if it already exists
        if os.path.isfile(self.crop_path):
            os.remove(self.crop_path)

        file_info.save_ome_tiff(data=cropped_images, all_channels=True)

        return file_info

    def adjust_timepoints_for_multiple(self, tif, channels, timepoints):
        # Get the number of pages in the series
        total_pages = len(tif.series[0].pages)

        # Calculate the expected multiple
        expected_multiple = channels * timepoints

        # Check if the total number of pages is a multiple of channels * timepoints
        if total_pages % expected_multiple != 0:
            print(f"Pages are not a multiple of channels * timepoints. Adjusting timepoints...")
            # Adjust the number of timepoints to ensure it is a multiple
            while total_pages % (channels * timepoints) != 0:
                # Reduce the number of timepoints by 1
                timepoints -= 1
                self.n_timepoints -= 1
                self.timepoints = list(range(self.n_timepoints))
                if timepoints <= 0:
                    raise ValueError("Cannot adjust timepoints to make pages a multiple of channels.")

            print(f"Adjusted number of timepoints to: {timepoints}")
        else:
            print("No adjustment needed. Pages are already a multiple of channels * timepoints.")

    # def check_page(self, page, page_index, t, cropped_images, t_index):
    #     if page is None:
    #         print(f"Page {page_index} is empty... skipping")
    #         missing_timepoint = t
    #         print(f"Missing timepoint: {missing_timepoint}")
    #         if missing_timepoint in self.timepoints:
    #             self.n_timepoints -= 1
    #             self.timepoints = list(range(self.n_timepoints))
    #             cropped_images = np.delete(cropped_images, missing_timepoint, axis=t_index)
    #             print(f'updated cropped_images shape: {cropped_images.shape}')
    #         return True
    #     return False

    def load_and_crop_images(self):
        # Calculate the shape of the cropped image
        cropped_shape = list(self.shape)

        if 'Z' in self.axes:
            cropped_shape[self.axes.find('Z')] = self.z_end - self.z_start if self.z_end is not None else self.shape[
                self.axes.find('Z')]
        if 'T' in self.axes:
            cropped_shape[self.axes.find('T')] = self.n_timepoints
        if 'C' in self.axes:
            cropped_shape[self.axes.find('C')] = self.n_channels
        cropped_shape[self.axes.find('Y')] = self.y_end - self.y_start
        cropped_shape[self.axes.find('X')] = self.x_end - self.x_start

        # Initialize an empty array with the same type as the original image but with cropped shape
        original_dtype = tifffile.TiffFile(self.im_path).series[0].dtype
        cropped_images = np.zeros(cropped_shape, dtype=original_dtype)
        print(f"Initialized cropped image shape: {cropped_images.shape}")

        count = 0

        with tifffile.TiffFile(self.im_path) as tif:
            # Determine the index of each axis
            t_index = self.axes.find('T')
            #print(f'{t_index=}')
            c_index = self.axes.find('C')
            #print(f'{c_index=}')
            z_index = self.axes.find('Z')
            #print(f'{z_index=}')
            y_index = self.axes.find('Y')
            x_index = self.axes.find('X')

            # # check for incomplete acquisition
            # self.adjust_timepoints_for_multiple(tif, self.n_channels, self.n_timepoints)
            # check for empty pages
            # for page_index in range(len(tif.series[0].pages)):
            #     if tif.series[0].pages[page_index] is None:
            #         print(f"Page {page_index} is empty... skipping")
            #         missing_timepoint = page_index // self.n_channels
            #         print(f"Missing timepoint: {missing_timepoint}")
            #         self.n_timepoints -= 1
            #         self.timepoints = list(range(self.n_timepoints))
            #         if missing_timepoint in self.timepoints:
            #             cropped_images = np.delete(cropped_images, missing_timepoint, axis=t_index)
            #         print(f'updated cropped_images shape: {cropped_images.shape}')


            # Loop over timepoints if they exist
            if t_index != -1:
                for t_pos, t_time in enumerate(self.timepoints):
                    # print(f"Processing timepoint {t_time}, {t_pos} of {self.n_timepoints}")
                    # Loop over channels if they exist
                    if c_index != -1:
                        for c in range(self.shape[c_index]):
                            if z_index != -1:  # If Z exists, iterate over Z, (T,C, Z, Y, X)
                                for z in range(self.z_start, self.z_end):
                                    page_index = (t_time * self.n_channels * self.z_end) + (c * self.z_end) + z
                                    page = tif.series[0].pages[page_index]
                                    # check if page is empty
                                    if page is None:
                                        print(f"Page {page_index} is empty... skipping")
                                        missing_timepoint = t_time
                                        print(f"Missing timepoint: {missing_timepoint}")
                                        if missing_timepoint in self.timepoints:
                                            self.n_timepoints -= 1
                                            self.timepoints = list(range(self.n_timepoints))
                                            cropped_images = np.delete(cropped_images, missing_timepoint, axis=t_index)
                                            print(f'updated cropped_images shape: {cropped_images.shape}')
                                        continue
                                    array = page.asarray()

                                    # Crop the array to the specified coordinates
                                    cropped_array = array[self.y_start:self.y_end, self.x_start:self.x_end]
                                    cropped_images[t_pos, c, z - self.z_start, :, :] = cropped_array
                                    count += 1
                                if t_time == self.timepoints[-1] and c == self.channels[-1]:
                                    self.max_z = np.max(cropped_images, axis=2)
                            else:  # If Z does not exist, handle 2D or 3D images without Z (T, C, Y, X)
                                page_index = (t_time * self.n_channels) + c

                                if page_index < 0:
                                    raise ValueError("Attempting to index a negative page index from tif")

                                # print(f'{page_index=}')
                                # print(f'{len(tif.series[0].pages)=}')
                                page = tif.series[0].pages[page_index]
                                # check if page is empty
                                if page is None:
                                    print(f"Page {page_index} is empty... skipping")
                                    missing_timepoint = t_time
                                    print(f"Missing timepoint: {missing_timepoint}")
                                    if missing_timepoint in self.timepoints:
                                        self.n_timepoints -= 1
                                        self.timepoints = list(range(self.n_timepoints))
                                        cropped_images = np.delete(cropped_images, missing_timepoint, axis=t_index)
                                        print(f'updated cropped_images shape: {cropped_images.shape}')
                                    continue
                                array = page.asarray()

                                # Crop the array to the specified coordinates
                                cropped_array = array[self.y_start:self.y_end, self.x_start:self.x_end]
                                cropped_images[t_pos, c, :, :] = cropped_array
                                count += 1
                                if t_time == self.timepoints[-1] and c == self.channels[-1]:
                                    print("max z image:")
                                    self.max_z = cropped_images

                    else:
                        # Handle case where no C dimension exists
                        if z_index != -1:  # If Z exists, iterate over Z (T, Z, Y, X)
                            # print(f' axes are (T, Z, Y, X)')
                            for z in range(self.z_start, self.z_end):
                                page_index = (t_time * self.z_end) + z
                                page = tif.series[0].pages[page_index]
                                # check if page is empty
                                if page is None:
                                    print(f"Page {page_index} is empty... skipping")
                                    missing_timepoint = t_time
                                    print(f"Missing timepoint: {missing_timepoint}")
                                    if missing_timepoint in self.timepoints:
                                        self.n_timepoints -= 1
                                        self.timepoints = list(range(self.n_timepoints))
                                        cropped_images = np.delete(cropped_images, missing_timepoint, axis=t_index)
                                        print(f'updated cropped_images shape: {cropped_images.shape}')
                                    continue
                                array = page.asarray()

                                # Crop the array to the specified coordinates
                                cropped_array = array[self.y_start:self.y_end, self.x_start:self.x_end]
                                cropped_images[t_pos, z - self.z_start, :, :] = cropped_array

                                count += 1
                            if t_time == self.timepoints[-1]:
                                self.max_z = np.max(cropped_images, axis=1)
                        else:  # If Z does not exist, handle 2D or 3D images without Z (T, Y, X)
                            page_index = t_time
                            page = tif.series[0].pages[page_index]
                            # check if page is empty
                            if page is None:
                                print(f"Page {page_index} is empty... skipping")
                                missing_timepoint = t_time
                                print(f"Missing timepoint: {missing_timepoint}")
                                if missing_timepoint in self.timepoints:
                                    self.n_timepoints -= 1
                                    self.timepoints = list(range(self.n_timepoints))
                                    cropped_images = np.delete(cropped_images, missing_timepoint, axis=t_index)
                                    print(f'updated cropped_images shape: {cropped_images.shape}')
                                continue
                            array = page.asarray()

                            # Crop the array to the specified coordinates
                            cropped_array = array[self.y_start:self.y_end, self.x_start:self.x_end]
                            cropped_images[t_pos, :, :] = cropped_array
                            count += 1
                            if t_time == self.timepoints[-1] - 1:
                                self.max_z = cropped_images
            else:
                # Handle case where no T dimension exists (C, Z, Y, X) or (C, Y, X)
                if c_index != -1:
                    for c in range(self.shape[c_index]):
                        if z_index != -1:  # If Z exists, iterate over Z (C, Z, Y, X)
                            for z in range(self.z_start, self.z_end):
                                page_index = (c * self.z_end) + z
                                array = tif.series[0].pages[page_index].asarray()

                                # Crop the array to the specified coordinates
                                cropped_array = array[self.y_start:self.y_end, self.x_start:self.x_end]
                                cropped_images[c, z - self.z_start, :, :] = cropped_array
                                count += 1
                            if c == self.shape[c_index] - 1:
                                self.max_z = np.max(cropped_images, axis=1)
                        else:  # If Z does not exist, handle 2D or 3D images without Z (C, Y, X)
                            page_index = c
                            array = tif.series[0].pages[page_index].asarray()

                            # Crop the array to the specified coordinates
                            cropped_array = array[self.y_start:self.y_end, self.x_start:self.x_end]
                            cropped_images[c, :, :] = cropped_array
                            count += 1
                            if c == self.shape[c_index] - 1:
                                self.max_z = np.max(cropped_images, axis=0)

                else:
                    # Handle case where no T or C dimension exists (Z, Y, X) or (Y, X)
                    if z_index != -1:  # If Z exists, iterate over Z (Z, Y, X)
                        for z in range(self.z_start, self.z_end):
                            page_index = z
                            array = tif.series[0].pages[page_index].asarray()

                            # Crop the array to the specified coordinates
                            cropped_array = array[self.y_start:self.y_end, self.x_start:self.x_end]
                            cropped_images[z - self.z_start, :, :] = cropped_array
                            count += 1
                        self.max_z = np.max(cropped_images, axis=0)
                    else:  # Handle 2D images, (X,Y) or (Y,X)
                        array = tif.series[0].pages[0].asarray()

                        # Crop the array to the specified coordinates
                        cropped_array = array[self.y_start:self.y_end, self.x_start:self.x_end]
                        cropped_images[:, :] = cropped_array
                        count += 1
                        self.max_z = cropped_images

        print(f"Final cropped image shape: {cropped_images.shape}")
        return cropped_images

    def make_mask(self, images):
        """
        Create a base mask based on the selected polygon ROI, expanded to cover the Z dimension if applicable.
        """
        # Determine the dimensions
        y_index = self.axes.find('Y')
        x_index = self.axes.find('X')
        z_index = self.axes.find('Z') if 'Z' in self.axes else -1

        # Determine the height and width of the mask
        mask_height = images.shape[y_index]
        mask_width = images.shape[x_index]

        print(f'{mask_height=}')
        print(f'{mask_width=}')

        # Initialize the base mask for Y and X dimensions
        if self.mask_coords['type'] == 'rectangle':
            print("Rectangle mask selected: ")
            self.x_start, self.y_start, self.x_end, self.y_end = self.mask_coords['coords']
            base_mask_2d = np.zeros((mask_height, mask_width), dtype=bool)
            base_mask_2d[self.y_start:self.y_end, self.x_start:self.x_end] = True
        elif self.mask_coords['type'] == 'polygon':
            print("Polygon mask selected: ")
            polygon_coords = self.mask_coords['coords']

            x, y = zip(*polygon_coords)

            base_mask_2d = np.zeros((mask_height, mask_width), dtype=bool)

            rr, cc = polygon(np.array(y), np.array(x))

            # Clamp the coordinates to the mask size
            rr = np.clip(rr, 0, mask_height - 1)
            cc = np.clip(cc, 0, mask_width - 1)

            print(f"Original polygon coords (x, y): {polygon_coords}")
            print(f"Adjusted polygon coords (x, y): {(np.array(x) - self.x_start, np.array(y) - self.y_start)}")
            print(f"Polygon rr: {rr}")
            print(f"Polygon cc: {cc}")

            base_mask_2d[rr, cc] = True

        # If a Z dimension exists, expand the 2D mask to cover the Z dimension
        if z_index != -1:
            z_depth = self.z_end - self.z_start
            base_mask = np.repeat(base_mask_2d[np.newaxis, :, :], z_depth, axis=0)
        else:
            base_mask = base_mask_2d
        if self.show_crop:
            self.napari_view(image=base_mask, IMname='Base Mask', contrast_limit=0.1, layer3d=self.is3D)

        base_mask = xp.asarray(base_mask, dtype=bool)
        self.mask = base_mask
        # self.napari_view(image=base_mask, IMname='Base Mask', contrast_limit=0.1, layer3d=self.is3D)

    # scan across the z dimension and allow use to use line selection tool to select area of for cropping
    def select_roi_and_z_range(self, t=0, c=0, input_image=None):
        """Allow the user to select an XY ROI on the 3D Z-stack and select the Z cropping range using an XZ projection."""

        if input_image is None:
            self.roi_Zstack = self._open_ome(t=t, c=c)
            array = np.stack(self.roi_Zstack, axis=0)
            if self.is3D:  # 3D Z-stack
                # max_projection = np.max(array, axis=0)  # XY max projection for ROI selection
                print("projecting along XZ")
                print(f'{array.shape=}')
                xz_projection = np.max(array, axis=2)  # XZ max projection for Z range selection
                from scipy.ndimage import rotate
                xz_projection = rotate(xz_projection, angle=197, reshape=True)
            else:  # 2D image
                print("2D image, can not select Z range")
                exit()
        else:
            xz_projection = input_image

        # print(f"Loaded array shape for timepoint {t}, channel {c}: {max_projection.shape}")

        viewer = napari.Viewer()
        # viewer.add_image(max_projection, name='max projection volume1')

        # Display XZ projection for Z range selection
        viewer.add_image(xz_projection, name='XZ projection')
        shapes = viewer.add_shapes(name='Z line', ndim=2, shape_type='line')

        @shapes.mouse_drag_callbacks.append
        def get_shape(layer, event):
            yield  # Start the callback
            while event.type == 'mouse_move':
                yield
            if len(layer.data) > 0:
                shape = layer.data[-1]
                if layer.shape_type[-1] == 'line':
                    print("Selected line points: ", shape)
                    y1 = shape[0][0]
                    y2 = shape[1][0]
                    self.z_start = min(y1, y2)
                    self.z_end = max(y1, y2)
                    # Adjust the coordinates to fit within the image boundaries
                    # if self.image_type == 'ome':
                    self.z_start = max(0, int(self.z_start))
                    self.z_end = min(xz_projection.shape[-1], int(self.z_end))
                    notifications.show_info(
                        f"Selected line coordinates: {self.z_start} to {self.z_end}")
                    viewer.close()

        # @viewer.bind_key('Enter', overwrite=True)
        # def finalize_selection(event):
        #     if len(z_line.data) > 0:
        #         line = z_line.data[-1]
        #         print("Selected line points: ", line)
        #         z_start, x_start = line[0]
        #         z_end, x_end = line[1]
        #
        #         self.z_start = max(0, int(min(z_start, z_end)))
        #         self.z_end = min(array.shape[0], int(max(z_start, z_end)))
        #
        #         print(f"Selected Z range: {self.z_start} to {self.z_end}")
        #         viewer.close()

        napari.run()

        if self.z_start is None or self.z_end is None:
            raise ValueError("No ROI or Z range selected")

        return self.z_start, self.z_end

    def load_metadata(self, metadata_path):
        with open(metadata_path) as f:
            lines = f.readlines()
        metadata_dict = dict()
        for line in lines:
            split = line.split(": ")
            metadata_dict[split[0]] = split[1][:-1]
        return metadata_dict

    # def crop_unskewed(self):
    #     topY_percent = 0.05
    #     # self.axes = 'TCZYX'
    #     if self.selected_timepoints is not None:
    #         self.timepoints = self.selected_timepoints
    #         self.n_timepoints = len(self.timepoints)
    #
    #     # select crop roi on 1st file from data folder
    #     """Crop the OME-TIFF image using the user-selected ROI."""
    #     # select the rectanglular area to crop
    #     # if os.path.isdir(self.im_path):
    #     parent_dir = self.im_path# os.path.dirname(self.im_path)
    #     self.data_folder = os.path.join(parent_dir, 'data')
    #     metadata_dir = os.path.join(parent_dir, 'metadata')
    #     if self.image_type == 'unskewed-multifile':
    #         sub_dirs = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f)) and 'seriesCount' in f]
    #         self.data_folder = os.path.join(parent_dir, sub_dirs[0], 'data')
    #         metadata_dir = os.path.join(parent_dir, sub_dirs[0], 'metadata')
    #     self.volume1 = os.path.join(self.data_folder, os.listdir(self.data_folder)[0])
    #
    #     # file_info = self.getMetadata(input_image_path=self.volume1)
    #     # self.axes = file_info.axes
    #     # print(f'{file_info.axes=}')
    #     # print(f'{self.axes=}')
    #     # self.shape = file_info.shape
    #     # print("original image shape: ", self.shape)
    #
    #
    #     with tifffile.TiffFile(self.volume1) as tif:
    #         # Read the image data
    #         image_data = tif.asarray()
    #     print(f"Loaded array shape: {image_data.shape}")
    #     # crop off the tiem stamp off the y axis
    #     image_data = image_data[:,:,int(image_data.shape[2] * topY_percent):, :]
    #     print(f"Trimmed array shape: {image_data.shape}")
    #     # project down to 2D
    #     max_projection = np.max(image_data, axis=1)
    #     max_projection2 = np.max(max_projection, axis=1)
    #
    #     # viewer = napari.Viewer()
    #     # viewer.add_image(max_projection, name='unskewed max projection (frame1)')
    #     # # viewer.camera.angles = (180, 0, 0)  # Rotate around the X-axis by 180 degrees
    #     # napari.run()
    #     notifications.show_info("SELECT RECTANGULAR ROI FOR X,Y CROPPING")
    #     self.select_roi(input_image=max_projection2, center_coords=None, zoom_factor=0.0)
    #
    #     if self.select_z:
    #         max_projection2 = np.max(max_projection, axis=0)
    #         notifications.show_info("SELECT LINE ROI FOR Z RANGE SELECTION")
    #         self.z_start, self.z_end = self.select_roi_and_z_range(t=0, c=self.selected_channel, input_image=max_projection2)
    #
    #     # crop each file in data folder into new cropped_data folder
    #     # self.cropped_images = self.load_and_crop_images()
    #     #print(f"Cropped image shape: {self.cropped_images.shape}")
    #     #file_info = self.save_file(self.cropped_images)
    #     # print(f"Saved cropped image to {self.crop_path}")
    #
    #     # crop data folder
    #     # make umbrella folder for all crop files
    #     # save_parent_dir = os.path.dirname(os.path.dirname(self.crop_path))
    #     self.crop_folder = os.path.dirname(self.crop_path)
    #     # print(f"Saving deskew cropped files to {self.crop_folder}")
    #     if not os.path.isdir(self.crop_folder):
    #         os.mkdir(self.crop_folder)
    #
    #     self.crop_data_folder = os.path.join(self.crop_folder,'data')
    #     self.crop_metadata_folder = os.path.join(self.crop_folder,'metadata')
    #     if not os.path.isdir(self.crop_data_folder):
    #         os.mkdir(self.crop_data_folder)
    #     if not os.path.isdir(self.crop_metadata_folder):
    #         os.mkdir(self.crop_metadata_folder)
    #
    #     data_files = []
    #     data_files_paths = []
    #     metadata_files = []
    #     metadata_files_paths = []
    #     if self.image_type == 'ome':
    #         metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith(".txt")]
    #         data_files = [f for f in os.listdir(self.data_folder) if f.endswith(".tif")]
    #         data_files.sort()         # order data files by time
    #         metadata_files.sort()         # order data files by time
    #         # filter to just ones in selected timepoints
    #         data_files = [data_files[i] for i in self.timepoints]
    #         metadata_files = [metadata_files[i] for i in self.timepoints]
    #     elif self.image_type == 'unskewed-multifile':
    #         sub_dirs = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f)) and 'seriesCount' in f]
    #         for sub_dir in sub_dirs:
    #             metadata_path = os.path.join(parent_dir, sub_dir, 'metadata')
    #             # metadata_files = [metadata_files.append(f) for f in os.listdir(metadata_path) if f.endswith(".txt")]
    #             metadata_files_paths = [metadata_files_paths.append(os.path.join(metadata_path, f)) for f in metadata_files if f.endswith(".txt")]
    #             data_path = os.path.join(parent_dir, sub_dir, 'data')
    #             data_files = [data_files.append(f) for f in os.listdir(data_path) if f.endswith(".tif")]
    #             data_files_paths = [data_files_paths.append(os.path.join(data_path, f)) for f in data_files if f.endswith(".tif")]
    #             data_files_paths = [data_files_paths[i] for i in self.timepoints]
    #             metadata_files_paths = [metadata_files_paths[i] for i in self.timepoints]
    #
    #
    #     count = 0
    #     for file in tqdm(data_files):
    #         # check if cropped file alread exists
    #         if os.path.isfile(os.path.join(self.crop_data_folder, file)):
    #             print(f"Skipping {file} as it already exists")
    #             continue
    #         with tifffile.TiffFile(os.path.join(self.data_folder, file)) as tif:
    #             # Read the image data
    #             image_data = tif.asarray()
    #             metadata = tif.ome_metadata
    #         # crop off the time stamp off the y axis
    #         image_data = image_data[:, :, int(image_data.shape[2] * topY_percent):, :]
    #         # Crop the array to the specified coordinates
    #         cropped_images = image_data[self.y_start:self.y_end, :, :,  self.x_start:self.x_end] # 200:image_data.shape[0]
    #         if self.select_z:
    #             cropped_images = cropped_images[:, :, self.z_start:self.z_end, :]
    #         if count == 0:
    #             print(f"Original image shape: {image_data.shape}")
    #             print(f"Cropped image shape: {cropped_images.shape}")
    #
    #             # project down to 2D
    #             max_projection = np.max(cropped_images, axis=1)
    #             max_projection = np.max(max_projection, axis=1)
    #             viewer = napari.Viewer()
    #             viewer.add_image(max_projection, name='cropped max projection (frame1)')
    #             # viewer.camera.angles = (180, 0, 0)  # Rotate around the X-axis by 180 degrees
    #             napari.run()
    #
    #         # save the cropped image to the cropped_data folder
    #         cropped_file = os.path.join(self.crop_data_folder, file)
    #         imwrite(cropped_file, cropped_images)
    #         # print(f"Saved cropped image to {cropped_file}")
    #
    #         metadata_path = os.path.join(metadata_dir, metadata_files[count])
    #         metadata = self.load_metadata(metadata_path)
    #
    #         # update the metadata
    #         metadata['slices_per_volume'] = cropped_images.shape[0]
    #         metadata['width_px'] = cropped_images.shape[3]
    #         metadata['height_px'] = cropped_images.shape[2]
    #
    #         # save the metadata to the cropped_metadata folder
    #         crop_metadata_file = os.path.join(self.crop_metadata_folder, metadata_files[count])
    #         with open(crop_metadata_file, 'w') as f:
    #             for key in metadata:
    #                 f.write(f"{key}: {metadata[key]}\n")
    #         count += 1
    #
    #     # deskew the cropped data folder into single deskew file
    #     filename = f"deskewed-crop{self.crop_count}.ome.tif"
    #     self.deskew(filename)
    #
    #     # pass the deskew file to the next step for the rest of the pipeline
    #     self.im_path = os.path.join(self.crop_folder, filename)
    #     self.crop_path = os.path.join(self.crop_folder, f'crop{self.crop_count}.ome.tif')
    #     self.nellie_path = os.path.join(self.crop_folder, (f'crop{self.crop_count}_nellie_out'))
    #     self.center_coords = None
    #     self.image_type = 'ome'
    #     self.select_z = False
    #     file_info = self.crop_ome()
    #     # file_info = self.save_file(deskew_path)
    #
    #     return file_info

    def crop_unskewed(self):
        topY_percent = 0.05
        parent_dir = self.im_path  # os.path.dirname(self.im_path)
        self.data_folder = os.path.join(parent_dir, 'data')
        # self.axes = 'TCZYX'
        if self.selected_timepoints is not None:
            self.timepoints = self.selected_timepoints
            self.n_timepoints = len(self.timepoints)
        else:
            self.n_timepoints = len(os.listdir(self.data_folder))
            self.timepoints = list(range(self.n_timepoints))

        # crop data folder
        # make umbrella folder for all crop files
        # save_parent_dir = os.path.dirname(os.path.dirname(self.crop_path))
        self.crop_folder = os.path.dirname(self.crop_path)
        if not os.path.isdir(self.crop_folder):
            os.mkdir(self.crop_folder)
        filename = f"deskewed-crop{self.crop_count}.ome.tif"
        if not os.path.isfile(os.path.join(self.crop_folder, filename)):
            # select crop roi on 1st file from data folder
            """Crop the OME-TIFF image using the user-selected ROI."""
            # select the rectanglular area to crop
            # if os.path.isdir(self.im_path):

            self.volume1 = os.path.join(self.data_folder, os.listdir(self.data_folder)[0])

            with tifffile.TiffFile(self.volume1) as tif:
                # Read the image data
                image_data = tif.asarray()
            print(f"Loaded array shape: {image_data.shape}")
            if len(image_data.shape) == 3:
                image_data = image_data[:, np.newaxis, :, :]
            # crop off the tiem stamp off the y axis
            image_data = image_data[:, :, int(image_data.shape[2] * topY_percent):, :]
            print(f"Trimmed array shape: {image_data.shape}")

            # project down to 2D
            max_projection = np.max(image_data, axis=1)
            max_projection2 = np.max(max_projection, axis=1)

            # viewer = napari.Viewer()
            # viewer.add_image(max_projection, name='unskewed max projection (frame1)')
            # # viewer.camera.angles = (180, 0, 0)  # Rotate around the X-axis by 180 degrees
            # napari.run()
            notifications.show_info("SELECT RECTANGULAR ROI FOR X,Y CROPPING")
            self.select_roi(input_image=max_projection2, center_coords=None, zoom_factor=0.0)

            if self.select_z:
                max_projection2 = np.max(max_projection, axis=0)
                notifications.show_info("SELECT LINE ROI FOR Z RANGE SELECTION")
                self.z_start, self.z_end = self.select_roi_and_z_range(t=0, c=self.selected_channel,
                                                                       input_image=max_projection2)

            # crop each file in data folder into new cropped_data folder
            self.crop_data_folder = os.path.join(self.crop_folder, 'data')
            self.crop_metadata_folder = os.path.join(self.crop_folder, 'metadata')
            if not os.path.isdir(self.crop_data_folder):
                os.mkdir(self.crop_data_folder)
            if not os.path.isdir(self.crop_metadata_folder):
                os.mkdir(self.crop_metadata_folder)
            count = 0
            metadata_dir = os.path.join(parent_dir, 'metadata')
            metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith(".txt")]
            data_files = [f for f in os.listdir(self.data_folder) if f.endswith(".tif")]
            # order data files by time
            data_files.sort()
            metadata_files.sort()
            # filter to just ones in selected timepoints
            data_files = [data_files[i - 1] for i in self.timepoints]
            metadata_files = [metadata_files[i - 1] for i in self.timepoints]
            for file in tqdm(data_files):
                # check if cropped file alread exists
                if os.path.isfile(os.path.join(self.crop_data_folder, file)):
                    print(f"Skipping {file} as it already exists")
                    continue
                with tifffile.TiffFile(os.path.join(self.data_folder, file)) as tif:
                    # Read the image data
                    image_data = tif.asarray()
                    metadata = tif.ome_metadata
                # crop off the time stamp off the y axis
                if len(image_data.shape) == 3:
                    image_data = image_data[:, np.newaxis, :, :]
                image_data = image_data[:, :, int(image_data.shape[2] * topY_percent):, :]
                # Crop the array to the specified coordinates
                cropped_images = image_data[self.y_start:self.y_end, :, :,
                                 self.x_start:self.x_end]  # 200:image_data.shape[0]
                if self.select_z:
                    cropped_images = cropped_images[:, :, self.z_start:self.z_end, :]
                if count == self.pearl_frame:
                    print(f"Original image shape: {image_data.shape}")
                    print(f"Cropped image shape: {cropped_images.shape}")

                    # project down to 2D
                    max_projection = np.max(cropped_images, axis=1)
                    max_projection = np.max(max_projection, axis=1)
                    viewer = napari.Viewer()
                    viewer.add_image(max_projection, name='cropped max projection (frame1)')
                    # viewer.camera.angles = (180, 0, 0)  # Rotate around the X-axis by 180 degrees
                    napari.run()

                # print(f"Saved cropped image to {cropped_file}")

                metadata_path = os.path.join(metadata_dir, metadata_files[count])
                metadata = self.load_metadata(metadata_path)

                # update the metadata
                metadata['slices_per_volume'] = cropped_images.shape[0]
                metadata['width_px'] = cropped_images.shape[3]
                metadata['height_px'] = cropped_images.shape[2]

                # remove extra dimension if needed
                if len(cropped_images.shape) == 4 and cropped_images.shape[1] == 1:
                    cropped_images = cropped_images[:, 0, :, :]
                    # cropped_images = np.squeeze(cropped_images, axis=1)
                    # print(f"updated cropped_images shape: {cropped_images.shape}")
                # save the cropped image to the cropped_data folder
                cropped_file = os.path.join(self.crop_data_folder, file)
                imwrite(cropped_file, cropped_images)

                # save the metadata to the cropped_metadata folder
                crop_metadata_file = os.path.join(self.crop_metadata_folder, metadata_files[count])
                with open(crop_metadata_file, 'w') as f:
                    for key in metadata:
                        f.write(f"{key}: {metadata[key]}\n")
                count += 1

            # deskew the cropped data folder into single deskew file
            self.deskew(filename)

        # pass the deskew file to the next step for the rest of the pipeline
        self.im_path = os.path.join(self.crop_folder, filename)
        self.crop_path = os.path.join(self.crop_folder, f'crop{self.crop_count}.ome.tif')
        self.nellie_path = os.path.join(self.crop_folder, (f'crop{self.crop_count}_nellie_out'))
        self.center_coords = None
        self.image_type = 'ome'
        self.select_z = False
        file_info = self.crop_ome()
        # file_info = self.save_file(deskew_path)

        return file_info

    def deskew(self, filename):
        from snouty_viewer.im_loader import ImPathInfo, allocate_memory_return_memmap, load_full
        from snouty_viewer._widget import PseudoImage
        from snouty_viewer._widget import ImInfo as SV_Im_Info
        print(f'deskewing {filename=}')
        im_path_info = ImPathInfo(self.crop_folder)
        print(f'{im_path_info.im_shape=}')
        # print(f'{im_path_info.metadata=}')
        skewed_memmap = PseudoImage(load_full(im_path_info, self.crop_folder)[0])
        im_info = SV_Im_Info(skewed_memmap, im_path_info.im_shape)

        # name =  im_path_info.path.rsplit(os.sep)[-1]
        # skewed_memmap_path = os.path.join(path_out, f"skewed-{name}.ome.tif")
        save_path = os.path.join(self.crop_folder, filename)
        print(f'{save_path=}')
        print(f'{im_info.im_desheared_shape=}')

        deskewed_memmap = allocate_memory_return_memmap(
            "TCZYX",
            im_info.im_desheared_shape,
            im_path_info.metadata,
            save_path,
            im_info.dtype,
        )
        im_info.im_desheared = deskewed_memmap
        im_info.deshear_all_channels(batch=False, show_multi=True)

    def set_timepoints(self):
        # Check if selected_timepoints is None or contains NaN values
        if self.selected_timepoints is None:
            if 'T' in self.axes:
                self.n_timepoints = self.shape[self.axes.index('T')]
                self.timepoints = list(range(self.n_timepoints))
            else:
                self.n_timepoints = 1
                self.timepoints = []
        else:
            self.timepoints = self.selected_timepoints
            self.n_timepoints = len(self.timepoints)

        print(f"Timepoints: {self.timepoints}")
    def set_channels(self):
        if self.selected_channel is None:
            self.selected_channel = 0
        if 'C' in self.axes:
            self.n_channels = self.shape[self.axes.index('C')]
            self.channels = list(range(self.n_channels))
        # print(f"Channels: {self.channels}")
        # print(f"Selected channel: {self.selected_channel}")
        # print(f"n channels: {self.n_channels}")

    def setZ(self):
        if 'Z' in self.axes:
            self.is3D = True
            print("3D image")
            if self.selected_z is None:
                self.z_start = 0
                self.z_end = self.shape[self.axes.index('Z')]
            else:
                self.z_start = max((self.shape[self.axes.index('Z')] // 2) - self.selected_z, 0)
                self.z_end = min((self.shape[self.axes.index('Z')] // 2) + self.selected_z,
                                 self.shape[self.axes.index('Z')])
        else:
            print("2D image")
            self.z_start = None
            self.z_end = None


    def set_z_range(self):
        self.setZ()
        if self.select_z:
            notifications.show_info("SELECT LINE ROI FOR Z RANGE SELECTION")
            self.z_start, self.z_end = self.select_roi_and_z_range(t=0, c=self.selected_channel)
        else:
            self.z_start = 0
            self.z_end = self.shape[self.axes.index('Z')] if 'Z' in self.axes else None

    def crop_ome(self):
        file_info = self.getMetadata()
        self.axes = file_info.axes
        print(f'{file_info.axes=}')
        print(f'{self.axes=}')

        self.shape = file_info.shape
        print("original image shape: ", self.shape)

        self.set_timepoints()
        self.set_channels()
        self.set_z_range()


        """Crop the OME-TIFF image using the user-selected ROI."""
        # select the rectanglular area to crop
        t = self.pearl_frame + self.timepoints[0]
        c = self.selected_channel
        print(f'selecting frame {t} channel {c}')
        max_projection = self._open_ome(t=t, c=c)
        # print(f"Loaded array shape for timepoint {t}, channel {c}: {max_projection.shape}")
        notifications.show_info("SELECT RECTANGULAR ROI FOR X,Y CROPPING")
        self.select_roi(input_image=max_projection, center_coords=self.center_coords, zoom_factor=2.0)

        self.cropped_images = self.load_and_crop_images()
        print(f"Cropped image shape: {self.cropped_images.shape}")
        file_info = self.save_file(self.cropped_images)
        print(f"Saved cropped image to {self.crop_path}")

        if self.show_crop:
            self.napari_view(image=self.cropped_images, IMname='Cropped Image (all volumes)', contrast_limit=0.2,
                             layer3d=True)

        return file_info

    def crop_image(self):
        if self.image_type == 'unskewed' or self.image_type == 'unskewed-multifile':
            self.file_info = self.crop_unskewed()
        else:
            self.file_info = self.crop_ome()
        return self.file_info

    def select_mask(self):
        # select the polygon area for the mask
        print("Select the polygon area for the mask")
        if self.max_z is None:
            raise ValueError("No image data provided")
        print(f'{self.max_z.shape=}')
        notifications.show_info("SELECT POLYGON ROI FOR X,Y MASK")
        self.mask_coords = self.select_roi(input_image=self.max_z)
        self.make_mask(self.cropped_images)
        self.save_mask()

    def save_mask(self):
        """
        Save the mask as an OME-TIFF file.
        """
        # Define the path where the mask will be saved
        save_path = os.path.join(self.nellie_path, 'mask.npy')
        # Ensure the directory exists
        os.makedirs(self.nellie_path, exist_ok=True)

        # delete the mask if it already exists
        if os.path.isfile(save_path):
            os.remove(save_path)

        # Save the mask as a NumPy file
        xp.save(save_path, self.mask)

        # Convert mask to uint8 for saving (still as a CuPy array)
        mask = self.mask.astype(xp.uint8)

        # Convert CuPy array to NumPy array just for saving
        mask_np = xp.asnumpy(mask)

        # Define the path where the mask will be saved
        save_path = os.path.join(self.nellie_path, 'mask.ome.tif')

        with tifffile.TiffWriter(save_path, bigtiff=True) as tif:
            options = dict(
                photometric='minisblack',
                metadata={'axes': 'ZYX' if mask_np.ndim == 3 else 'YX'}
            )
            tif.write(mask_np, **options)

        print(f"Mask saved as '{save_path}'.")
