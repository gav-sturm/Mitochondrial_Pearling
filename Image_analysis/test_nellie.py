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



im_path = r"F:\UCSF\Micro_needle\20240814_Gav_Mito_Needle_U2OS\DIsh1_cell_2\DIsh1_cell_2_MMStack_Default_imageJ.ome.tif"
file_info = FileInfo(filepath=im_path)
file_info.find_metadata()
file_info.load_metadata()
print(f'file_info: {file_info.axes=}')
file_info.change_selected_channel(0)
print(f'selected channel {file_info.ch=}')
print(f'image shape: {file_info.shape=}')
run(file_info, mask=None, remove_edges=False)