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



if __name__ == "__main__":
    folder = r"F:\Calico\SNOUTY_data\2022-12-01_kayley_ionomycin_injections\2022-12-01_16-48-40_000_COX8A_CellLigt-ER-RFP_300volumes_beading"
    basename = ""
    im_path = os.path.join(folder, basename)
    print(f'running nellie on {im_path}')
    nellie_path = os.path.join(folder, 'nellie-crop-output')
    crop_path = os.path.join(folder, 'crop1.ome.tif')
    crop_snout = CropSnout(im_path=im_path, crop_path=crop_path, nellie_path=nellie_path,
                           select_timepoints=None, select_channel=0, select_z=True, image_type='unskewed',
                           time_interval=1, center_coords=None, show_crop=True)
    file_info = crop_snout.crop_image()  # Allow the user to select the ROI
    crop_path = file_info.ome_output_path
    nellie_path = file_info.output_dir
    crop_snout.select_mask()
    # mask = crop_snout.getMask()

    print(f'running nellie on {crop_path}')
    print(f'saving nellie outputs to {nellie_path=}')
    file_info = FileInfo(filepath=crop_path, output_dir=nellie_path)
    file_info.find_metadata()
    file_info.load_metadata()
    print(f'image shape: {file_info.shape=}')
    mask = xp.load(os.path.join(nellie_path, 'mask.npy'))
    print(f'mask shape: {mask.shape=}')
    run(file_info, mask=mask)


# folder = r"E:\SNOUTY_data\Pearling-examples\austin-pearling-metrics-example-3d-snouty"
#     # FCCP: r"F:\UCSF\W1_FRAP\2023-08-15_tmre-sybrgold\TMRE\FCCP\RPE1_CellLight_Mito_GFP_FCCP_4uM_3min_1-good"
#     # 2D: r"F:\UCSF\W1_FRAP\2023-08-08_ER-PKMO\RPE1-ER_3-pearling"
#     # 3D: r"E:\SNOUTY_data\Pearling-examples\austin-pearling-metrics-example-3d-snouty"
# basename =  "deskewed-austin-pearling-metrics-example-3d-snouty.ome.tif"
#     # FCCP: "RPE1_CellLight_Mito_GFP_FCCP_4uM_3min_1_MMStack_Default.ome.tif"
#     # 2D: "RPE1-ER_3_MMStack_Default.ome.tif"
#     # 3D: "deskewed-austin-pearling-metrics-example-3d-snouty.ome.tif"
