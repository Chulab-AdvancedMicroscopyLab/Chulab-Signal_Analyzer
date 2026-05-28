"""
Usage:
python Chulab-Signal_Analyzer/vessel_analyzer.py --config configs/config.json
"""

import sys
import json
from pathlib import Path

def _pre_init_concurrency():
    config_path = "configs/config.json"
    for i, arg in enumerate(sys.argv):
        if (arg == "--config" or arg == "-c") and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    config = {}
    if Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception:
            pass
    resources = config.get("resources", {})
    from utils.concurrency import initialize_concurrency
    initialize_concurrency(
        numba_threads=resources.get("numba_threads", 8),
        dask_threads=resources.get("dask_threads", 8),
    )

_pre_init_concurrency()

import time
import os
import argparse
import numpy as np
import dask.array as da

from tqdm import tqdm
from dask.diagnostics.progress import ProgressBar
from skimage.morphology import skeletonize
from skimage.measure import regionprops
from scipy.ndimage import convolve, label, distance_transform_edt, median_filter

from utils.analyzer_count_tools import numba_unique_vessel
from utils.analyzer_report_tools import create_vessel_report
from utils.concurrency import initialize_concurrency

def check_and_load_zarr(path, component=None, chunk_size=None):
    """
    Check and load a Zarr array from a given path and component.

    Parameters:
        path (str): Path to the Zarr directory.
        component (str, optional): Component within the Zarr store.
        chunk_size (tuple, optional): Desired chunk size.

    Returns:
        dask.array.Array or None
    """
    if not path:
        return None

    full_path = os.path.join(path, component) if component else path
    if not os.path.exists(full_path):
        return None
    if not os.path.exists(os.path.join(full_path, '.zarray')):
        print(f"⚠️  Skipping (not a zarr array): {full_path}")
        return None
    print(f"✅ Found: {full_path}")
    return da.from_zarr(full_path, chunks=chunk_size) if chunk_size else da.from_zarr(full_path)

def process_filter_chunk(block, filter_size):
    """
    Applies a median filter followed by a Gaussian blur to a 3D image block on the CPU.

    Parameters:
        block (np.ndarray): The 3D image chunk to process.
        filter_size (int): The radius for the median filter in all dimensions.

    Returns:
        np.ndarray: The filtered and thresholded image block.
    """
    # 1. Binarize the input block to work with a mask.
    binary_block = (block > 0).astype(np.float32)

    # 2. Apply a median filter using scipy.ndimage.
    # Note: size = 2 * radius + 1. The original code used a radius.
    filter_diameter = 2 * filter_size + 1
    median_filtered = median_filter(
        input=binary_block,
        size=filter_diameter
    )

    # 3. Apply the final threshold condition to get a binary mask.
    final_mask = (median_filtered > 0.5).astype(np.uint8)
    
    return final_mask

def process_skeletonize_chunk(block):
    """Skeletonize a binary 3D block and mark bifurcation/trifurcation points."""
    block[block > 0] = 1
    skeleton = skeletonize(block.astype(np.uint8)).astype(np.uint8)
    skeleton *= block

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = convolve(skeleton, kernel, mode='constant')
    
    bifurcation_candidates = (skeleton > 0) & (neighbor_count == 3)
    labeled_bifurcations, _ = label(bifurcation_candidates)  # type: ignore
    labeled_bifurcations = labeled_bifurcations.astype(np.int32)
    
    for region in regionprops(labeled_bifurcations):
        com = tuple(np.round(region.centroid).astype(int))
        skeleton[com] = 2

    trifurcation_candidates = (skeleton > 0) & (neighbor_count >= 4)
    labeled_trifurcations, _ = label(trifurcation_candidates)  # type: ignore
    labeled_trifurcations = labeled_trifurcations.astype(np.int32)
    
    for region in regionprops(labeled_trifurcations):
        com = tuple(np.round(region.centroid).astype(int))
        skeleton[com] = 3

    return skeleton

def process_distance_transform(block):
    """Compute Euclidean distance transform for a 3D block."""
    return distance_transform_edt(block > 0)

def process_calculation_chunk(anno, hema, mask, skel, dist):
    """
    Perform voxel-wise statistics for labeled regions in a 3D volume.

    Parameters:
        anno (ndarray): Label image.
        hema (ndarray): Hemisphere mask.
        mask (ndarray): Signal mask.
        skel (ndarray): Skeletonized signal.
        dist (ndarray): Distance transform.

    Returns:
        dict: Mapping of region ID to a stats vector.
    """
    return dict(numba_unique_vessel(anno, hema, mask, skel, dist))

def process_analysis_report(region_signals, voxel, output_name, output_path, annotation_map_path=None, mask_folder=""):
    """
    Generate Excel and CSV reports from signal dictionaries for multiple brain regions.

    Parameters:
        region_signals (dict): Keys are region names, values are signal dictionaries.
        voxel (tuple): Voxel size as a 3D tuple (x, y, z).
        output_name (str): Base name for the output files.
        output_path (str): Directory to save the reports.
        annotation_map_path (str, optional): Path to structures.csv.
    """
    if annotation_map_path:
        structure_path = annotation_map_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        structure_path = os.path.join(script_dir, 'utils', 'annotation_maps', 'allen_mouse_brain_structure.csv')
    target_id = None

    os.makedirs(output_path, exist_ok=True)
    voxel_volume = np.prod(voxel)

    for region_name, signal in region_signals.items():
        output_file_xlsx = os.path.join(output_path, f'{mask_folder}_{output_name}_{region_name}_report.xlsx')
        
        # Generate Excel report
        create_vessel_report(signal, voxel_volume, output_file_xlsx, structure_path, target_id)

def run_task(task):
    chunk_size = tuple(task["chunk_size"]) if task.get("chunk_size") else None

    start_time = time.time()
    mask_data = check_and_load_zarr(task["mask_path"], chunk_size=chunk_size)
    anno_data = check_and_load_zarr(task["annotation_path"], chunk_size=chunk_size)
    hema_data = check_and_load_zarr(task.get("hemasphere_path"), chunk_size=chunk_size)

    print(f"Mask shape: {mask_data.shape}") # type: ignore
    print(f"Annotation shape: {anno_data.shape}") # type: ignore

    # # Step 1: Filtering
    # filtered_data = check_and_load_zarr(task["output_path"], "filtered_mask.zarr", chunk_size=chunk_size)
    # if filtered_data is None:
    #     print("🔄 Applying Gaussian filter...")
    #     with ProgressBar():
    #         filtered_data = da.map_blocks(
    #             process_filter_chunk,
    #             mask_data, dtype=np.uint8
    #         )
    #         filtered_data.to_zarr(os.path.join(task["output_path"], "filtered_mask.zarr"), overwrite=True)
    #         filtered_data = da.from_zarr(os.path.join(task["output_path"], "filtered_mask.zarr"))

    # Step 2: Skeletonization
    skeleton_data = check_and_load_zarr(task["output_path"], "skeletonize_mask.zarr", chunk_size=chunk_size)
    if skeleton_data is None:
        print("🔄 Skeletonizing vessel mask...")
        with ProgressBar():
            skeleton_data = da.map_blocks(
                process_skeletonize_chunk,
                mask_data, dtype=np.uint8
            )
            skeleton_data.to_zarr(os.path.join(task["output_path"], "skeletonize_mask.zarr"), overwrite=True)
            skeleton_data = da.from_zarr(os.path.join(task["output_path"], "skeletonize_mask.zarr"))

    # Step 3: Distance Transform
    distance_data = check_and_load_zarr(task["output_path"], "distance_mask.zarr", chunk_size=chunk_size)
    if distance_data is None:
        print("🔄 Calculating distance transform...")
        with ProgressBar():
            distance_data = da.map_blocks(
                process_distance_transform,
                mask_data, dtype=np.float32
            )
            distance_data.to_zarr(os.path.join(task["output_path"], "distance_mask.zarr"), overwrite=True)
            distance_data = da.from_zarr(os.path.join(task["output_path"], "distance_mask.zarr"))

    # Step 4: Feature Extraction
    print("🔄 Extracting features...")
    full_brain_signal = {}
    left_brain_signal = {}
    right_brain_signal = {}
    img_dimension = mask_data.shape # type: ignore

    for i in tqdm(range(0, img_dimension[0], task.get("z_per_slab", 16))):
        start_i, end_i = i, min(i + task.get("z_per_slab", 16), img_dimension[0])
        if hema_data is None:
            anno_chunk, mask_chunk, skel_chunk, dist_chunk = da.compute(
                anno_data[start_i:end_i], # type: ignore
                mask_data[start_i:end_i],  # type: ignore
                skeleton_data[start_i:end_i],
                distance_data[start_i:end_i],
            )
            hema_chunk = np.zeros_like(anno_chunk)
        else:
            anno_chunk, hema_chunk, mask_chunk, skel_chunk, dist_chunk = da.compute(
                anno_data[start_i:end_i], # type: ignore
                hema_data[start_i:end_i],
                mask_data[start_i:end_i],  # type: ignore
                skeleton_data[start_i:end_i],
                distance_data[start_i:end_i],
            )

        result = process_calculation_chunk(
            anno=anno_chunk,
            hema=hema_chunk,
            mask=mask_chunk,
            skel=skel_chunk,
            dist=dist_chunk,
        )

        for value, nums in result.items():
            if value not in full_brain_signal:
                full_brain_signal[value] = nums[0:9].copy()
            else:
                full_brain_signal[value][:6] += nums[:6]
                full_brain_signal[value][6] += nums[6]                          # radius_sq
                if nums[7] > full_brain_signal[value][7]: full_brain_signal[value][7] = nums[7]  # max
                if nums[8] < full_brain_signal[value][8]: full_brain_signal[value][8] = nums[8]  # min

            if value not in left_brain_signal:
                left_brain_signal[value] = nums[9:18].copy()
            else:
                left_brain_signal[value][:6] += nums[9:15]
                left_brain_signal[value][6] += nums[15]
                if nums[16] > left_brain_signal[value][7]: left_brain_signal[value][7] = nums[16]
                if nums[17] < left_brain_signal[value][8]: left_brain_signal[value][8] = nums[17]

            if value not in right_brain_signal:
                right_brain_signal[value] = nums[18:27].copy()
            else:
                right_brain_signal[value][:6] += nums[18:24]
                right_brain_signal[value][6] += nums[24]
                if nums[25] > right_brain_signal[value][7]: right_brain_signal[value][7] = nums[25]
                if nums[26] < right_brain_signal[value][8]: right_brain_signal[value][8] = nums[26]

    # Step 5: Report Generation
    print("📄 Generating final report...")
    region_signals = {'full_brain': full_brain_signal}
    if hema_data is not None:
        region_signals['left_brain'] = left_brain_signal
        region_signals['right_brain'] = right_brain_signal

    mask_folder = os.path.basename(os.path.dirname(task["mask_path"].rstrip("/")))
    process_analysis_report(region_signals, tuple(task.get("voxel", [0.004, 0.00182, 0.00182])), 'vessel', task["output_path"], task.get("annotation_map_path"), mask_folder)

    print(f"✅ Processing completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full 3D vessel analysis pipeline.")
    parser.add_argument("--config", "-c", type=str, default="configs/config.json", help="Path to config JSON")
    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as f:
        full_config = json.load(f)

    tasks = full_config.get("vessel_analyzer", [])
    if isinstance(tasks, dict):
        tasks = [tasks]

    for task in tasks:
        run_task(task)