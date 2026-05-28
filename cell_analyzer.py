"""
Usage:
python Chulab-Signal_Analyzer/cell_analyzer.py --config configs/config.json
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
from scipy import ndimage as ndi
from tqdm import tqdm
import json
import dask.array as da
from dask.diagnostics.progress import ProgressBar
from dask.distributed import Client, LocalCluster

from utils.analyzer_count_tools import numba_unique_cell
from utils.analyzer_report_tools import create_cell_report
from utils.concurrency import initialize_concurrency


def check_and_load_zarr(path, component=None, chunk_size=None):
    """ 
    Check if a Zarr component exists inside the path; if yes, load it.
    If the component does not exist, try loading the full path as a Zarr store.

    Parameters:
        path (str): Directory where the Zarr store is located.
        component (str, optional): Zarr component name. Defaults to None.
        chunk_size (tuple, optional): Chunk size for loading data. Defaults to None (auto-chunking).

    Returns:
        dask.array.Array or None: Loaded Dask array if found, otherwise None.
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
    median_filtered = ndi.median_filter(
        input=binary_block,
        size=filter_diameter
    )

    # 3. Apply the final threshold condition to get a binary mask.
    final_mask = (median_filtered > 0.5).astype(np.uint8)
    
    return final_mask


def process_local_maxima_chunk(block):
    """
    Detects local maxima in a 3D image block using SciPy and NumPy.

    The function performs the following steps:
    1. Creates a binary copy of the input block to work with.
    2. Applies a Gaussian blur using scipy.ndimage to reduce noise.
    3. Detects local maxima by comparing the blurred block to its maximum-filtered version.
    4. Retains only the maxima that correspond to positive values in the original block.

    Parameters:
        block (np.ndarray): The 3D image block to process.

    Returns:
        np.ndarray: A binary 3D mask (dtype=uint8) where detected local maxima are set to 1.
    """
    # Create a binary version of the block to match the original function's logic.
    # This also prevents modifying the input array in-place.
    binary_block = (block > 0).astype(np.uint8)

    # Apply a Gaussian blur. SciPy needs a float input for this.
    blurred_block = ndi.gaussian_filter(binary_block.astype(np.float32), sigma=(1, 1, 1))

    # Detect local maxima using a maximum filter.
    # A pixel is a local maximum if it equals the maximum value in its neighborhood.
    # Note: size = 2 * radius + 1. The original code used radius (x=5, y=5, z=3).
    # We assume a (Z, Y, X) axis order for the size.
    footprint_size = (3, 3, 3) # For radius_z=3, radius_y=5, radius_x=5
    maxima_mask = (blurred_block == ndi.maximum_filter(blurred_block, size=footprint_size))

    # Ensure the final maxima are located within the original foreground region.
    final_maxima = maxima_mask & (binary_block > 0)
    
    # 1. Create the 'labels' array.
    labels, num_labels = ndi.label(final_maxima)

    # 2. If no objects are found, return a blank block of the correct type.
    if num_labels == 0:
        # We can reuse 'labels' here too.
        labels[:] = 0
        return labels.astype(np.uint8)

    # 3. Calculate the centroids. After this line, 'final_maxima' is no longer needed.
    mass_centers = ndi.center_of_mass(final_maxima, labels, range(1, num_labels + 1))

    # Explicitly delete the reference to free up memory sooner (optional but good practice).
    del final_maxima

    # --- MEMORY OPTIMIZATION ---
    # Instead of creating a new np.zeros_like() array, we re-purpose the 'labels' array.

    # 4. First, set all values in the existing 'labels' array to 0. This is an in-place operation.
    labels[:] = 0

    # 5. Now, use this cleared array to mark the center points.
    for z, y, x in mass_centers:
        labels[int(round(z)), int(round(y)), int(round(x))] = 1

    # 6. Return the re-purposed array, converting it to the final desired data type.
    return labels.astype(np.uint8)

def process_calculation_chunk(anno, hema, mask):
    """Extract unique nonzero values and their counts per chunk."""
    # Convert Numba typed dict to a standard Python dict for serialization
    result = numba_unique_cell(anno, hema, mask)
    return {k: v.tolist() for k, v in result.items()}

def _aggregate_signals(result_dict, full_brain, left_brain, right_brain):
    """Helper to aggregate counts from a slab into the main dictionaries."""
    for value_str, nums in result_dict.items():
        value = int(value_str)
        # Full brain
        if value not in full_brain: full_brain[value] = nums[:2]
        else: full_brain[value] = [x + y for x, y in zip(full_brain[value], nums[:2])]
        # Left brain
        if value not in left_brain: left_brain[value] = nums[2:4]
        else: left_brain[value] = [x + y for x, y in zip(left_brain[value], nums[2:4])]
        # Right brain
        if value not in right_brain: right_brain[value] = nums[4:6]
        else: right_brain[value] = [x + y for x, y in zip(right_brain[value], nums[4:6])]

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
        create_cell_report(signal, voxel_volume, output_file_xlsx, structure_path, target_id)

def run_task(task):
    chunk_size = tuple(task["chunk_size"]) if task.get("chunk_size") else None

    start_time = time.time()
    mask_data = check_and_load_zarr(task["mask_path"], chunk_size=chunk_size)
    anno_data = check_and_load_zarr(task["annotation_path"], chunk_size=chunk_size)
    hema_data = check_and_load_zarr(task.get("hemasphere_path"), chunk_size=chunk_size)

    print(f"Mask shape: {mask_data.shape}") # type: ignore
    print(f"Annotation shape: {anno_data.shape}") # type: ignore

    # # **Step 1: Apply Filtering (Skip if Exists)**
    # filtered_data = check_and_load_zarr(task["output_path"], "filtered_mask.zarr", chunk_size=chunk_size)
    # if filtered_data is None:
    #     with ProgressBar():
    #         print("🔄 Applying filtering...")
    #         filtered_data = da.map_blocks(
    #             process_filter_chunk,
    #             mask_data,
    #             dtype=np.uint8,
    #             filter_size=task.get("filter_size", 3),
    #         )
    #         filtered_data.to_zarr(os.path.join(task["output_path"], "filtered_mask.zarr"), overwrite=True)
    #         filtered_data = da.from_zarr(os.path.join(task["output_path"], "filtered_mask.zarr"))

    # **Step 2: Compute Local Maxima (Skip if Exists)**
    maxima_data = check_and_load_zarr(task["output_path"], "maxima_mask.zarr", chunk_size=chunk_size)
    if maxima_data is None:
        with ProgressBar():
            print("🔄 Finding local maxima...")
            maxima_data = da.map_overlap(
                process_local_maxima_chunk,
                mask_data,
                trim=True,
                depth=8,
                dtype=np.uint8,
            )
            maxima_data.to_zarr(os.path.join(task["output_path"], "maxima_mask.zarr"), overwrite=True)
            maxima_data = da.from_zarr(os.path.join(task["output_path"], "maxima_mask.zarr"))

    # **Step 3: Process Unique Values and Counts**
    full_brain_signal = {}
    left_brain_signal = {}
    right_brain_signal = {}
    
    checkpoint_path = os.path.join(task["output_path"], "cell_counts.json")
    coords_path = os.path.join(task["output_path"], "cell_coordinates.json")
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Found checkpoint file, loading from: {checkpoint_path}")
        # Load from checkpoint and aggregate directly
        with open(checkpoint_path, 'r') as f:
            for line in f:
                result_dict = json.loads(line)
                _aggregate_signals(result_dict, full_brain_signal, left_brain_signal, right_brain_signal)
    else:
        print("🔄 Processing unique values and counts...")
        z_per_slab = task.get("z_per_slab", 128)
        img_dimension = mask_data.shape

        with open(checkpoint_path, 'w') as f_checkpoint, open(coords_path, 'w') as f_coords:
            for i in tqdm(range(0, img_dimension[0], z_per_slab)):
                start_z, end_z = i, min(i + z_per_slab, img_dimension[0])
                
                anno_slab = anno_data[start_z:end_z].compute()
                maxima_slab = maxima_data[start_z:end_z].compute()
                hema_slab = hema_data[start_z:end_z].compute() if hema_data is not None else np.zeros_like(anno_slab)

                # --- NEW: Extract and Save Individual Cell Coordinates ---
                # Find all detected centroids (value 1) in the current slab
                cell_indices = np.argwhere(maxima_slab == 1)
                for local_z, y, x in cell_indices:
                    global_z = int(local_z + start_z)
                    anno_id = int(anno_slab[local_z, y, x])
                    hema_id = int(hema_slab[local_z, y, x]) if hema_data is not None else 0
                    
                    coord_entry = {
                        "z": global_z,
                        "y": int(y),
                        "x": int(x),
                        "anno_id": anno_id,
                        "hema_id": hema_id
                    }
                    f_coords.write(json.dumps(coord_entry) + '\n')

                # Process the slab and get the dictionary of counts
                result_dict = process_calculation_chunk(anno_slab, hema_slab, maxima_slab)

                # Write to checkpoint immediately
                f_checkpoint.write(json.dumps(result_dict) + '\n')

                # Aggregate results in memory
                _aggregate_signals(result_dict, full_brain_signal, left_brain_signal, right_brain_signal)


    # **Step 4: Save Results as a CSV Report**
    print("📄 Generating final report...")
    region_signals = {'full_brain': full_brain_signal}
    if hema_data is not None:
        region_signals['left_brain'] = left_brain_signal
        region_signals['right_brain'] = right_brain_signal

    mask_folder = os.path.basename(os.path.dirname(task["mask_path"].rstrip("/")))
    process_analysis_report(region_signals, tuple(task.get("voxel", [0.004, 0.00182, 0.00182])), 'cell', task["output_path"], task.get("annotation_map_path"), mask_folder)

    end_time = time.time()
    print(f"✅ Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full 3D cell analysis pipeline.")
    parser.add_argument("--config", "-c", type=str, default="configs/config.json", help="Path to config JSON")
    cli_args = parser.parse_args()

    with open(cli_args.config, 'r') as f:
        full_config = json.load(f)

    tasks = full_config.get("cell_analyzer", [])
    if isinstance(tasks, dict):
        tasks = [tasks]

    for task in tasks:
        run_task(task)
