"""
Usage:
python cell_analyzer.py \
    --mask_path             ./datas/neun_mask_ome.zarr/0 \
    --annotation_path       ./datas/annotation_ome.zarr/0 \
    --output_path           ./datas/neun_output \
    --chunk-size            128 128 128
"""

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
    if os.path.exists(full_path):
        print(f"✅ Found: {full_path}")

        # Load Zarr dataset with specified chunk size or auto-chunks
        return da.from_zarr(full_path, chunks=chunk_size) if chunk_size else da.from_zarr(full_path)

    return None


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

def process_analysis_report(region_signals, voxel, output_name, output_path):
    """
    Generate Excel reports from signal dictionaries for multiple brain regions.

    Parameters:
        region_signals (dict): Keys are region names, values are signal dictionaries.
        voxel (tuple): Voxel size as a 3D tuple (x, y, z).
        output_name (str): Base name for the output files.
        output_path (str): Directory to save the reports.
        structure_path (str): Path to the structure CSV file.
        target_id (int, optional): ID for filtering specific brain structures.
    """
    structure_path='./utils/structures.csv'
    target_id=None

    os.makedirs(output_path, exist_ok=True)
    voxel_volume = np.prod(voxel)

    for region_name, signal in region_signals.items():
        output_file = os.path.join(output_path, f'{output_name}_{region_name}_report.xlsx')
        create_cell_report(signal, voxel_volume, output_file, structure_path, target_id)

def main():
    """
    Main function to process a 3D cell mask and generate analysis reports.

    This function performs the following steps:
    1. Parses command-line arguments for input/output Zarr paths and processing parameters.
    2. Loads the input Zarr datasets for mask, annotation, and optionally hemisphere segmentation.
    3. Applies a median box filter followed by a Gaussian blur using GPU acceleration.
    4. Detects local maxima across the filtered image volume.
    5. Computes signal counts and distributions across full, left, and right brain hemispheres.
    6. Generates Excel reports for each region using the computed signals.

    Command-line arguments:
        mask_path (str): Path to the Zarr file containing the input cell mask.
        annotation_path (str): Path to the annotation Zarr file.
        output_path (str): Directory where final Excel reports will be saved.
        --hemasphere_path (str, optional): Path to hemisphere segmentation Zarr file.
        --voxel (float, optional): Voxel size for volume computation.
        --chunk-size (int, optional): Override default chunk size for Dask processing.
        --filter-size (int, optional): Size of the median filter kernel.
        --filter-sigma (float, optional): Sigma value for the Gaussian blur.
    """
    parser = argparse.ArgumentParser(
        description="Apply median filter to a Zarr file using Dask with map_overlap."
    )
    parser.add_argument("--mask_path", type=str, required=True, help="Zarr path to cell mask to be filtered.")
    parser.add_argument("--annotation_path", type=str, required=True, help="Zarr path to annotation")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for temporary zarr and final report")
    parser.add_argument("--hemasphere_path", type=str, default=None, 
                        help="Zarr path to hemisphere segmentation.")
    parser.add_argument("--voxel", type=float, nargs='+', default=(0.004, 0.00182, 0.00182),
                        help="For final volume calculation. (default: 0.004, 0.00182, 0.00182)")
    parser.add_argument("--chunk-size", type=int, nargs='+', default=None,
                        help="Optional: Override chunk size for Dask processing (space-separated)")
    parser.add_argument("--filter-size", type=int, default=3,
                        help="Size of the median filter kernel (default: 3)")
    parser.add_argument("--z-per-slab", type=int, default=128,
                        help="Number of Z-slices to process per slab (default: 128). Adjust based on memory.")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Number of Dask worker processes to start (default: 8).")
    parser.add_argument("--memory-limit", type=str, default='32GB',
                        help="Memory limit per Dask worker (e.g., '16GB', '256GB') (default: '32GB').")

    args = parser.parse_args()
    chunk_size = tuple(args.chunk_size) if args.chunk_size else None
    cluster = LocalCluster(
        n_workers=args.n_workers,
        memory_limit=args.memory_limit
    )

    # client = Client(cluster)
    # print(f"Dashboard link: {client.dashboard_link}")

    start_time = time.time()
    # Load Zarr arrays
    mask_data = check_and_load_zarr(args.mask_path, chunk_size=chunk_size)
    anno_data = check_and_load_zarr(args.annotation_path, chunk_size=chunk_size)
    hema_data = check_and_load_zarr(args.hemasphere_path, chunk_size=chunk_size)

    print(f"Mask shape: {mask_data.shape}") # type: ignore
    print(f"Annotation shape: {anno_data.shape}") # type: ignore

    # # **Step 1: Apply Filtering (Skip if Exists)**
    # filtered_data = check_and_load_zarr(args.output_path, "filtered_mask.zarr", chunk_size=chunk_size)
    # if filtered_data is None:
    #     with ProgressBar():
    #         print("🔄 Applying filtering...")
    #         filtered_data = da.map_blocks(
    #             process_filter_chunk,
    #             mask_data,
    #             dtype=np.uint8,
    #             filter_size=args.filter_size,
    #         )
    #         filtered_data.to_zarr(os.path.join(args.output_path, "filtered_mask.zarr"), overwrite=True)
    #         filtered_data = da.from_zarr(os.path.join(args.output_path, "filtered_mask.zarr"))

    # **Step 2: Compute Local Maxima (Skip if Exists)**
    maxima_data = check_and_load_zarr(args.output_path, "maxima_mask.zarr", chunk_size=chunk_size)
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
            maxima_data.to_zarr(os.path.join(args.output_path, "maxima_mask.zarr"), overwrite=True)
            maxima_data = da.from_zarr(os.path.join(args.output_path, "maxima_mask.zarr"))

    # **Step 3: Process Unique Values and Counts**
    full_brain_signal = {}
    left_brain_signal = {}
    right_brain_signal = {}
    
    checkpoint_path = os.path.join(args.output_path, "cell_counts.json")
    if os.path.exists(checkpoint_path):
        print(f"✅ Found checkpoint file, loading from: {checkpoint_path}")
        # Load from checkpoint and aggregate directly
        with open(checkpoint_path, 'r') as f:
            for line in f:
                result_dict = json.loads(line)
                _aggregate_signals(result_dict, full_brain_signal, left_brain_signal, right_brain_signal)
    else:
        print("🔄 Processing unique values and counts...")
        z_per_slab = args.z_per_slab
        img_dimension = mask_data.shape

        with open(checkpoint_path, 'w') as f_checkpoint:
            for i in tqdm(range(0, img_dimension[0], z_per_slab)):
                start_z, end_z = i, min(i + z_per_slab, img_dimension[0])
                
                anno_slab = anno_data[start_z:end_z].compute()
                maxima_slab = maxima_data[start_z:end_z].compute()
                hema_slab = hema_data[start_z:end_z].compute() if hema_data is not None else np.zeros_like(anno_slab)

                # Process the slab and get the dictionary of counts
                result_dict = process_calculation_chunk(anno_slab, hema_slab, maxima_slab)

                # Write to checkpoint immediately
                f_checkpoint.write(json.dumps(result_dict) + '\n')

                # Aggregate results in memory
                _aggregate_signals(result_dict, full_brain_signal, left_brain_signal, right_brain_signal)


    # **Step 4: Save Results as a CSV Report**
    print("📄 Generating final report...")
    region_signals = {
        'full_brain': full_brain_signal,
        'left_brain': left_brain_signal,
        'right_brain': right_brain_signal
    }

    process_analysis_report(region_signals, tuple(args.voxel), 'cell', args.output_path)

    end_time = time.time()
    print(f"✅ Processing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
