"""
Usage:
python vessel_analyzer.py \
    --mask_path             ./datas/lectin_mask_ome.zarr/0 \
    --annotation_path       ./datas/annotation_ome.zarr/0 \
    --output_path           ./datas/lectin_output_1 \
    --chunk-size            128 128 128
"""

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
    if os.path.exists(full_path):
        print(f"✅ Found: {full_path}")
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
    median_filtered = median_filter(
        input=binary_block,
        size=filter_diameter
    )

    # 3. Apply the final threshold condition to get a binary mask.
    final_mask = (median_filtered > 0.5).astype(np.uint8)
    
    return final_mask

def process_skeletonize_chunk(block):
    """Skeletonize a binary 3D block and mark bifurcation points."""
    block[block > 0] = 1
    skeleton = skeletonize(block.astype(np.uint8)).astype(np.uint8)
    skeleton *= block

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = convolve(skeleton, kernel, mode='constant')
    bifurcation_candidates = (skeleton > 0) & (neighbor_count >= 3)
    labeled_array, _ = label(bifurcation_candidates)  # type: ignore
    labeled_array = labeled_array.astype(np.int32)
    
    for region in regionprops(labeled_array):
        com = tuple(np.round(region.centroid).astype(int))
        skeleton[com] = 2

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
        create_vessel_report(signal, voxel_volume, output_file, structure_path, target_id)

def main():
    """
    Main function for full 3D vessel analysis pipeline using Dask and GPU-accelerated image processing.

    This pipeline processes a vessel mask and generates a statistical report containing
    volume, length, radius, and other vessel-related features for full brain and each hemisphere.

    Steps performed:
    1. Parse command-line arguments.
    2. Load Zarr datasets for vessel mask, annotation, and optional hemisphere segmentation.
    3. Apply a Gaussian filter to smooth the input vessel mask.
    4. Skeletonize the filtered mask to extract vessel centerlines.
    5. Compute a distance transform to estimate vessel radius.
    6. Extract vessel features in chunks, aggregating statistics for each brain region.
    7. Generate Excel reports summarizing the extracted features.

    Command-line arguments:
        mask_path (str): Zarr path to the vessel mask.
        annotation_path (str): Zarr path to annotation labels.
        output_path (str): Output directory for the final Excel report.
        --hemasphere_path (str, optional): Zarr path to hemisphere segmentation.
        --voxel (float, optional): Voxel size for volume calculation. Default: (0.004, 0.00182, 0.00182).
        --chunk-size (int, optional): Custom Dask chunk size (space-separated).
        --filter-sigma (float, optional): Sigma for Gaussian filtering. Default: 0.3.
    """
    parser = argparse.ArgumentParser(description="Full 3D vessel analysis pipeline.")
    parser.add_argument("--mask_path", type=str, required=True, help="Zarr path to the vessel mask.")
    parser.add_argument("--annotation_path", type=str, required=True, help="Zarr path to annotation labels.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for report and temporary zarr.")
    parser.add_argument("--hemasphere_path", type=str, default=None, 
                        help="Zarr path to hemisphere segmentation.")
    parser.add_argument("--voxel", type=float, nargs='+', default=(0.004, 0.00182, 0.00182), 
                        help="For final volume calculation. (default: 0.004, 0.00182, 0.00182)")
    parser.add_argument("--chunk-size", type=int, nargs='+', default=None, 
                        help="Optional: Override chunk size for Dask processing (space-separated)")
    parser.add_argument("--filter-sigma", type=float, default=2,
                        help="Sigma of the gaussian filter (default: 0.3)")
    parser.add_argument("--z-per-slab", type=int, default=16,
                        help="Number of Z-slices to process per slab (default: 16). Adjust based on memory.")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Number of Dask worker processes to start (default: 8).")
    parser.add_argument("--memory-limit", type=str, default='32GB',
                        help="Memory limit per Dask worker (e.g., '16GB', '256GB') (default: '32GB').")

    args = parser.parse_args()
    chunk_size = tuple(args.chunk_size) if args.chunk_size else None

    # Note: Dask cluster setup is parsed but not yet fully integrated.

    start_time = time.time()
    # Load datasets
    mask_data = check_and_load_zarr(args.mask_path, chunk_size=chunk_size)
    anno_data = check_and_load_zarr(args.annotation_path, chunk_size=chunk_size)
    hema_data = check_and_load_zarr(args.hemasphere_path, chunk_size=chunk_size)

    print(f"Mask shape: {mask_data.shape}") # type: ignore
    print(f"Annotation shape: {anno_data.shape}") # type: ignore

    # Step 1: Filtering
    filtered_data = check_and_load_zarr(args.output_path, "filtered_mask.zarr", chunk_size=chunk_size)
    if filtered_data is None:
        print("🔄 Applying Gaussian filter...")
        with ProgressBar():
            filtered_data = da.map_blocks(
                process_filter_chunk,
                mask_data, dtype=np.uint8, filter_sigma=args.filter_sigma
            )
            filtered_data.to_zarr(os.path.join(args.output_path, "filtered_mask.zarr"), overwrite=True)
            filtered_data = da.from_zarr(os.path.join(args.output_path, "filtered_mask.zarr"))

    # Step 2: Skeletonization
    skeleton_data = check_and_load_zarr(args.output_path, "skeletonize_mask.zarr", chunk_size=chunk_size)
    if skeleton_data is None:
        print("🔄 Skeletonizing vessel mask...")
        with ProgressBar():
            skeleton_data = da.map_blocks(
                process_skeletonize_chunk,
                mask_data, dtype=np.uint8
            )
            skeleton_data.to_zarr(os.path.join(args.output_path, "skeletonize_mask.zarr"), overwrite=True)
            skeleton_data = da.from_zarr(os.path.join(args.output_path, "skeletonize_mask.zarr"))

    # Step 3: Distance Transform
    distance_data = check_and_load_zarr(args.output_path, "distance_mask.zarr", chunk_size=chunk_size)
    if distance_data is None:
        print("🔄 Calculating distance transform...")
        with ProgressBar():
            distance_data = da.map_blocks(
                process_distance_transform,
                mask_data, dtype=np.float32
            )
            distance_data.to_zarr(os.path.join(args.output_path, "distance_mask.zarr"), overwrite=True)
            distance_data = da.from_zarr(os.path.join(args.output_path, "distance_mask.zarr"))

    # Step 4: Feature Extraction
    print("🔄 Extracting features...")
    full_brain_signal = {}
    left_brain_signal = {}
    right_brain_signal = {}
    img_dimension = mask_data.shape # type: ignore

    for i in tqdm(range(0, img_dimension[0], args.z_per_slab)):
        start_i, end_i = i, min(i + args.z_per_slab, img_dimension[0])
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
                full_brain_signal[value] = nums[:6]
            else:
                full_brain_signal[value][:5] += nums[:5]
                if nums[5] > full_brain_signal[value][5]:
                    full_brain_signal[value][5] = nums[5]

            if value not in left_brain_signal:
                left_brain_signal[value] = nums[6:12]
            else:
                left_brain_signal[value][:5] += nums[6:11]
                if nums[11] > left_brain_signal[value][5]:
                    left_brain_signal[value][5] = nums[11]

            if value not in right_brain_signal:
                right_brain_signal[value] = nums[12:]
            else:
                right_brain_signal[value][:5] += nums[12:17]
                if nums[17] > right_brain_signal[value][5]:
                    right_brain_signal[value][5] = nums[17]

    # Step 5: Report Generation
    print("📄 Generating final report...")
    region_signals = {
        'full_brain': full_brain_signal,
        'left_brain': left_brain_signal,
        'right_brain': right_brain_signal
    }

    process_analysis_report(region_signals, tuple(args.voxel), 'vessel', args.output_path)

    print(f"✅ Processing completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()