import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types

@njit
def numba_unique_cell(anno, hema, mask):
    # Initialize a typed dictionary for storing unique values and counts
    unique_counts = Dict.empty(key_type=types.int64, value_type=types.int64[:])
    
    # Parallel iteration over 3D array
    for i in range(anno.shape[0]):
        for j in range(anno.shape[1]):
            for k in range(anno.shape[2]):
                val = anno[i, j, k]
                if val < 1:
                    continue
                
                if val not in unique_counts:
                    unique_counts[val] = np.array([1, 0, 0, 0, 0, 0], dtype=np.int64)
                else:
                    unique_counts[val][0] += 1
                
                if mask[i, j, k] > 0:
                    unique_counts[val][1] += 1
                    
                h = int(hema[i, j, k])

                if h == 1:
                    unique_counts[val][2] += 1
                    
                    if mask[i, j, k] > 0:
                        unique_counts[val][3] += 1
                        
                elif h == 2:
                    unique_counts[val][4] += 1
                    
                    if mask[i, j, k] > 0:
                        unique_counts[val][5] += 1
                        
    return unique_counts

@njit
def numba_unique_vessel(anno, hema, mask, skel, dist):
    """
    Extract voxel-wise statistics for labeled regions in a 3D volume.

    Parameters:
        anno (ndarray): Label image with region IDs.
        hema (ndarray): Hemisphere mask (0: all, 1: left, 2: right).
        mask (ndarray): Binary mask indicating detected signal.
        skel (ndarray): Skeletonized mask.
        dist (ndarray): Distance transform of mask.

    Returns:
        Dict[int, ndarray]: Mapping of region ID to an 18-element vector:
            [0]  total_voxels
            [1]  signal_voxels
            [2]  skeleton_voxels
            [3]  bifurcation_voxels
            [4]  total_distance
            [5]  max_distance
            [6]  left_voxels
            [7]  left_signal_voxels
            [8]  left_skeleton_voxels
            [9]  left_bifurcations
            [10] left_distance
            [11] left_max_distance
            [12] right_voxels
            [13] right_signal_voxels
            [14] right_skeleton_voxels
            [15] right_bifurcations
            [16] right_distance
            [17] right_max_distance
    """
    unique_counts = Dict.empty(key_type=types.int64, value_type=types.int64[:])

    for i in range(anno.shape[0]):
        for j in range(anno.shape[1]):
            for k in range(anno.shape[2]):
                val = anno[i, j, k]
                if val < 1:
                    continue

                if val not in unique_counts:
                    unique_counts[val] = np.zeros(18, dtype=np.int64)
                    unique_counts[val][0] = 1
                else:
                    unique_counts[val][0] += 1

                if mask[i, j, k] > 0:
                    unique_counts[val][1] += 1

                if skel[i, j, k] > 0:
                    unique_counts[val][2] += 1

                if skel[i, j, k] > 1:
                    unique_counts[val][3] += 1

                if dist[i, j, k] > 0:
                    unique_counts[val][4] += dist[i, j, k]

                if dist[i, j, k] > unique_counts[val][5]:
                    unique_counts[val][5] = dist[i, j, k]

                h = int(hema[i, j, k])

                if h == 1:
                    unique_counts[val][6] += 1
                    if mask[i, j, k] > 0:
                        unique_counts[val][7] += 1
                    if skel[i, j, k] > 0:
                        unique_counts[val][8] += 1
                    if skel[i, j, k] > 1:
                        unique_counts[val][9] += 1
                    if dist[i, j, k] > 0:
                        unique_counts[val][10] += dist[i, j, k]
                    if dist[i, j, k] > unique_counts[val][11]:
                        unique_counts[val][11] = dist[i, j, k]

                elif h == 2:
                    unique_counts[val][12] += 1
                    if mask[i, j, k] > 0:
                        unique_counts[val][13] += 1
                    if skel[i, j, k] > 0:
                        unique_counts[val][14] += 1
                    if skel[i, j, k] > 1:
                        unique_counts[val][15] += 1
                    if dist[i, j, k] > 0:
                        unique_counts[val][16] += dist[i, j, k]
                    if dist[i, j, k] > unique_counts[val][17]:
                        unique_counts[val][17] = dist[i, j, k]

    return unique_counts