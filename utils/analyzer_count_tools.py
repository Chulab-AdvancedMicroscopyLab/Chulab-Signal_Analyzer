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
        Dict[int, ndarray]: Mapping of region ID to a 27-element float64 vector.
        Each group of 9 = [voxels, signal, skeleton, bifurc, trifurc, radius, radius_sq, max_radius, min_radius]
            [0-8]   full brain
            [9-17]  left hemisphere
            [18-26] right hemisphere
    """
    unique_counts = Dict.empty(key_type=types.int64, value_type=types.float64[:])

    for i in range(anno.shape[0]):
        for j in range(anno.shape[1]):
            for k in range(anno.shape[2]):
                val = anno[i, j, k]
                if val < 1:
                    continue

                if val not in unique_counts:
                    unique_counts[val] = np.zeros(27, dtype=np.float64)
                    unique_counts[val][0] = 1
                    unique_counts[val][8]  = np.inf   # full min_radius sentinel
                    unique_counts[val][17] = np.inf   # left min_radius sentinel
                    unique_counts[val][26] = np.inf   # right min_radius sentinel
                else:
                    unique_counts[val][0] += 1

                if mask[i, j, k] > 0:
                    unique_counts[val][1] += 1

                if skel[i, j, k] > 0:
                    unique_counts[val][2] += 1

                if skel[i, j, k] == 2:
                    unique_counts[val][3] += 1

                if skel[i, j, k] == 3:
                    unique_counts[val][4] += 1

                if skel[i, j, k] > 0:
                    d = dist[i, j, k]
                    unique_counts[val][5] += d
                    unique_counts[val][6] += d * d
                    if d < unique_counts[val][8]:
                        unique_counts[val][8] = d

                if dist[i, j, k] > unique_counts[val][7]:
                    unique_counts[val][7] = dist[i, j, k]

                h = int(hema[i, j, k])

                if h == 1:
                    unique_counts[val][9] += 1
                    if mask[i, j, k] > 0:
                        unique_counts[val][10] += 1
                    if skel[i, j, k] > 0:
                        d = dist[i, j, k]
                        unique_counts[val][11] += 1
                        unique_counts[val][14] += d
                        unique_counts[val][15] += d * d
                        if d < unique_counts[val][17]:
                            unique_counts[val][17] = d
                    if skel[i, j, k] == 2:
                        unique_counts[val][12] += 1
                    if skel[i, j, k] == 3:
                        unique_counts[val][13] += 1
                    if dist[i, j, k] > unique_counts[val][16]:
                        unique_counts[val][16] = dist[i, j, k]

                elif h == 2:
                    unique_counts[val][18] += 1
                    if mask[i, j, k] > 0:
                        unique_counts[val][19] += 1
                    if skel[i, j, k] > 0:
                        d = dist[i, j, k]
                        unique_counts[val][20] += 1
                        unique_counts[val][23] += d
                        unique_counts[val][24] += d * d
                        if d < unique_counts[val][26]:
                            unique_counts[val][26] = d
                    if skel[i, j, k] == 2:
                        unique_counts[val][21] += 1
                    if skel[i, j, k] == 3:
                        unique_counts[val][22] += 1
                    if dist[i, j, k] > unique_counts[val][25]:
                        unique_counts[val][25] = dist[i, j, k]

    return unique_counts