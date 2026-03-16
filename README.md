# Zarr File Analyzer

This project provides tools to analyze biological structures from OME-Zarr volumes. Two main modules are provided: Cell Analyzer and Vessel Analyzer. Both tools automatically generate a comprehensive **Excel (.xlsx)** report and a flat **CSV (.csv)** summary file upon completion.

---

## Quick Start (Recommended)

### 1. Create and Activate the Environment

```bash
conda create -n signal-analyzer python=3.10 -y
conda activate signal-analyzer
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

## Alternative: Docker Setup

If you prefer to run the tools using Docker, follow these standard steps:

### 1. Build the Docker Image

Navigate to the project directory and build the Docker image:

```bash
docker build -t signal-analyzer .
```

### 2. Run the Docker Container

You can run the container by mounting your dataset directory to `/workspace/datas` inside the container:

```bash
docker run -it --rm -v /path/to/your/dataset:/workspace/datas signal-analyzer bash
```

From inside the container, you can run the analyzer scripts as shown below.

---

## 3. Start Analyzing

Once your environment is set up (via Conda or Docker), you can run the following analysis modules.

### Cell Analyzer

Extracts unique non-zero values, detects local maxima, and tallies cell counts per brain region and hemisphere.

```bash
python cell_analyzer.py \
    --mask_path <mask_path> \
    --annotation_path <annotation_path> \
    --output_path <output_path> \
    --hemasphere_path <hemasphere_zarr> \
    --chunk-size 128 128 128 \
    --n-workers 8 \
    --numba-threads 8
```

- `--mask_path`: Path to the cell mask (e.g., `neun_mask_ome.zarr`)
- `--annotation_path`: Path to annotation zarr file
- `--output_path`: Directory to store result files
- `--hemasphere_path`: (Optional) Path to hemisphere segmentation zarr dataset (for left/right hemisphere splitting)
- `--chunk-size`: (Optional) Chunk size for Dask arrays
- `--n-workers`: (Optional) Number of Dask worker processes (default: 8)
- `--numba-threads`: (Optional) Number of Numba threads (default: 8)

---

### Vessel Analyzer

Calculates advanced morphological metrics for vessels including volume, length, average radius, and counts of both **bifurcations (3 branches)** and **trifurcations (4+ branches)**.

```bash
python vessel_analyzer.py \
    --mask_path <mask_path> \
    --annotation_path <annotation_path> \
    --output_path <output_path> \
    --hemasphere_path <hemasphere_zarr> \
    --chunk-size 128 128 128 \
    --n-workers 8 \
    --numba-threads 8
```

- `--mask_path`: Path to the vessel mask (e.g., `lectin_mask_ome.zarr`)
- `--annotation_path`: Path to annotation zarr file
- `--output_path`: Directory to store result files
- `--hemasphere_path`: (Optional) Path to hemisphere segmentation zarr dataset
- `--chunk-size`: (Optional) Chunk size for Dask arrays
- `--n-workers`: (Optional) Number of Dask worker processes (default: 8)
- `--numba-threads`: (Optional) Number of Numba threads (default: 8)

---

## 📜 License

MIT License
