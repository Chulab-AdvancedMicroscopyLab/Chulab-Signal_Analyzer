# Zarr File Analyzer

This project provides tools to analyze biological structures from OME-Zarr volumes. Two main modules are provided:

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

If you prefer to run the tools using Docker, follow these steps:

### 1. Install Docker

Download and install Docker for your system:

- [Docker Desktop for Windows/macOS](https://www.docker.com/products/docker-desktop)
- For Ubuntu/Linux:

```bash
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl enable docker
sudo systemctl start docker
```

Verify Docker is working:

```bash
docker --version
docker-compose --version
```

### 2. Run the App

#### For Linux/macOS:

```bash
chmod +x run.sh
./run.sh
```

#### For Windows (PowerShell):

```powershell
./run.ps1
```

You’ll be prompted to enter the path to your **Zarr dataset directory**. The script will handle everything from Docker build to startup.

#### What the Script Does

- Dynamically generates a `docker-compose.yml`
- Mounts your dataset at `/workspace/datas`
- Cleans up Docker resources after exit

---

## 3. Start Analyzing

Once your environment is set up (via Conda or Docker), you can run the following analysis modules:

### Cell Analyzer

```bash
python cell_analyzer.py \
    <mask_path> \
    <annotation_path> \
    <output_path> \
    --hemasphere_path <hemasphere_zarr> \
    --chunk-size 128 128 128
```

- `mask_path`: Path to the neuron mask (e.g., `neun_mask_ome.zarr`)
- `annotation_path`: Path to annotation zarr file
- `output_path`: Directory to store result files
- `--hemasphere_path`: (Optional) Path to another zarr dataset
- `--chunk-size`: (Optional) Chunk size for Dask arrays

---

### Vessel Analyzer

```bash
python vessel_analyzer.py \
    <mask_path> \
    <annotation_path> \
    <output_path> \
    --hemasphere_path <hemasphere_zarr> \
    --chunk-size 128 128 128
```

- `mask_path`: Path to the neuron mask (e.g., `lectin_mask_ome.zarr`)
- `annotation_path`: Path to annotation zarr file
- `output_path`: Directory to store result files
- `--hemasphere_path`: (Optional) Path to another zarr dataset
- `--chunk-size`: (Optional) Chunk size for Dask arrays

---

## 📜 License

MIT License
