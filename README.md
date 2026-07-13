# BraTS 2026

This repository contains the code for participating in **Task 1** and **Task 4** of the [BraTS 2026 Challenge](https://www.synapse.org/Synapse:syn74274097/challenge/).


## Repository structure

```
BraTS-2026/
├── Task1/          # Brain Metastasis Segmentation (nnDetection + MONAI Deploy)
└── Task4/          # Local Synthesis / Inpainting (to be added)
```

---

## Task 1 — Brain Metastasis Segmentation

Run inference with Docker Compose. Mount a folder of multi-parametric MRI NIfTI volumes at `/input` inside the container; the predicted segmentation mask is written to `/output`.

### Prerequisites

- Linux workstation with an NVIDIA GPU
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Input data

Place one case per folder. Each case must include **four MRI modalities** as NIfTI (`.nii.gz`) files:

| Modality | Filename suffix | Description |
|----------|-----------------|-------------|
| T1c | `-t1c.nii.gz` | T1 contrast-enhanced |
| T1n | `-t1n.nii.gz` | T1 native |
| T2-FLAIR | `-t2f.nii.gz` | T2 FLAIR |
| T2w | `-t2w.nii.gz` | T2 weighted |

**Example** (case ID `BraTS-MET-00833-000`):

```
input/
└── BraTS-MET-00833-000/
    ├── BraTS-MET-00833-000-t1c.nii.gz
    ├── BraTS-MET-00833-000-t1n.nii.gz
    ├── BraTS-MET-00833-000-t2f.nii.gz
    └── BraTS-MET-00833-000-t2w.nii.gz
```

### Build the Docker image

From the Task 1 deploy directory:

```bash
cd Task1/deploy/monet-x64-workstation-dgpu-linux-amd64-nifti
docker build -t brats-met-nndet:1.1 .
```

### Run prediction

1. Edit `Task1/deploy/compose.yaml` and set the volume mounts to your local paths:

   ```yaml
   volumes:
     - /path/to/your/input:/input
     - /path/to/your/output:/output
   ```

2. Start the container:

   ```bash
   cd Task1/deploy
   docker compose up
   ```

3. Read the predicted segmentation mask from the mounted **`/output`** directory on the host (e.g. `BraTS-MET-00833-000.nii.gz`).

The container runs the nnDetection RetinaUNet model (`RetinaUNetV001_D3V001_3d`) followed by the MONAI Deploy post-processing pipeline.

### Hardware requirements

The default `compose.yaml` requests:

- 1× NVIDIA GPU (`runtime: nvidia`)
- 16 GB RAM
- 2 CPUs
- 2 GB shared memory (`shm_size`)

---


---

## License

See [LICENSE](LICENSE).
