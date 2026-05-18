# 3D Shape Completion Using an Encoder–Predictor Network (EPN)

A lightweight 3D reconstruction pipeline for completing incomplete mobile-style 3D scans using TSDF voxel representations and a deep learning-based Encoder–Predictor Network.

---

## Overview

This project explores 3D shape completion using volumetric deep learning. The system takes an incomplete 3D shape represented as a partial Truncated Signed Distance Field (TSDF) and predicts a completed 3D reconstruction.

The long-term goal of this project is to improve incomplete or noisy mobile phone-based 3D scans by reconstructing missing geometry automatically. Since consumer-grade scans often suffer from occlusion, missing surfaces, and low-quality geometry, this project investigates whether a lightweight neural reconstruction pipeline can learn to infer missing 3D structure from partial observations.

The current implementation focuses on synthetic primitive-based data in order to build and validate the reconstruction pipeline in a controlled environment before scaling toward more complex geometry and real-world scan data.

---

## Current Features

* TSDF-based volumetric shape representation
* Partial scan simulation through half-space occlusion
* 3D Encoder–Predictor Network (EPN-style architecture)
* Marching Cubes mesh extraction
* Synthetic primitive dataset generation
* 32³ and 64³ voxel reconstruction experiments
* BatchNorm3D and increased model-capacity experiments
* Visual reconstruction comparison pipeline

---

## Pipeline

The current reconstruction pipeline:

```text
3D Shape
   ↓
TSDF Generation
   ↓
Partial TSDF Occlusion
   ↓
Encoder–Predictor Network
   ↓
Predicted TSDF
   ↓
Marching Cubes
   ↓
Reconstructed Mesh
```

---

## Methodology

### Data Generation

Synthetic 3D primitives are generated using Python scripts, including:

* Cubes
* Spheres
* Combined cube structures (L-shaped geometry)

Each generated shape is converted into:

* Full TSDF volume
* Partial TSDF volume
* Occupancy grid
* Alignment bounds

The current implementation simulates incompleteness using half-slice occlusion to mimic missing geometry commonly found in partial scans.

### Model Architecture

The reconstruction model is based on a lightweight Encoder–Predictor Network (EPN).

The architecture consists of:

* 3D convolutional encoder
* Latent bottleneck representation
* 3D convolutional decoder

The encoder extracts spatial geometric features from the partial TSDF input, while the decoder reconstructs the completed volumetric field.

### Mesh Reconstruction

After prediction, the Marching Cubes algorithm extracts the zero-level surface from the predicted TSDF volume to generate a reconstructed mesh.

---

## Results

The project currently includes experiments comparing:

* 32³ vs 64³ voxel resolution
* Small vs medium-capacity models
* BatchNorm3D integration
* Reconstruction quality across primitive types

### Best Current Result

* Grid Resolution: 64³
* Architecture: Medium-capacity + BatchNorm3D
* Validation Loss: 0.007

Key findings:

* Higher voxel resolution significantly improves reconstruction quality
* BatchNorm3D improves stability and convergence
* Increased channel capacity improves reconstruction of more complex geometry
* Correct TSDF preprocessing was critical for stable training

---

## Repository Structure

```text
3d-shape-completion/
│
├── models/          # Network architectures
├── scripts/         # Training and preprocessing scripts
├── outputs/         # Reconstruction results and visualizations
├── docs/            # Reports, slides, and project documentation
├── data/            # Sample data / dataset structure
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Future Goals

Planned future development includes:

* More complex synthetic geometry
* Realistic scan corruption simulation
* Real mobile-phone scan integration
* Improved reconstruction architectures
* Sparse voxel and implicit representations
* Better generalization across object categories

The long-term objective is to create a practical reconstruction pipeline capable of improving incomplete consumer-grade 3D scans while remaining lightweight and computationally accessible.

---

## Technologies Used

* PyTorch
* NumPy
* Trimesh
* Marching Cubes
* Python

---

## Documentation

Additional project materials:

* Final report
* Presentation slides
* Reconstruction comparisons
* Experimental results

These can be found in the `/docs` and `/outputs` directories.

---

## References

* Dai et al., *Shape Completion Using 3D Encoder–Predictor CNNs and Shape Synthesis* (CVPR 2017)
* Dai et al., *ScanComplete* (CVPR 2018)
* DeepSDF
* Occupancy Networks
* SC-Diff

---

## Author

Tarek Eltantawy
University of Miami
Computer Science
