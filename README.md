# Can YOLO Detect Retinal Pathologies?

*A Step Towards Automated OCT Analysis*

[![DOI](https://img.shields.io/badge/DOI-10.3390/diagnostics15141823-blue)](https://doi.org/10.3390/diagnostics15141823)

This repository contains the code and configurations used in the study titled:
**"Can YOLO Detect Retinal Pathologies? A Step Towards Automated OCT Analysis"**
by Adriana-Ioana Ardelean, Eugen-Richard Ardelean, and Anca Marginean, published in *Diagnostics*, 2025.

## 📄 Overview

This project benchmarks modern object detection models (YOLOv8–v12, YOLOE, YOLO-World, RT-DETR) for retinal Optical Coherence Tomography (OCT) analysis. It evaluates their ability to detect retinal biomarkers across two publicly available datasets:

* **AROI**: focused on fluid detection in AMD.
* **OCT5k**: includes multiple retinal pathologies.

Our experiments identify **YOLOE** and **YOLOv12** as the most effective models, offering an excellent trade-off between detection accuracy and computational efficiency.

---

## 📊 Datasets

This project requires a .yaml file to be created for the models to run on each dataset, as shown in this example:
```
train: ...\data\OCT5k\yolo\images\train
val: ...\data\OCT5k\yolo\images\val
test: ...\data\OCT5k\yolo\images\test
nc: 9
names: [Choroidalfolds, Fluid, Geographicatrophy, Harddrusen, Hyperfluorescentspots, PRlayerdisruption, Reticulardrusen, Softdrusen, SoftdrusenPED]
```

### 1. AROI Dataset

* **Name**: Annotated Retinal OCT Images (AROI)
* **Description**: 1136 manually annotated OCT B-scans from 24 patients with neovascular AMD. Fluid types include:

  * Pigment Epithelial Detachment (PED)
  * Subretinal Fluid (SRF)
  * Intraretinal Fluid (IRF)
* **Access**: Available upon request from the authors.
  🔗 [AROI Dataset Page](https://ipg.fer.hr/ipg/resources/oct_image_database)

### 2. OCT5k Dataset

* **Name**: OCT5k - Multi-Disease Retinal OCT Dataset
* **Description**: 1672 scans labeled with 9 types of pathologies (e.g., soft/hard drusen, geographic atrophy, PR layer disruption, etc.)
* **Access**: Openly available on UCL Research Data Repository.
  🔗 [OCT5k Dataset](https://rdr.ucl.ac.uk/articles/dataset/OCT5k_A_dataset_of_multi-disease_and_multi-graded_annotations_for_retinal_layers/22128671)



---

## 🧠 Models Evaluated

* YOLOv8 to YOLOv12 (Ultralytics & Community)
* YOLOE: Prompt-guided detection
* YOLO-World: Open-vocabulary object detection
* RT-DETR: Transformer-based object detection



## 📈 Results Summary

| Model   | Dataset | mAP\@50   | Inference Time | GFLOPs |
| ------- | ------- | --------- | -------------- | ------ |
| YOLOv12 | AROI    | 0.712     | 4.9 ms         | 21.2   |
| YOLOE   | AROI    | **0.725** | **4.1 ms**     | 35.3   |
| YOLOv12 | OCT5k   | 0.301     | 10.4 ms        | 21.2   |
| YOLOE   | OCT5k   | **0.355** | **10.3 ms**    | 35.3   |

---

## 📜 Citation

If you use this code or reference the models/datasets in your work, please cite:

```bibtex
@article{Ardelean2025YOLO,
  title     = {Can YOLO Detect Retinal Pathologies? A Step Towards Automated OCT Analysis},
  author    = {Ardelean, Adriana-Ioana and Ardelean, Eugen-Richard and Marginean, Anca},
  journal   = {Diagnostics},
  year      = {2025},
  volume    = {15},
  number    = {14},
  pages     = {1823},
  doi       = {10.3390/diagnostics15141823}
}
```

---

## 📬 Contact

For questions, please contact:
📧 [ardeleaneugenrichard@gmail.com](mailto:ardeleaneugenrichard@gmail.com)

---

