# Role Simulation: Data Engineer — Computer Vision Data & DVC Pipeline Specification

**Project**: Project 02 · Deep Learning + Computer Vision (Disease Detection System)  
**Owner Role**: Data Engineer  
**DVC & MLflow Server**: [`https://dagshub.com/29omchavan/mlops-project-2-cnn-disease-classifier`](https://dagshub.com/29omchavan/mlops-project-2-cnn-disease-classifier)

---

## 1. Dataset Taxonomy & Biomarker Characteristics
Following Krish Naik's *End-to-End Deep Learning Project Using MLOps DVC Pipeline*, the dataset models 4 poultry pathology microscopy classes organized in PyTorch `ImageFolder` hierarchy (`artifacts/data_ingestion/dataset/{train,val}/<Class>/*.png`):

| Class Index | Class Label | Morphological & Chromatic Biomarkers | Train Images | Val Images |
| :---: | :--- | :--- | :---: | :---: |
| `0` | `Coccidiosis` | Reddish-brown hemorrhagic background with circular protozoan oocyst ring structures | 60 | 20 |
| `1` | `Healthy` | Uniform golden-ochre background with smooth organic grain texture | 60 | 20 |
| `2` | `Newcastle_Disease` | High-contrast greenish-yellow mucoid lesions with irregular linear streaks | 60 | 20 |
| `3` | `Salmonella` | Chalky pale-white background with dense punctate bacterial colonies | 60 | 20 |

---

## 2. Data Augmentation Pipeline (`components/model_training.py`)
To prevent overfitting on small medical/veterinary microscopy datasets and achieve translational/rotational invariance, the training `DataLoader` applies stochastic on-the-fly transformations:
1. `Resize((64, 64))`
2. `RandomHorizontalFlip(p=0.5)` and `RandomVerticalFlip(p=0.3)` (microscopy slides have no canonical up/down orientation)
3. `RandomRotation(degrees=15)`
4. `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)` (simulates varying microscope illumination)
5. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (aligns input channel distributions with ImageNet pretrained ResNet-18 convolutional filters)

> **Crucial Engineering Rule**: Validation and inference transforms apply **only** deterministic `Resize` and `Normalize` — never stochastic flips or jitter.

---

## 3. Data Version Control (DVC) DAG Architecture (`dvc.yaml`)

Why use DVC in Deep Learning? Git cannot store gigabytes of image datasets or `.pt` model weights efficiently. DVC constructs a **Directed Acyclic Graph (DAG)** of pipeline stages where each stage declares:
- `cmd`: The Python command to execute
- `deps`: Code and input data dependencies (hashed via MD5/SHA256)
- `params`: Hyperparameters from `params.yaml`
- `outs`: Output directories/models cached by DVC
- `metrics`: `scores.json` tracked across Git commits (`dvc metrics show` / `dvc metrics diff`)

When `dvc repro` runs, DVC inspects `dvc.lock`: if a stage's `deps` and `params` hashes have not changed, DVC skips re-running that stage!
