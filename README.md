# YOLO Dataset Analyzer 🚀

[![PyPI version](https://badge.fury.io/py/yolo-dataset-analyzer.svg)](https://badge.fury.io/py/yolo-dataset-analyzer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**Professional GUI toolkit for analyzing, visualizing, and cleaning YOLO datasets.** Load your `dataset.yaml`, get instant diagnostics, plots, imbalance detection, and dataset optimization tools.

## ✨ Features

| **Analysis Tabs** | **What it shows** |
|-------------------|-------------------|
| 📊 **Summary** | Total images/objects/classes, size breakdown |
| 📈 **Class Distribution** | Per-class bar charts (train/test), imbalance ratio |
| 🔢 **Objects/Image** | Histograms, avg/std density warnings |
| 📏 **BBox Size** | Pixel categories (tiny/small/medium/large), detectability warnings |
| 📐 **Aspect Ratio** | Histograms (square/wide/tall) |
| 🗺️ **Spatial** | Heatmap of bbox centers (bias detection) |
| 🖼️ **Resolution** | Scatter plot, variance warnings |
| 👁️ **Viewer** | Image grid with drawn bounding boxes |
| ⚖️ **Imbalance** | Detailed counts, ratio (>3.0 ⚠️), health score |
| ⚓ **Anchors** | KMeans clustering (9 clusters) on bbox dimensions |

## 🛠️ Utilities Included
- **`cluster_remove.py`** 🎯 **Smart Reducer**: CLIP embeddings + KMeans → Keep 1 representative/cluster (e.g., reduce 10k→100 imgs)
- **`duplicator_remove.py`** 🧹 **Duplicate Hunter**: FAISS similarity search (0.94 threshold), bulk trash mover
- **`download.py`** 📥 **Model Downloader**: OpenCLIP ViT-B/32 for offline use

## 🚀 Quick Start

```bash
# Clone & Install
git clone <repo>
cd yolo_analyzer
pip install -r requirements.txt

# Download models (optional, for clustering/dupe detection)
python download.py

# Launch GUI
python -m yolo_dataset_analyzer.main
```

**Load any YOLO `dataset.yaml` → Instant analysis!**

## 🖼️ Screenshots
*(Add your screenshots here)*

```
[Summary Tab]          [Class Distribution]     [BBox Sizes]
+------------------+  +-------------------+   +------------------+
| Train: 5000 imgs |  | Class 0: 1200    |   | Tiny: 15% ⚠️    |
| Objects: 25k     |  | Imbalance: 4.2x   |   | Small: 30%       |
+------------------+  +-------------------+   +------------------+
```

## 📦 Installation

```bash
pip install PyQt5 matplotlib pandas numpy opencv-python scikit-learn tqdm ultralytics
pip install open-clip-torch faiss-cpu torch torchvision  # For utilities
```

**Full `requirements.txt`:**
```
PyQt5>=5.15
matplotlib>=3.5
pandas>=1.5
numpy>=1.23
opencv-python>=4.6
scikit-learn>=1.1
tqdm
ultralytics  # YOLO YAML parsing
open-clip-torch  # Clustering/dupe detection
faiss-cpu
torch torchvision torchaudio
```

## 📖 Usage

### 1. **Core Analyzer GUI**
```
📁 Click "Select dataset.yaml"
⏳ Auto-analysis (progress bar)
📊 Explore 10 tabs with plots/tables
💾 Export insights manually
```

**Sample Output:**
```
Dataset Health Score: 85/100
Class Imbalance Ratio: 2.8 (moderate)
Spatial Bias: None detected ✅
Object Density: 5.2/img (good)
```

### 2. **cluster_remove.py** - Dataset Reducer
```bash
python cluster_remove.py
# Select image folder → Target count (e.g., 100) → CLIP + KMeans → Export kept/removed
```
**GUI**: Stats, keep/remove preview, cluster browser, JSON metadata.

### 3. **duplicator_remove.py** - Duplicate Remover
```bash
python duplicator_remove.py
# Select dataset folder → CLIP model → Inspect pairs → Bulk trash
```
**Similarity Threshold**: 0.94 (tunable).

## 🔍 Example Workflow
```
1. dataset.yaml ───┐
                    ├─> python main.py ──> Diagnostics + Plots
2. images/          │
   labels/ ────────┘
3. Cleaned ──> python cluster_remove.py ──> 90% reduction
4. Final ────> python duplicator_remove.py ──> No dups
```

## 🤝 Contributing
1. Fork & PR
2. Add tests
3. Update docs

**Issues?** [Open one](https://github.com/yourusername/yolo_analyzer/issues)

## 📄 License
MIT - Use freely in commercial/open-source projects!

---

⭐ **Star if useful!** Questions? [Open Issue](https://github.com/yourusername/yolo_analyzer/issues/new)

