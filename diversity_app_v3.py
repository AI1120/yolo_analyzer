import sys
import os
import torch
import yaml
import numpy as np
import open_clip
from ultralytics import YOLO
from PIL import Image
import io

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QRadioButton, QButtonGroup, QComboBox, QProgressBar, 
                             QTextEdit, QGroupBox, QSpinBox, QFrame, QStatusBar,
                             QDialog, QDialogButtonBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPixmap

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sklearn
try:
    from safetensors.torch import load_file as load_safetensors
except:
    load_safetensors = None


# Check sklearn version for correct parameter
SKLEARN_VERSION = tuple(map(int, sklearn.__version__.split('.')[:2]))
TSNE_ITER_PARAM = 'max_iter' if SKLEARN_VERSION >= (1, 2) else 'n_iter'

# ============================================================
# MODERN STYLESHEET
# ============================================================

MODERN_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #1a1a2e;
}

QWidget {
    background-color: #1a1a2e;
    color: #eaeaea;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 10px;
    margin-top: 15px;
    padding-top: 15px;
    font-weight: bold;
    font-size: 14px;
    color: #00d9ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 10px;
    color: #00d9ff;
}

QPushButton {
    background-color: #0f3460;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    font-weight: bold;
    font-size: 13px;
    min-width: 120px;
}

QPushButton:hover {
    background-color: #00d9ff;
    color: #1a1a2e;
}

QPushButton:pressed {
    background-color: #0099cc;
}

QPushButton:disabled {
    background-color: #2a2a4e;
    color: #666666;
}

#runButton {
    background-color: #00d9ff;
    color: #1a1a2e;
    font-size: 15px;
    padding: 15px 30px;
}

#runButton:hover {
    background-color: #00ffff;
}

#distButton {
    background-color: #16213e;
    border: 1px solid #00d9ff;
    color: #00d9ff;
    font-size: 13px;
    padding: 10px 20px;
}

#distButton:hover {
    background-color: #00d9ff;
    color: #1a1a2e;
}

QRadioButton {
    color: #eaeaea;
    spacing: 8px;
    font-size: 13px;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #0f3460;
    background-color: #1a1a2e;
}

QRadioButton::indicator:checked {
    background-color: #00d9ff;
    border: 2px solid #00d9ff;
}

QComboBox {
    background-color: #0f3460;
    color: #ffffff;
    border: 1px solid #0f3460;
    border-radius: 8px;
    padding: 10px 15px;
    font-size: 13px;
    min-width: 200px;
}

QComboBox:hover {
    border: 1px solid #00d9ff;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 8px solid #00d9ff;
    margin-right: 10px;
}

QSpinBox {
    background-color: #0f3460;
    color: #ffffff;
    border: 1px solid #0f3460;
    border-radius: 8px;
    padding: 10px 15px;
    font-size: 13px;
}

QSpinBox:hover {
    border: 1px solid #00d9ff;
}

QProgressBar {
    background-color: #0f3460;
    border: none;
    border-radius: 8px;
    height: 25px;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #00d9ff;
    border-radius: 8px;
}

QTextEdit {
    background-color: #0f3460;
    color: #00ff88;
    border: 1px solid #0f3460;
    border-radius: 8px;
    padding: 10px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}

QLabel {
    color: #eaeaea;
    font-size: 13px;
}

#pathLabel {
    color: #00d9ff;
    font-size: 11px;
    background-color: #0f3460;
    padding: 8px 12px;
    border-radius: 6px;
}

#selectedImageInfo {
    background-color: #0f3460;
    border-radius: 8px;
    padding: 10px;
    color: #00ffff;
    font-weight: bold;
    font-size: 12px;
}

QStatusBar {
    background-color: #16213e;
    color: #00d9ff;
    border-top: 1px solid #0f3460;
}

#imagePreviewFrame {
    background-color: #0f3460;
    border: 2px solid #00d9ff;
    border-radius: 10px;
}
"""

# ============================================================
# WORKER THREAD
# ============================================================

class EmbeddingWorker(QThread):
    progress_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(object, object, object, object)  # tsne_coords, paths, splits, image_dir
    error_signal = pyqtSignal(str)

    def __init__(self, mode, yolo_weights, yolo_model_name, yaml_path, clip_model_name, max_images, clip_weights_path=None):
        super().__init__()
        self.mode = mode
        self.yolo_weights = yolo_weights
        self.yolo_model_name = yolo_model_name
        self.yaml_path = yaml_path
        self.clip_model_name = clip_model_name
        self.max_images = max_images
        self.clip_weights_path = clip_weights_path

    def run(self):
        try:
            if self.mode == 'yolo':
                embeddings, paths, splits, image_dir = self.extract_yolo_features()
            else:
                embeddings, paths, splits, image_dir = self.extract_clip_features()
            
            if len(embeddings) < 2:
                raise Exception("Not enough images processed to run t-SNE (Need >= 2).")

            self.progress_signal.emit("Running t-SNE...", 90)
            tsne_coords = self.run_tsne(embeddings)
            
            self.progress_signal.emit("Complete!", 100)
            self.finished_signal.emit(tsne_coords, paths, splits, image_dir)

        except Exception as e:
            import traceback
            self.error_signal.emit(f"{str(e)}\n{traceback.format_exc()}")

    def get_image_paths_from_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Dataset root
        root = data.get('path', '')
        if not os.path.isabs(root):
            root = os.path.join(os.path.dirname(yaml_path), root)
            print('!'*50, root)

        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        files_with_splits = []

        split_keys = [
            ('train', 'train'),
            ('val', 'val'),
            ('valid', 'val'),
            ('test', 'test')
        ]

        # Helper to normalize path fields
        def normalize_paths(p):
            if not p:
                return []
            if isinstance(p, str):
                return [p]
            if isinstance(p, list):
                return p
            return []

        # Collect images
        for key, label in split_keys:
            subpaths = normalize_paths(data.get(key))

            for subpath in subpaths:
                dir_path = subpath if os.path.isabs(subpath) else os.path.join(root, subpath)

                if not os.path.exists(dir_path):
                    continue

                # Walk directories recursively
                for r, _, files in os.walk(dir_path):
                    for f in files:
                        if f.lower().endswith(extensions):
                            files_with_splits.append((os.path.join(r, f), label))

        # Limit images if requested
        if self.max_images and len(files_with_splits) > self.max_images:
            files_with_splits = files_with_splits[:self.max_images]

        # Determine primary directory
        primary_dir = root
        train_paths = normalize_paths(data.get("train"))

        if train_paths:
            first = train_paths[0]
            candidate = first if os.path.isabs(first) else os.path.join(root, first)
            if os.path.exists(candidate):
                primary_dir = candidate

        return files_with_splits, primary_dir

    def extract_yolo_features(self):
        self.progress_signal.emit("Loading YOLO model...", 10)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = YOLO(self.yolo_weights) if self.yolo_weights and os.path.exists(self.yolo_weights) else YOLO(self.yolo_model_name)
        model.to(device)
        model.eval()
        
        yolo_model = model.model if hasattr(model, 'model') else model
        features_list, paths_list, splits_list = [], [], []
        
        image_data, image_dir = self.get_image_paths_from_yaml(self.yaml_path)
        total = len(image_data)
        self.progress_signal.emit(f"Found {total} images. Processing...", 20)

        for i, (img_path, split_label) in enumerate(tqdm(image_data)):
            try:
                if hasattr(yolo_model, 'model') and len(yolo_model.model) > 0:
                    backbone_layers = yolo_model.model[:10]
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(640, 640), mode='bilinear', align_corners=False)
                    img_tensor = img_tensor.to(device)
                    
                    feat = img_tensor
                    for layer in backbone_layers:
                        feat = layer(feat)
                    
                    if len(feat.shape) == 4:
                        feat_pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
                    else:
                        feat_pooled = feat
                    
                    features_list.append(feat_pooled.cpu().numpy())
                    paths_list.append(img_path)
                    splits_list.append(0 if split_label == 'train' else (1 if split_label == 'val' else 2))
                
                progress = int(20 + (i/total)*70)
                self.progress_signal.emit(f"Processing {i+1}/{total}", progress)
            except Exception as e:
                print(f"Skip {img_path}: {e}")
                continue
        
        if len(features_list) == 0:
            raise Exception("No features extracted.")
        
        return np.vstack(features_list), paths_list, np.array(splits_list), image_dir

    def extract_clip_features(self):
        self.progress_signal.emit("Loading OpenCLIP model...", 10)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.clip_weights_path and os.path.exists(self.clip_weights_path):
            model, preprocess, _ = self.load_openclip(self.clip_model_name, weights_path=self.clip_weights_path, device=device)
        else:
            model, preprocess, _ = self.load_openclip(self.clip_model_name, device=device)

        image_data, image_dir = self.get_image_paths_from_yaml(self.yaml_path)
        total = len(image_data)
        self.progress_signal.emit(f"Found {total} images. Processing...", 20)
        
        embeddings, paths_list, splits_list = [], [], []
        batch, batch_paths, batch_splits = [], [], []
        
        for i, (img_path, split_label) in enumerate(tqdm(image_data)):
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = preprocess(img).unsqueeze(0).to(device)
                batch.append(img_t)
                batch_paths.append(img_path)
                batch_splits.append(0 if split_label == 'train' else (1 if split_label == 'val' else 2))
                
                if len(batch) >= 16:
                    with torch.no_grad():
                        feat = model.encode_image(torch.cat(batch, dim=0))
                        embeddings.append(feat.cpu().numpy())
                    paths_list.extend(batch_paths)
                    splits_list.extend(batch_splits)
                    batch, batch_paths, batch_splits = [], [], []
                    
                progress = int(20 + (i/total)*70)
                self.progress_signal.emit(f"Processing {i+1}/{total}", progress)
            except Exception as e:
                print(f"Skip {img_path}: {e}")
        
        if batch:
            with torch.no_grad():
                feat = model.encode_image(torch.cat(batch, dim=0))
                embeddings.append(feat.cpu().numpy())
            paths_list.extend(batch_paths)
            splits_list.extend(batch_splits)
                
        if len(embeddings) > 0:
            return np.vstack(embeddings), paths_list, np.array(splits_list), image_dir
        else:
            raise Exception("No CLIP embeddings extracted")

    def load_openclip(
        self, model_name: str = "ViT-B-32",
        pretrained: str | None = "openai",
        weights_path: str | None = None,
        device: str | None = None,
        precision: str = "fp32",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None if weights_path else pretrained,
            device=device
        )

        tokenizer = open_clip.get_tokenizer(model_name)

        if weights_path:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(weights_path)

            ext = os.path.splitext(weights_path)[1].lower()

            print(f"Loading OpenCLIP weights: {weights_path}")

            # -------- SAFETENSORS --------
            if ext == ".safetensors":
                if load_safetensors is None:
                    raise ImportError(
                        "safetensors not installed. Install with: pip install safetensors"
                    )

                checkpoint = load_safetensors(weights_path, device=device)

            # -------- PT / BIN --------
            elif ext in [".pt", ".bin"]:
                checkpoint = torch.load(
                    weights_path,
                    map_location=device,
                    weights_only=False  # PyTorch >=2.6 compatibility
                )

            else:
                raise ValueError(f"Unsupported checkpoint format: {ext}")

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]

                elif "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]

            new_state_dict = {}

            for k, v in checkpoint.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v

            checkpoint = new_state_dict

            missing, unexpected = model.load_state_dict(checkpoint, strict=False)

            if missing:
                print(f"Missing keys: {len(missing)}")

            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")

        if precision == "fp16":
            model = model.half()

        elif precision == "bf16":
            model = model.to(dtype=torch.bfloat16)

        model.eval()
        model.to(device)

        print(f"OpenCLIP model loaded: {model_name} on {device}")

        return model, preprocess, tokenizer

    def run_tsne(self, embeddings):
        X_scaled = StandardScaler().fit_transform(embeddings)
        perplexity_val = min(30, len(embeddings) - 1)
        if perplexity_val < 1: perplexity_val = 1
        
        tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, init='pca', **{TSNE_ITER_PARAM: 1500})
        return tsne.fit_transform(X_scaled)

# ============================================================
# DISTRIBUTION MODAL DIALOG
# ============================================================

class DistributionDialog(QDialog):
    def __init__(self, tsne_coords, splits, colors, names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Distribution Analysis")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(MODERN_STYLESHEET)
        self.setModal(True)
        
        self.tsne_coords = tsne_coords
        self.splits = splits
        self.colors = colors
        self.names = names
        
        self.init_ui()
        self.draw_histogram()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QLabel("Dimension 1 Distribution by Split")
        title.setStyleSheet("font-weight: bold; font-size: 16px; color: #00d9ff; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Canvas
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='#1a1a2e')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(400)
        layout.addWidget(self.canvas)
        
        # Close Button
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        btn_box.setStyleSheet("""
            QPushButton {
                background-color: #0f3460;
                color: white;
                border-radius: 5px;
                padding: 10px 30px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #00d9ff; color: #1a1a2e; }
        """)
        layout.addWidget(btn_box)

    def draw_histogram(self):
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#16213e')
        
        for i in range(3):
            mask = self.splits == i
            if np.sum(mask) > 0:
                ax.hist(self.tsne_coords[mask, 0], bins=30, alpha=0.6, 
                        label=self.names[i], color=self.colors[i], 
                        edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('t-SNE Dimension 1', color='#00d9ff')
        ax.set_ylabel('Count', color='#00d9ff')
        ax.tick_params(colors='#eaeaea')
        ax.spines['bottom'].set_color('#0f3460')
        ax.spines['top'].set_color('#0f3460')
        ax.spines['left'].set_color('#0f3460')
        ax.spines['right'].set_color('#0f3460')
        ax.grid(alpha=0.2, color='#0f3460', axis='y')
        ax.legend(facecolor='#0f3460', edgecolor='#00d9ff', labelcolor='#eaeaea')
        
        self.fig.tight_layout()
        self.canvas.draw()

# ============================================================
# MAIN UI WINDOW
# ============================================================

class DiversityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Diversity Analyzer")
        self.setMinimumSize(1600, 1000)
        self.setStyleSheet(MODERN_STYLESHEET)
        self.worker = None
        
        self.image_paths = []
        self.image_splits = []
        self.image_dir = ""
        self.tsne_coordinates = None
        self.current_image_index = -1
        self.highlight_artist = None
        self.scatter_plot = None
        
        self.split_colors = ['#00d9ff', '#ffaa00', "#f50505"]
        self.split_names = ['Train', 'Validation', 'Test']
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- LEFT PANEL: Controls ---
        control_panel = QGroupBox("Configuration")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(400)

        # Mode Selection
        mode_card = QFrame()
        mode_card.setStyleSheet("QFrame { background-color: #0f3460; border-radius: 10px; padding: 15px; }")
        mode_layout = QVBoxLayout()
        mode_card.setLayout(mode_layout)
        
        mode_layout.addWidget(QLabel("Embedding Mode"))
        self.mode_group = QButtonGroup()
        self.radio_yolo = QRadioButton("YOLO Backbone")
        self.radio_clip = QRadioButton("OpenCLIP - Recommended")
        self.radio_yolo.setChecked(True)
        self.mode_group.addButton(self.radio_yolo)
        self.mode_group.addButton(self.radio_clip)
        mode_layout.addWidget(self.radio_yolo)
        mode_layout.addWidget(self.radio_clip)
        control_layout.addWidget(mode_card)

        # YOLO Settings
        self.yolo_card = QFrame()
        self.yolo_card.setStyleSheet("QFrame { background-color: #0f3460; border-radius: 10px; padding: 15px; }")
        yolo_layout = QVBoxLayout()
        self.yolo_card.setLayout(yolo_layout)
        yolo_layout.addWidget(QLabel("YOLO Settings"))
        
        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.addItems(["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "Custom..."])
        self.yolo_model_combo.currentTextChanged.connect(self.on_yolo_model_change)
        yolo_layout.addWidget(self.yolo_model_combo)
        
        self.yolo_file_widget = QWidget()
        yolo_file_layout = QHBoxLayout()
        self.yolo_file_widget.setLayout(yolo_file_layout)
        self.yolo_btn = QPushButton("Select Weights")
        self.yolo_btn.clicked.connect(self.select_yolo_weights)
        self.yolo_path_label = QLabel("No custom weights")
        self.yolo_path_label.setObjectName("pathLabel")
        yolo_file_layout.addWidget(self.yolo_btn)
        yolo_file_layout.addWidget(self.yolo_path_label)
        yolo_layout.addWidget(self.yolo_file_widget)
        self.yolo_file_widget.setVisible(False)
        control_layout.addWidget(self.yolo_card)

        # CLIP Settings
        self.clip_card = QFrame()
        self.clip_card.setStyleSheet("QFrame { background-color: #0f3460; border-radius: 10px; padding: 15px; }")
        clip_layout = QVBoxLayout()
        self.clip_card.setLayout(clip_layout)
        clip_layout.addWidget(QLabel("OpenCLIP Settings"))
        self.clip_combo = QComboBox()
        self.clip_combo.addItems(["ViT-B-32", "ViT-B-16", "ViT-L-14", "RN50"])
        clip_layout.addWidget(self.clip_combo)

        clip_file_layout = QHBoxLayout()
        self.clip_btn = QPushButton("Select CLIP Weights")
        self.clip_btn.clicked.connect(self.select_clip_weights)
        self.clip_path_label = QLabel("Default pretrained")
        self.clip_path_label.setObjectName("pathLabel")
        clip_file_layout.addWidget(self.clip_btn)
        clip_file_layout.addWidget(self.clip_path_label)
        clip_layout.addLayout(clip_file_layout)

        control_layout.addWidget(self.clip_card)
        self.clip_card.setVisible(False)

        # Dataset Settings
        dataset_card = QFrame()
        dataset_card.setStyleSheet("QFrame { background-color: #0f3460; border-radius: 10px; padding: 15px; }")
        dataset_layout = QVBoxLayout()
        dataset_card.setLayout(dataset_layout)
        dataset_layout.addWidget(QLabel("Dataset Settings"))
        
        yaml_layout = QHBoxLayout()
        self.yaml_btn = QPushButton("Select YAML")
        self.yaml_btn.clicked.connect(self.select_yaml)
        self.yaml_path_label = QLabel("No YAML selected")
        self.yaml_path_label.setObjectName("pathLabel")
        yaml_layout.addWidget(self.yaml_btn)
        yaml_layout.addWidget(self.yaml_path_label)
        dataset_layout.addLayout(yaml_layout)
        
        self.max_img_spin = QSpinBox()
        self.max_img_spin.setRange(10, 100000)
        self.max_img_spin.setValue(2000)
        dataset_layout.addWidget(QLabel("Max Images:"))
        dataset_layout.addWidget(self.max_img_spin)
        control_layout.addWidget(dataset_card)

        # Run Button
        self.run_btn = QPushButton("RUN ANALYSIS")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self.start_analysis)
        control_layout.addWidget(self.run_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)

        # Log
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(150)
        control_layout.addWidget(self.log_console)

        # --- RIGHT PANEL: Plot + Image Preview ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        # Plot Panel
        plot_panel = QGroupBox("t-SNE Visualization")
        plot_layout = QVBoxLayout()
        plot_panel.setLayout(plot_layout)
        
        self.fig = Figure(figsize=(12, 9), dpi=100, facecolor='#1a1a2e')
        self.plot_canvas = FigureCanvas(self.fig)
        self.plot_canvas.setMinimumWidth(800)
        self.plot_canvas.setMinimumHeight(500)
        self.plot_canvas.mpl_connect('pick_event', self.on_point_click)
        
        self.nav_toolbar = NavigationToolbar(self.plot_canvas, self)
        self.nav_toolbar.setStyleSheet("""
            QToolBar { background-color: #0f3460; border: none; border-radius: 8px; padding: 5px; }
            QToolBar QToolButton { background-color: #1a1a2e; border: 1px solid #0f3460; border-radius: 5px; color: #eaeaea; }
            QToolBar QToolButton:hover { background-color: #00d9ff; color: #1a1a2e; }
        """)
        plot_layout.addWidget(self.nav_toolbar)
        plot_layout.addWidget(self.plot_canvas)
        
        # Distribution Button
        self.dist_btn = QPushButton("View Distribution")
        self.dist_btn.setObjectName("distButton")
        self.dist_btn.clicked.connect(self.show_distribution_dialog)
        self.dist_btn.setEnabled(False)
        plot_layout.addWidget(self.dist_btn)
        
        self.plot_info = QLabel("Click points to view images")
        self.plot_info.setStyleSheet("color: #888888; font-size: 12px; padding: 10px;")
        self.plot_info.setAlignment(Qt.AlignCenter)
        plot_layout.addWidget(self.plot_info)

        # Image Preview
        image_panel = QGroupBox("Image Preview")
        image_layout = QVBoxLayout()
        image_panel.setLayout(image_layout)
        
        self.selected_image_info = QLabel("No image selected")
        self.selected_image_info.setObjectName("selectedImageInfo")
        self.selected_image_info.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.selected_image_info)
        
        self.image_preview_frame = QFrame()
        self.image_preview_frame.setObjectName("imagePreviewFrame")
        image_frame_layout = QVBoxLayout()
        self.image_preview_frame.setLayout(image_frame_layout)
        
        self.image_preview_label = QLabel()
        self.image_preview_label.setMinimumSize(500, 400)
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setStyleSheet("QLabel { background-color: #1a1a2e; border-radius: 8px; color: #888888; }")
        self.image_preview_label.setText("Click a point\nto view image")
        image_frame_layout.addWidget(self.image_preview_label)
        image_layout.addWidget(self.image_preview_frame)
        
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.show_previous_image)
        self.prev_btn.setEnabled(False)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.open_image_folder)
        self.open_folder_btn.setEnabled(False)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(self.open_folder_btn)
        image_layout.addLayout(nav_layout)
        
        right_panel.addWidget(plot_panel)
        right_panel.addWidget(image_panel)

        main_layout.addWidget(control_panel)
        main_layout.addLayout(right_panel)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        self.radio_yolo.toggled.connect(self.toggle_inputs)
        self.toggle_inputs()
        self.log("Ready. Select YAML and run analysis.")

    def select_clip_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CLIP Weights",
            "",
            "Checkpoint Files (*.pt *.pth *.bin *.safetensors)"
        )

        if path:
            self.clip_weights_path = path
            self.clip_path_label.setText(os.path.basename(path))

    def toggle_inputs(self):
        is_yolo = self.radio_yolo.isChecked()
        self.yolo_card.setVisible(is_yolo)
        self.clip_card.setVisible(not is_yolo)

    def on_yolo_model_change(self, text):
        if text == "Custom...":
            self.yolo_file_widget.setVisible(True)
            self.select_yolo_weights()
        else:
            self.yolo_file_widget.setVisible(False)

    def select_yolo_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Weights", "", "PyTorch Files (*.pt)")
        if path:
            self.yolo_path_label.setText(os.path.basename(path))
            self.yolo_weights_path = path

    def select_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YAML", "", "YAML Files (*.yaml *.yml)")
        if path:
            self.yaml_path_label.setText(os.path.basename(path))
            self.yaml_file_path = path

    def log(self, message):
        self.log_console.append(message)
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def start_analysis(self):
        if not hasattr(self, 'yaml_file_path'):
            self.log("ERROR: Select YAML file.")
            return
        
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing...")
        self.progress_bar.setValue(0)
        self.dist_btn.setEnabled(False)

        mode = 'yolo' if self.radio_yolo.isChecked() else 'clip'
        yolo_model = self.yolo_model_combo.currentText().split()[0] if self.radio_yolo.isChecked() else "yolo11n.pt"
        yolo_weights = getattr(self, 'yolo_weights_path', None) if self.radio_yolo.isChecked() and "Custom" in self.yolo_model_combo.currentText() else None
        clip_model = self.clip_combo.currentText().split()[0]
        
        clip_weights = getattr(self, "clip_weights_path", None)
        self.worker = EmbeddingWorker(mode, yolo_weights, yolo_model, 
                                      self.yaml_file_path, clip_model, self.max_img_spin.value(), clip_weights)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.error_signal.connect(self.analysis_error)
        self.worker.start()

    def update_progress(self, msg, val):
        self.log(msg)
        self.progress_bar.setValue(val)

    def analysis_finished(self, tsne_coords, paths, splits, image_dir):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN ANALYSIS")
        self.dist_btn.setEnabled(True)
        
        self.tsne_coordinates = tsne_coords
        self.image_paths = paths
        self.image_splits = splits
        self.image_dir = image_dir
        
        self.draw_plot()
        self.plot_info.setVisible(False)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        self.log("Analysis Complete!")

    def analysis_error(self, err_msg):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN ANALYSIS")
        self.log(f"ERROR: {err_msg}")

    def show_distribution_dialog(self):
        if self.tsne_coordinates is None:
            return
        dialog = DistributionDialog(
            self.tsne_coordinates, 
            self.image_splits, 
            self.split_colors, 
            self.split_names, 
            self
        )
        dialog.exec_()

    def draw_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#16213e')
        
        splits = self.image_splits
        self.scatter_plot = ax.scatter(
            self.tsne_coordinates[:, 0], 
            self.tsne_coordinates[:, 1], 
            s=60, alpha=0.7, c=splits, 
            cmap=ListedColormap(self.split_colors), 
            edgecolors='#ffffff', linewidths=0.5,
            picker=5
        )
        
        unique_splits = np.unique(splits)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=self.split_names[i],
                                  markerfacecolor=self.split_colors[i], markersize=8) 
                           for i in unique_splits if i < len(self.split_names)]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', facecolor='#0f3460', 
                      edgecolor='#00d9ff', labelcolor='#eaeaea')
        
        ax.set_xlabel('Dimension 1', color='#00d9ff')
        ax.set_ylabel('Dimension 2', color='#00d9ff')
        ax.tick_params(colors='#eaeaea')
        ax.spines['bottom'].set_color('#0f3460')
        ax.spines['top'].set_color('#0f3460')
        ax.spines['left'].set_color('#0f3460')
        ax.spines['right'].set_color('#0f3460')
        ax.grid(alpha=0.2, color='#0f3460')
        
        self.fig.tight_layout()
        self.plot_canvas.draw_idle()

    def on_point_click(self, event):
        if event.artist != self.scatter_plot:
            return
        if not self.image_paths:
            return
        
        ind = event.ind
        if len(ind) == 0:
            return
        
        self.current_image_index = int(ind[0])
        self.highlight_selected_point()
        self.display_current_image()
        
        split_name = self.split_names[self.image_splits[self.current_image_index]]
        self.statusBar.showMessage(f"Image {self.current_image_index + 1} ({split_name})")

    def highlight_selected_point(self):
        if self.highlight_artist is not None:
            self.highlight_artist.remove()
        
        if 0 <= self.current_image_index < len(self.tsne_coordinates):
            ax = self.fig.axes[0]
            x, y = self.tsne_coordinates[self.current_image_index]
            self.highlight_artist = ax.scatter([x], [y], s=300, alpha=0.6, 
                                               facecolors='none', edgecolors='#ff0066', 
                                               linewidths=4, zorder=10)
            self.plot_canvas.draw_idle()

    def display_current_image(self):
        if not (0 <= self.current_image_index < len(self.image_paths)):
            return
        
        img_path = self.image_paths[self.current_image_index]
        split_name = self.split_names[self.image_splits[self.current_image_index]]
        self.selected_image_info.setText(f"{os.path.basename(img_path)} | {split_name}")
        
        try:
            img = Image.open(img_path).convert('RGB')
            max_width, max_height = 500, 400
            ratio = img.width / img.height
            if ratio > max_width/max_height:
                new_width = max_width
                new_height = int(max_width / ratio)
            else:
                new_height = max_height
                new_width = int(max_height * ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue(), 'PNG')
            self.image_preview_label.setPixmap(pixmap)
            self.image_preview_label.setText("")
        except Exception as e:
            self.image_preview_label.setText(f"Error: {e}")

    def show_previous_image(self):
        if not self.image_paths: return
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.highlight_selected_point()
        self.display_current_image()

    def show_next_image(self):
        if not self.image_paths: return
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.highlight_selected_point()
        self.display_current_image()

    def open_image_folder(self):
        if self.image_dir and os.path.exists(self.image_dir):
            import subprocess
            try:
                if sys.platform == 'win32': os.startfile(self.image_dir)
                elif sys.platform == 'darwin': subprocess.run(['open', self.image_dir])
                else: subprocess.run(['xdg-open', self.image_dir])
            except: pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = DiversityApp()
    window.show()
    sys.exit(app.exec_())