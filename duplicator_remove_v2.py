import sys
import os
import cv2
import torch
import faiss
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QTabWidget, QProgressBar, 
    QFileDialog, QTextEdit, QMessageBox, QFrame, 
    QSizePolicy, QProgressDialog, QComboBox, QGroupBox,
    QRadioButton, QDialog, QSlider, QSpinBox
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# -----------------------------
# Import Handling with Proper Fallbacks
# -----------------------------
USE_OPEN_CLIP = False
USE_CLIP = False

try:
    import open_clip
    USE_OPEN_CLIP = True
    print("✓ Using open_clip")
except ImportError:
    try:
        import clip
        USE_CLIP = True
        print("✓ Using clip")
    except ImportError:
        print("⚠ Warning: Neither open_clip nor clip found. Install with: pip install open-clip-torch")

# -----------------------------
# Modern Dark Theme Stylesheet
# -----------------------------
DARK_STYLE = """
QMainWindow {
    background-color: #1e1e1e;
    color: #ffffff;
}
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Roboto", sans-serif;
}
QFrame {
    background-color: #2b2b2b;
    border-radius: 8px;
    border: 1px solid #3e3e3e;
}
QLabel {
    color: #e0e0e0;
    padding: 5px;
}
QLabel#Title {
    font-size: 24px;
    font-weight: bold;
    color: #ffffff;
    padding: 15px;
}
QLabel#Status {
    font-size: 14px;
    color: #4fc3f7;
    padding: 10px;
    background-color: #263238;
    border-radius: 4px;
}
QLabel#ModelInfo {
    font-size: 14px;
    color: #a5b4fc;
    padding: 12px;
    background-color: #312e81;
    border-radius: 6px;
    font-weight: bold;
}
QLabel#ModelSelected {
    font-size: 16px;
    color: #10b981;
    padding: 10px;
    background-color: #064e3b;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton {
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2563eb;
}
QPushButton:pressed {
    background-color: #1d4ed8;
}
QPushButton:disabled {
    background-color: #4b5563;
    color: #9ca3af;
}
QPushButton#Danger {
    background-color: #ef4444;
}
QPushButton#Danger:hover {
    background-color: #dc2626;
}
QPushButton#Warning {
    background-color: #f59e0b;
    color: #000000;
}
QPushButton#Warning:hover {
    background-color: #d97706;
}
QPushButton#Success {
    background-color: #10b981;
}
QPushButton#Success:hover {
    background-color: #059669;
}
QPushButton#Secondary {
    background-color: #4b5563;
}
QPushButton#Secondary:hover {
    background-color: #6b7280;
}
QPushButton#ModelSelect {
    background-color: #8b5cf6;
    font-size: 14px;
    padding: 12px 25px;
}
QPushButton#ModelSelect:hover {
    background-color: #7c3aed;
}
QPushButton#DatasetSelect {
    background-color: #3b82f6;
    font-size: 14px;
    padding: 12px 25px;
}
QPushButton#DatasetSelect:hover {
    background-color: #2563eb;
}
QComboBox {
    background-color: #2b2b2b;
    color: #e0e0e0;
    border: 1px solid #3e3e3e;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
}
QComboBox QAbstractItemView {
    background-color: #2b2b2b;
    color: #e0e0e0;
    border: 1px solid #3e3e3e;
    selection-background-color: #3b82f6;
}
QProgressBar {
    border: 1px solid #3e3e3e;
    border-radius: 4px;
    text-align: center;
    background-color: #2b2b2b;
    height: 10px;
}
QProgressBar::chunk {
    background-color: #3b82f6;
    border-radius: 4px;
}
QTabWidget::pane {
    border: 1px solid #3e3e3e;
    border-radius: 8px;
    background-color: #2b2b2b;
}
QTabBar::tab {
    background-color: #2b2b2b;
    color: #9ca3af;
    padding: 10px 20px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QTabBar::tab:selected {
    background-color: #3b82f6;
    color: white;
}
QTextEdit {
    background-color: #121212;
    color: #e0e0e0;
    border: 1px solid #3e3e3e;
    border-radius: 4px;
    padding: 5px;
    font-family: "Consolas", "Monaco", monospace;
}
QProgressDialog {
    background-color: #2b2b2b;
    color: white;
    border: 1px solid #3e3e3e;
}
QProgressDialog QProgressBar {
    background-color: #121212;
}
QProgressDialog QProgressBar::chunk {
    background-color: #f59e0b;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #3e3e3e;
    border-radius: 8px;
    margin-top: 15px;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 8px;
    color: #4fc3f7;
}
QRadioButton {
    color: #e0e0e0;
    spacing: 8px;
}
QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid #3e3e3e;
    background-color: #2b2b2b;
}
QRadioButton::indicator:checked {
    background-color: #8b5cf6;
    border: 2px solid #8b5cf6;
}
"""

# -----------------------------
# Predefined Model Options
# -----------------------------
PREDEFINED_MODELS = {
    "ViT-B-32 (Fast, Recommended)": "ViT-B-32",
    "ViT-B-16 (Balanced)": "ViT-B-16",
    "ViT-L-14 (Accurate)": "ViT-L-14",
    "ViT-H-14 (Most Accurate, Slow)": "ViT-H-14",
    "RN50 (ResNet, Fast)": "RN50",
    "RN101 (ResNet, Balanced)": "RN101",
}

# -----------------------------
# Worker Thread for Analysis
# -----------------------------
class AnalyzerWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    status_message = pyqtSignal(str)

    def __init__(self, image_paths, model_config, threshold=0.94, batch_size=32, embedding_mode="option1", dataset_folder=None):
        super().__init__()
        self.image_paths = image_paths
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.model_config = model_config
        self.threshold = threshold
        self.model = None
        self.preprocess = None
        self.embedding_mode = embedding_mode
        self.dataset_folder = dataset_folder
        self.crop_data = {}  # Store crop info for Option 2

    def load_model(self):
        """Load the embedding model with proper error handling"""
        try:
            if self.model_config["type"] == "predefined":
                model_name = self.model_config["name"]
                
                if USE_OPEN_CLIP:
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                        model_name,
                        pretrained='laion2b_s34b_b79k',
                        device=self.device
                    )
                    self.model.eval()
                    return f"OpenCLIP {model_name} (Pretrained)"
                    
                elif USE_CLIP:
                    self.model, self.preprocess = clip.load(
                        model_name,
                        device=self.device,
                        jit=False
                    )
                    self.model.eval()
                    return f"CLIP {model_name} (Pretrained)"
                else:
                    raise ImportError("No CLIP library available")
                    
            elif self.model_config["type"] == "local":
                model_path = self.model_config["name"]
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                if USE_OPEN_CLIP:
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                        'ViT-B-32',
                        pretrained=model_path,
                        device=self.device
                    )
                    self.model.eval()
                    return f"Local Model: {os.path.basename(model_path)}"
                elif USE_CLIP:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    self.model, self.preprocess = clip.load(
                        'ViT-B-32',
                        device=self.device,
                        jit=False
                    )
                    
                    if hasattr(self.model, 'load_state_dict'):
                        self.model.load_state_dict(state_dict, strict=False)
                    
                    self.model.eval()
                    return f"Local Model: {os.path.basename(model_path)}"
                else:
                    raise ImportError("No CLIP library available for local model")
                    
        except Exception as e:
            raise Exception(f"Model loading failed: {str(e)}")

    def run(self):
        result = {
            "duplicates": [], 
            "total_images": 0, 
            "error": None,
            "model_info": ""
        }
        
        try:
            self.progress.emit(5)
            self.status_message.emit("Loading model...")
            
            result["model_info"] = self.load_model()
            
            if self.model is None or self.preprocess is None:
                raise Exception("Failed to load model or preprocess function")
            
            self.progress.emit(10)
            self.status_message.emit(f"Processing {len(self.image_paths)} images...")
            
        except Exception as e:
            self.finished.emit({**result, "error": str(e)})
            return

        embeddings = []
        valid_paths = []
        total_images = len(self.image_paths)
        processed = 0

        try:
            from PIL import Image
        except ImportError:
            self.finished.emit({**result, "error": "PIL library not found. Install with: pip install Pillow"})
            return

        for i in range(0, len(self.image_paths), self.batch_size):
            batch_paths = self.image_paths[i:i+self.batch_size]
            batch_tensors = []
            batch_valid_paths = []

            for path in batch_paths:
                try:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    
                    if self.embedding_mode == "option2":
                        # Load JSON and crop bbox area
                        json_path = os.path.splitext(path)[0] + ".json"
                        if not os.path.exists(json_path):
                            continue
                        
                        import json
                        with open(json_path, 'r') as f:
                            label_data = json.load(f)
                        
                        if not label_data.get('shapes'):
                            continue
                        
                        # Use first bbox
                        shape = label_data['shapes'][0]
                        points = shape['points']
                        x1, y1 = int(min(points[0][0], points[1][0])), int(min(points[0][1], points[1][1]))
                        x2, y2 = int(max(points[0][0], points[1][0])), int(max(points[0][1], points[1][1]))
                        
                        # Expand bbox by 1.2x
                        w, h = x2 - x1, y2 - y1
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        new_w, new_h = int(w * 1.2), int(h * 1.2)
                        x1 = max(0, cx - new_w // 2)
                        y1 = max(0, cy - new_h // 2)
                        x2 = min(img.shape[1], cx + new_w // 2)
                        y2 = min(img.shape[0], cy + new_h // 2)
                        
                        self.crop_data[path] = (x1, y1, x2, y2)
                        img = img[y1:y2, x1:x2]
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img)
                    tensor = self.preprocess(img_pil)
                    batch_tensors.append(tensor)
                    batch_valid_paths.append(path)
                    
                except Exception as e:
                    print(f"Warning: Could not process {path}: {e}")
                    continue

            if not batch_tensors:
                processed += len(batch_paths)
                continue

            try:
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    if hasattr(self.model, 'encode_image'):
                        batch_emb = self.model.encode_image(batch_tensor)
                    else:
                        batch_emb = self.model(batch_tensor)
                    
                    if isinstance(batch_emb, tuple):
                        batch_emb = batch_emb[0]
                    
                    batch_emb = batch_emb / batch_emb.norm(dim=1, keepdim=True)
                    embeddings.append(batch_emb.cpu().numpy())
                
                valid_paths.extend(batch_valid_paths)
                processed += len(batch_paths)
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                progress_pct = 10 + int((processed / total_images) * 50)
                self.progress.emit(progress_pct)
                
            except Exception as e:
                print(f"Batch processing error: {e}")
                continue

        if not embeddings:
            self.finished.emit({**result, "error": "No valid images processed. Check image formats."})
            return

        self.progress.emit(65)
        self.status_message.emit("Building similarity index...")

        try:
            embeddings = np.vstack(embeddings).astype("float32")
            result["total_images"] = len(valid_paths)

            d = embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            
            try:
                if self.device == "cuda" and faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    self.status_message.emit("Using GPU for similarity search")
                else:
                    self.status_message.emit("Using CPU for similarity search")
            except Exception:
                self.status_message.emit("GPU not available, using CPU")

            index.add(embeddings)
            k = min(5, len(embeddings))
            sims, neighbors = index.search(embeddings, k)

            self.progress.emit(75)
            self.status_message.emit("Finding duplicates...")

            duplicates = []
            seen_pairs = set()
            threshold = self.threshold

            for i in range(len(sims)):
                for j_idx, score in zip(neighbors[i], sims[i]):
                    if i != j_idx and score >= threshold:
                        p1, p2 = valid_paths[i], valid_paths[j_idx]
                        pair_key = tuple(sorted((p1, p2)))
                        
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            duplicates.append((p1, p2, float(score)))
                
                if i % 500 == 0:
                    progress_pct = 75 + int((i / len(sims)) * 20)
                    self.progress.emit(progress_pct)

            self.progress.emit(100)
            result["duplicates"] = duplicates
            result["crop_data"] = self.crop_data
            result["embedding_mode"] = self.embedding_mode
            
            self.status_message.emit("Analysis complete!")
            
        except Exception as e:
            result["error"] = f"Analysis error: {str(e)}"
            print(f"Analysis error: {e}")

        finally:
            if self.model is not None:
                del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()

        self.finished.emit(result)

# -----------------------------
# Model Selection Dialog
# -----------------------------
class ModelSelectionDialog(QDialog):
    model_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(DARK_STYLE)
        self.setModal(True)
        self.setWindowTitle("Select Model")
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        self.local_model_file = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("🧠 Select Embedding Model")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        if not USE_OPEN_CLIP and not USE_CLIP:
            warning = QLabel("⚠️ No CLIP library found! Install with: pip install open-clip-torch")
            warning.setStyleSheet("color: #ef4444; font-weight: bold; padding: 10px;")
            warning.setWordWrap(True)
            layout.addWidget(warning)
        
        type_group = QGroupBox("Model Source")
        type_layout = QVBoxLayout(type_group)
        
        self.predefined_radio = QRadioButton("📡 Predefined Model (Download Online)")
        self.local_radio = QRadioButton("📁 Local Model File (Offline)")
        
        self.predefined_radio.setChecked(True)
        self.predefined_radio.toggled.connect(self.on_model_type_changed)
        
        type_layout.addWidget(self.predefined_radio)
        type_layout.addWidget(self.local_radio)
        layout.addWidget(type_group)
        
        self.predefined_widget = QWidget()
        predefined_layout = QVBoxLayout(self.predefined_widget)
        predefined_layout.setContentsMargins(0, 0, 0, 0)
        
        predefined_label = QLabel("Choose a pretrained model:")
        predefined_label.setStyleSheet("color: #9ca3af;")
        predefined_layout.addWidget(predefined_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(PREDEFINED_MODELS.keys()))
        self.model_combo.setMinimumHeight(40)
        self.model_combo.currentTextChanged.connect(self.update_info_label)
        predefined_layout.addWidget(self.model_combo)
        
        layout.addWidget(self.predefined_widget)
        
        self.local_widget = QWidget()
        local_layout = QVBoxLayout(self.local_widget)
        local_layout.setContentsMargins(0, 0, 0, 0)
        
        self.local_model_path = QLabel("No file selected")
        self.local_model_path.setStyleSheet("color: #9ca3af; padding: 10px; background-color: #121212; border-radius: 4px;")
        self.local_model_path.setWordWrap(True)
        local_layout.addWidget(self.local_model_path)
        
        local_btn_layout = QHBoxLayout()
        self.browse_btn = QPushButton("📁 Browse Model File")
        self.browse_btn.setObjectName("Secondary")
        self.browse_btn.clicked.connect(self.browse_local_model)
        
        self.clear_btn = QPushButton("✕ Clear")
        self.clear_btn.setObjectName("Secondary")
        self.clear_btn.clicked.connect(self.clear_local_model)
        
        local_btn_layout.addWidget(self.browse_btn)
        local_btn_layout.addWidget(self.clear_btn)
        local_layout.addLayout(local_btn_layout)
        
        layout.addWidget(self.local_widget)
        self.local_widget.setVisible(False)
        
        self.info_label = QLabel("📌 ViT-B-32 - Fast and recommended for most use cases")
        self.info_label.setObjectName("ModelInfo")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        btn_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("Secondary")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.confirm_btn = QPushButton("✅ Confirm Model")
        self.confirm_btn.setObjectName("Success")
        self.confirm_btn.setMinimumHeight(45)
        self.confirm_btn.clicked.connect(self.confirm_selection)
        
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.confirm_btn)
        layout.addLayout(btn_layout)
        
        self.update_info_label()
        
    def on_model_type_changed(self, checked):
        if self.predefined_radio.isChecked():
            self.predefined_widget.setVisible(True)
            self.local_widget.setVisible(False)
        else:
            self.predefined_widget.setVisible(False)
            self.local_widget.setVisible(True)
        self.update_info_label()
            
    def browse_local_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Model File", 
            "", 
            "Model Files (*.pt *.pth *.bin *.safetensors);;All Files (*)"
        )
        if file_path:
            self.local_model_file = file_path
            try:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                self.local_model_path.setText(f"📄 {os.path.basename(file_path)}\n📊 Size: {size_mb:.1f} MB")
                self.local_model_path.setStyleSheet("color: #10b981; padding: 10px; background-color: #064e3b; border-radius: 4px;")
            except Exception:
                self.local_model_path.setText(f"📄 {os.path.basename(file_path)}")
            self.update_info_label()
            
    def clear_local_model(self):
        self.local_model_file = None
        self.local_model_path.setText("No file selected")
        self.local_model_path.setStyleSheet("color: #9ca3af; padding: 10px; background-color: #121212; border-radius: 4px;")
        self.update_info_label()
        
    def update_info_label(self):
        if self.predefined_radio.isChecked():
            model_name = self.model_combo.currentText()
            self.info_label.setText(f"📌 {model_name}")
        else:
            if self.local_model_file:
                self.info_label.setText(f"📌 Local Model: {os.path.basename(self.local_model_file)}")
            else:
                self.info_label.setText("⚠️ Please select a local model file")
                
    def confirm_selection(self):
        if self.predefined_radio.isChecked():
            if not USE_OPEN_CLIP and not USE_CLIP:
                QMessageBox.critical(
                    self, 
                    "Library Missing", 
                    "No CLIP library found! Please install with: pip install open-clip-torch"
                )
                return
                
            model_config = {
                "type": "predefined",
                "name": PREDEFINED_MODELS[self.model_combo.currentText()],
                "display_name": self.model_combo.currentText()
            }
        else:
            if not self.local_model_file:
                QMessageBox.warning(self, "Model Required", "Please select a local model file.")
                return
            model_config = {
                "type": "local",
                "name": self.local_model_file,
                "display_name": f"Local: {os.path.basename(self.local_model_file)}"
            }
        
        self.model_selected.emit(model_config)
        self.accept()

# -----------------------------
# Modern Duplicate Viewer
# -----------------------------
class ClusterViewer(QMainWindow):
    def __init__(self, duplicate_pairs, crop_data=None, embedding_mode="option1"):
        super().__init__()
        self.original_pairs = list(duplicate_pairs)
        self.pairs = list(duplicate_pairs)
        self.idx = 0
        self.trash_folder = "_dataset_trash"
        self.processed_count = 0
        self.moved_files = []
        self.crop_data = crop_data or {}
        self.embedding_mode = embedding_mode

        self.setWindowTitle("Duplicate Inspector")
        self.resize(1400, 900)
        self.setStyleSheet(DARK_STYLE)
        self.setFocusPolicy(Qt.StrongFocus)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        header = QFrame()
        header_layout = QHBoxLayout(header)
        self.title_label = QLabel("🔍 Duplicate Inspector")
        self.title_label.setObjectName("Title")
        self.info_label = QLabel("Loading...")
        self.info_label.setStyleSheet("font-size: 16px; color: #9ca3af;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.info_label)
        main_layout.addWidget(header)

        image_frame = QFrame()
        image_layout = QHBoxLayout(image_frame)
        image_layout.setSpacing(20)

        self.left_widget = self.create_image_card("Keep (Original)", "#10b981")
        self.right_widget = self.create_image_card("Duplicate (Action)", "#ef4444")

        image_layout.addWidget(self.left_widget, 1)
        image_layout.addWidget(self.right_widget, 1)
        main_layout.addWidget(image_frame, 1)

        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        self.btn_prev = QPushButton("← Prev")
        self.btn_prev.clicked.connect(self.prev_pair)
        
        self.btn_next = QPushButton("Next →")
        self.btn_next.clicked.connect(self.next_pair)

        self.btn_move = QPushButton("📦 Move to Trash")
        self.btn_move.setObjectName("Danger")
        self.btn_move.clicked.connect(self.move_to_trash)
        
        self.btn_bulk_move = QPushButton("🗑️ Bulk Move All to Trash")
        self.btn_bulk_move.setObjectName("Warning")
        self.btn_bulk_move.clicked.connect(self.bulk_move_to_trash)
        
        self.btn_skip = QPushButton("⏭ Skip")
        self.btn_skip.setObjectName("Success")
        self.btn_skip.clicked.connect(self.skip_pair)

        self.btn_close = QPushButton("Finish & Log")
        self.btn_close.clicked.connect(self.close_and_log)

        controls_layout.addWidget(self.btn_prev)
        controls_layout.addWidget(self.btn_next)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_move)
        controls_layout.addWidget(self.btn_bulk_move)
        controls_layout.addWidget(self.btn_skip)
        controls_layout.addWidget(self.btn_close)
        
        main_layout.addWidget(controls_frame)

        log_frame = QFrame()
        log_layout = QVBoxLayout(log_frame)
        self.log_label = QLabel("Status: Ready")
        self.deletion_log = QTextEdit()
        self.deletion_log.setReadOnly(True)
        self.deletion_log.setMaximumHeight(100)
        log_layout.addWidget(self.log_label)
        log_layout.addWidget(self.deletion_log)
        main_layout.addWidget(log_frame)

        self.show_pair()
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev_pair()
        elif event.key() == Qt.Key_Right:
            self.next_pair()
        elif event.key() == Qt.Key_Delete:
            self.move_to_trash()
        else:
            super().keyPressEvent(event)

    def create_image_card(self, title, color_hex):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        
        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet(f"font-weight: bold; color: {color_hex}; font-size: 16px;")
        layout.addWidget(lbl_title)

        lbl_img = QLabel("No Image")
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setStyleSheet(f"background-color: #121212; border: 2px dashed {color_hex}; border-radius: 8px;")
        lbl_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl_img.setMinimumHeight(400)
        layout.addWidget(lbl_img)

        lbl_path = QLabel("")
        lbl_path.setAlignment(Qt.AlignCenter)
        lbl_path.setWordWrap(True)
        lbl_path.setStyleSheet("color: #9ca3af; font-size: 12px;")
        layout.addWidget(lbl_path)

        container.img_label = lbl_img
        container.path_label = lbl_path
        return container

    def load_pixmap(self, path):
        if not os.path.exists(path):
            return None
        try:
            if self.embedding_mode == "option2" and path in self.crop_data:
                # Draw bbox on original image and add cropped inset
                img = cv2.imread(path)
                if img is None:
                    return None
                x1, y1, x2, y2 = self.crop_data[path]
                
                # Draw green rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Extract cropped region
                cropped = img[y1:y2, x1:x2].copy()
                
                # Calculate inset size (20% of image width/height)
                inset_h = int(img.shape[0] * 0.2)
                inset_w = int(img.shape[1] * 0.2)
                
                # Resize cropped to fit inset
                cropped_resized = cv2.resize(cropped, (inset_w, inset_h))
                
                # Position at bottom-right corner with 10px margin
                margin = 10
                y_start = img.shape[0] - inset_h - margin
                x_start = img.shape[1] - inset_w - margin
                
                # Add white border around inset
                cv2.rectangle(img, (x_start-2, y_start-2), (x_start+inset_w+2, y_start+inset_h+2), (255, 255, 255), 2)
                
                # Overlay cropped image
                img[y_start:y_start+inset_h, x_start:x_start+inset_w] = cropped_resized
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img.shape
                bytes_per_line = ch * w
                from PyQt5.QtGui import QImage
                q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
            else:
                pixmap = QPixmap(path)
            
            if pixmap.isNull():
                return None
            return pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        except Exception:
            return None

    def show_pair(self):
        if not self.pairs:
            self.info_label.setText("✅ All pairs processed!")
            self.left_widget.img_label.setPixmap(QPixmap())
            self.right_widget.img_label.setPixmap(QPixmap())
            self.left_widget.img_label.setText("Done")
            self.right_widget.img_label.setText("Done")
            self.toggle_controls(False)
            return

        path1, path2, score = self.pairs[self.idx]
        self.info_label.setText(f"Pair {self.idx + 1} / {len(self.pairs)} | Similarity: {score:.4f}")

        pix1 = self.load_pixmap(path1)
        if pix1:
            self.left_widget.img_label.setPixmap(pix1)
            self.left_widget.img_label.setText("")
        else:
            self.left_widget.img_label.setText("❌ Load Error")
        self.left_widget.path_label.setText(os.path.basename(path1))

        pix2 = self.load_pixmap(path2)
        if pix2:
            self.right_widget.img_label.setPixmap(pix2)
            self.right_widget.img_label.setText("")
        else:
            self.right_widget.img_label.setText("❌ Load Error")
        self.right_widget.path_label.setText(os.path.basename(path2))

        self.toggle_controls(True)

    def toggle_controls(self, enabled):
        self.btn_prev.setEnabled(enabled and self.idx > 0)
        self.btn_next.setEnabled(enabled and self.idx < len(self.pairs) - 1)
        self.btn_move.setEnabled(enabled)
        self.btn_bulk_move.setEnabled(enabled and len(self.pairs) > 1)
        self.btn_skip.setEnabled(enabled)

    def move_to_trash(self):
        if not self.pairs:
            return
        
        path1, path2, score = self.pairs[self.idx]

        reply = QMessageBox.question(
            self, 
            'Confirm Move', 
            f'Move to Trash?\n\n{os.path.basename(path2)}',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self._process_file_action(path2, "Moved")

    def bulk_move_to_trash(self):
        if not self.pairs:
            QMessageBox.information(self, 'No Duplicates', 'No duplicate pairs to remove.')
            return

        reply = QMessageBox.warning(
            self,
            '⚠️ WARNING: Bulk Operation',
            f'You are about to move {len(self.pairs)} files to the trash.\n\n'
            f'This will move the "Duplicate" image from every pair.\n\n'
            f'Are you absolutely sure?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        progress = QProgressDialog("Moving files to trash...", "Cancel", 0, len(self.pairs), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Bulk Move Progress")
        progress.setMinimumDuration(0)
        progress.setStyleSheet("QLabel { color: white; }")
        progress.show()

        success_count = 0
        fail_count = 0
        pairs_to_remove = []

        for i, (path1, path2, score) in enumerate(self.pairs):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            progress.setLabelText(f"Moving {i+1}/{len(self.pairs)}:\n{os.path.basename(path2)}")
            QApplication.processEvents()

            try:
                if not os.path.exists(self.trash_folder):
                    os.makedirs(self.trash_folder)
                
                dest = os.path.join(self.trash_folder, os.path.basename(path2))
                if os.path.exists(dest):
                    base, ext = os.path.splitext(dest)
                    dest = f"{base}_{datetime.now().strftime('%H%M%S')}{ext}"
                
                os.rename(path2, dest)
                
                log_msg = f"[BULK MOVED] {os.path.basename(path2)}"
                self.deletion_log.append(log_msg)
                self.moved_files.append(path2)
                success_count += 1
                pairs_to_remove.append((path1, path2, score))
            except Exception as e:
                log_msg = f"[BULK FAILED] {os.path.basename(path2)} - {str(e)}"
                self.deletion_log.append(log_msg)
                fail_count += 1

        progress.setValue(len(self.pairs))
        progress.close()

        for pair in pairs_to_remove:
            if pair in self.pairs:
                self.pairs.remove(pair)

        self.idx = 0
        self.processed_count += success_count

        summary_msg = f"Bulk Move Complete!\n\n"
        summary_msg += f"✅ Successfully moved: {success_count} files\n"
        summary_msg += f"❌ Failed: {fail_count} files\n"
        if fail_count > 0:
            summary_msg += f"\nCheck log for details."

        QMessageBox.information(self, 'Bulk Move Complete', summary_msg)
        self.log_label.setText(f"Status: Bulk Moved {success_count} files")
        self.show_pair()

    def _process_file_action(self, path, action_type):
        try:
            if not os.path.exists(self.trash_folder):
                os.makedirs(self.trash_folder)
            
            dest = os.path.join(self.trash_folder, os.path.basename(path))
            if os.path.exists(dest):
                base, ext = os.path.splitext(dest)
                dest = f"{base}_{datetime.now().strftime('%H%M%S')}{ext}"
            
            os.rename(path, dest)
            
            log_msg = f"[{action_type}] {os.path.basename(path)}"
            self.deletion_log.append(log_msg)
            self.moved_files.append(path)
            self.log_label.setText(f"Last Action: {action_type}")
            
            if self.idx < len(self.pairs):
                self.pairs.pop(self.idx)
                if self.idx >= len(self.pairs):
                    self.idx = max(0, len(self.pairs) - 1)
            
            self.show_pair()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move file: {str(e)}")

    def skip_pair(self):
        if self.idx < len(self.pairs) - 1:
            self.idx += 1
            self.show_pair()
        else:
            self.info_label.setText("End of list reached.")

    def prev_pair(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_pair()

    def next_pair(self):
        if self.idx < len(self.pairs) - 1:
            self.idx += 1
            self.show_pair()

    def close_and_log(self):
        moved_count = len(self.moved_files)
        msg = f"Session Complete.\nFiles Moved to Trash: {moved_count}\n\nTrash Folder: _dataset_trash"
        QMessageBox.information(self, "Summary", msg)
        self.close()

    def closeEvent(self, event):
        moved_count = len(self.moved_files)
        if moved_count > 0:
            reply = QMessageBox.question(
                self,
                'Confirm Close',
                f'{moved_count} files were moved to trash.\nContinue closing?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

# -----------------------------
# Main Application
# -----------------------------
class YOLOInspector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Dataset Analyzer Pro")
        self.resize(1100, 800)
        self.setStyleSheet(DARK_STYLE)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(12)

        header_label = QLabel("🤖 YOLO Dataset Duplicate Analyzer")
        header_label.setObjectName("Title")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff; padding: 8px;")
        self.layout.addWidget(header_label)

        if not USE_OPEN_CLIP and not USE_CLIP:
            warning = QLabel("⚠️ CLIP library not found! Install with: pip install open-clip-torch")
            warning.setStyleSheet("color: #ef4444; font-weight: bold; padding: 10px; border: 2px solid #ef4444; border-radius: 8px;")
            warning.setWordWrap(True)
            self.layout.addWidget(warning)

        model_card = QFrame()
        model_card.setStyleSheet("background-color: #2b2b2b; border-radius: 8px; padding: 8px;")
        model_layout = QVBoxLayout(model_card)
        model_layout.setSpacing(8)
        
        # model_title = QLabel("Select Embedding Model")
        # model_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #8b5cf6; margin-bottom: 5px;")
        # model_layout.addWidget(model_title)
        
        self.model_select_btn = QPushButton("🧠 Select Model")
        self.model_select_btn.setStyleSheet("background-color: #8b5cf6; color: white; border: 1px solid #7c3aed; border-radius: 6px; font-size: 14px; padding: 12px 25px; font-weight: bold;")
        self.model_select_btn.setObjectName("")
        self.model_select_btn.setMinimumHeight(40)
        self.model_select_btn.clicked.connect(self.open_model_selection)
        model_layout.addWidget(self.model_select_btn)
        
        self.model_status_label = QLabel("⚠️ No model selected")
        self.model_status_label.setStyleSheet("font-size: 12px; color: #a5b4fc; padding: 8px; background-color: #312e81; border-radius: 6px; font-weight: bold;")
        self.model_status_label.setAlignment(Qt.AlignCenter)
        self.model_status_label.setWordWrap(True)
        model_layout.addWidget(self.model_status_label)
        
        # Create a horizontal layout for the top section
        top_section = QHBoxLayout()
        
        # Left column for model and dataset
        left_column = QVBoxLayout()
        left_column.addWidget(model_card)
        
        dataset_card = QFrame()
        dataset_card.setStyleSheet("background-color: #2b2b2b; border-radius: 8px; padding: 8px;")
        dataset_layout = QVBoxLayout(dataset_card)
        dataset_layout.setSpacing(8)
        
        self.select_btn = QPushButton("📂 Select Dataset Folder")
        self.select_btn.setStyleSheet("background-color: #3b82f6; color: white; border: 1px solid #2563eb; border-radius: 6px; font-size: 14px; padding: 12px 25px; font-weight: bold;")
        self.select_btn.setMinimumHeight(40)
        self.select_btn.clicked.connect(self.select_folder)
        self.select_btn.setEnabled(False)
        dataset_layout.addWidget(self.select_btn)
        
        self.folder_status_label = QLabel("⚠️ No folder selected")
        self.folder_status_label.setStyleSheet("font-size: 12px; color: #a5b4fc; padding: 8px; background-color: #312e81; border-radius: 6px; font-weight: bold;")
        self.folder_status_label.setAlignment(Qt.AlignCenter)
        self.folder_status_label.setWordWrap(True)
        dataset_layout.addWidget(self.folder_status_label)
        
        left_column.addWidget(dataset_card)
        
        # Right column for threshold and analysis
        right_column = QVBoxLayout()
        
        threshold_card = QFrame()
        threshold_card.setStyleSheet("background-color: #2b2b2b; border-radius: 8px; padding: 8px;")
        threshold_layout = QVBoxLayout(threshold_card)
        threshold_layout.setSpacing(5)
        threshold_layout.setContentsMargins(5, 5, 5, 5)
        
        threshold_controls = QHBoxLayout()
        
        threshold_label = QLabel("Threshold:")
        threshold_label.setStyleSheet("color: #e0e0e0; font-size: 14px;")
        threshold_controls.addWidget(threshold_label)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(80, 99)
        self.threshold_slider.setValue(94)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.valueChanged.connect(self.update_threshold_from_slider)
        threshold_controls.addWidget(self.threshold_slider)
        
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(80, 99)
        self.threshold_spinbox.setValue(94)
        self.threshold_spinbox.setSuffix("%")
        self.threshold_spinbox.valueChanged.connect(self.update_threshold_from_spinbox)
        threshold_controls.addWidget(self.threshold_spinbox)
        
        threshold_layout.addLayout(threshold_controls)
        
        right_column.addWidget(threshold_card)
        
        # Add run analysis button directly without card wrapper
        self.run_analysis_btn = QPushButton("🚀 Run Duplicate Analysis")
        self.run_analysis_btn.setStyleSheet("background-color: #10b981; color: white; border: 1px solid #059669; border-radius: 6px; font-size: 14px; padding: 12px 25px; font-weight: bold; margin-top: 8px;")
        self.run_analysis_btn.setMinimumHeight(40)
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn.setEnabled(False)
        right_column.addWidget(self.run_analysis_btn)
        
        # Add columns to horizontal layout with equal stretch factors
        top_section.addLayout(left_column, 1)  # 50% width
        top_section.addLayout(right_column, 1)  # 50% width
        
        # Add the horizontal section to main layout
        self.layout.addLayout(top_section)

        self.status = QLabel("Status: Idle - Select a model first")
        self.status.setObjectName("Status")
        self.status.setWordWrap(True)
        self.layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs, 1)

        self.summary = QTextEdit()
        self.summary.setMaximumHeight(120)
        summary_label = QLabel("Analysis Summary:")
        summary_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 5px;")
        self.layout.addWidget(summary_label)
        self.layout.addWidget(self.summary)

        self.image_paths = []
        self.viewer = None
        self.model_config = None
        self.worker = None
        self.selected_folder = None
        
        # Add embedding mode selection
        mode_card = QFrame()
        mode_card.setStyleSheet("background-color: #2b2b2b; border-radius: 8px; padding: 8px;")
        mode_layout = QVBoxLayout(mode_card)
        mode_layout.setSpacing(8)
        
        mode_label = QLabel("Embedding Mode:")
        mode_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4fc3f7;")
        mode_layout.addWidget(mode_label)
        
        self.option1_radio = QRadioButton("Option 1: Whole Image Embedding")
        self.option2_radio = QRadioButton("Option 2: BBox Crop Embedding (1.2x, requires JSON)")
        self.option1_radio.setChecked(True)
        
        mode_layout.addWidget(self.option1_radio)
        mode_layout.addWidget(self.option2_radio)
        
        self.layout.insertWidget(1, mode_card)

    def open_model_selection(self):
        dialog = ModelSelectionDialog(self)
        dialog.model_selected.connect(self.on_model_selected)
        dialog.exec_()
        
    def on_model_selected(self, model_config):
        self.model_config = model_config
        display_name = model_config.get("display_name", "Unknown")
        
        self.model_status_label.setText(f"✅ {display_name}")
        self.model_status_label.setStyleSheet("font-size: 12px; color: #10b981; padding: 8px; background-color: #064e3b; border-radius: 6px; font-weight: bold;")
        
        self.status.setText(f"Status: Model loaded - Ready to select dataset")
        self.select_btn.setEnabled(True)
        self.update_run_button_state()
        
    def select_folder(self):
        if not self.model_config:
            QMessageBox.warning(
                self, 
                "Model Required", 
                "Please select a model first before selecting the dataset."
            )
            self.open_model_selection()
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.selected_folder = folder
            self.status.setText(f"Scanning folder: {folder}...")
            QApplication.processEvents()
            
            self.image_paths = self.import_images(folder)
            
            if not self.image_paths:
                self.status.setText("⚠️ No images found in folder.")
                self.folder_status_label.setText("⚠️ No images found in selected folder")
                QMessageBox.warning(self, "No Images", "No valid image files found in the selected folder.")
                self.selected_folder = None
                self.update_run_button_state()
                return

            self.folder_status_label.setText(f"✅ {len(self.image_paths)} images found")
            self.folder_status_label.setStyleSheet("font-size: 12px; color: #10b981; padding: 8px; background-color: #064e3b; border-radius: 6px; font-weight: bold;")
            
            self.status.setText(f"Found {len(self.image_paths)} images. Ready to analyze.")
            self.update_run_button_state()

    def import_images(self, folder):
        images = []
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(exts):
                    images.append(os.path.join(root, f))
        return images
        
    def update_threshold_from_slider(self, value):
        self.threshold_spinbox.setValue(value)
        
    def update_threshold_from_spinbox(self, value):
        self.threshold_slider.setValue(value)
        
    def update_run_button_state(self):
        can_run = self.model_config is not None and self.selected_folder is not None and len(self.image_paths) > 0
        self.run_analysis_btn.setEnabled(can_run)
        
    def run_analysis(self):
        if not self.model_config or not self.selected_folder or not self.image_paths:
            QMessageBox.warning(self, "Requirements Missing", "Please select both a model and dataset folder first.")
            return
            
        threshold = self.threshold_spinbox.value() / 100.0  # Convert percentage to decimal
        embedding_mode = "option2" if self.option2_radio.isChecked() else "option1"
        
        self.status.setText(f"Starting analysis with {threshold:.2f} threshold ({embedding_mode})...")
        self.progress.setValue(0)
        self.select_btn.setEnabled(False)
        self.model_select_btn.setEnabled(False)
        self.run_analysis_btn.setEnabled(False)
        self.threshold_slider.setEnabled(False)
        self.threshold_spinbox.setEnabled(False)
        self.option1_radio.setEnabled(False)
        self.option2_radio.setEnabled(False)

        self.worker = AnalyzerWorker(self.image_paths, self.model_config, threshold, embedding_mode=embedding_mode, dataset_folder=self.selected_folder)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status_message.connect(self.status.setText)
        self.worker.finished.connect(self.analysis_done)
        self.worker.start()

    def analysis_done(self, result):
        self.progress.setValue(100)
        self.select_btn.setEnabled(True)
        self.model_select_btn.setEnabled(True)
        self.run_analysis_btn.setEnabled(True)
        self.threshold_slider.setEnabled(True)
        self.threshold_spinbox.setEnabled(True)
        self.option1_radio.setEnabled(True)
        self.option2_radio.setEnabled(True)
        
        if result.get("error"):
            self.status.setText(f"❌ Error: {result['error']}")
            self.summary.setText(f"Error: {result['error']}")
            QMessageBox.critical(self, "Analysis Error", result['error'])
            return

        self.status.setText("✅ Analysis Complete!")
        
        dup_tab = QWidget()
        layout = QVBoxLayout(dup_tab)

        if result["duplicates"]:
            count = len(result["duplicates"])
            btn = QPushButton(f"👁 Inspect {count} Duplicate Pairs")
            btn.setMinimumHeight(60)
            btn.setFont(QFont("Arial", 16, QFont.Bold))
            btn.setObjectName("Danger")
            btn.clicked.connect(lambda: self.open_cluster_viewer(result["duplicates"], result.get("crop_data", {}), result.get("embedding_mode", "option1")))
            layout.addWidget(btn)
            
            preview = QTextEdit()
            preview.setReadOnly(True)
            preview.setMaximumHeight(120)
            text = "Top Similar Pairs:\n"
            for i, (p1, p2, score) in enumerate(result["duplicates"][:5]):
                text += f"{i+1}. {score:.3f} | {os.path.basename(p1)} ↔ {os.path.basename(p2)}\n"
            preview.setText(text)
            layout.addWidget(preview)
        else:
            layout.addWidget(QLabel("✅ No duplicates found above threshold."))
            layout.addStretch()

        self.tabs.clear()
        self.tabs.addTab(dup_tab, "Duplicate Analysis")
        threshold_pct = self.threshold_spinbox.value()
        self.summary.setText(
            f"Model: {result.get('model_info', 'Unknown')}\n"
            f"Total Images: {result['total_images']}\n"
            f"Threshold: {threshold_pct}%\n"
            f"Duplicates Found: {len(result['duplicates'])}"
        )

    def open_cluster_viewer(self, duplicates, crop_data=None, embedding_mode="option1"):
        if self.viewer and self.viewer.isVisible():
            self.viewer.activateWindow()
            return
        self.viewer = ClusterViewer(duplicates, crop_data, embedding_mode)
        self.viewer.show()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                'Confirm Close',
                'Analysis is still running. Close anyway?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = YOLOInspector()
    window.show()
    
    sys.exit(app.exec_())