import sys
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QTabWidget, QProgressBar, 
    QFileDialog, QTextEdit, QSizePolicy, QMessageBox, 
    QSpinBox, QGroupBox, QScrollArea, QFrame, QCheckBox,
    QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QRadioButton, QButtonGroup, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import shutil

try:
    import open_clip
except ImportError:
    print("Warning: open_clip not found. Please install it: pip install open-clip-torch")
    open_clip = None


# -----------------------------
# Worker thread for embedding and clustering
# -----------------------------
class ClusteringWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    status_update = pyqtSignal(str)

    def __init__(self, image_paths, target_count, model_path=None, batch_size=16):
        super().__init__()
        self.image_paths = image_paths
        self.target_count = target_count
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = None

    def run(self):
        result = {}
        self.progress.emit(0)
        
        if self.model_path and os.path.exists(self.model_path):
            self.status_update.emit(f"Loading local model from: {self.model_path}")
            model_source = "local"
        else:
            self.status_update.emit("Loading CLIP model (download if needed)...")
            model_source = "remote"

        if open_clip is None:
            self.finished.emit({"error": "open_clip not installed"})
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.status_update.emit(f"Using device: {self.device}")
            
            if model_source == "local":
                if os.path.isdir(self.model_path):
                    model_file = os.path.join(self.model_path, "open_clip_pytorch_model.bin")
                    config_file = os.path.join(self.model_path, "open_clip_config.json")
                    
                    if not os.path.exists(model_file):
                        raise FileNotFoundError(f"Model file not found: {model_file}")
                    if not os.path.exists(config_file):
                        raise FileNotFoundError(f"Config file not found: {config_file}")
                    
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        'ViT-B-32',
                        pretrained=None,
                        device=self.device
                    )
                    
                    checkpoint = torch.load(model_file, map_location=self.device)
                    if 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                        
                elif os.path.isfile(self.model_path) and self.model_path.endswith(('.pt', '.bin')):
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        'ViT-B-32',
                        pretrained=None,
                        device=self.device
                    )
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    if 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    raise FileNotFoundError(f"Invalid model path: {self.model_path}")
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained='laion2b_s34b_b79k',
                    device=self.device
                )
            
            model.eval()
        except Exception as e:
            self.finished.emit({"error": f"Model loading failed: {str(e)}"})
            return

        embeddings = []
        valid_paths = []

        self.status_update.emit(f"Embedding {len(self.image_paths)} images...")
        total_steps = len(self.image_paths)
        
        for i in range(0, len(self.image_paths), self.batch_size):
            batch_paths = self.image_paths[i:i+self.batch_size]
            batch_tensors = []

            for path in batch_paths:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                from PIL import Image
                img = Image.fromarray(img)
                tensor = preprocess(img)
                batch_tensors.append(tensor)

            if len(batch_tensors) == 0:
                continue

            batch_tensor = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                batch_emb = model.encode_image(batch_tensor)
                batch_emb = batch_emb.float()
                batch_emb = batch_emb / batch_emb.norm(dim=1, keepdim=True)

            embeddings.append(batch_emb.cpu().numpy())
            valid_paths.extend(batch_paths)

            progress_pct = int((i / total_steps) * 40)
            self.progress.emit(progress_pct)

        if len(embeddings) == 0:
            self.finished.emit({"error": "No valid images found"})
            return

        embeddings = np.vstack(embeddings).astype("float32")
        self.status_update.emit(f"Embedded {len(valid_paths)} images successfully")

        actual_clusters = min(self.target_count, len(valid_paths))
        self.status_update.emit(f"Running K-Means with {actual_clusters} clusters...")

        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        self.progress.emit(60)
        self.status_update.emit("Finding representative images...")

        representatives = []
        cluster_info = {}

        for cluster_id in range(actual_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = kmeans.cluster_centers_[cluster_id]
            
            closest_idx_in_cluster = pairwise_distances_argmin_min(
                [centroid], cluster_embeddings
            )[0][0]
            
            closest_global_idx = cluster_indices[closest_idx_in_cluster]
            representatives.append({
                'path': valid_paths[closest_global_idx],
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_indices),
                'distance_to_centroid': float(pairwise_distances_argmin_min(
                    [centroid], cluster_embeddings
                )[1][0])
            })

            cluster_info[cluster_id] = {
                'all_paths': [valid_paths[idx] for idx in cluster_indices],
                'representative': valid_paths[closest_global_idx],
                'size': len(cluster_indices)
            }

        self.progress.emit(80)
        self.status_update.emit("Calculating statistics...")

        cluster_sizes = [cluster_info[c]['size'] for c in range(actual_clusters)]
        kept_paths = set([rep['path'] for rep in representatives])
        removed_paths = [path for path in valid_paths if path not in kept_paths]

        self.progress.emit(100)
        self.status_update.emit("Clustering complete!")

        result = {
            'total_images': len(valid_paths),
            'target_count': self.target_count,
            'actual_clusters': actual_clusters,
            'representatives': representatives,
            'cluster_info': cluster_info,
            'kept_paths': list(kept_paths),
            'removed_paths': removed_paths,
            'cluster_sizes': cluster_sizes,
            'all_paths': valid_paths,
            'embeddings': embeddings,
            'cluster_labels': cluster_labels.tolist()
        }

        self.finished.emit(result)


# -----------------------------
# Cluster Preview Dialog
# -----------------------------
class ClusterPreviewDialog(QDialog):
    def __init__(self, cluster_id, cluster_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Cluster {cluster_id} - {cluster_data['size']} images")
        self.resize(900, 700)
        
        layout = QVBoxLayout()
        
        # Info header
        info_label = QLabel(
            f"<b>Cluster {cluster_id}</b><br>"
            f"Total images: {cluster_data['size']}<br>"
            f"Representative: {os.path.basename(cluster_data['representative'])}"
        )
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #e3f2fd; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # Scrollable grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        grid_widget = QWidget()
        grid_layout = QVBoxLayout()
        
        # Representative image (highlighted)
        rep_frame = QFrame()
        rep_frame.setStyleSheet("border: 3px solid #4CAF50; background-color: #e8f5e9; border-radius: 5px;")
        rep_layout = QHBoxLayout()
        
        rep_label = QLabel()
        pixmap = self.load_pixmap(cluster_data['representative'])
        if pixmap:
            rep_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
        rep_label.setAlignment(Qt.AlignCenter)
        rep_label.setMinimumSize(200, 200)
        
        rep_info = QLabel(
            f"✓ <b>REPRESENTATIVE</b><br>"
            f"{os.path.basename(cluster_data['representative'])}<br>"
            f"<i>(This image will be KEPT)</i>"
        )
        rep_info.setStyleSheet("color: #2e7d32; font-size: 13px; font-weight: bold;")
        rep_info.setWordWrap(True)
        
        rep_layout.addWidget(rep_label)
        rep_layout.addWidget(rep_info)
        rep_frame.setLayout(rep_layout)
        grid_layout.addWidget(rep_frame)
        
        # Other images in cluster
        other_label = QLabel(f"<b>Other {cluster_data['size'] - 1} images in this cluster (will be removed):</b>")
        other_label.setStyleSheet("font-size: 13px; font-weight: bold; margin-top: 15px;")
        grid_layout.addWidget(other_label)
        
        # Grid for other images
        row_layout = QHBoxLayout()
        cols = 5
        count = 0
        
        for path in cluster_data['all_paths']:
            if path == cluster_data['representative']:
                continue
            
            img_label = QLabel()
            pixmap = self.load_pixmap(path)
            if pixmap:
                img_label.setPixmap(pixmap.scaled(120, 120, Qt.KeepAspectRatio))
            img_label.setToolTip(path)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setMinimumSize(120, 120)
            img_label.setStyleSheet("border: 1px solid #f44336; border-radius: 3px;")
            
            row_layout.addWidget(img_label)
            count += 1
            
            if count % cols == 0:
                grid_layout.addLayout(row_layout)
                row_layout = QHBoxLayout()
        
        if row_layout.count() > 0:
            grid_layout.addLayout(row_layout)
        
        grid_widget.setLayout(grid_layout)
        scroll.setWidget(grid_widget)
        layout.addWidget(scroll)
        
        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_close.setStyleSheet("padding: 10px; font-size: 14px;")
        layout.addWidget(btn_close)
        
        self.setLayout(layout)
    
    def load_pixmap(self, path):
        if os.path.exists(path):
            return QPixmap(path)
        return None


# -----------------------------
# Reduction Viewer (3 Tabs)
# -----------------------------
class ReductionViewer(QWidget):
    def __init__(self, result_data):
        super().__init__()
        self.result = result_data
        self.current_cluster_id = None
        
        layout = QVBoxLayout()
        
        # Stats Summary
        stats_box = QGroupBox("📊 Summary Statistics")
        stats_layout_inner = QVBoxLayout()
        
        reduction_pct = 100 - (len(result_data['kept_paths']) / result_data['total_images'] * 100)
        self.stats_label = QLabel(
            f"Total Images: {result_data['total_images']}\n"
            f"Target Count: {result_data['target_count']}\n"
            f"Clusters Created: {result_data['actual_clusters']}\n"
            f"To Keep: {len(result_data['kept_paths'])}\n"
            f"To Remove: {len(result_data['removed_paths'])}\n"
            f"Reduction: {reduction_pct:.1f}%"
        )
        self.stats_label.setStyleSheet("font-size: 14px; padding: 10px;")
        stats_layout_inner.addWidget(self.stats_label)
        stats_box.setLayout(stats_layout_inner)
        layout.addWidget(stats_box)
        
        # Three Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_representatives_tab(), "✅ Images to Keep")
        self.tabs.addTab(self.create_removed_tab(), "❌ Images to Remove")
        self.tabs.addTab(self.create_cluster_browser_tab(), "📁 Browse Clusters")
        layout.addWidget(self.tabs)
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_export = QPushButton("📥 Export Selection")
        self.btn_export.clicked.connect(self.export_selection)
        self.btn_export.setStyleSheet("padding: 10px; font-size: 14px; background-color: #4CAF50; color: white; font-weight: bold;")
        
        self.btn_copy = QPushButton("📋 Copy Paths to Clipboard")
        self.btn_copy.clicked.connect(self.copy_paths)
        self.btn_copy.setStyleSheet("padding: 10px; font-size: 14px;")
        
        btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(self.btn_copy)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def create_representatives_tab(self):
        """Tab 1: Images to Keep (Representatives)"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        info = QLabel(f"Showing {len(self.result['representatives'])} representative images (one per cluster)")
        info.setStyleSheet("font-weight: bold; padding: 5px; color: #2e7d32;")
        layout.addWidget(info)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        grid_widget = QWidget()
        grid_layout = QVBoxLayout()
        
        row_layout = QHBoxLayout()
        cols = 4
        
        reps_sorted = sorted(self.result['representatives'], 
                           key=lambda x: x['cluster_size'], reverse=True)
        
        for i, rep in enumerate(reps_sorted):
            frame = QFrame()
            frame.setStyleSheet("border: 2px solid #4CAF50; border-radius: 5px; background-color: #e8f5e9;")
            frame_layout = QVBoxLayout()
            
            img_label = QLabel()
            pixmap = self.load_pixmap(rep['path'])
            if pixmap:
                img_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setMinimumSize(150, 150)
            frame_layout.addWidget(img_label)
            
            info_text = (f"📊 Cluster {rep['cluster_id']}\n"
                        f"👥 Size: {rep['cluster_size']} images\n"
                        f"📏 Dist: {rep['distance_to_centroid']:.3f}\n"
                        f"📄 {os.path.basename(rep['path'])}")
            info_label = QLabel(info_text)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setWordWrap(True)
            info_label.setStyleSheet("font-size: 11px; padding: 5px;")
            frame_layout.addWidget(info_label)
            
            frame.setLayout(frame_layout)
            row_layout.addWidget(frame)
            
            if (i + 1) % cols == 0:
                grid_layout.addLayout(row_layout)
                row_layout = QHBoxLayout()
        
        if row_layout.count() > 0:
            grid_layout.addLayout(row_layout)
        
        grid_widget.setLayout(grid_layout)
        scroll.setWidget(grid_widget)
        layout.addWidget(scroll)
        
        tab.setLayout(layout)
        return tab
    
    def create_removed_tab(self):
        """Tab 2: Images to Remove"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        info = QLabel(f"Showing {len(self.result['removed_paths'])} images that would be removed")
        info.setStyleSheet("font-weight: bold; padding: 5px; color: #f44336;")
        layout.addWidget(info)
        
        # Search box
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("🔍 Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to filter filenames...")
        self.search_input.textChanged.connect(self.filter_removed_list)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        # List view
        self.removed_list = QListWidget()
        
        for path in self.result['removed_paths']:
            item = QListWidgetItem(f"📄 {os.path.basename(path)}")
            item.setToolTip(path)
            item.setData(Qt.UserRole, path)
            self.removed_list.addItem(item)
        
        self.removed_list.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.removed_list)
        
        # Count label
        self.removed_count_label = QLabel(f"Total: {len(self.result['removed_paths'])} files")
        self.removed_count_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.removed_count_label)
        
        tab.setLayout(layout)
        return tab
    
    def create_cluster_browser_tab(self):
        """Tab 3: Browse All Clusters"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        info = QLabel("Browse all clusters and see which images belong to each")
        info.setStyleSheet("font-weight: bold; padding: 5px; color: #1976D2;")
        layout.addWidget(info)
        
        # Split view: Cluster list + Preview
        split_layout = QHBoxLayout()
        
        # Left: Cluster list
        list_box = QGroupBox("Clusters")
        list_layout = QVBoxLayout()
        
        self.cluster_list = QListWidget()
        self.cluster_list.itemClicked.connect(self.on_cluster_clicked)
        
        for cluster_id in sorted(self.result['cluster_info'].keys()):
            data = self.result['cluster_info'][cluster_id]
            item_text = f"📁 Cluster {cluster_id}: {data['size']} images"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, cluster_id)
            self.cluster_list.addItem(item)
        
        list_layout.addWidget(self.cluster_list)
        list_box.setLayout(list_layout)
        split_layout.addWidget(list_box, 1)
        
        # Right: Preview area
        preview_box = QGroupBox("Cluster Preview")
        preview_layout = QVBoxLayout()
        
        self.cluster_preview_label = QLabel("Select a cluster to preview")
        self.cluster_preview_label.setAlignment(Qt.AlignCenter)
        self.cluster_preview_label.setStyleSheet("border: 2px dashed #ccc; padding: 20px; background-color: #fafafa;")
        self.cluster_preview_label.setMinimumSize(300, 300)
        preview_layout.addWidget(self.cluster_preview_label)
        
        self.cluster_info_label = QLabel("")
        self.cluster_info_label.setAlignment(Qt.AlignCenter)
        self.cluster_info_label.setWordWrap(True)
        self.cluster_info_label.setStyleSheet("font-size: 12px; padding: 10px;")
        preview_layout.addWidget(self.cluster_info_label)
        
        self.btn_preview_cluster = QPushButton("Open Detailed Preview")
        self.btn_preview_cluster.clicked.connect(self.open_cluster_preview)
        self.btn_preview_cluster.setEnabled(False)
        self.btn_preview_cluster.setStyleSheet("padding: 10px; font-size: 14px;")
        preview_layout.addWidget(self.btn_preview_cluster)
        
        preview_box.setLayout(preview_layout)
        split_layout.addWidget(preview_box, 2)
        
        layout.addLayout(split_layout)
        tab.setLayout(layout)
        return tab
    
    def on_cluster_clicked(self, item):
        cluster_id = item.data(Qt.UserRole)
        self.current_cluster_id = cluster_id
        self.btn_preview_cluster.setEnabled(True)
        
        data = self.result['cluster_info'][cluster_id]
        rep_path = data['representative']
        
        pixmap = self.load_pixmap(rep_path)
        if pixmap:
            self.cluster_preview_label.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio))
        else:
            self.cluster_preview_label.setText(f"Cluster {cluster_id}\n{data['size']} images")
        
        self.cluster_info_label.setText(
            f"<b>Cluster {cluster_id}</b><br>"
            f"Images: {data['size']}<br>"
            f"Representative: {os.path.basename(rep_path)}"
        )
    
    def open_cluster_preview(self):
        if hasattr(self, 'current_cluster_id'):
            cluster_id = self.current_cluster_id
            data = self.result['cluster_info'][cluster_id]
            dialog = ClusterPreviewDialog(cluster_id, data, self)
            dialog.exec_()
    
    def filter_removed_list(self, text):
        """Filter removed images list based on search text"""
        for i in range(self.removed_list.count()):
            item = self.removed_list.item(i)
            if text.lower() in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)
        
        visible_count = sum(1 for i in range(self.removed_list.count()) if not self.removed_list.item(i).isHidden())
        self.removed_count_label.setText(f"Showing: {visible_count} / {len(self.result['removed_paths'])} files")
    
    def load_pixmap(self, path):
        if os.path.exists(path):
            return QPixmap(path)
        return None
    
    def export_selection(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        
        kept_folder = os.path.join(folder, "kept_images")
        removed_folder = os.path.join(folder, "removed_images")
        
        os.makedirs(kept_folder, exist_ok=True)
        
        copied = 0
        errors = 0
        
        for path in self.result['kept_paths']:
            try:
                dest = os.path.join(kept_folder, os.path.basename(path))
                shutil.copy2(path, dest)
                copied += 1
            except Exception as e:
                errors += 1
                print(f"Error copying {path}: {e}")
        
        # Ask if user wants to copy removed images too
        copy_removed = QMessageBox.question(
            self, "Copy Removed Images?",
            f"Would you like to also copy the {len(self.result['removed_paths'])} removed images to a separate folder?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if copy_removed == QMessageBox.Yes:
            os.makedirs(removed_folder, exist_ok=True)
            for path in self.result['removed_paths']:
                try:
                    dest = os.path.join(removed_folder, os.path.basename(path))
                    shutil.copy2(path, dest)
                except Exception as e:
                    errors += 1
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_original': self.result['total_images'],
            'kept_count': len(self.result['kept_paths']),
            'removed_count': len(self.result['removed_paths']),
            'clusters': self.result['actual_clusters'],
            'kept_paths': [os.path.basename(p) for p in self.result['kept_paths']],
            'removed_paths': [os.path.basename(p) for p in self.result['removed_paths']],
            'cluster_info': {
                str(k): {
                    'size': v['size'],
                    'representative': os.path.basename(v['representative'])
                }
                for k, v in self.result['cluster_info'].items()
            }
        }
        
        metadata_file = os.path.join(folder, "selection_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        QMessageBox.information(
            self, "Export Complete",
            f"Exported {copied} images to:\n{folder}\n\n"
            f"Errors: {errors}\n"
            f"Metadata saved to: selection_metadata.json"
        )
    
    def copy_paths(self):
        clipboard = QApplication.clipboard()
        paths_text = "\n".join(self.result['kept_paths'])
        clipboard.setText(paths_text)
        
        QMessageBox.information(
            self, "Copied to Clipboard",
            f"Copied {len(self.result['kept_paths'])} file paths to clipboard"
        )


# -----------------------------
# Main Application
# -----------------------------
class ImageReducerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Dataset Reducer - K-Means Clustering")
        self.resize(900, 700)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        # Model Settings
        model_group = QGroupBox("🤖 Model Settings")
        model_layout = QVBoxLayout()
        
        self.model_btn_group = QButtonGroup(self)
        self.radio_remote = QRadioButton("🌐 Download from Internet (Standard)")
        self.radio_local = QRadioButton("📂 Use Local Model (Offline)")
        self.model_btn_group.addButton(self.radio_remote, 1)
        self.model_btn_group.addButton(self.radio_local, 2)
        self.radio_remote.setChecked(True)
        
        model_layout.addWidget(self.radio_remote)
        model_layout.addWidget(self.radio_local)
        
        local_path_layout = QHBoxLayout()
        self.local_path_input = QLineEdit()
        self.local_path_input.setPlaceholderText("Path to model file or folder")
        self.local_path_input.setEnabled(False)
        
        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.setEnabled(False)
        self.browse_model_btn.clicked.connect(self.browse_model)
        
        local_path_layout.addWidget(self.local_path_input)
        local_path_layout.addWidget(self.browse_model_btn)
        model_layout.addLayout(local_path_layout)
        
        self.model_help = QLabel("💡 Supports: .pt file, .bin file, or folder with config + weights")
        self.model_help.setStyleSheet("color: #666; font-size: 12px;")
        self.model_help.setWordWrap(True)
        model_layout.addWidget(self.model_help)
        
        model_group.setLayout(model_layout)
        self.layout.addWidget(model_group)

        self.radio_local.toggled.connect(self.on_model_source_changed)

        # Folder selection
        self.select_btn = QPushButton("📁 Select Image Folder")
        self.select_btn.clicked.connect(self.select_folder)
        self.layout.addWidget(self.select_btn)

        # Target count
        target_group = QGroupBox("Target Settings")
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Number of Images:"))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(1, 10000)
        self.target_spin.setValue(100)
        target_layout.addWidget(self.target_spin)
        
        self.start_btn = QPushButton("🚀 Start Clustering")
        self.start_btn.clicked.connect(self.start_clustering)
        self.start_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #2196F3; color: white;")
        target_layout.addWidget(self.start_btn)
        target_group.setLayout(target_layout)
        self.layout.addWidget(target_group)

        # Status
        self.status = QLabel("Status: Ready")
        self.status.setWordWrap(True)
        self.layout.addWidget(self.status)

        # Progress
        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)

        self.image_paths = []
        self.viewer = None
        self.result_data = None

    def on_model_source_changed(self):
        is_local = self.radio_local.isChecked()
        self.local_path_input.setEnabled(is_local)
        self.browse_model_btn.setEnabled(is_local)
        self.model_help.setVisible(is_local)

    def browse_model(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Select Model Type")
        msg.setText("What type of model file do you have?")
        msg.setInformativeText(
            "• Select 'Model File' if you have a .pt or .bin file\n"
            "• Select 'Model Folder' if you have a folder with config + weights"
        )
        
        btn_file = QPushButton("📄 Model File (.pt/.bin)")
        btn_folder = QPushButton("📁 Model Folder")
        btn_cancel = QPushButton("Cancel")
        
        msg.addButton(btn_file, QMessageBox.AcceptRole)
        msg.addButton(btn_folder, QMessageBox.AcceptRole)
        msg.addButton(btn_cancel, QMessageBox.RejectRole)
        
        msg.exec_()
        
        clicked_btn = msg.clickedButton()
        
        if clicked_btn == btn_file:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Model File", "", 
                "Model Files (*.pt *.bin);;All Files (*)"
            )
            if file_path:
                self.local_path_input.setText(file_path)
                self.status.setText(f"Model file selected: {os.path.basename(file_path)}")
                
        elif clicked_btn == btn_folder:
            folder = QFileDialog.getExistingDirectory(self, "Select Model Folder")
            if folder:
                self.local_path_input.setText(folder)
                self.status.setText(f"Model folder selected: {folder}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_paths = self.import_images(folder)
            self.status.setText(f"Found {len(self.image_paths)} images in: {folder}")

    def import_images(self, folder):
        images = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    images.append(os.path.join(root, f))
        return images

    def start_clustering(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please select a folder with images first.")
            return
        
        model_path = None
        if self.radio_local.isChecked():
            model_path = self.local_path_input.text().strip()
            
            if not model_path:
                QMessageBox.critical(self, "Invalid Model Path", "Please specify a valid local model path.")
                return
            
            if not os.path.exists(model_path):
                QMessageBox.critical(self, "Invalid Model Path", f"Path does not exist:\n{model_path}")
                return
            
            if os.path.isfile(model_path):
                if not model_path.endswith(('.pt', '.bin')):
                    QMessageBox.warning(self, "Warning", 
                        f"Selected file may not be a valid model:\n{model_path}\n\nExpected: .pt or .bin file")
            elif os.path.isdir(model_path):
                model_file = os.path.join(model_path, "open_clip_pytorch_model.bin")
                config_file = os.path.join(model_path, "open_clip_config.json")
                
                if not os.path.exists(model_file):
                    QMessageBox.warning(self, "Warning", 
                        f"Model folder missing required file:\n{model_file}\n\nExpected: open_clip_pytorch_model.bin")
                if not os.path.exists(config_file):
                    QMessageBox.warning(self, "Warning", 
                        f"Model folder missing required file:\n{config_file}\n\nExpected: open_clip_config.json")

        target = self.target_spin.value()
        if target >= len(self.image_paths):
            if not QMessageBox.warning(self, "Target Too High", 
                f"Target ({target}) >= Total ({len(self.image_paths)}). Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
                return

        self.status.setText(f"Starting clustering (Model: {'Local' if model_path else 'Remote'})...")
        self.progress.setValue(0)

        self.worker = ClusteringWorker(self.image_paths, target, model_path=model_path)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status_update.connect(self.status.setText)
        self.worker.finished.connect(self.clustering_done)
        self.worker.start()
        
    def clustering_done(self, result):
        if 'error' in result:
            QMessageBox.critical(self, "Error", f"Clustering failed: {result['error']}")
            return

        self.result_data = result
        self.status.setText("Clustering complete!")
        self.progress.setValue(100)
        
        if self.viewer:
            self.viewer.close()
        self.viewer = ReductionViewer(result)
        viewer_window = QMainWindow()
        viewer_window.setWindowTitle("Clustering Results")
        viewer_window.setCentralWidget(self.viewer)
        viewer_window.resize(1200, 800)
        viewer_window.show()
        self.viewer_window = viewer_window
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageReducerApp()
    window.show()
    sys.exit(app.exec_())