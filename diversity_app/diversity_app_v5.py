import sys
import os
import numpy as np
from PIL import Image
import io
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QRadioButton, QButtonGroup, QComboBox, QProgressBar, 
                             QTextEdit, QGroupBox, QSpinBox, QFrame, QStatusBar,
                             QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from style import MODERN_STYLESHEET
from embedding import EmbeddingWorker
from distribution import DistributionDialog


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
        
        self.split_colors = []
        self.split_names = []
        self.class_names_map = {}  # Store class_id to class_name mapping
        
        # Fullscreen state
        self.is_fullscreen = False
        self.fullscreen_window = None
        
        # Clustering state
        self.clustering_enabled = False
        self.cluster_labels = None
        self.optimal_k = None
        self.kmeans_model = None
        
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
        
        self.viz_method_combo = QComboBox()
        self.viz_method_combo.addItems(["t-SNE", "UMAP", "OpenTSNE"])
        # dataset_layout.addWidget(QLabel("Visualization Method:"))
        dataset_layout.addWidget(self.viz_method_combo)
        
        # Sampling Controls
        sampling_card = QFrame()
        sampling_card.setStyleSheet("QFrame { background-color: #0f3460; border-radius: 10px; }")
        sampling_layout = QVBoxLayout()
        sampling_card.setLayout(sampling_layout)

        self.sampling_check = QCheckBox("Enable Random Sampling")
        self.sampling_check.setChecked(False)
        sampling_layout.addWidget(self.sampling_check)

        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(10, 50000)
        self.sample_spin.setValue(2000)
        self.sample_spin.setEnabled(False)
        sampling_layout.addWidget(self.sample_spin)

        self.sampling_check.toggled.connect(self.sample_spin.setEnabled)
        dataset_layout.addWidget(sampling_card)

        control_layout.addWidget(dataset_card)

        # Run Button
        self.run_btn = QPushButton("RUN ANALYSIS")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self.start_analysis)
        control_layout.addWidget(self.run_btn)
        
        # Load History Button
        self.load_history_btn = QPushButton("Load History")
        self.load_history_btn.setObjectName("distButton")
        self.load_history_btn.clicked.connect(self.load_history)
        control_layout.addWidget(self.load_history_btn)

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
        plot_panel = QGroupBox("Embedding Visualization")
        plot_layout = QHBoxLayout()  # Changed to horizontal layout
        plot_panel.setLayout(plot_layout)
        
        # Left side: Class filter checkboxes
        filter_panel = QGroupBox("Class Filter")
        filter_panel.setObjectName("Class Filter")  # Add object name for finding
        filter_panel.setFixedWidth(200)
        self.filter_layout = QVBoxLayout()
        filter_panel.setLayout(self.filter_layout)
        
        # Select All / Deselect All buttons
        filter_buttons_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("All")
        self.select_all_btn.setMaximumHeight(30)
        self.select_all_btn.clicked.connect(self.select_all_classes)
        self.deselect_all_btn = QPushButton("None")
        self.deselect_all_btn.setMaximumHeight(30)
        self.deselect_all_btn.clicked.connect(self.deselect_all_classes)
        filter_buttons_layout.addWidget(self.select_all_btn)
        filter_buttons_layout.addWidget(self.deselect_all_btn)
        self.filter_layout.addLayout(filter_buttons_layout)
        
        # Scroll area for checkboxes
        from PyQt5.QtWidgets import QScrollArea
        self.filter_scroll = QScrollArea()
        self.filter_scroll.setWidgetResizable(True)
        self.filter_scroll.setMaximumHeight(400)
        self.filter_widget = QWidget()
        self.filter_checkboxes_layout = QVBoxLayout()
        self.filter_widget.setLayout(self.filter_checkboxes_layout)
        self.filter_scroll.setWidget(self.filter_widget)
        self.filter_layout.addWidget(self.filter_scroll)
        
        self.class_checkboxes = {}  # Store checkboxes by class_id
        
        plot_layout.addWidget(filter_panel)
        
        # Right side: Plot area
        plot_area = QWidget()
        plot_area_layout = QVBoxLayout()
        plot_area.setLayout(plot_area_layout)
        
        self.fig = Figure(figsize=(12, 9), dpi=100, facecolor='#1a1a2e')
        self.plot_canvas = FigureCanvas(self.fig)
        self.plot_canvas.setMinimumWidth(800)
        self.plot_canvas.setMinimumHeight(500)
        self.plot_canvas.mpl_connect('pick_event', self.on_point_click)
        self.plot_canvas.mouseDoubleClickEvent = self.on_plot_double_click  # Add double click handler
        
        self.nav_toolbar = NavigationToolbar(self.plot_canvas, self)
        self.nav_toolbar.setStyleSheet("""
            QToolBar { background-color: #0f3460; border: none; border-radius: 8px; padding: 5px; }
            QToolBar QToolButton { background-color: #1a1a2e; border: 1px solid #0f3460; border-radius: 5px; color: #eaeaea; }
            QToolBar QToolButton:hover { background-color: #00d9ff; color: #1a1a2e; }
        """)
        plot_area_layout.addWidget(self.nav_toolbar)
        plot_area_layout.addWidget(self.plot_canvas)
        
        # Distribution Button
        self.dist_btn = QPushButton("View Distribution")
        self.dist_btn.setObjectName("distButton")
        self.dist_btn.clicked.connect(self.show_distribution_dialog)
        self.dist_btn.setEnabled(False)
        plot_area_layout.addWidget(self.dist_btn)
        
        # Clustering Button
        self.cluster_btn = QPushButton("K-Means Clustering")
        self.cluster_btn.setObjectName("distButton")
        self.cluster_btn.clicked.connect(self.toggle_clustering)
        self.cluster_btn.setEnabled(False)
        plot_area_layout.addWidget(self.cluster_btn)
        
        self.plot_info = QLabel("Click points to view images")
        self.plot_info.setStyleSheet("color: #888888; font-size: 12px; padding: 10px;")
        self.plot_info.setAlignment(Qt.AlignCenter)
        plot_area_layout.addWidget(self.plot_info)
        
        plot_layout.addWidget(plot_area)

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
    
    def load_history(self):
        """Load embeddings or final results from history"""
        history_dir = "./history"
        if not os.path.exists(history_dir):
            self.log("No history directory found.")
            return
        
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load History File",
            history_dir,
            "Pickle Files (*.pkl)"
        )
        
        if path:
            self.log(f"Loading from: {os.path.basename(path)}")
            self.start_analysis_from_history(path)

    def start_analysis(self):
        """Start normal analysis from scratch"""
        self._start_analysis_internal(load_from_file=None)
    
    def start_analysis_from_history(self, history_file_path):
        """Start analysis from history file"""
        self._start_analysis_internal(load_from_file=history_file_path)
    
    def _start_analysis_internal(self, load_from_file=None):
        """Internal method to start analysis with optional history loading"""
        if not load_from_file and not hasattr(self, 'yaml_file_path'):
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
        viz_method = self.viz_method_combo.currentText().lower().replace("-", " ")
        
        clip_weights = getattr(self, "clip_weights_path", None)
        
        # Get sampling parameters from UI
        sampling_enabled = hasattr(self, 'sampling_check') and self.sampling_check.isChecked()
        sample_size = getattr(self, 'sample_spin', None).value() if sampling_enabled else None
        
        # Set multiprocessing parameters (hardcoded)
        use_multiprocessing = True  # Always enable multiprocessing
        num_processes = 8  # Fixed to 8 cores
        
        # Use dummy values for yaml_path if loading from history
        yaml_path = getattr(self, 'yaml_file_path', '') if not load_from_file else ''
        
        self.worker = EmbeddingWorker(mode, yolo_weights, yolo_model, 
                                    yaml_path, clip_model, 
                                    sampling_enabled, sample_size, clip_weights, viz_method,
                                    use_multiprocessing, num_processes, load_from_file)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.error_signal.connect(self.analysis_error)
        self.worker.start()

    def update_progress(self, msg, val):
        self.log(msg)
        self.progress_bar.setValue(val)

    def analysis_finished(self, coords, paths, class_ids, image_dir, class_names_map):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN ANALYSIS")
        self.dist_btn.setEnabled(False)
        
        self.tsne_coordinates = coords
        self.image_paths = paths
        self.image_splits = class_ids  # Now contains class_ids instead of split labels
        self.image_dir = image_dir
        self.class_names_map = class_names_map or {}  # Store class names mapping
        
        # Get unique class IDs that are actually present in the data
        unique_classes = np.unique(class_ids)
        
        # Create mapping from class_id to index position
        self.class_id_to_index = {int(c): i for i, c in enumerate(unique_classes)}
        
        # Build split_names list indexed by position (not class_id)
        self.split_names = [self.class_names_map.get(int(c), f"Class {c}") for c in unique_classes]
        
        # Generate colors for each unique class (COMPATIBLE with all matplotlib versions)
        import matplotlib.cm as cm
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_classes)))
        self.split_colors = [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' for c in colors]
        
        # Create class filter checkboxes
        self.create_class_checkboxes(unique_classes)
        
        self.draw_plot()
        self.plot_info.setVisible(False)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        self.dist_btn.setEnabled(True)
        self.cluster_btn.setEnabled(True)
        self.log("Analysis Complete!")

    def analysis_error(self, err_msg):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN ANALYSIS")
        self.log(f"ERROR: {err_msg}")

    def show_distribution_dialog(self):
        if self.tsne_coordinates is None:
            return

        # Get unique class IDs for histogram
        unique_classes = np.unique(self.image_splits)

        dialog = DistributionDialog(
            self.tsne_coordinates, 
            self.image_splits, 
            self.split_colors, 
            self.split_names, 
            self,
            self.class_names_map  # Pass class names mapping
        )
        dialog.exec_()

    def draw_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#16213e')
        
        splits = self.image_splits
        
        # Get visible classes based on checkboxes
        visible_classes = self.get_visible_classes()
        if not visible_classes:
            # If no classes are visible, show empty plot
            ax.set_xlabel('Dimension 1', color='#00d9ff')
            ax.set_ylabel('Dimension 2', color='#00d9ff')
            ax.tick_params(colors='#eaeaea')
            self.fig.tight_layout()
            self.plot_canvas.draw_idle()
            return
        
        # Filter data to show only visible classes
        visible_mask = np.isin(splits, visible_classes)
        visible_coords = self.tsne_coordinates[visible_mask]
        visible_splits = splits[visible_mask]
        
        if len(visible_coords) == 0:
            # No data to show
            ax.set_xlabel('Dimension 1', color='#00d9ff')
            ax.set_ylabel('Dimension 2', color='#00d9ff')
            ax.tick_params(colors='#eaeaea')
            self.fig.tight_layout()
            self.plot_canvas.draw_idle()
            return
        
        # Draw clustering areas first (behind points)
        if self.clustering_enabled and self.cluster_labels is not None:
            self.draw_cluster_areas(ax, visible_coords, visible_mask)
        
        # Convert class_ids to position indices for coloring (only for visible classes)
        split_indices = np.array([self.class_id_to_index.get(int(c), 0) for c in visible_splits])
        
        self.scatter_plot = ax.scatter(
            visible_coords[:, 0], 
            visible_coords[:, 1], 
            s=30, alpha=0.8, c=split_indices,  # Always use class colors
            cmap=ListedColormap(self.split_colors), 
            edgecolors='#ffffff', linewidths=0.5,
            picker=5
        )
        
        # Always show class legend (not cluster legend)
        visible_unique_splits = np.unique(visible_splits)
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                label=self.split_names[self.class_id_to_index.get(int(i), 0)],
                                markerfacecolor=self.split_colors[self.class_id_to_index.get(int(i), 0)], 
                                markersize=8) 
                        for i in visible_unique_splits]
        
        # Store visible data for point clicking
        self.visible_mask = visible_mask
        self.visible_indices = np.where(visible_mask)[0]
        
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
        
        # Map clicked point back to original image index
        visible_index = int(ind[0])
        if hasattr(self, 'visible_indices') and visible_index < len(self.visible_indices):
            self.current_image_index = self.visible_indices[visible_index]
        else:
            self.current_image_index = visible_index
        
        self.highlight_selected_point()
        self.display_current_image()
        
        class_id = int(self.image_splits[self.current_image_index])
        class_name = self.class_names_map.get(class_id, f"Class {class_id}")
        self.statusBar.showMessage(f"Image {self.current_image_index + 1} | Class: {class_name}")

    def highlight_selected_point(self):
        if self.highlight_artist is not None:
            self.highlight_artist.remove()
        
        if 0 <= self.current_image_index < len(self.tsne_coordinates):
            ax = self.fig.axes[0]
            x, y = self.tsne_coordinates[self.current_image_index]
            self.highlight_artist = ax.scatter([x], [y], s=150, alpha=0.6, 
                                               facecolors='none', edgecolors='#ff0066', 
                                               linewidths=4, zorder=10)
            self.plot_canvas.draw_idle()

    def display_current_image(self):
        if not (0 <= self.current_image_index < len(self.image_paths)):
            return
        
        img_path = self.image_paths[self.current_image_index]
        class_id = int(self.image_splits[self.current_image_index])
        class_name = self.class_names_map.get(class_id, f"Class {class_id}")
        image_info_text = f"{os.path.basename(img_path)} | {class_name}"
        
        # Update both normal and fullscreen image info
        self.selected_image_info.setText(image_info_text)
        if hasattr(self, 'fullscreen_image_info') and self.fullscreen_image_info:
            self.fullscreen_image_info.setText(image_info_text)
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Determine target size based on mode
            if self.is_fullscreen and hasattr(self, 'fullscreen_image_label') and self.fullscreen_image_label:
                max_width, max_height = 380, 300
                target_label = self.fullscreen_image_label
            else:
                max_width, max_height = 500, 400
                target_label = self.image_preview_label
            
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
            
            # Update both normal and fullscreen image labels
            self.image_preview_label.setPixmap(pixmap)
            self.image_preview_label.setText("")
            
            if target_label and target_label != self.image_preview_label:
                target_label.setPixmap(pixmap)
                target_label.setText("")
                
        except Exception as e:
            error_text = f"Error: {e}"
            self.image_preview_label.setText(error_text)
            if hasattr(self, 'fullscreen_image_label') and self.fullscreen_image_label:
                self.fullscreen_image_label.setText(error_text)

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
    
    def create_class_checkboxes(self, unique_classes):
        """Create checkboxes for each class"""
        # Clear existing checkboxes
        for checkbox in self.class_checkboxes.values():
            checkbox.setParent(None)
        self.class_checkboxes.clear()
        
        # Create new checkboxes
        for class_id in sorted(unique_classes):
            class_name = self.class_names_map.get(int(class_id), f"Class {class_id}")
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(True)  # Default: all checked
            checkbox.stateChanged.connect(self.on_class_filter_changed)
            
            # Style the checkbox
            checkbox.setStyleSheet("""
                QCheckBox {
                    color: #eaeaea;
                    font-size: 12px;
                    spacing: 5px;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border-radius: 3px;
                    border: 1px solid #0f3460;
                    background-color: #1a1a2e;
                }
                QCheckBox::indicator:checked {
                    background-color: #00d9ff;
                    border: 1px solid #00d9ff;
                }
            """)
            
            self.class_checkboxes[int(class_id)] = checkbox
            self.filter_checkboxes_layout.addWidget(checkbox)
        
        # Add stretch to push checkboxes to top
        self.filter_checkboxes_layout.addStretch()
    
    def get_visible_classes(self):
        """Get list of class IDs that are currently visible (checked)"""
        visible_classes = []
        for class_id, checkbox in self.class_checkboxes.items():
            if checkbox.isChecked():
                visible_classes.append(class_id)
        return visible_classes
    
    def on_class_filter_changed(self):
        """Called when any class checkbox is changed"""
        self.draw_plot()
    
    def select_all_classes(self):
        """Check all class checkboxes"""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_classes(self):
        """Uncheck all class checkboxes"""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(False)
    
    def toggle_clustering(self):
        """Toggle K-means clustering on/off"""
        if self.clustering_enabled:
            self.disable_clustering()
        else:
            self.enable_clustering()
    
    def enable_clustering(self):
        """Enable K-means clustering with elbow method"""
        if self.tsne_coordinates is None or len(self.tsne_coordinates) < 4:
            self.log("ERROR: Need at least 4 points for clustering")
            return
        
        self.log("Running K-means clustering with elbow method...")
        
        try:
            # Get visible data for clustering
            visible_classes = self.get_visible_classes()
            if not visible_classes:
                self.log("ERROR: No visible classes for clustering")
                return
            
            visible_mask = np.isin(self.image_splits, visible_classes)
            visible_coords = self.tsne_coordinates[visible_mask]
            
            if len(visible_coords) < 4:
                self.log("ERROR: Need at least 4 visible points for clustering")
                return
            
            # Find optimal K using elbow method
            self.optimal_k = self.find_optimal_k(visible_coords)
            self.log(f"Optimal K found: {self.optimal_k}")
            
            # Perform K-means clustering on all data
            self.kmeans_model = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
            self.cluster_labels = np.full(len(self.tsne_coordinates), -1)  # Initialize with -1
            
            # Only cluster visible points
            visible_cluster_labels = self.kmeans_model.fit_predict(visible_coords)
            self.cluster_labels[visible_mask] = visible_cluster_labels
            
            self.clustering_enabled = True
            self.cluster_btn.setText("Disable Clustering")
            
            # Redraw plot with clustering
            self.draw_plot()
            
            self.log(f"Clustering enabled with {self.optimal_k} clusters")
            
        except Exception as e:
            self.log(f"Clustering error: {str(e)}")
    
    def disable_clustering(self):
        """Disable clustering and return to class-based coloring"""
        self.clustering_enabled = False
        self.cluster_labels = None
        self.optimal_k = None
        self.kmeans_model = None
        self.cluster_btn.setText("K-Means Clustering")
        
        # Redraw plot without clustering
        self.draw_plot()
        
        self.log("Clustering disabled")
    
    def find_optimal_k(self, coords, max_k=10):
        """Find optimal number of clusters using elbow method"""
        max_k = min(max_k, len(coords) - 1)
        if max_k < 2:
            return 2
        
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(coords)
            inertias.append(kmeans.inertia_)
        
        # Use KneeLocator to find elbow
        try:
            kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
            optimal_k = kl.elbow
            if optimal_k is None:
                # Fallback: use silhouette score
                optimal_k = self.find_optimal_k_silhouette(coords, max_k)
        except:
            # Fallback: use silhouette score
            optimal_k = self.find_optimal_k_silhouette(coords, max_k)
        
        return optimal_k if optimal_k else 3  # Default fallback
    
    def find_optimal_k_silhouette(self, coords, max_k=10):
        """Fallback method using silhouette score"""
        max_k = min(max_k, len(coords) - 1)
        if max_k < 2:
            return 2
        
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coords)
            score = silhouette_score(coords, cluster_labels)
            silhouette_scores.append(score)
        
        # Return k with highest silhouette score
        best_k_idx = np.argmax(silhouette_scores)
        return k_range[best_k_idx]
    
    def generate_cluster_colors(self, n_clusters):
        """Generate distinct colors for clusters"""
        import matplotlib.cm as cm
        colors = cm.get_cmap('Set3')(np.linspace(0, 1, n_clusters))
        return [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' for c in colors]
    
    def draw_cluster_areas(self, ax, visible_coords, visible_mask):
        """Draw cluster areas as convex hulls or circles"""
        if self.cluster_labels is None:
            return
        
        visible_cluster_labels = self.cluster_labels[visible_mask]
        colors = self.generate_cluster_colors(self.optimal_k)
        
        for cluster_id in range(self.optimal_k):
            cluster_mask = visible_cluster_labels == cluster_id
            if np.sum(cluster_mask) < 3:  # Need at least 3 points for convex hull
                continue
            
            cluster_points = visible_coords[cluster_mask]
            
            try:
                # Draw convex hull
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                
                # Create polygon patch
                polygon = patches.Polygon(hull_points, closed=True, 
                                        facecolor=colors[cluster_id], 
                                        alpha=0.2, edgecolor=colors[cluster_id], 
                                        linewidth=2, linestyle='--')
                ax.add_patch(polygon)
                
            except Exception as e:
                # Fallback: draw circle around cluster center
                center = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - center, axis=1)
                radius = np.max(distances) * 1.1
                
                circle = patches.Circle(center, radius, 
                                      facecolor=colors[cluster_id], 
                                      alpha=0.15, edgecolor=colors[cluster_id], 
                                      linewidth=2, linestyle='--')
                ax.add_patch(circle)
    
    def on_plot_double_click(self, event):
        """Handle double click on plot canvas to toggle fullscreen"""
        self.toggle_fullscreen()
    
    def toggle_fullscreen(self):
        """Toggle between normal and fullscreen view of the plot"""
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()
    
    def enter_fullscreen(self):
        """Enter fullscreen mode for the plot"""
        if self.is_fullscreen:
            return
        
        # Create fullscreen window
        from PyQt5.QtWidgets import QDialog
        from PyQt5.QtCore import Qt
        
        self.fullscreen_window = QDialog(self)
        self.fullscreen_window.setWindowTitle("Embedding Visualization - Fullscreen")
        self.fullscreen_window.setWindowFlags(Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.fullscreen_window.setStyleSheet(MODERN_STYLESHEET)
        
        # Create main layout for fullscreen window
        main_fullscreen_layout = QHBoxLayout()
        self.fullscreen_window.setLayout(main_fullscreen_layout)
        
        # Left side: Filter panel
        filter_panel = self.findChild(QGroupBox, "Class Filter")
        if filter_panel:
            filter_panel.setParent(self.fullscreen_window)
            main_fullscreen_layout.addWidget(filter_panel)
        
        # Center: Plot area
        fullscreen_plot_area = QWidget()
        fullscreen_plot_layout = QVBoxLayout()
        fullscreen_plot_area.setLayout(fullscreen_plot_layout)
        
        # Move plot canvas to fullscreen window
        self.plot_canvas.setParent(self.fullscreen_window)
        fullscreen_plot_layout.addWidget(self.plot_canvas)
        
        # Move toolbar to fullscreen window
        self.nav_toolbar.setParent(self.fullscreen_window)
        fullscreen_plot_layout.addWidget(self.nav_toolbar)
        
        # Move distribution button to fullscreen window
        self.dist_btn.setParent(self.fullscreen_window)
        fullscreen_plot_layout.addWidget(self.dist_btn)
        
        # Move clustering button to fullscreen window
        self.cluster_btn.setParent(self.fullscreen_window)
        fullscreen_plot_layout.addWidget(self.cluster_btn)
        
        main_fullscreen_layout.addWidget(fullscreen_plot_area)
        
        # Right side: Image preview panel
        image_preview_panel = QGroupBox("Image Preview")
        image_preview_panel.setFixedWidth(400)
        image_preview_layout = QVBoxLayout()
        image_preview_panel.setLayout(image_preview_layout)
        
        # Move image preview components to fullscreen
        # Create new image info label for fullscreen
        self.fullscreen_image_info = QLabel("No image selected")
        self.fullscreen_image_info.setObjectName("selectedImageInfo")
        self.fullscreen_image_info.setAlignment(Qt.AlignCenter)
        image_preview_layout.addWidget(self.fullscreen_image_info)
        
        # Create new image preview frame for fullscreen
        self.fullscreen_image_frame = QFrame()
        self.fullscreen_image_frame.setObjectName("imagePreviewFrame")
        fullscreen_image_frame_layout = QVBoxLayout()
        self.fullscreen_image_frame.setLayout(fullscreen_image_frame_layout)
        
        self.fullscreen_image_label = QLabel()
        self.fullscreen_image_label.setMinimumSize(380, 300)
        self.fullscreen_image_label.setAlignment(Qt.AlignCenter)
        self.fullscreen_image_label.setStyleSheet("QLabel { background-color: #1a1a2e; border-radius: 8px; color: #888888; }")
        self.fullscreen_image_label.setText("Click a point\nto view image")
        fullscreen_image_frame_layout.addWidget(self.fullscreen_image_label)
        image_preview_layout.addWidget(self.fullscreen_image_frame)
        
        # Navigation buttons for fullscreen
        fullscreen_nav_layout = QHBoxLayout()
        self.fullscreen_prev_btn = QPushButton("Previous")
        self.fullscreen_prev_btn.clicked.connect(self.show_previous_image)
        self.fullscreen_prev_btn.setEnabled(False)
        self.fullscreen_next_btn = QPushButton("Next")
        self.fullscreen_next_btn.clicked.connect(self.show_next_image)
        self.fullscreen_next_btn.setEnabled(False)
        self.fullscreen_open_folder_btn = QPushButton("Open Folder")
        self.fullscreen_open_folder_btn.clicked.connect(self.open_image_folder)
        self.fullscreen_open_folder_btn.setEnabled(False)
        
        fullscreen_nav_layout.addWidget(self.fullscreen_prev_btn)
        fullscreen_nav_layout.addWidget(self.fullscreen_next_btn)
        fullscreen_nav_layout.addWidget(self.fullscreen_open_folder_btn)
        image_preview_layout.addLayout(fullscreen_nav_layout)
        
        main_fullscreen_layout.addWidget(image_preview_panel)
        
        # Set up fullscreen window
        self.fullscreen_window.showMaximized()
        self.fullscreen_window.closeEvent = self.on_fullscreen_close
        
        # Add double click handler to fullscreen canvas
        self.plot_canvas.mouseDoubleClickEvent = self.on_fullscreen_double_click
        
        # Enable navigation buttons if we have data
        if self.image_paths:
            self.fullscreen_prev_btn.setEnabled(True)
            self.fullscreen_next_btn.setEnabled(True)
            self.fullscreen_open_folder_btn.setEnabled(True)
            
            # Display current image if one is selected
            if self.current_image_index >= 0:
                self.display_current_image()
        
        self.is_fullscreen = True
        self.log("Entered fullscreen mode. Double-click plot to exit.")
    
    def exit_fullscreen(self):
        """Exit fullscreen mode and restore normal view"""
        if not self.is_fullscreen or not self.fullscreen_window:
            return
        
        # Find the original plot panel
        plot_panel = None
        for child in self.findChildren(QGroupBox):
            if child.title() == "Embedding Visualization":
                plot_panel = child
                break
        
        if not plot_panel:
            return
        
        # Get the plot layout (should be HBoxLayout)
        plot_layout = plot_panel.layout()
        
        # Move filter panel back to original location
        filter_panel = self.fullscreen_window.findChild(QGroupBox)
        if filter_panel and filter_panel.title() == "Class Filter":
            filter_panel.setParent(plot_panel)
            plot_layout.insertWidget(0, filter_panel)  # Insert at beginning
        
        # Find or create plot area widget
        plot_area = None
        for i in range(plot_layout.count()):
            widget = plot_layout.itemAt(i).widget()
            if widget and not isinstance(widget, QGroupBox):
                plot_area = widget
                break
        
        if not plot_area:
            # Create new plot area if it doesn't exist
            plot_area = QWidget()
            plot_area_layout = QVBoxLayout()
            plot_area.setLayout(plot_area_layout)
            plot_layout.addWidget(plot_area)
        
        plot_area_layout = plot_area.layout()
        
        # Move components back to original location
        self.nav_toolbar.setParent(plot_area)
        plot_area_layout.addWidget(self.nav_toolbar)
        
        self.plot_canvas.setParent(plot_area)
        plot_area_layout.addWidget(self.plot_canvas)
        
        self.dist_btn.setParent(plot_area)
        plot_area_layout.addWidget(self.dist_btn)
        
        self.cluster_btn.setParent(plot_area)
        plot_area_layout.addWidget(self.cluster_btn)
        
        # Restore original double click handler
        self.plot_canvas.mouseDoubleClickEvent = self.on_plot_double_click
        
        # Clean up fullscreen-specific components
        if hasattr(self, 'fullscreen_image_info'):
            self.fullscreen_image_info = None
        if hasattr(self, 'fullscreen_image_label'):
            self.fullscreen_image_label = None
        if hasattr(self, 'fullscreen_image_frame'):
            self.fullscreen_image_frame = None
        if hasattr(self, 'fullscreen_prev_btn'):
            self.fullscreen_prev_btn = None
        if hasattr(self, 'fullscreen_next_btn'):
            self.fullscreen_next_btn = None
        if hasattr(self, 'fullscreen_open_folder_btn'):
            self.fullscreen_open_folder_btn = None
        
        # Close fullscreen window
        self.fullscreen_window.close()
        self.fullscreen_window = None
        self.is_fullscreen = False
        
        self.log("Exited fullscreen mode.")
    
    def on_fullscreen_double_click(self, event):
        """Handle double click in fullscreen mode"""
        self.exit_fullscreen()
    
    def on_fullscreen_close(self, event):
        """Handle fullscreen window close event"""
        if self.is_fullscreen:
            self.exit_fullscreen()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = DiversityApp()
    window.show()
    sys.exit(app.exec_())
