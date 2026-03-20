import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, QLabel)

from style import MODERN_STYLESHEET


# ============================================================
# DISTRIBUTION MODAL DIALOG
# ============================================================

class DistributionDialog(QDialog):
    def __init__(self, tsne_coords, splits, colors, names, parent=None, class_names_map=None):
        super().__init__(parent)
        self.setWindowTitle("Distribution Analysis")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(MODERN_STYLESHEET)
        self.setModal(True)
        
        self.tsne_coords = tsne_coords
        self.splits = splits
        self.colors = colors
        self.names = names
        self.class_names_map = class_names_map or {}
        
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

        # Get unique class IDs present in the data
        unique_classes = np.unique(self.splits)
        
        for i, class_id in enumerate(unique_classes):
            mask = self.splits == class_id
            if np.sum(mask) > 0:
                # Get class name from mapping
                class_name = self.class_names_map.get(int(class_id), f"Class {class_id}")
                
                # Use color from the dynamic colors list
                color_idx = i % len(self.colors)
                ax.hist(self.tsne_coords[mask, 0], bins=30, alpha=0.6, 
                        label=class_name, color=self.colors[color_idx], 
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
