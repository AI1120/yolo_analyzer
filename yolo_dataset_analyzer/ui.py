from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTabWidget, QStatusBar, QTextEdit,
                             QFileDialog, QProgressBar, QSplitter, QComboBox,
                             QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()

        splitter = QSplitter(Qt.Horizontal)
        
        plots_widget = QWidget()
        plots_layout = QVBoxLayout()
        
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select Class:"))
        self.class_selector = QComboBox()
        self.class_selector.addItem("All Classes")
        self.class_selector.currentTextChanged.connect(self.on_class_changed)
        selector_layout.addWidget(self.class_selector)
        selector_layout.addStretch()
        
        plots_layout.addLayout(selector_layout)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.train = QVBoxLayout()
        self.test = QVBoxLayout()

        layout.addLayout(self.train)
        layout.addLayout(self.test)
        
        plots_layout.addLayout(layout)
        plots_widget.setLayout(plots_layout)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumWidth(350)
        self.summary_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                color: #333333;
            }
        """)
        
        splitter.addWidget(plots_widget)
        splitter.addWidget(self.summary_text)
        splitter.setSizes([700, 350])
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        self.train_df = None
        self.test_df = None
        self.class_names = None
        self.plot_function = None

    def set_plots(self, train_fig, test_fig, summary_text="", train_df=None, 
                  test_df=None, class_names=None, plot_function=None):
        
        if plot_function is not None:
            self.train_df = train_df
            self.test_df = test_df
            self.class_names = class_names
            self.plot_function = plot_function
            
            if class_names:
                self.class_selector.clear()
                self.class_selector.addItem("All Classes")
                for class_id, class_name in sorted(class_names.items()):
                    self.class_selector.addItem(f"{class_name} ({class_id})")
            
            self.class_selector.setVisible(True)
        else:
            self.class_selector.setVisible(False)
        
        self.update_plots(train_fig, test_fig)
        self.summary_text.setText(summary_text)

    def on_class_changed(self, class_text):
        if self.train_df is None or self.plot_function is None:
            return
        
        if class_text == "All Classes":
            train_filtered = self.train_df
            test_filtered = self.test_df
        else:
            try:
                if '(' in class_text and ')' in class_text:
                    class_id = int(class_text.split('(')[-1].split(')')[0])
                    train_filtered = self.train_df[self.train_df['class'] == class_id]
                    test_filtered = self.test_df[self.test_df['class'] == class_id]
                else:
                    return
            except (ValueError, IndexError):
                return
        
        train_fig = self.plot_function(train_filtered)
        test_fig = self.plot_function(test_filtered)
        self.update_plots(train_fig, test_fig)

    def update_plots(self, train_fig, test_fig):
        self.clear(self.train)
        self.clear(self.test)

        train_label = QLabel("Train")
        train_label.setFont(QFont("Segoe UI", 8, QFont.Bold))
        train_label.setAlignment(Qt.AlignCenter)
        train_label.setFixedHeight(15)
        train_label.setStyleSheet("color: #0078d4; margin: 0px; padding: 0px;")
        
        train_canvas = FigureCanvas(train_fig)
        train_canvas.setMinimumSize(350, 250)
        
        self.train.addWidget(train_label)
        self.train.addWidget(train_canvas)

        test_label = QLabel("Test")
        test_label.setFont(QFont("Segoe UI", 8, QFont.Bold))
        test_label.setAlignment(Qt.AlignCenter)
        test_label.setFixedHeight(15)
        test_label.setStyleSheet("color: #0078d4; margin: 0px; padding: 0px;")
        
        test_canvas = FigureCanvas(test_fig)
        test_canvas.setMinimumSize(350, 250)
        
        self.test.addWidget(test_label)
        self.test.addWidget(test_canvas)

    def clear(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                if hasattr(w, 'figure'):
                    w.figure.clear()
                    import matplotlib.pyplot as plt
                    plt.close(w.figure)
                w.deleteLater()

class MainUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Dataset Analyzer")
        self.resize(1200, 800)
        self.setMinimumSize(1000, 700)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QTabWidget::pane {
                border: 1px solid #dcdcdc;
                background-color: white;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #e1e1e1;
                color: #333333;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #0078d4;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #d0d0d0;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                color: #333333;
            }
            QStatusBar {
                background-color: #ffffff;
                color: #333333;
                border-top: 1px solid #dcdcdc;
                font-size: 12px;
            }
            QProgressBar {
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                text-align: center;
                background-color: #e1e1e1;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 5px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.load_btn = QPushButton("📁 Select dataset.yaml")
        self.tabs = QTabWidget()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status = QStatusBar()

        layout.addWidget(self.load_btn)
        layout.addWidget(self.tabs)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status)
        self.setLayout(layout)

        self.build_tabs()

    def build_tabs(self):
        self.class_tab = AnalysisTab()
        self.obj_tab = AnalysisTab()
        self.size_tab = AnalysisTab()
        self.ratio_tab = AnalysisTab()
        self.spatial_tab = AnalysisTab()
        self.res_tab = AnalysisTab()

        self.viewer_tab = QWidget()
        self.imbalance_tab = QWidget()
        self.anchor_tab = QWidget()
        self.health_tab = QWidget()

        self.summary_tab = QTextEdit()
        self.summary_tab.setReadOnly(True)

        self.tabs.addTab(self.summary_tab, "📊 Summary")
        self.tabs.addTab(self.class_tab, "📈 Class Distribution")
        self.tabs.addTab(self.obj_tab, "🔢 Objects/Image")
        self.tabs.addTab(self.size_tab, "📏 BBox Size")
        self.tabs.addTab(self.ratio_tab, "📐 Aspect Ratio")
        self.tabs.addTab(self.spatial_tab, "🗺️ Spatial")
        self.tabs.addTab(self.res_tab, "🖼️ Resolution")
        self.tabs.addTab(self.viewer_tab, "👁️ Viewer")
        self.tabs.addTab(self.imbalance_tab, "⚖️ Imbalance")
        self.tabs.addTab(self.anchor_tab, "⚓ Anchors")
        self.tabs.addTab(self.health_tab, "💚 Health")

    def select_yaml(self):
        return QFileDialog.getOpenFileName(self, "Select YAML", "", "YAML Files (*.yaml *.yml)")[0]