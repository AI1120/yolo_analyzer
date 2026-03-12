import sys
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QTextEdit, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from viewer import ImageViewer
from ui import MainUI
from worker import AnalysisWorker
import summaries
from datetime import datetime

class App(MainUI):
    def __init__(self):
        super().__init__()
        self.load_btn.clicked.connect(self.load_yaml)
        self.worker = None

    def load_yaml(self):
        path = self.select_yaml()
        if not path:
            return

        self.load_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.worker = AnalysisWorker(path)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def update_progress(self, message):
        self.status.showMessage(message)

    def on_analysis_error(self, error):
        self.status.showMessage(f"Error: {error}")
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)

    def on_analysis_complete(self, results):
        train_df = results['train_df']
        test_df = results['test_df']
        train_imgs = results['train_imgs']
        summary = results['summary']
        plots_data = results['plots']
        diag = results['diagnostics']
        
        import plots
        
        self.class_tab.set_plots(*plots_data['class'], 
            summaries.generate_class_summary(train_df, test_df, results.get('class_names')))
        self.obj_tab.set_plots(*plots_data['obj'], 
            summaries.generate_objects_summary(results['train_counts'], results['test_counts']))
        self.size_tab.set_plots(*plots_data['size'], 
            summaries.generate_bbox_summary(train_df, test_df), 
            train_df, test_df, results.get('class_names'), plots.bbox_size)
        self.ratio_tab.set_plots(*plots_data['ratio'], 
            summaries.generate_ratio_summary(train_df, test_df), 
            train_df, test_df, results.get('class_names'), plots.ratio_analysis)
        self.spatial_tab.set_plots(*plots_data['spatial'], 
            summaries.generate_spatial_summary(train_df, test_df), 
            train_df, test_df, results.get('class_names'), plots.spatial_distribution)
        self.res_tab.set_plots(*plots_data['res'], 
            summaries.generate_resolution_summary(train_df, test_df), 
            train_df, test_df, results.get('class_names'), plots.resolution_analysis)

        viewer = ImageViewer(train_imgs[:100])
        vlayout = QVBoxLayout()
        vlayout.addWidget(viewer)
        self.viewer_tab.setLayout(vlayout)

        self.imbalance_tab.setLayout(self.text_layout(
            f"Class Distribution:\n{diag['counts']}\n\n" +
            f"Imbalance Ratio: {diag['ratio']:.2f}\n\n" +
            f"Note: Ratio > 3.0 indicates significant class imbalance"
        ))

        self.anchor_tab.setLayout(self.text_layout(str(diag['anchors'])))

        self.health_tab.setLayout(self.text_layout(
            f"Dataset Health Score: {diag['score']}/100"
        ))

        summary_text = f"""
Train Images: {summary['train_images']}
Test Images: {summary['test_images']}
Train Objects: {summary['train_objects']}
Test Objects: {summary['test_objects']}
Classes: {summary['classes']}
Small Objects: {diag['small']}
Medium Objects: {diag['medium']}
Large Objects: {diag['large']}
"""
        self.summary_tab.setText(summary_text)

        self.status.showMessage("Analysis Complete")
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)

    def text_layout(self, text):
        layout = QVBoxLayout()
        t = QTextEdit()
        t.setText(text)
        t.setReadOnly(True)
        t.setStyleSheet("""
            QTextEdit {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(t)
        return layout

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())