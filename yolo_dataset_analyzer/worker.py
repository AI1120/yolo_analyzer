from PyQt5.QtCore import QThread, pyqtSignal
from analyzer import DatasetAnalyzer
import plots
from diagnostics import *

class AnalysisWorker(QThread):
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, yaml_path):
        super().__init__()
        self.yaml_path = yaml_path
    
    def run(self):
        try:
            self.progress.emit("Loading dataset configuration...")
            analyzer = DatasetAnalyzer(self.yaml_path, self.progress.emit)
            
            self.progress.emit("Analyzing training data...")
            train_df, test_df, train_counts, test_counts, train_imgs, summary, class_names = analyzer.run()
            
            self.progress.emit("Generating plots...")
            
            results = {
                'train_df': train_df,
                'test_df': test_df,
                'train_counts': train_counts,
                'test_counts': test_counts,
                'train_imgs': train_imgs,
                'summary': summary,
                'class_names': class_names,
                'plots': {
                    'class': (plots.class_distribution(train_df, class_names), plots.class_distribution(test_df, class_names)),
                    'obj': (plots.objects_per_image(train_counts), plots.objects_per_image(test_counts)),
                    'size': (plots.bbox_size(train_df), plots.bbox_size(test_df)),
                    'ratio': (plots.ratio_analysis(train_df), plots.ratio_analysis(test_df)),
                    'spatial': (plots.spatial_distribution(train_df), plots.spatial_distribution(test_df)),
                    'res': (plots.resolution_analysis(train_df), plots.resolution_analysis(test_df))
                }
            }
            
            self.progress.emit("Computing diagnostics...")
            
            counts, ratio = class_imbalance(train_df, class_names)
            small, medium, large = object_size_stats(train_df)
            anchors = anchor_boxes(train_df)
            score = health_score(summary, ratio)
            
            results['diagnostics'] = {
                'counts': counts,
                'ratio': ratio,
                'small': small,
                'medium': medium,
                'large': large,
                'anchors': anchors,
                'score': score
            }
            
            self.progress.emit("Complete")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))