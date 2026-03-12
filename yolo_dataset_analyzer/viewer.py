import cv2
import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageViewer(QWidget):
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.idx = 0
        self.label = QLabel()
        self.btn = QPushButton("Next")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn)
        self.setLayout(layout)

        self.btn.clicked.connect(self.next_img)
        self.show_img()

    def draw_boxes(self, path):
        img = cv2.imread(path)
        if img is None:
            return None

        label_path = path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

        if os.path.exists(label_path):
            h, w = img.shape[:2]
            with open(label_path) as f:
                for line in f:
                    try:
                        c, xc, yc, bw, bh = map(float, line.split())
                        x1 = int((xc - bw/2) * w)
                        y1 = int((yc - bh/2) * h)
                        x2 = int((xc + bw/2) * w)
                        y2 = int((yc + bh/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except:
                        continue

        return img

    def show_img(self):
        if not self.images or self.idx >= len(self.images):
            return
        
        img = self.draw_boxes(self.images[self.idx])
        if img is None:
            self.label.setText("Failed to load image")
            return
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q).scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def next_img(self):
        if not self.images:
            return
        self.idx = (self.idx + 1) % len(self.images)
        self.show_img()