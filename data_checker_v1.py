import sys
import os
import shutil
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QSlider, QFileDialog, QSizePolicy, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt

STYLE = """
QMainWindow, QWidget {
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}
QPushButton {
    background: #16213e;
    color: #e0e0e0;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 7px 20px;
    font-size: 13px;
}
QPushButton:hover  { background: #0f3460; }
QPushButton:pressed { background: #533483; }
QPushButton#btn_delete {
    background: #7b1e1e;
    border: 1px solid #c0392b;
    color: #ffcccc;
    font-weight: bold;
}
QPushButton#btn_delete:hover { background: #c0392b; color: white; }
QLabel#lbl_info   { color: #7ecfff; font-size: 12px; }
QLabel#lbl_counter { color: #a0cfff; font-size: 13px; font-weight: bold; }
QLabel#lbl_ctx_val { color: #f0c040; font-weight: bold; font-size: 14px; min-width: 52px; }
QLabel#panel_title {
    color: #7ecfff; font-size: 12px;
    qproperty-alignment: AlignCenter;
    padding: 2px 0;
}
QSlider::groove:horizontal {
    height: 6px; background: #0f3460; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f0c040; border: none;
    width: 16px; height: 16px; margin: -5px 0; border-radius: 8px;
}
QSlider::sub-page:horizontal { background: #533483; border-radius: 3px; }
"""


class ImagePanel(QFrame):
    """A panel that always scales its pixmap to fill the available space."""
    def __init__(self):
        super().__init__()
        self._pixmap = None
        self.setStyleSheet("background:#0d0d1a; border:1px solid #0f3460; border-radius:6px;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def set_text(self, text):
        self._pixmap = None
        self._text = text
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.width() - 4, self.height() - 4,
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        else:
            p.setPen(QColor("#555"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       getattr(self, '_text', ''))
        p.end()


def load_labelme_boxes(json_path):
    boxes = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data.get('shapes', []):
            if shape.get('shape_type') != 'rectangle':
                continue
            pts = shape['points']
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            boxes.append((shape.get('label', ''),
                          int(min(x1, x2)), int(min(y1, y2)),
                          int(max(x1, x2)), int(max(y1, y2))))
    except Exception:
        pass
    return boxes


def cv2_to_qpixmap(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    return QPixmap.fromImage(QImage(img_rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888))


def draw_boxes_on_pixmap(pixmap, boxes, img_w, img_h):
    result = pixmap.copy()
    p = QPainter(result)
    sx, sy = pixmap.width() / img_w, pixmap.height() / img_h
    for label, x1, y1, x2, y2 in boxes:
        p.setPen(QPen(QColor(255, 60, 60), 2))
        p.drawRect(int(x1*sx), int(y1*sy), int((x2-x1)*sx), int((y2-y1)*sy))
        p.setPen(QPen(QColor(255, 220, 50)))
        p.setFont(QFont('Segoe UI', 9, QFont.Bold))
        p.drawText(int(x1*sx)+3, int(y1*sy)-5, label)
    p.end()
    return result


def collect_pairs(root):
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    pairs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != 'trash_data']
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                continue
            json_path = os.path.join(dirpath, os.path.splitext(fname)[0] + '.json')
            if os.path.exists(json_path):
                pairs.append((os.path.join(dirpath, fname), json_path))
    return pairs


class DataChecker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Data Checker")
        self.pairs = []
        self.index = 0
        self.context_scale = 3.0
        self.root_folder = ""
        self.show_boxes = True
        self._build_ui()
        self.showMaximized()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # ── Top bar ──────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(10)
        btn_open = QPushButton("📂  Open Folder")
        btn_open.setFixedHeight(36)
        btn_open.setFixedWidth(150)
        btn_open.clicked.connect(self.open_folder)
        self.lbl_info = QLabel("No folder loaded")
        self.lbl_info.setObjectName("lbl_info")
        top.addWidget(btn_open)
        top.addWidget(self.lbl_info, 1)
        root.addLayout(top)

        # ── Image panels ─────────────────────────────────────────
        img_row = QHBoxLayout()
        img_row.setSpacing(10)

        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        t1 = QLabel("Full Image")
        t1.setObjectName("panel_title")
        t1.setFixedHeight(20)
        self.panel_full = ImagePanel()
        left_col.addWidget(t1)
        left_col.addWidget(self.panel_full, 1)

        right_col = QVBoxLayout()
        right_col.setSpacing(4)
        self.lbl_ctx_title = QLabel("Context Crop  (3.0×)")
        self.lbl_ctx_title.setObjectName("panel_title")
        self.lbl_ctx_title.setFixedHeight(20)
        self.panel_crop = ImagePanel()
        right_col.addWidget(self.lbl_ctx_title)
        right_col.addWidget(self.panel_crop, 1)

        img_row.addLayout(left_col, 6)
        img_row.addLayout(right_col, 4)
        root.addLayout(img_row, 1)

        # ── Controls ─────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        ctrl.addWidget(QLabel("Context (N × bbox):"))
        self.ctx_slider = QSlider(Qt.Horizontal)
        self.ctx_slider.setRange(100, 2000)
        self.ctx_slider.setValue(300)
        self.ctx_slider.setFixedWidth(280)
        self.ctx_slider.setFixedHeight(28)
        self.ctx_slider.valueChanged.connect(self._on_context)
        self.lbl_ctx_val = QLabel("3.0×")
        self.lbl_ctx_val.setObjectName("lbl_ctx_val")
        ctrl.addWidget(self.ctx_slider)
        ctrl.addWidget(self.lbl_ctx_val)

        ctrl.addSpacing(16)

        self.btn_toggle_boxes = QPushButton("☑  Boxes: ON")
        self.btn_toggle_boxes.setFixedSize(130, 36)
        self.btn_toggle_boxes.clicked.connect(self.toggle_boxes)
        ctrl.addWidget(self.btn_toggle_boxes)

        ctrl.addSpacing(16)

        self.btn_prev = QPushButton("◀  Prev")
        self.btn_prev.setFixedSize(100, 36)
        self.btn_prev.clicked.connect(self.prev_image)

        self.btn_next = QPushButton("Next  ▶")
        self.btn_next.setFixedSize(100, 36)
        self.btn_next.clicked.connect(self.next_image)

        self.btn_delete = QPushButton("🗑  Delete")
        self.btn_delete.setObjectName("btn_delete")
        self.btn_delete.setFixedSize(120, 36)
        self.btn_delete.clicked.connect(self.delete_current)

        ctrl.addWidget(self.btn_prev)
        ctrl.addWidget(self.btn_next)
        ctrl.addSpacing(8)
        ctrl.addWidget(self.btn_delete)
        ctrl.addStretch()

        self.lbl_counter = QLabel("")
        self.lbl_counter.setObjectName("lbl_counter")
        ctrl.addWidget(self.lbl_counter)

        root.addLayout(ctrl)

    # ── Logic ─────────────────────────────────────────────────────

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Root Folder")
        if not folder:
            return
        self.root_folder = folder
        self.pairs = collect_pairs(folder)
        self.index = 0
        self.lbl_info.setText(f"{folder}   —   {len(self.pairs)} pairs found")
        self.show_current()

    def show_current(self):
        if not self.pairs:
            self.panel_full.set_text("No pairs found")
            self.panel_crop.set_text("")
            self.lbl_counter.setText("")
            return

        img_path, json_path = self.pairs[self.index]
        img = cv2.imread(img_path)
        if img is None:
            self.panel_full.set_text("Cannot load image")
            self.panel_crop.set_text("")
            return

        img_h, img_w = img.shape[:2]
        boxes = load_labelme_boxes(json_path)

        # Full image with all boxes
        full_px = cv2_to_qpixmap(img)
        if self.show_boxes:
            full_px = draw_boxes_on_pixmap(full_px, boxes, img_w, img_h)
        self.panel_full.set_pixmap(full_px)

        # Context crop of first box — square, centered, no resize
        if boxes:
            _, x1, y1, x2, y2 = boxes[0]
            bw, bh = x2 - x1, y2 - y1
            # Use the longer side as the base so the bbox always fits
            half = max(bw, bh) * self.context_scale / 2
            half = int(half)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            # Square region in image coordinates
            rx1, ry1 = cx - half, cy - half
            rx2, ry2 = cx + half, cy + half
            # Pad with black if the square goes outside the image
            crop_size = half * 2
            canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            # Overlap between square region and image
            src_x1 = max(rx1, 0)
            src_y1 = max(ry1, 0)
            src_x2 = min(rx2, img_w)
            src_y2 = min(ry2, img_h)
            dst_x1 = src_x1 - rx1
            dst_y1 = src_y1 - ry1
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            if src_x2 > src_x1 and src_y2 > src_y1:
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
            if self.show_boxes:
                bx1 = x1 - rx1
                by1 = y1 - ry1
                bx2 = x2 - rx1
                by2 = y2 - ry1
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (60, 60, 255), 2)
            self.panel_crop.set_pixmap(cv2_to_qpixmap(canvas))
        else:
            self.panel_crop.set_text("No bounding box")

        rel = os.path.relpath(img_path, self.root_folder)
        self.lbl_counter.setText(
            f"{self.index + 1} / {len(self.pairs)}   |   {rel}   |   {len(boxes)} box(es)")

    def toggle_boxes(self):
        self.show_boxes = not self.show_boxes
        self.btn_toggle_boxes.setText("☑  Boxes: ON" if self.show_boxes else "☐  Boxes: OFF")
        self.show_current()

    def _on_context(self, value):
        self.context_scale = value / 100.0
        self.lbl_ctx_val.setText(f"{self.context_scale:.1f}×")
        self.lbl_ctx_title.setText(f"Context Crop  ({self.context_scale:.1f}×)")
        self.show_current()

    def prev_image(self):
        if self.pairs and self.index > 0:
            self.index -= 1
            self.show_current()

    def next_image(self):
        if self.pairs and self.index < len(self.pairs) - 1:
            self.index += 1
            self.show_current()

    def delete_current(self):
        if not self.pairs:
            return
        img_path, json_path = self.pairs[self.index]
        trash_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trash_data')
        os.makedirs(trash_dir, exist_ok=True)
        shutil.move(img_path, os.path.join(trash_dir, os.path.basename(img_path)))
        shutil.move(json_path, os.path.join(trash_dir, os.path.basename(json_path)))
        self.pairs.pop(self.index)
        if self.index >= len(self.pairs):
            self.index = max(0, len(self.pairs) - 1)
        self.lbl_info.setText(f"{self.root_folder}   —   {len(self.pairs)} pairs found")
        self.show_current()

    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key_Left, Qt.Key_A):
            self.prev_image()
        elif k in (Qt.Key_Right, Qt.Key_D):
            self.next_image()
        elif k in (Qt.Key_Delete, Qt.Key_X):
            self.delete_current()
        elif k == Qt.Key_R:
            self.toggle_boxes()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = DataChecker()
    sys.exit(app.exec_())
