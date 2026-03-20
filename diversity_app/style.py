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
