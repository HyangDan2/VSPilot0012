# ui.py (수정됨 - 실시간 적용 + Blur 알고리즘 선택)
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QTextEdit, QCheckBox, QSlider, QGridLayout
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from pathlib import Path
import numpy as np
from PIL import Image
from src.Processor import ImageProcessor
from src.Calculator import MSSSIMCalculator
from src.Logger import LogManager

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MS-SSIM GUI (실시간 & Blur 선택)")

        self.img1_path = None
        self.img2_path = None
        self.raw_img1 = None
        self.raw_img2 = None
        self.proc_img1 = None
        self.proc_img2 = None

        self.calculator = MSSSIMCalculator()
        self.logger = LogManager()

        self.init_ui()

    def init_ui(self):
        self.combo_method = QComboBox()
        self.combo_method.addItems(["pytorch-msssim", "pyiqa-ssim", "pyiqa-msssim(mono)", "pyiqa-msssim(color)" ])

        self.btn_load1 = QPushButton("이미지 1")
        self.btn_load2 = QPushButton("이미지 2")
        self.btn_load1.clicked.connect(self.load_image1)
        self.btn_load2.clicked.connect(self.load_image2)

        self.btn_compare = QPushButton("📊 유사도 계산")
        self.btn_compare.clicked.connect(self.compare)

        # 옵션1
        self.check_gray1 = QCheckBox("Grayscale 1")
        self.check_clahe1 = QCheckBox("CLAHE 1")
        self.slider_clahe1 = QSlider(Qt.Orientation.Horizontal)
        self.slider_clahe1.setRange(10, 50)
        self.slider_clahe1.setValue(20)
        self.check_blur1 = QCheckBox("Blur 1")
        self.slider_blur1 = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur1.setRange(0, 100)
        self.slider_blur1.setValue(0)
        self.combo_blur_mode1 = QComboBox()
        self.combo_blur_mode1.addItems(["Light", "Strong", "SoftEdge"])

        # 옵션2
        self.check_gray2 = QCheckBox("Grayscale 2")
        self.check_clahe2 = QCheckBox("CLAHE 2")
        self.slider_clahe2 = QSlider(Qt.Orientation.Horizontal)
        self.slider_clahe2.setRange(10, 50)
        self.slider_clahe2.setValue(20)
        self.check_blur2 = QCheckBox("Blur 2")
        self.slider_blur2 = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur2.setRange(0, 100)
        self.slider_blur2.setValue(0)
        self.combo_blur_mode2 = QComboBox()
        self.combo_blur_mode2.addItems(["Light", "Strong", "SoftEdge"])

        # QLabel 4개
        self.label1 = QLabel("이미지1 원본")
        self.label1_post = QLabel("이미지1 처리")
        self.label2 = QLabel("이미지2 원본")
        self.label2_post = QLabel("이미지2 처리")
        for l in [self.label1, self.label1_post, self.label2, self.label2_post]:
            l.setFixedSize(300, 300)
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 저장 버튼
        self.btn_save1 = QPushButton("💾 이미지1 저장")
        self.btn_save2 = QPushButton("💾 이미지2 저장")
        self.btn_save1.clicked.connect(self.save_image1)
        self.btn_save2.clicked.connect(self.save_image2)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)

        # 실시간 signal 연결
        self.check_gray1.stateChanged.connect(self.update_preview1)
        self.check_clahe1.stateChanged.connect(self.update_preview1)
        self.slider_clahe1.valueChanged.connect(self.update_preview1)
        self.check_blur1.stateChanged.connect(self.update_preview1)
        self.slider_blur1.valueChanged.connect(self.update_preview1)
        self.combo_blur_mode1.currentTextChanged.connect(self.update_preview1)

        self.check_gray2.stateChanged.connect(self.update_preview2)
        self.check_clahe2.stateChanged.connect(self.update_preview2)
        self.slider_clahe2.valueChanged.connect(self.update_preview2)
        self.check_blur2.stateChanged.connect(self.update_preview2)
        self.slider_blur2.valueChanged.connect(self.update_preview2)
        self.combo_blur_mode2.currentTextChanged.connect(self.update_preview2)

        layout = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(self.combo_method)
        top.addWidget(self.btn_load1)
        top.addWidget(self.btn_load2)
        top.addWidget(self.btn_compare)

        opt = QGridLayout()
        for row, items in enumerate([
            (self.check_gray1, self.check_clahe1, self.slider_clahe1, self.check_blur1, self.slider_blur1, self.combo_blur_mode1),
            (self.check_gray2, self.check_clahe2, self.slider_clahe2, self.check_blur2, self.slider_blur2, self.combo_blur_mode2),
        ]):
            for col, w in enumerate(items):
                opt.addWidget(w, row, col)

        imggrid = QGridLayout()
        imggrid.addWidget(self.label1, 0, 0)
        imggrid.addWidget(self.label1_post, 1, 0)
        imggrid.addWidget(self.label2, 0, 1)
        imggrid.addWidget(self.label2_post, 1, 1)

        save = QHBoxLayout()
        save.addWidget(self.btn_save1)
        save.addWidget(self.btn_save2)

        layout.addLayout(top)
        layout.addLayout(opt)
        layout.addLayout(imggrid)
        layout.addLayout(save)
        layout.addWidget(self.result_box)
        self.setLayout(layout)

    def load_image1(self):
        path, _ = QFileDialog.getOpenFileName(self, "Image 1")
        if path:
            self.img1_path = path
            self.raw_img1 = Image.open(path).convert("RGB")
            self.label1.setPixmap(QPixmap(path).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
            self.update_preview1()

    def load_image2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Image 2")
        if path:
            self.img2_path = path
            self.raw_img2 = Image.open(path).convert("RGB")
            self.label2.setPixmap(QPixmap(path).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
            self.update_preview2()

    def update_preview1(self):
        if self.raw_img1:
            processor = ImageProcessor(
                to_gray=self.check_gray1.isChecked(),
                clahe_clip=self.slider_clahe1.value() / 10 if self.check_clahe1.isChecked() else 0,
                blur_strength=self.slider_blur1.value() if self.check_blur1.isChecked() else 0,
                blur_mode=self.combo_blur_mode1.currentText()
            )
            self.proc_img1 = processor.preprocess(self.raw_img1)
            self.set_preview(self.label1_post, self.proc_img1)

    def update_preview2(self):
        if self.raw_img2:
            processor = ImageProcessor(
                to_gray=self.check_gray2.isChecked(),
                clahe_clip=self.slider_clahe2.value() / 10 if self.check_clahe2.isChecked() else 0,
                blur_strength=self.slider_blur2.value() if self.check_blur2.isChecked() else 0,
                blur_mode=self.combo_blur_mode2.currentText()
            )
            self.proc_img2 = processor.preprocess(self.raw_img2)
            self.set_preview(self.label2_post, self.proc_img2)

    def set_preview(self, label, pil_img):
        arr = np.array(pil_img.convert("RGB"))
        h, w, ch = arr.shape
        q_img = QImage(arr.data, w, h, ch * w, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_img).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))

    def compare(self):
        if not self.proc_img1 or not self.proc_img2:
            self.result_box.setText("⚠ 전처리 이미지가 없습니다.")
            return
        processor = ImageProcessor()
        t1 = processor.to_tensor(self.proc_img1)
        t2 = processor.to_tensor(self.proc_img2, resize_to=t1.shape[-2:])
        method = self.combo_method.currentText()
        score = self.calculator.calculate(method, t1, t2)
        self.result_box.setText(f"✅ {method} 유사도: {score:.4f}")

    def save_image1(self):
        if self.proc_img1:
            path, _ = QFileDialog.getSaveFileName(self, "Save", "image1_processed.png")
            if path:
                self.proc_img1.save(path)

    def save_image2(self):
        if self.proc_img2:
            path, _ = QFileDialog.getSaveFileName(self, "Save", "image2_processed.png")
            if path:
                self.proc_img2.save(path)
