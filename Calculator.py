# calculator.py
import pyiqa
from pytorch_msssim import ms_ssim

class MSSSIMCalculator:
    def __init__(self):
        self.pyiqa_model = pyiqa.create_metric('ssim')

    def calculate(self, method: str, img1, img2) -> float:
        if method == "pyiqa-ssim":
            return self.pyiqa_model(img1, img2).item()
        else:
            return ms_ssim(img1, img2, data_range=1.0).item()