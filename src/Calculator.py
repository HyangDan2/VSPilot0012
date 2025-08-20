# calculator.py
import pyiqa
from pytorch_msssim import ms_ssim

class MSSSIMCalculator:
    def __init__(self):
        self.pyiqa_model = pyiqa.create_metric('ssim')
        self.pyiqa_model_msssim_mono = pyiqa.create_metric('ms_ssim')
        self.pyiqa_model_msssim_color = pyiqa.create_metric('ms_ssim', 
                                                            as_loss=False,
                                                            test_y_channel=False,
                                                            color_space='rgb',
                                                            downsample=False,
                                                            is_prod=True
                                                            )

    def calculate(self, method: str, img1, img2) -> float:
        if method == "pyiqa-ssim":
            return self.pyiqa_model(img1, img2).item()
        elif method == "pyiqa-msssim(mono)":
            return self.pyiqa_model_msssim_mono(img1, img2).item()
        elif method == "pyiqa-msssim(color)":
            return self.pyiqa_model_msssim_color(img1, img2).item()
        else:
            return ms_ssim(img1, img2, data_range=1.0).item()