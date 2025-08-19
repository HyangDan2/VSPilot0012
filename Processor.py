# processor.py
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
torch = __import__('torch')

class ImageProcessor:
    def __init__(self, to_gray=False, clahe_clip=0.0, blur_strength=0, blur_mode="Light"):
        self.to_gray = to_gray
        self.clahe_clip = clahe_clip
        self.blur_strength = blur_strength
        self.blur_mode = blur_mode  # Light, Strong, SoftEdge

    def preprocess(self, pil_img: Image.Image) -> Image.Image:
        img = pil_img.convert("RGB")
        cv_img = np.array(img)

        if self.to_gray:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

        if self.clahe_clip > 0:
            lab = cv2.cvtColor(cv_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            cv_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        if self.blur_strength > 0:
            if self.blur_mode == "Light":
                sigma = self.blur_strength * 3.0 / 100.0
                cv_img = cv2.GaussianBlur(cv_img, (5, 5), sigma)
            elif self.blur_mode == "Strong":
                k = int(self.blur_strength / 10) * 2 + 1
                cv_img = cv2.medianBlur(cv_img, k)
            elif self.blur_mode == "SoftEdge":
                d = 9
                sigmaColor = self.blur_strength * 2
                sigmaSpace = self.blur_strength
                cv_img = cv2.bilateralFilter(cv_img, d, sigmaColor, sigmaSpace)

        return Image.fromarray(cv_img)

    def to_tensor(self, pil_img: Image.Image, resize_to=None):
        tf = T.ToTensor()
        t = tf(pil_img).unsqueeze(0)  # (1, C, H, W)
        if resize_to:
            t = torch.nn.functional.interpolate(t, size=resize_to, mode="bilinear", align_corners=False)
        return t
