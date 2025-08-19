# logger.py
from pathlib import Path
import csv
from PIL import Image

class LogManager:
    def __init__(self, save_dir="outputs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.log_path = self.save_dir / "save_log.csv"

    def save_image(self, img: Image.Image, filename: str) -> str:
        path = self.save_dir / filename
        img.save(path)
        return str(path)

    def write_log(self, path1, path2, opt1, opt2, score, method):
        write_header = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Image1", "Image2", "Options1", "Options2", "MS-SSIM", "Method"])
            writer.writerow([path1, path2, opt1, opt2, f"{score:.4f}", method])