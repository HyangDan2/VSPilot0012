# MS-SSIM Image Similarity Analyzer (PyTorch & PyIQA)

This is a PySide6 GUI-based application for comparing two images and computing their similarity using the MS-SSIM metric.

## Features

- Load two input images (Image 1 & Image 2)
- Real-time preprocessing: Grayscale, CLAHE, and Blur (with 3 algorithm options)
- Side-by-side view of original and preprocessed images (4 total views)
- MS-SSIM score calculation (select between PyTorch or PyIQA implementation)
- Save preprocessed images
- Export similarity logs with preprocessing parameters in CSV format

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Example Output

- PyTorch-MSSSIM score: `0.7267`
- PyIQA-SSIM score: `0.7678`

## Screenshot

![앱 실행 화면](./image/screenshot.png)

## Project Structure

```
project/
├── main.py
├── ui.py
├── processor.py
├── calculator.py
├── logger.py
├── outputs/
└── README.md
```

## License

Apache 2.0 — see LICENSE.txt for details.
