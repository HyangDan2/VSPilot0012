# MS-SSIM 이미지 유사도 분석기 (PyTorch & PyIQA)

PySide6 GUI 기반으로 두 이미지를 비교하고 MS-SSIM 유사도를 계산하는 도구입니다.

## 주요 기능

- 이미지1 / 이미지2 불러오기
- 실시간 전처리: Grayscale, CLAHE, Blur (3종 알고리즘)
- 후처리 이미지 확인 (총 4개: 원본/처리)
- MSSSIM 유사도 측정 (PyTorch 또는 PyIQA 선택)
- 전처리 이미지 저장 및 CSV 로그 저장

## 실행 방법

```bash
pip install -r requirements.txt
python main.py
```

## 결과 예시

- PyTorch-MSSSIM 유사도: `0.7267`
- PyIQA-SSIM 유사도: `0.7678`

## 스크린샷

(예시 이미지를 넣어주세요)

## 디렉토리 구조

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

## 라이선스

Apache 2.0 (LICENSE.txt 참조)
