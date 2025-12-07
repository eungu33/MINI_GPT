## 1. 프로젝트 개요 (Overview)

본 프로젝트는 **Transformer의 Decoder-only 아키텍처**를 **파이토치로 밑바닥부터(From Scratch)** 직접 구현하여, 경량화된 생성형 언어 모델(MiniGPT)을 구축하는 것을 목표로 합니다.

**학습 데이터:** 해리포터 소설 전권(1권~7권)의 약 620만 자 텍스트를 사용하여, 문맥을 이해하고 해리포터 특유의 문체로 텍스트를 생성하는 모델을 개발합니다.

## 2. 주요 기능 및 특징 (Key Features)

* **Decoder-only Architecture:** GPT 모델의 핵심인 Multi-Head Self-Attention 및 Causal Masking을 구현하여 텍스트 생성 능력을 확보.
* **고성능 토크나이징:** OpenAI의 **`tiktoken`** 라이브러리를 활용한 Byte Pair Encoding (BPE) 기반 토크나이저 통합.
* **학습 효율 최적화:**
    * **Mixed Precision (AMP):** Colab T4 GPU 환경에서 **혼합 정밀도 학습**을 적용하여 메모리 효율 증대 및 학습 속도 2배 향상.
    * **PyTorch 2.0+ `torch.compile`** 적용을 통한 추가적인 속도 가속.
* **유연한 데이터 전처리:** 다수의 `.txt` 파일을 자동으로 병합하고 정제하는 데이터 파이프라인 구축.

## 3. 개발 환경 및 요구 사항 (Prerequisites)

이 프로젝트는 Google Colab 환경(Tesla T4 GPU)에 최적화되어 있습니다.

| 구분 | 환경 |
| :--- | :--- |
| **언어** | Python 3.x |
| **프레임워크** | PyTorch 2.0 이상 |
| **패키지** | tiktoken, tqdm, numpy, matplotlib |

### 4. 프로젝트 구조 (Project Structure)

```text
MiniGPT_Project/
├── HarryPotter/
│   ├── 01_Harry_Potter_...txt       # 원본 소설 텍스트 (1~7권)
│   └── combined_harry_potter.txt    # (자동 생성) 전체 병합 파일
│
├── src/
│   ├── __init__.py
│   ├── utils.py                     # 데이터 병합, 정제, 시각화 유틸리티
│   ├── dataset.py                   # PyTorch Dataset 및 토크나이징 처리
│   └── model.py                     # MiniGPT 모델 아키텍처 (Decoder-only)
│
├── config.py                        # 하이퍼파라미터 및 경로 설정
├── train.py                         # 학습 실행 (Mixed Precision 적용)
└── inference.py                     # 텍스트 생성 테스트

# 학습이 완료되면 'minigpt_final_weights.pth' 파일이 생성됩니다.

학습이 완료된 후, inference.py를 실행하여 모델이 생성하는 텍스트를 확인합니다.

### 5. 학습 결과 및 성능
