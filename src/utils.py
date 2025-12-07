# src/utils.py
import os
import re
import unicodedata
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. 데이터 전처리 함수
# -----------------------------------------
def clean_text(filename, base_path):
    """
    원본 텍스트 파일을 정규화 및 정리하여 새 파일로 저장하고 경로를 반환합니다.
    """
    if not os.path.exists(base_path): os.makedirs(base_path)
    raw_path = os.path.join(base_path, filename)
    cleaned_path = os.path.join(base_path, "cleaned_v2_" + filename)

    # 이미 전처리된 파일이 있으면 재사용 (중복 처리 방지)
    if os.path.exists(cleaned_path): return cleaned_path

    print(f"[Pre-processing] Cleaning text: {filename}")
    
    # 원본 파일이 없을 경우 테스트용 더미 파일 생성
    if not os.path.exists(raw_path):
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write("Harry Potter dummy text " * 1000)
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # [데이터 정규화]
    # 1. 유니코드 정규화 (NFKC)
    text = unicodedata.normalize('NFKC', text)

    # 2. 특수기호 통일: 유니코드 코드로 안전하게 변경
    replacements = {
        '\u201c': '"', '\u201d': '"',  # “ ” (Smart Double Quotes)
        "\u2018": "'", "\u2019": "'",  # ‘ ’ (Smart Single Quotes)
        '\u2014': '-', '\u2013': '-',  # — – (Em Dash, En Dash)
        '\u2026': '...',               # … (Ellipsis)
        '\\': ''                       # 역슬래시 제거
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 3. 출력 불가능한 제어 문자 제거
    text = "".join(ch for ch in text if ch.isprintable())

    # 4. 공백 및 줄바꿈 정리
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    with open(cleaned_path, 'w', encoding='utf-8') as f:
        f.write(text)
        
    print(f"[Done] Cleaned file saved: {cleaned_path}")
    return cleaned_path

# -----------------------------------------
# 2. 시각화 함수
# -----------------------------------------
def plot_loss_curve(train_losses, val_losses):
    """
    학습/평가 Loss 변화 그래프를 저장합니다.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("[Result] Graph saved as loss_curve.png")