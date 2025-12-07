# src/utils.py
import os
import glob
import re
import unicodedata
import matplotlib.pyplot as plt

def prepare_data(base_path, output_filename):
    """
    폴더 내 모든 txt 파일을 찾아서 하나로 합치고 정제합니다.
    """
    if not os.path.exists(base_path): os.makedirs(base_path)
    
    merged_path = os.path.join(base_path, output_filename)
    
    # 이미 합쳐진 파일이 있으면 그 경로 반환
    if os.path.exists(merged_path):
        print(f"[Info] Found existing merged file: {merged_path}")
        return merged_path

    print(f"[Pre-processing] Merging all .txt files in {base_path}...")
    
    # 모든 txt 파일 읽기
    all_text = ""
    file_list = glob.glob(os.path.join(base_path, "*.txt"))
    # 합쳐진 파일 자체는 제외
    file_list = [f for f in file_list if output_filename not in f and "cleaned" not in f]
    file_list.sort() # 순서대로 정렬 (1권 -> 7권)

    if not file_list:
        print("Warning: No txt files found. Creating dummy.")
        return create_dummy(merged_path)

    for file in file_list:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            all_text += f.read() + "\n"
    
    # 정제 과정 (Clean)
    cleaned_text = clean_string(all_text)
    
    with open(merged_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
        
    print(f"[Done] Saved merged file ({len(cleaned_text)} chars): {merged_path}")
    return merged_path

def create_dummy(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Harry Potter dummy text " * 10000)
    return path

def clean_string(text):
    # 유니코드 정규화
    text = unicodedata.normalize('NFKC', text)
    # 특수기호 통일
    replacements = {
        '\u201c': '"', '\u201d': '"',
        "\u2018": "'", "\u2019": "'",
        '\u2014': '-', '\u2013': '-',
        '\u2026': '...',
        '\\': ''
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    text = "".join(ch for ch in text if ch.isprintable())
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
