# src/dataset.py
import torch
import tiktoken
from torch.utils.data import Dataset

# -----------------------------------------
# 데이터셋 클래스 (8:1:1 분할)
# -----------------------------------------
class GPTDataset(Dataset):
    def __init__(self, txt_file, block_size, split='train'):
        with open(txt_file, 'r', encoding='utf-8') as f: text = f.read()
        
        # GPT-2 BPE Tokenizer 사용
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab # 50257
        tokens = self.tokenizer.encode(text)
        total = len(tokens)
        
        # 데이터셋을 8:1:1 비율로 분할 (Train:Val:Test)
        n1, n2 = int(0.8 * total), int(0.9 * total)
        
        if split == 'train':
            self.ids = tokens[:n1]
        elif split == 'val':
            self.ids = tokens[n1:n2]
        else:
            self.ids = tokens[n2:]
            
        self.block_size = block_size

    def __len__(self):
        # block_size 만큼의 길이를 확보해야 하므로 조정
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, idx):
        # Next Token Prediction을 위한 입력(x)과 정답(y) 생성
        chunk = self.ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def decode(self, token_ids):
        # 생성된 토큰 ID 리스트를 텍스트로 디코딩
        return self.tokenizer.decode(token_ids)