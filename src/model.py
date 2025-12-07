# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------
# MiniGPT 모델 구조 (Decoder-only Transformer)
# -----------------------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_head, n_layer, block_size):
        super().__init__()
        # Token Embedding: 단어의 의미 학습
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        # Position Embedding: 단어의 순서 정보 학습
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        
        # Causal (Decoder-only) Transformer Block
        # norm_first=True 설정으로 Pre-LN Transformer 구조 적용
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=n_head, 
            batch_first=True, 
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        
        # 최종 Normalization 및 출력층
        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        
        # Input Embedding = Token Emb + Pos Emb
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        # [핵심] Causal Mask 생성 (미래 토큰 참조 방지)
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(idx.device)
        
        # PyTorch TransformerEncoderLayer에 마스크 적용하여 Decoder처럼 동작
        x = self.blocks(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    # 텍스트 생성 함수 (Inference)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 현재 문맥 길이 제한 (block_size)
            idx_cond = idx[:, -self.block_size:]
            
            # 모델 예측
            logits = self(idx_cond)
            logits = logits[:, -1, :] # 마지막 토큰의 예측값만 사용
            
            # 확률 분포 변환 및 샘플링 (Top-1 Greedy 대신 Multinominal Sampling 사용)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 생성된 토큰을 이어 붙임
            idx = torch.cat((idx, idx_next), dim=1)
        return idx