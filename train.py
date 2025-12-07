# train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# import config.py와 src의 모듈들
import config
from src.utils import clean_text, plot_loss_curve
from src.dataset import GPTDataset
from src.model import MiniGPT

# -----------------------------------------
# 학습 및 평가 로직
# -----------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    """평가 데이터셋에 대한 평균 Loss를 계산합니다."""
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
        losses.append(loss.item())
    return sum(losses) / len(losses) if losses else 0

def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs, dataset_decoder):
    """메인 학습 루프"""
    train_hist, val_hist = [], []
    
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)
        avg_loss = 0
        
        for i, (x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Loss 계산
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
            
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())
            
        # 스케줄러를 통한 학습률 조정
        scheduler.step()
        
        # Epoch 종료 후 평가
        train_loss = avg_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        
        # Perplexity(PPL) 계산
        ppl = math.exp(val_loss)
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f}")
        
        # 중간 생성 결과 확인
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
        gen_tokens = model.generate(ctx, max_new_tokens=30)[0].tolist()
        generated_text = dataset_decoder(gen_tokens)
        print(f"  Generated Text: {generated_text.strip()}\n")

    return train_hist, val_hist

# -----------------------------------------
# 메인 실행 블록
# -----------------------------------------
if __name__ == "__main__":
    # 1. Device 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 2. 데이터 준비 및 로드
    raw_filename = config.RAW_FILENAMES[0]
    clean_path = clean_text(raw_filename, config.BASE_PATH)
    
    # 데이터셋 로드 (GPT-2 Vocab Size: 50257)
    train_ds = GPTDataset(clean_path, config.BLOCK_SIZE, 'train')
    val_ds = GPTDataset(clean_path, config.BLOCK_SIZE, 'val')
    test_ds = GPTDataset(clean_path, config.BLOCK_SIZE, 'test')
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. 모델 초기화
    VOCAB_SIZE = train_ds.vocab_size # 50257
    model = MiniGPT(VOCAB_SIZE, config.EMB_DIM, config.N_HEAD, config.N_LAYER, config.BLOCK_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    # Cosine Annealing 스케줄러 적용
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    print("\nTraining Start...")
    train_loss, val_loss = train(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        device, 
        config.EPOCHS, 
        train_ds.decode # 디코딩 함수 전달
    )

    # 4. 결과 그래프 저장
    plot_loss_curve(train_loss, val_loss)

    # 5. 최종 Test Set 평가
    print("\nFinal Evaluation (Test Set)...")
    test_loss = evaluate(model, test_loader, device)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Perplexity: {math.exp(test_loss):.2f}")
    
    # 6. 모델 가중치 저장 (추론을 위해)
    torch.save(model.state_dict(), "minigpt_final_weights.pth")
    print("\nAll processes finished. Model weights saved.")