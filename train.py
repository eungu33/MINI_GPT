# train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast # [핵심] Mixed Precision

import config
from src.utils import prepare_data, plot_loss_curve
from src.dataset import GPTDataset
from src.model import MiniGPT

# -----------------------------------------
# 학습 함수 (AMP 적용됨)
# -----------------------------------------
def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs, dataset_decoder):
    train_hist, val_hist = [], []
    scaler = GradScaler() # [핵심] 그라디언트 스케일러

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)
        avg_loss = 0
        
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True) # 성능 최적화
            
            # [핵심] autocast로 연산을 float16으로 수행 (속도 향상)
            with autocast(enabled=(device=='cuda')):
                logits = model(x)
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
            
            # [핵심] 스케일러를 통해 역전파 수행
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            avg_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        # 평가 및 기록
        train_loss = avg_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        
        ppl = math.exp(val_loss) if val_loss < 100 else float('inf')
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f}")
        
        # 중간 생성
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            gen_tokens = model.generate(ctx, max_new_tokens=50)[0].tolist()
        print(f"  Gen: {dataset_decoder(gen_tokens).strip()}\n")

    return train_hist, val_hist

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    # 평가 때는 autocast만 사용 (Scaler 불필요)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(enabled=(device=='cuda')):
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
        losses.append(loss.item())
    return sum(losses) / len(losses) if losses else 0

if __name__ == "__main__":
    # GPU 설정 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. 데이터 준비 (전권 병합)
    # config.BASE_PATH 안에 있는 모든 txt를 합쳐서 config.MERGED_FILENAME으로 저장
    data_path = prepare_data(config.BASE_PATH, config.MERGED_FILENAME)
    
    # 2. 데이터셋 로드
    # num_workers=2 추가: 데이터 로딩 속도 향상
    # pin_memory=True 추가: CPU->GPU 전송 속도 향상
    kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
    
    print("Loading Dataset...")
    train_ds = GPTDataset(data_path, config.BLOCK_SIZE, 'train')
    val_ds = GPTDataset(data_path, config.BLOCK_SIZE, 'val')
    test_ds = GPTDataset(data_path, config.BLOCK_SIZE, 'test')
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, **kwargs)

    print(f"Vocab Size: {train_ds.vocab_size}, Train Batches: {len(train_loader)}")

    # 3. 모델 초기화
    model = MiniGPT(train_ds.vocab_size, config.EMB_DIM, config.N_HEAD, config.N_LAYER, config.BLOCK_SIZE).to(device)
    
    # torch.compile: PyTorch 2.0+ 최적화 (Colab에서 지원하면 자동 적용)
    try:
        model = torch.compile(model)
        print("Enable torch.compile() optimization")
    except:
        pass

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    print("\nTraining Start (with Mixed Precision)...")
    train_loss, val_loss = train(
        model, train_loader, val_loader, optimizer, scheduler, device, config.EPOCHS, train_ds.decode
    )

    plot_loss_curve(train_loss, val_loss)
    torch.save(model.state_dict(), "minigpt_final_weights.pth")
    print("Done.")
