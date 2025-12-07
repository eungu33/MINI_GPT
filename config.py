# config.py

# 경로 설정
BASE_PATH = "HarryPotter"
RAW_FILENAMES = ["01 Harry Potter and the Sorcerers Stone.txt"]

# 하이퍼파라미터 설정 (CPU 환경 고려)
BATCH_SIZE = 32
BLOCK_SIZE = 64      # 문맥 길이 (Context Window)
EMB_DIM = 256        # 임베딩 차원
N_HEAD = 4           # Attention Head 수
N_LAYER = 4          # Transformer Layer 수
LEARNING_RATE = 5e-4 
EPOCHS = 3           # Loss 변화 그래프 확인용