# config.py

# 경로 설정
BASE_PATH = "HarryPotter"
MERGED_FILENAME = "combined_harry_potter.txt"

# 하이퍼파라미터 설정
BATCH_SIZE = 64
BLOCK_SIZE = 256     # 문맥 길이 (Context Window)
EMB_DIM = 384        # 임베딩 차원
N_HEAD = 6           # Attention Head 수
N_LAYER = 6          # Transformer Layer 수
LEARNING_RATE = 3e-4 

EPOCHS = 10         
