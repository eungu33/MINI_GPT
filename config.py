# config.py

# 경로 설정
BASE_PATH = "HarryPotter"
MERGED_FILENAME = "combined_harry_potter.txt"

# 하이퍼파라미터 설정
#BATCH_SIZE = 64
#BLOCK_SIZE = 256     # 문맥 길이 (Context Window)
#EMB_DIM = 384        # 임베딩 차원
#N_HEAD = 6           # Attention Head 수
#N_LAYER = 6          # Transformer Layer 수
#LEARNING_RATE = 3e-4 

#EPOCHS = 10         

# config.py 수정

# 반복 횟수 줄이기
EPOCHS = 1          
# 모델 크기 줄이기
BLOCK_SIZE = 128     # 256 -> 128 (문맥 길이 줄임)
EMB_DIM = 256        # 384 -> 256 (모델 뚱뚱함 줄임)
N_LAYER = 4          # 6 -> 4 (층수 줄임)
N_HEAD = 4           # 6 -> 4

# 나머지는 그대로
BATCH_SIZE = 64
LEARNING_RATE = 5e-4 # 모델 작아졌으니 학습률 살짝 키움
