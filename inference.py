# inference.py
import torch
import torch.nn.functional as F
import tiktoken
import config
from src.model import MiniGPT
from src.dataset import GPTDataset # Vocab Size, Decoder를 위해 임시 로드

# -----------------------------------------
# 텍스트 생성 테스트
# -----------------------------------------
def run_inference(model, device, prompt, max_tokens):
    # 1. 토크나이저 초기화
    tokenizer = tiktoken.get_encoding("gpt2")
    VOCAB_SIZE = tokenizer.n_vocab # 50257

    # 2. 모델 로드 및 가중치 적용
    # 주의: 이 파일은 학습된 모델 가중치 파일("minigpt_final_weights.pth")이 필요합니다.
    try:
        model.load_state_dict(torch.load("minigpt_final_weights.pth", map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print("❌ 에러: 'minigpt_final_weights.pth' 파일을 찾을 수 없습니다. train.py를 먼저 실행하세요.")
        return

    # 3. 프롬프트 인코딩
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        # 프롬프트가 비었을 경우 <|endoftext|> 토큰으로 시작 (토큰 50256)
        input_ids = [VOCAB_SIZE - 1] 
    
    # 모델 입력 형태 (1, T)로 변환
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"\n--- Generating Text ---\nPrompt: {prompt}\n")
    
    # 4. 텍스트 생성
    with torch.no_grad():
        generated_tokens = model.generate(x, max_new_tokens=max_tokens)[0].tolist()
    
    # 5. 결과 디코딩 및 출력
    generated_text = tokenizer.decode(generated_tokens)
    print(generated_text)
    print("\n-------------------------")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 구조 생성 (가중치 로드를 위해)
    model = MiniGPT(
        vocab_size=50257, 
        emb_dim=config.EMB_DIM, 
        n_head=config.N_HEAD, 
        n_layer=config.N_LAYER, 
        block_size=config.BLOCK_SIZE
    )

    # 테스트 프롬프트 설정
    test_prompt = "Harry Potter was walking down the street when he saw"
    
    run_inference(model, device, test_prompt, max_tokens=100)