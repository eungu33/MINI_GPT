# inference.py
import torch
import torch.nn.functional as F
import tiktoken
import config
from src.model import MiniGPT

# -----------------------------------------
# 텍스트 생성 테스트
# -----------------------------------------
def run_inference(model, device, prompt, max_tokens):
    # 1. 토크나이저 초기화
    tokenizer = tiktoken.get_encoding("gpt2")
    VOCAB_SIZE = tokenizer.n_vocab # 50257

    # 2. 모델 로드 (torch.compile로 인한 _orig_mod 접두사 제거 로직 추가)
    try:
        # 가중치 파일 불러오기
        state_dict = torch.load("minigpt_final_weights.pth", map_location=device)
        
        # [핵심 수정] 키 이름에서 '_orig_mod.' 제거하기
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        print("✅ 모델 가중치 로드 성공!")
        
    except FileNotFoundError:
        print("❌ 에러: 'minigpt_final_weights.pth' 파일을 찾을 수 없습니다.")
        return

    # 3. 프롬프트 인코딩
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        # 프롬프트가 비었을 경우 <|endoftext|> 토큰으로 시작
        input_ids = [VOCAB_SIZE - 1] 
    
    # 모델 입력 형태 (1, T)로 변환
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"\n--- Generating Text ---\nPrompt: {prompt}\n")
    print(prompt, end="", flush=True) # 프롬프트 먼저 출력
    
    # 4. 텍스트 생성
    with torch.no_grad():
        # 한 글자씩 생성하면서 바로 출력하는 스트리밍 방식
        for _ in range(max_tokens):
            idx_cond = x[:, -config.BLOCK_SIZE:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, idx_next), dim=1)
            
            # 방금 만든 글자 디코딩해서 출력
            next_token = idx_next.item()
            print(tokenizer.decode([next_token]), end="", flush=True)
            
    print("\n\n-------------------------")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 구조 생성
    # 주의: config.py의 설정값과 학습할 때 설정값이 같아야 합니다.
    model = MiniGPT(
        vocab_size=50257, 
        emb_dim=config.EMB_DIM, 
        n_head=config.N_HEAD, 
        n_layer=config.N_LAYER, 
        block_size=config.BLOCK_SIZE
    )

    # 테스트할 문장을 여기에 적으세요
    test_prompt = "Harry Potter was"
    
    run_inference(model, device, test_prompt, max_tokens=200)
