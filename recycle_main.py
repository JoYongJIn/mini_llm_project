# 본 main.py파일인 recycle_main파일은 학습과 동시에 가중치저장하고 챗봇을 실행하는것이 아니라 학습후 저장된 가중치를 불러와 챗봇을 실행하는 파일이다.
# 이 파일은 finetuning_main.py와 다른점은 그냥 저장된 모델을 불러와서 챗봇을 실행하는것이다.

import torch
import os

from tokenizer.BPE import BPETokenizer
from models.basic_model import GPT
from utils.basic_generate import chat
from configs.basic_config import seq_len, d_model

# -------------------------
# 디바이스 설정 (MPS 포함)
# -------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 데이터 불러오기 (⚠️ tokenizer용)
# -------------------------
file_path = os.path.join("data", "bank_service_1.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# tokenizer 생성 (⚠️ 반드시 동일 vocab)
# -------------------------
tokenizer_path = "weight_bias/tokenizer1.pt"

if not os.path.exists(tokenizer_path):
    raise FileNotFoundError("❌ tokenizer 없음. 먼저 train 실행해라")

# 기존 build_vocab 대신 저장된 tokenizer 불러오기
tokenizer = torch.load(tokenizer_path)

vocab_size = len(tokenizer.vocab)

print("Vocab size:", vocab_size)

# -------------------------
# 모델 생성
# -------------------------
model = GPT(vocab_size, d_model, seq_len, n_layers=4).to(device)

# -------------------------
# 🔥 저장된 weight 불러오기
# -------------------------
model_path = "weight_bias/chat_bot.pt" # 여기에 불러올 저장된 파라미터의 경로와 파일명을 입력

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ 저장된 모델이 없음. 먼저 train 실행해라")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ 모델 로드 완료")

# -------------------------
# 챗봇 실행
# -------------------------
chat(model, tokenizer, seq_len)