# BPE 안붙은것들도 이용가능하나 BPE붙은 파일은 반드시써야함 그렇게 호환시켜져있음
# basic_tokenizer를 이용했을때보다 성능이 훨씬 좋아지는것으로 보임. 역시 조사까지 나누어 다른 조사가 붙었다고 같은 단어를 다른 토큰벡터로 일일히 매칭시킬때보다 성능이 올라감 그로인해 조사 임베딩 벡터가 새로 생겼으나 효과적 학습 규칙이 생긴것으로 보임
# 조사도 토큰화됨으로써 조사가 붙는 자리의 개념도 학습이될수있게되며 그렇게 조사가 붙은 앞은 명사라는 개념과 그러한 자리에 들어간 단어들이 다음 블록에서 맥락적으로 이해되며 조사 앞자리에 들어가는 단어들의 공통적인 방향이 생기게 되어 의미있게 되었다고 볼수있다 (품사개념이 신경망에 학습될수있도록 되었다 : 조사까지 토큰화시킬수있도록 함으로써 생긴 영향)

import torch
import os

from tokenizer.BPE import BPETokenizer
from models.basic_model import GPT
from training.basic_training import train
from utils.basic_generate import chat
from configs.basic_config import seq_len, d_model, epochs

# -------------------------
# 데이터 불러오기
# -------------------------
file_path = os.path.join("data", "bank_service_1.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# tokenizer (수정 핵심)
# -------------------------
tokenizer = BPETokenizer(num_merges=200)   # ❗ text 넣는거 아님
tokenizer.build_vocab(text)                # ❗ 반드시 먼저 호출

vocab_size = len(tokenizer.vocab)

print("Vocab size:", vocab_size)  # 디버깅용

# -------------------------
# 데이터 생성
# -------------------------
data = tokenizer.encode(text)

print("Sample encoded:", data[:20])  # 디버깅용

X, y = [], []
for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len])
    y.append(data[i+1:i+seq_len+1])

X = torch.tensor(X)
y = torch.tensor(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------------
# 모델
# -------------------------
model = GPT(vocab_size, d_model, seq_len)

# -------------------------
# 학습
# -------------------------
train(model, X, y, vocab_size, epochs)

# -------------------------
# 챗봇 실행
# -------------------------
chat(model, tokenizer, seq_len)