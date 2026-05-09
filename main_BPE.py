# BPE 안붙은것들도 이용가능하나 BPE붙은 파일은 반드시써야함 그렇게 호환시켜져있음
# basic_tokenizer를 이용했을때보다 성능이 훨씬 좋아지는것으로 보임. 역시 조사까지 나누어 다른 조사가 붙었다고 같은 단어를 다른 토큰벡터로 일일히 매칭시킬때보다 성능이 올라감 그로인해 조사 임베딩 벡터가 새로 생겼으나 효과적 학습 규칙이 생긴것으로 보임
# 조사도 토큰화됨으로써 조사가 붙는 자리의 개념도 학습이될수있게되며 그렇게 조사가 붙은 앞은 명사라는 개념과 그러한 자리에 들어간 단어들이 다음 블록에서 맥락적으로 이해되며 조사 앞자리에 들어가는 단어들의 공통적인 방향이 생기게 되어 의미있게 되었다고 볼수있다 (품사개념이 신경망에 학습될수있도록 되었다 : 조사까지 토큰화시킬수있도록 함으로써 생긴 영향)
# main파일에서만 조절로 MPS사용가능한가지 다른 파일들에는 MPS device코드 안넣어도

import torch
import os

from tokenizer.BPE import BPETokenizer
from models.basic_model import GPT # 내가 만든 모델 이름이 GPT임
from training.basic_training import train
from utils.basic_generate import chat
from configs.basic_config import seq_len, d_model, epochs

# -------------------------
# 데이터 불러오기
# -------------------------
file_path = os.path.join("data", "chat_bot.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# tokenizer (수정 핵심)
# -------------------------
tokenizer = BPETokenizer(num_merges=200)   # # num_merges는 BPE에서 병합할 횟수로, 이 값이 높을수록 더 긴 단어들이 토큰으로 만들어집니다. num_merges=200은 200번의 병합을 수행하여 어휘를 구축한다는 의미입니다. 이 값을 조절하여 어휘의 크기와 토큰화의 세밀함을 조절할 수 있습니다. num_merges 값이 작아질수록 단어사전의 토큰수가 적어졌었다. 그 이유는 num_merges가 작으면 병합이 적게 이루어져서 더 짧은 토큰들이 만들어지기 때문입니다. 예를 들어, num_merges=100으로 설정하면 100번의 병합이 이루어지므로, 단어들이 비교적 짧은 토큰으로 분할될 가능성이 높습니다. 반면에 num_merges=200으로 설정하면 200번의 병합이 이루어지므로, 단어들이 더 긴 토큰으로 병합될 가능성이 높아집니다. 따라서 num_merges 값을 조절하여 어휘의 크기와 토큰화의 세밀함을 조절할 수 있습니다.
tokenizer.build_vocab(text)                # tokenizer의 vocab을 구축하는 함수로, text를 입력받아서 vocab을 구축하는 함수로, 여기서 만들어진 vocab은 tokenizer.pt로 저장되어서 모델과 함께 불러와서 사용된다.

# 🔥 추가 (중요): tokenizer 저장
torch.save(tokenizer, "weight_bias/tokenizer1.pt")

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
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # MPS 사용하기
model = GPT(vocab_size, d_model, seq_len, n_layers=4).to(device)
X = X.to(device)
y = y.to(device) # 이야 MPS쓰니까 CPU로 학습할떄보다 거의 5배이상 압도적으로 빨라졌다. 이부분 지워서 CPU로 학습할때와 비교가능하다.

# -------------------------
# 학습
# -------------------------
patience = 10 # early stopping patience: loss가 10번 연속 좋아지지 않으면 학습 중단
train(model, X, y, vocab_size, epochs, patience=patience)

# early stopping에서 저장된 best_model을 최종 chat_bot.pt로 저장
model.load_state_dict(torch.load("weight_bias/best_model.pt", map_location=device))
torch.save(model.state_dict(), "weight_bias/chat_bot.pt") # 모델 파라미터 저장 (weight_bias 폴더에 gpt.pt로 저장)

# -------------------------
# 챗봇 실행
# -------------------------
chat(model, tokenizer, seq_len)

'''
# MPS 사용 여부 확인하는 코드로 훗날 CUDA를 이용할때도 이와같이 확인해보자
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
print("Using device:", device)
'''