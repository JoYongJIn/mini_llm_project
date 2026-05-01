# main.py는 전체적인 흐름을 제어하는 파일입니다. 데이터 불러오기, 토크나이저 생성, 모델 생성, 학습, 챗봇 실행 등 모든 과정을 이 파일에서 관리합니다.
import torch
import os

from tokenizer.basic_tokenizer import SimpleTokenizer
from models.basic_model import GPT
from training.basic_training import train
from utils.basic_generate import chat # generate도 불러와서 사용
from configs.basic_config import seq_len, d_model, epochs # error가 뜨는 경우 보통 스펠링을 틀린경우가 많으니 확인해보자

# 데이터 불러오기
file_path = os.path.join("data", "bank_service.txt") # 이 부분을 손보면 다른 데이터로도 학습시킬수있다.
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# tokenizer
tokenizer = SimpleTokenizer(text) # 이 부분을 손보면 다른 tokenizer로도 학습시킬수있다.
vocab_size = len(tokenizer.vocab)

# 데이터 생성
data = tokenizer.encode(text) # 토크나이저를 이용해 텍스트를 숫자로 변환한다.

X, y = [], [] # 이 부분은 모델이 학습할 수 있는 형태로 데이터를 변환하는 부분입니다. seq_len 길이만큼의 입력과 그 다음 토큰을 출력으로 만들어줍니다. 예를 들어, "The cat sat on the mat"라는 문장이 있고 seq_len이 3이라면, X에는 ["The", "cat", "sat"], ["cat", "sat", "on"], ["sat", "on", "the"] 등이 들어가고, y에는 ["cat"], ["sat"], ["on"] 등이 들어갑니다. 이렇게 하면 모델이 "The cat sat"을 입력으로 받았을 때 다음 토큰인 "on"을 예측하도록 학습할 수 있습니다.
for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len]) # 학습 구조를 보면 모델이 seq_len 길이만큼의 입력을 받아서 다음 토큰을 예측하도록 되어 있습니다. 
    y.append(data[i+1:i+seq_len+1]) # 이 부분은 모델이 다음 토큰을 예측하도록 학습하는 부분입니다.

X = torch.tensor(X)
y = torch.tensor(y)

# 모델
model = GPT(vocab_size, d_model, seq_len)

# 학습
train(model, X, y, vocab_size, epochs)

# 챗봇 실행
chat(model, tokenizer, seq_len)
