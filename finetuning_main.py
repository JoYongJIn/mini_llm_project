# fine tuning_main.py 파일에서 학습률을 조절하는 방법은 어떻게 해야할까?(traing.py 파일에 있는 하이퍼파라미터인데)
# 이 파일은 finetuning하여 tuning된 가중치를 저장까지만 하고 대화실행은 recycle_main.py에서 한다.
import torch

from models.basic_model import GPT
from training.finetuning_training import train
from configs.finetuning_config import d_model, seq_len, epochs

# 1. tokenizer 불러오기
tokenizer = torch.load("weight_bias/tokenizer.pt") # tokenizer.pt에는 토크나이저 전체 상태가 저장되어 있다. vocab, inv_vocab, merges, num_merges 등등이 저장되어 있다. 
# 결론 tokenizer.pt = vocab + BPE 규칙 + 디코딩 정보 전부
vocab_size = len(tokenizer.vocab) # tokenizer.vocab은 단어사전에 있는 단어의 개수로 vocab_size에 저장된다. 모델의 출력 차원은 vocab_size가 된다. 모델이 예측하는 것은 다음 토큰의 확률 분포이므로, 출력 차원은 어휘 크기와 일치해야 한다. 따라서 vocab_size는 모델의 출력 레이어의 크기를 결정하는 중요한 하이퍼파라미터입니다. 만약 vocab_size가 너무 작으면 모델이 표현할 수 있는 단어의 다양성이 제한될 수 있고, 너무 크면 모델이 학습하기 어려워질 수 있습니다. 따라서 적절한 vocab_size를 선택하는 것이 중요합니다.

# 2. 모델 생성
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = GPT(vocab_size, d_model, seq_len).to(device)

# 3.기존 weight 불러오기
model.load_state_dict(torch.load("weight_bias/chat_bot.pt", map_location=device)) # 여기에는 내가 fine tuning하려는 모델의 저장된 weight의 경로와 파일명을 입력하면 된다.
# model.eval()

# -------------------------
# 🔥 새로운 데이터 생성 (추가된 부분)
# -------------------------
file_path = "data/lion.txt" # 여기에 fine tuning에 사용할 새로운 데이터 파일의 경로와 이름을 입력하면된다.
with open(file_path, "r", encoding="utf-8") as f:
    new_text = f.read()

data = tokenizer.encode(new_text)

X_new, y_new = [], []

for i in range(len(data) - seq_len):
    X_new.append(data[i:i+seq_len])
    y_new.append(data[i+1:i+seq_len+1])

X_new = torch.tensor(X_new).to(device)
y_new = torch.tensor(y_new).to(device)

# 4. 새로운 데이터로 이어서 학습
train(model, X_new, y_new, vocab_size, epochs)

# 5. 다시 저장
torch.save(model.state_dict(), "weight_bias/chat_bot_finetuned.pt")

'''
fine tuning시 주의할점은 catastrophic forgetting이 발생할 수 있다는 것이다. catastrophic forgetting은 모델이 새로운 데이터로 학습할 때, 이전에 학습한 내용을 잊어버리는 현상을 말한다. 이를 방지하기 위해서는 다음과 같은 방법들을 고려할 수 있다:
1. 작은 학습률 사용: fine tuning 시에는 작은 학습률을 사용하는 것이 좋다. 이렇게 하면 모델이 기존에 학습한 내용을 급격하게 변경하지 않고, 새로운 데이터에 적응할 수 있다. (새 데이터만 + lr낮게)
2. 정규화 기법 사용: L2 정규화나 드롭아웃과 같은 정규화 기법을 사용하여 모델이 새로운 데이터에 과적합되는 것을 방지할 수 있다.
3. 점진적 학습: 새로운 데이터로 모델을 학습할 때, 기존 데이터와 새로운 데이터를 혼합하여 학습하는 방법도 있다. 이렇게 하면 모델이 기존 데이터에 대한 지식을 유지하면서 새로운 데이터에 적응할 수 있다.(기존 데이터 70% + 새로운 데이터 30% 섞어서 학습)
4. 모델 아키텍처 조정: 모델의 아키텍처를 조정하여, 새로운 데이터에 대한 학습이 기존 데이터에 대한 학습을 방해하지 않도록 할 수 있다. 예를 들어, 모델의 일부 레이어를 고정하거나, 새로운 레이어를 추가하여 fine tuning을 수행할 수 있다.

fine tuning을 통해 내가 원하는 특정 도메인이나 작업에 모델을 맞출 수 있다.
domain adaptation(특정 분야만 강화 가능) => 금융 데이터 추가(금융 챗봇됨), 감정 대화 추가(공감 잘함)
'''