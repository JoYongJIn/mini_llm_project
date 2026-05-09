# 이 파일은 basic_trainin.py와 달리 finetuning할때 사용하는 파일로 learning rate를 1e-3에서 1e-4로 낮추어서 모델이 이미 학습된 weight를 너무 크게 변경하지 않도록 한다.

import torch
import torch.nn as nn
from models.basic_model import GPT # 연결된 모델에서 내부에 있는 클래스(GPT)를 불러와야 하는구나

def train(model, X, y, vocab_size, epochs=20): # fine tuning에서는 epochs도 줄이는 경우가 많다. fine tuning은 이미 학습된 모델을 기반으로 새로운 데이터에 맞게 조정하는 과정이므로, 일반적으로 학습률을 낮추고 epochs를 줄이는 것이 좋습니다. 이렇게 하면 모델이 기존에 학습한 내용을 급격하게 변경하지 않고, 새로운 데이터에 적응할 수 있습니다. 따라서 fine tuning에서는 작은 학습률과 적절한 epochs 값을 선택하는 것이 중요합니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # fine tuning시에는 작은 학습률을 사용하는 것이 일반적
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(X)

        loss = criterion(out.view(-1, vocab_size), y.view(-1))

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(epoch, loss.item())