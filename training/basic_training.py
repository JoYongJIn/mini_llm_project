# 만들어진 모델을 학습시키는 파일로 하이퍼파리미터등을 조절할수있다.

import torch
import torch.nn as nn
from models.basic_model import GPT # 연결된 모델에서 내부에 있는 클래스(GPT)를 불러와야 하는구나

def train(model, X, y, vocab_size, epochs=1000): # 함수로 저장해 두어서 함수를 호출하며 하이퍼파라미터를 조절할수있다.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(X)

        loss = criterion(out.view(-1, vocab_size), y.view(-1))

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(epoch, loss.item())