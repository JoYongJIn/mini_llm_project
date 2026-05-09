# 만들어진 모델을 학습시키는 파일로 하이퍼파리미터등을 조절할수있다.

import torch
import torch.nn as nn
from models.basic_model import GPT # 연결된 모델에서 내부에 있는 클래스(GPT)를 불러와야 하는구나

def train(model, X, y, vocab_size, epochs=400, lr=1e-3, patience=10): # 함수로 저장해 두어서 함수를 호출하며 하이퍼파라미터를 조절할수있다.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf') # 가장 좋은 loss 저장
    patience_counter = 0     # early stopping 카운터

    for epoch in range(epochs):
        model.train() # 학습 모드

        optimizer.zero_grad()

        out = model(X)

        loss = criterion(out.view(-1, vocab_size), y.view(-1))

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(epoch, loss.item())

        # early stopping 핵심 부분
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0

            # 가장 좋은 모델 저장
            torch.save(model.state_dict(), "weight_bias/best_model.pt")
        else:
            patience_counter += 1

        # patience 만큼 개선 없으면 중단
        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break