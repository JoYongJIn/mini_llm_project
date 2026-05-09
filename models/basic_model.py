import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_mask(size):
    return torch.tril(torch.ones(size, size))

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape # B: 배치 크기, T: 시퀀스 길이, D: 모델의 차원입니다. 이 세 가지 값은 모델이 입력 데이터를 처리하는 데 필요한 정보를 제공합니다. B는 한 번에 처리하는 데이터 샘플의 수를 나타내고, T는 모델이 한 번에 처리하는 시퀀스의 길이를 나타내며, D는 모델이 각 토큰을 표현하는 데 사용하는 벡터의 차원을 나타냅니다.

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = Q @ K.transpose(-2, -1) / (D ** 0.5)

        mask = causal_mask(T).to(x.device) # mask가 있는것을 보면 GPT와 같은 autoregressive 모델이구나 라고 생각할수있다. causal_mask는 시퀀스의 길이에 맞는 하삼각 행렬을 생성하여, 모델이 현재 토큰 이후의 토큰을 참조하지 못하도록 하는 역할을 합니다. 이렇게 하면 모델이 이전 토큰들만을 기반으로 다음 토큰을 예측하도록 강제할 수 있습니다. GPT와 같은 autoregressive 모델에서는 이러한 causal_mask가 필수적입니다. 왜냐하면 모델이 미래의 토큰을 참조하지 않고, 오직 과거의 토큰들만을 기반으로 다음 토큰을 예측해야 하기 때문입니다. 만약 causal_mask가 없다면, 모델은 시퀀스 내의 모든 토큰을 참조할 수 있게 되어, autoregressive한 특성이 사라지게 됩니다. 이러한 학습방식과 BERT의 학습방식을 섞어 더해서 단어를 예측하는 LLM모델을 생각해볼수도 있다.
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        return attn @ V

class Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = SelfAttention(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x
    
class GPT(nn.Module): # 위 부품들을 조립한 최종 모델(부품들로 조립한 완전체 모델) => 이게 일반적으로 불러서 사용하는 학습구조 모델이다.
    def __init__(self, vocab_size, d_model, seq_len, n_layers=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.blocks = nn.ModuleList( # 트랜스포머 블록 n_layers번 통과시키고 있다.
            [Block(d_model) for _ in range(n_layers)]
        )

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape

        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))

        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)

        return self.head(x)