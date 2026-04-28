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
        B, T, D = x.shape

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = Q @ K.transpose(-2, -1) / (D ** 0.5)

        mask = causal_mask(T).to(x.device)
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
    def __init__(self, vocab_size, d_model, seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.blocks = nn.Sequential(
            Block(d_model),
            Block(d_model)
        )

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape

        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)

        return self.head(x)