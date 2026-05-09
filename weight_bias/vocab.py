# 이 파일은 tokenizer의 vocab을 확인하는 파일로, tokenizer.pt에서 vocab을 불러와서 vocab의 타입과 크기를 출력한다.

import sys
from pathlib import Path
import torch

# 현재 파일: /Users/pc/LLM_Project/weight_bias/vocab.py
# 프로젝트 루트: /Users/pc/LLM_Project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from tokenizer.BPE import BPETokenizer

tokenizer = torch.load(PROJECT_ROOT / "weight_bias" / "tokenizer1.pt")

print("vocab type:", type(tokenizer.vocab))
print("vocab size:", len(tokenizer.vocab))

# vocab 일부 출력
if isinstance(tokenizer.vocab, dict):
    print("sample vocab:", list(tokenizer.vocab.items())[:100])
else:
    print("sample vocab:", list(tokenizer.vocab)[:100])