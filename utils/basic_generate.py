# util폴더에는 모델이 학습된 후 텍스트를 생성하는 함수를 작성합니다. generate.py에서는 모델이 학습된 후에 텍스트를 생성하는 함수인 generate와 chat 함수를 정의합니다.
import torch
import torch.nn.functional as F

def generate(model, tokenizer, prompt, seq_len, max_len=20, temperature=0.9):
    model.eval()

    tokens = tokenizer.encode(prompt)

    for _ in range(max_len):
        x = torch.tensor(tokens[-seq_len:]).unsqueeze(0) # 여기서 unsqueeze(0)을 한 이유는 모델이 배치 형태의 입력을 기대하기 때문입니다. unsqueeze(0)을 사용하면 tokens[-seq_len:]의 차원이 (seq_len,)에서 (1, seq_len)로 변경되어 모델에 입력할 수 있게 됩니다. 이렇게 하면 모델이 하나의 시퀀스를 배치로 처리할 수 있습니다.

        out = model(x)
        logits = out[0, -1] / temperature
        probs = F.softmax(logits, dim=0) # probabilities로 변환

        next_token = torch.multinomial(probs, 1).item() # multinomial(다항분포) 함수를 사용하여 확률 분포에서 다음 토큰을 샘플링합니다. temperature를 조절하여 생성되는 텍스트의 다양성을 조절할 수 있습니다. temperature가 높을수록 더 다양한 텍스트가 생성되고, 낮을수록 더 보수적인 텍스트가 생성됩니다. multinomial 함수는 확률 분포에서 샘플링을 수행하는 함수로, probs에서 다음 토큰을 샘플링합니다. 이렇게 하면 모델이 생성하는 텍스트가 더 다양해질 수 있습니다. 만약 temperature를 0으로 설정하면 모델은 항상 가장 높은 확률을 가진 토큰을 선택하게 됩니다.
        tokens.append(next_token) # temperature 조절은 위에 함수 변수에서 temperature=1.0으로 설정해주면 됩니다. temperature가 높을수록 더 다양한 텍스트가 생성되고, 그 이유는 temperature가 높을수록 logits이 더 평평해지고, 그 결과로 softmax에서 생성되는 확률 분포가 더 균등해지기 때문입니다.
# temperature(어원은 물리에서옴 볼츠만분포)는 softmax의 지수부분을 T로 나누는 것으로 함수값들간 차이를 줄이는 효과가있다 이때 나누는 T의 크기를 변수로 설정해주는것이다. temperature가 높을수록 logits이 더 평평해지고, 그 결과로 softmax에서 생성되는 확률 분포가 더 균등해지기 때문입니다. 이렇게 하면 모델이 생성하는 텍스트가 더 다양해질 수 있습니다.
        if tokenizer.itos[next_token] == "<EOS>":
            break

    return tokenizer.decode(tokens)


def chat(model, tokenizer, seq_len):
    print("챗봇 시작 (exit 입력하면 종료)")

    while True:
        user_input = input("User: ")

        if user_input == "exit":
            break

        prompt = f"<SOS> <USER> {user_input} <BOT>"

        output = generate(model, tokenizer, prompt, seq_len)

        response = output.split("<BOT>")[-1]
        response = response.replace("<EOS>", "").strip()

        print("Bot:", response)