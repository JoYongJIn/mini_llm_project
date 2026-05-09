# util폴더에는 모델이 학습된 후 텍스트를 생성(챗봇)하는 함수를 작성합니다. generate.py에서는 모델이 학습된 후에 텍스트를 생성하는 함수인 generate와 chat 함수를 정의합니다.
# utils에 함수들 정의해 놓고 가져다쓰는것이다.
import torch
import torch.nn.functional as F

def generate(model, tokenizer, prompt, seq_len, max_len=20, temperature=0.5): # temperature는 생성되는 텍스트의 다양성을 조절하는 하이퍼파라미터로, temperature가 높을수록 더 다양한 텍스트가 생성되고, 낮을수록 더 보수적인 텍스트가 생성됩니다. temperature가 높을수록 logits이 더 평평해지고, 그 결과로 softmax에서 생성되는 확률 분포가 더 균등해지기 때문입니다. 이렇게 하면 모델이 생성하는 텍스트가 더 다양해질 수 있습니다. 만약 temperature를 0으로 설정하면 모델은 항상 가장 높은 확률을 가진 토큰을 선택하게 됩니다.
    model.eval() # 모델을 평가 모드로 설정하여 드롭아웃과 배치 정규화를 비활성화합니다.

    tokens = tokenizer.encode(prompt)

    for _ in range(max_len):
        device = next(model.parameters()).device # 이 코드는 모델의 파라미터가 위치한 디바이스를 자동으로 감지하여 x 텐서를 해당 디바이스로 이동시키는 역할을 합니다. 이렇게 하면 모델이 CPU에서 학습되었든 GPU에서 학습되었든 상관없이 올바른 디바이스에서 텐서를 처리할 수 있습니다. 모델이 GPU에서 학습된 경우, 이 코드는 x 텐서를 GPU로 이동시켜 모델과 동일한 디바이스에서 연산이 이루어지도록 합니다. 반대로 모델이 CPU에서 학습된 경우, 이 코드는 x 텐서를 CPU에 유지시킵니다. 이렇게 함으로써 코드의 유연성이 향상되고, 다양한 환경에서 모델을 실행할 수 있게 됩니다.
        # next함수는 다음 값을 꺼내는 함수로 모델의 파라미터 중 하나를 반환하는데, 이 파라미터의 device 속성을 통해 모델이 위치한 디바이스를 알 수 있습니다. 이렇게 감지된 디바이스로 x 텐서를 이동시킴으로써 모델과 텐서가 동일한 디바이스에서 연산되도록 보장할 수 있습니다.
        x = torch.tensor(tokens[-seq_len:], dtype=torch.long).unsqueeze(0).to(device) # 여기서 unsqueeze(0)을 한 이유는 모델이 배치 형태의 입력을 기대하기 때문입니다. unsqueeze(0)을 사용하면 tokens[-seq_len:]의 차원이 (seq_len,)에서 (1, seq_len)로 변경되어 모델에 입력할 수 있게 됩니다. 이렇게 하면 모델이 하나의 시퀀스를 배치로 처리할 수 있습니다.

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