# util폴더에는 모델이 학습된 후 텍스트를 생성하는 함수를 작성합니다. generate.py에서는 모델이 학습된 후에 텍스트를 생성하는 함수인 generate와 chat 함수를 정의합니다.
import torch
import torch.nn.functional as F

def generate(model, tokenizer, prompt, seq_len, max_len=20, temperature=1.0):
    model.eval()

    tokens = tokenizer.encode(prompt)

    for _ in range(max_len):
        x = torch.tensor(tokens[-seq_len:]).unsqueeze(0)

        out = model(x)
        logits = out[0, -1] / temperature
        probs = F.softmax(logits, dim=0)

        next_token = torch.multinomial(probs, 1).item()
        tokens.append(next_token)

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