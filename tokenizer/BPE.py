class BPETokenizer:
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.vocab = {}
        self.inv_vocab = {}
        self.merges = {}
        self.UNK = "<UNK>"

    def get_stats(self, tokens):
        pairs = {}
        for token in tokens:
            symbols = token.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def merge_vocab(self, pair, tokens):
        new_tokens = []
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for token in tokens:
            # 안전하게 replace (부분문자열 오염 방지)
            new_token = token.replace(bigram, replacement)
            new_tokens.append(new_token)

        return new_tokens

    def build_vocab(self, text):
        # 1. 단어 → "문자 공백 분리 + </w>"
        tokens = text.split()
        tokens = [" ".join(list(word)) + " </w>" for word in tokens]

        # 2. BPE merge 학습
        for _ in range(self.num_merges):
            pairs = self.get_stats(tokens)
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merges[best] = "".join(best)
            tokens = self.merge_vocab(best, tokens)

        # 3. vocab 생성
        vocab = set()
        for token in tokens:
            vocab.update(token.split())

        self.vocab = {w: i for i, w in enumerate(sorted(vocab))}

        # UNK 추가 (안전장치)
        if self.UNK not in self.vocab:
            self.vocab[self.UNK] = len(self.vocab)

        self.inv_vocab = {i: w for w, i in self.vocab.items()}
        
        self.itos = self.inv_vocab # 토크나이저는 항상 두개가 한쌍인데 stoi는 단어->인덱스, itos는 인덱스->단어입니다. itos = "index to string" itos를 넣으면 모델이 낸 숫자를 사람이 읽을수있게해줌

    # 🔥 핵심 수정 부분
    def encode_word(self, word):
        # build_vocab과 동일한 포맷으로 시작
        word = " ".join(list(word)) + " </w>"

        while True:
            symbols = word.split()
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]

            merged = False
            for pair in pairs:
                if pair in self.merges:
                    word = word.replace(" ".join(pair), "".join(pair))
                    merged = True
                    break

            if not merged:
                break

        return word.split()

    def encode(self, text):
        tokens = []
        for word in text.split():
            tokens.extend(self.encode_word(word))

        # UNK fallback 안전 처리
        unk_id = self.vocab[self.UNK]
        return [self.vocab.get(t, unk_id) for t in tokens]

    def decode(self, ids):
        tokens = [self.inv_vocab.get(i, self.UNK) for i in ids]

        words = []
        current_word = ""

        for t in tokens:
            # 🔥 핵심: </w>가 토큰 안에 포함된 경우 처리
            if "</w>" in t:
                # </w> 제거하고 단어에 붙이기
                t_clean = t.replace("</w>", "")
                current_word += t_clean

                if current_word:
                    words.append(current_word)
                current_word = ""
            else:
                current_word += t

        # 혹시 남은 단어 처리
        if current_word:
            words.append(current_word)

        return " ".join(words)