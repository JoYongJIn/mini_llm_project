# tokenizer는 모델을 학습할때 학습데이터 토큰화 뿐만 아니라 내 입력을 모델이 인식할때도 쓰인다. 그래서 tokenizer는 모델과 함께 저장되어야한다. tokenizer는 모델이 학습된 후에도 텍스트를 인코딩하고 디코딩하는 데 사용되기 때문에, 모델과 함께 저장하는 것이 중요합니다. 이렇게 하면 나중에 모델을 불러올 때 동일한 tokenizer를 사용할 수 있어, 일관된 토큰화와 디토큰화를 보장할 수 있습니다. 만약 tokenizer가 저장되지 않으면, 모델이 학습된 후에 텍스트를 인코딩하거나 디코딩할 때 문제가 발생할 수 있습니다. 따라서 tokenizer를 모델과 함께 저장하는 것은 모델의 활용성과 일관성을 유지하는 데 필수적입니다.
# tokenizer 일관성의 중요함에 대한 내용이 위 설명이다.
class BPETokenizer:
    def __init__(self, num_merges=200): # num_merges는 BPE에서 병합할 횟수로, 이 값이 높을수록 더 긴 단어들이 토큰으로 만들어집니다. num_merges=200은 200번의 병합을 수행하여 어휘를 구축한다는 의미입니다.
        self.num_merges = num_merges
        self.vocab = {} # vocab은 단어와 인덱스의 매핑을 저장하는 딕셔너리로, 모델이 텍스트를 숫자로 변환할 때 사용됩니다.
        self.inv_vocab = {} # inv_vocab은 인덱스와 단어의 매핑을 저장하는 딕셔너리로, 모델이 예측한 숫자를 다시 텍스트로 변환할 때 사용됩니다.
        self.merges = {} # merges는 BPE 병합 규칙을 저장하는 딕셔너리로, 어떤 토큰들이 병합되어 새로운 토큰이 만들어지는지를 나타냅니다. 예를 들어, ('h', 'e')가 ('he')로 병합되는 규칙이 있다면, merges 딕셔너리에 {('h', 'e'): 'he'}와 같이 저장됩니다. 이 정보는 토큰화 과정에서 단어를 어떻게 분할하고 병합할지를 결정하는 데 사용됩니다.
        self.UNK = "<UNK>" # UNK는 "unknown"의 약자로, 토크나이저가 처리할 수 없는 단어나 토큰을 나타내는 특별한 토큰입니다. 모델이 학습된 어휘에 없는 단어를 만나면, 해당 단어를 UNK 토큰으로 대체하여 처리합니다. 이렇게 하면 모델이 예측할 수 없는 단어에 대해서도 일관된 방식으로 대응할 수 있습니다. UNK 토큰은 모델이 새로운 단어나 희귀한 단어를 만났을 때도 안정적으로 작동하도록 도와줍니다.

    def get_stats(self, tokens):
        pairs = {}
        for token in tokens:
            symbols = token.split() # 토큰을 공백으로 분리하여 심볼 리스트로 만듭니다. 예를 들어, "h e l l o </w>"라는 토큰이 있다면, symbols 리스트는 ['h', 'e', 'l', 'l', 'o', '</w>']가 됩니다. 이렇게 분리된 심볼들은 BPE 병합 규칙을 적용하는 데 사용됩니다.
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] = pairs.get(pair, 0) + 1 # pairs 딕셔너리에 각 심볼 쌍의 빈도를 계산하여 저장합니다. 예를 들어, ('h', 'e')라는 쌍이 여러 토큰에서 나타난다면, pairs 딕셔너리에서 ('h', 'e')의 값은 해당 쌍이 나타난 총 횟수가 됩니다. 이렇게 계산된 빈도는 BPE 병합 규칙을 학습할 때 가장 빈도가 높은 쌍을 선택하는 데 사용됩니다.
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
        vocab = set() # set은 중복을 허용하지 않는 자료구조로, 여기서는 토큰들의 중복을 제거하여 고유한 토큰 집합을 만들기 위해 사용됩니다. 이렇게 하면 최종적으로 생성되는 vocab에는 각 토큰이 한 번씩만 포함되게 됩니다. BPE 병합 과정을 거치면서 생성된 토큰들이 중복될 수 있기 때문에, set을 사용하여 고유한 토큰 집합을 만드는 것이 중요합니다.
        for token in tokens:
            vocab.update(token.split()) # update 메서드는 set에 여러 요소를 한 번에 추가할 때 사용됩니다.

        self.vocab = {w: i for i, w in enumerate(sorted(vocab))} # vocab 딕셔너리를 단어와 인덱스의 매핑으로 생성합니다. sorted 함수를 사용하여 토큰들을 정렬한 후, enumerate 함수를 사용하여 각 토큰에 고유한 인덱스를 할당합니다. 이렇게 하면 vocab 딕셔너리는 각 단어를 해당하는 인덱스와 매핑하는 형태로 만들어집니다. 예를 들어, "h"라는 토큰이 0번 인덱스에 할당되고, "e"라는 토큰이 1번 인덱스에 할당되는 식입니다. 이 매핑은 모델이 텍스트를 숫자로 변환할 때 사용됩니다.

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
                if pair in self.merges: # merges 딕셔너리에 있는 병합 규칙이 현재 단어의 토큰 쌍과 일치하는지 확인합니다. 만약 일치하는 규칙이 있다면, 해당 쌍을 병합하여 새로운 토큰으로 만들어줍니다. 이렇게 함으로써 단어가 점점 더 긴 토큰으로 변환됩니다.
                    word = word.replace(" ".join(pair), "".join(pair)) # join함수는 리스트의 요소들을 문자열로 결합하는 함수입니다. 여기서는 pair가 ('h', 'e')라면 " ".join(pair)는 "h e"가 되고, "".join(pair)는 "he"가 됩니다. 따라서 word.replace(" ".join(pair), "".join(pair))는 word에서 "h e"를 "he"로 대체하는 역할을 합니다.
                    merged = True # replace 함수를 사용하여 단어에서 병합된 쌍을 새로운 토큰으로 대체합니다. 이렇게 하면 단어가 점점 더 긴 토큰으로 변환됩니다. 만약 병합이 이루어졌다면 merged 플래그를 True로 설정하여, 다음 반복에서 다시 병합 규칙을 적용할 수 있도록 합니다. 만약 병합이 이루어지지 않았다면, 더 이상 병합할 쌍이 없다는 의미이므로 루프를 종료합니다.
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