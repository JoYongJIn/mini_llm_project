class SimpleTokenizer:
    def __init__(self, text=None):
        self.UNK = "<UNK>"
        self.vocab = []
        self.stoi = {}
        self.itos = {}

        if text is not None:
            self.build_vocab(text)

    def build_vocab(self, text):
        tokens = text.split()
        self.vocab = sorted(list(set(tokens)))

        if self.UNK not in self.vocab:
            self.vocab.append(self.UNK)

        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, s):
        return [self.stoi.get(w, self.stoi[self.UNK]) for w in s.split()]

    def decode(self, l):
        return " ".join([self.itos[i] for i in l])