# your_tokenizer_file.py
import json

class CharTokenizer:
    def __init__(self, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        self.specials = specials
        self.vocab = None
        self.inv_vocab = None

    def build(self, texts):
        chars = sorted(set("".join(map(str, texts))))
        tokens = list(self.specials) + chars
        self.vocab = {tok: i for i, tok in enumerate(tokens)}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}
        return self.vocab

    def save(self, prefix):
        with open(prefix + "_vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load(self, prefix):
        with open(prefix + "_vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {int(v): k for k, v in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(ch, self.vocab["<unk>"]) for ch in str(text)]

    def decode(self, ids, stop_at_eos=True):
        out = []
        for i in ids:
            tok = self.inv_vocab.get(int(i), "<unk>")
            if tok in ("<pad>", "<sos>"):
                continue
            if tok == "<eos>" and stop_at_eos:
                break
            out.append(tok)
        return "".join(out)
