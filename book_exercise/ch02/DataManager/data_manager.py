
from tokenizer import tokenizer, SimpleTokenizerV1

class DataManager:
    def build_vocab(self, txt:str) -> list:
        preprocess:list = tokenizer(txt)
        all_words:list = sorted(set(preprocess))
        vocab_size:int = len(all_words)
        print('Vocab size: ', vocab_size)
        vocab:dict = {token:interger for interger, token in enumerate(all_words)}
        return vocab
    