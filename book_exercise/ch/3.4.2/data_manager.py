
from tokenizer import create_dataloader_v1, GPTDatasetV1

class DataManager:
    def build_vocab(self, txt:str) -> list:
        """
        preprocessed:list = tokenizer(txt)
        all_tokens:list = sorted(list(set(preprocessed)))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])
        vocab_size:int = len(all_tokens)
        print('Vocab size: ', vocab_size)
        vocab:dict = {token:interger for interger, token in enumerate(all_tokens)}
        #print('vocab items: ', vocab.items())
        print('='*30)
        for i, item in enumerate(list(vocab.items())[-5:]):
            print(' - ',item)
        print('='*30)
        return vocab
        """
        pass