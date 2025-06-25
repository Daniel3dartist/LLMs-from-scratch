import re

def tokenizer(data:str) -> list:
    regex_string:str = r'([,.:;?_!"()\']|--|\s)'
    tokens_pre_cleaned:list = re.split(regex_string, data)
    return [i.strip() for i in tokens_pre_cleaned if i.strip()] # Remove spaces (" ") before return tokenized data



class SimpleTokenizerV1:
    def __init__(self, vocab:dict):
        self.str_to_int:dict = vocab
        self.int_to_str = {id:string for string, id in vocab.items()}

    def encode(self, txt:str) -> list:
        regex_string:str = r'([,.:;?_!"()\']|--|\s)'
        preprocessed:list = re.split(regex_string, txt)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids:list = [self.str_to_int[string] for string in preprocessed]
        return ids

    def decode(self, ids:list) -> str:
        regex_string:str = r'\s+([,.?!"()\'])'
        txt:str = " ".join([self.int_to_str[id] for id in ids])
        return re.sub(regex_string, r'\1', txt)