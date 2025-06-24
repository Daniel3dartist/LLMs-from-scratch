import re

def tokenizer(data:str) -> list:
    regex_string:str = r'([,.:;?_!"()\']|--|\s)'
    tokens_pre_cleaned:list = re.split(regex_string, data)
    return [i.strip() for i in tokens_pre_cleaned if i.strip()] # Remove spaces (" ") before return tokenized data