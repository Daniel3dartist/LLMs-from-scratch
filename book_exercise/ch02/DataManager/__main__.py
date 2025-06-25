from typing import Type
from pathlib import Path

from data_manager import DataManager
from reader import Reader
from tokenizer import tokenizer, SimpleTokenizerV1, SimpleTokenizerV2


def execute_test(input:str, tokenizer):
    print('='*100)
    print('Input: "' + input + '"')
    print('-'*100)
    ids:list = tokenizer.encode(input)
    print('ENCODER: ', ids)
    print('-'*100)
    print('DECODE: ', tokenizer.decode(ids))
    print('='*100)

def test_tokenize_01(vocab:dict):
    tokenizer_v1:Type[SimpleTokenizerV1] = SimpleTokenizerV1(vocab)
    string_input:str = """
    It's the last he painted, you know," 
    Mrs. Gisburn said with pardonable pride.
    """
    execute_test(string_input, tokenizer_v1)


def test_tokenize_02(vocab:dict):
    tokenizer_v2:Type[SimpleTokenizerV2] = SimpleTokenizerV2(vocab)
    string_input_01 = "Hello, do you like tea?"
    string_input_02 = "In the sunlit terraces of the palace."

    text = " <|endoftext|> ".join((string_input_01, string_input_02))
    print('\n\nTEXT: \n' + text + '\n\n')
    execute_test(text, tokenizer_v2)

def main():
    script_directory = str(Path(__file__).parent.parent.resolve()).replace('\\', '/')
    path:str = f'{script_directory}/'
    file_name:str = 'the-verdict.txt'
    reader:Type[Reader] = Reader(path, file_name)
    text:str = reader.get_text()
    vocab:dict = DataManager().build_vocab(text)
    print("="*100)
    test_tokenize_01(vocab)
    test_tokenize_02(vocab)
    
    


main()