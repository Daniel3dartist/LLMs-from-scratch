from typing import Type
from pathlib import Path

from data_manager import DataManager
from reader import Reader
from tokenizer import tokenizer, SimpleTokenizerV1


def main():
    tokenizer:Type[SimpleTokenizerV1]
    script_directory = str(Path(__file__).parent.parent.resolve()).replace('\\', '/')
    path:str = f'{script_directory}/'
    file_name:str = 'the-verdict.txt'
    reader:Type[Reader] = Reader(path, file_name)
    text:str = reader.get_text()
    vocab:dict = DataManager().build_vocab(text)
    tokenizer = SimpleTokenizerV1(vocab)
    string_input:str = """
        It's the last he painted, you know," 
        Mrs. Gisburn said with pardonable pride.
    """
    ids:list = tokenizer.encode(string_input)
    print('ENCODER: ', ids)
    print('='*80)
    print('DECODE: ', tokenizer.decode(ids))


main()