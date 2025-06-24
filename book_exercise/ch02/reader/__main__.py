from typing import Type
from pathlib import Path

from reader import Reader


def main():
    script_directory = str(Path(__file__).parent.parent.resolve()).replace('\\', '/')
    path:str = f'{script_directory}/'
    file_name:str = 'the-verdict.txt'
    reader:Type[Reader] = Reader(path, file_name)
    data:list = reader.get_data()
    print('Data size: ', len(data))
    print(data[:30])



main()