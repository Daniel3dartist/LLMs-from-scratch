from typing import Type
from pathlib import Path

from data_manager import DataManager
from reader import Reader
from tokenizer import create_dataloader_v1


def main():
    script_directory = str(Path(__file__).parent.parent.resolve()).replace('\\', '/')
    path:str = f'{script_directory}/'
    file_name:str = 'the-verdict.txt'
    reader:Type[Reader] = Reader(path, file_name)
    text:str = reader.get_text()
    #vocab:dict = DataManager().build_vocab(text)
    print("="*100)
    dataloader = create_dataloader_v1(
        text,
        batch_size=1,
        max_lenght=4,
        stride=1,
        shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)   

    print("="*100)

    dataloader = create_dataloader_v1(
        text,
        batch_size=8,
        max_lenght=4,
        stride=4,
        shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    print("="*100)
    
    


main()