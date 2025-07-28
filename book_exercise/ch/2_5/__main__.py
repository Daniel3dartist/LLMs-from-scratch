from typing import Type
from pathlib import Path

from data_manager import DataManager
from reader import Reader
from tokenizer import create_dataloader_v1
import torch


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
    
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    max_length = 4
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    print(pos_embedding_layer.weight)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)
    print(pos_embeddings)
    print('='*60)
    input_embeddings = token_embeddings + pos_embeddings
   
    print(input_embeddings)
    print(input_embeddings.shape)
    


main()