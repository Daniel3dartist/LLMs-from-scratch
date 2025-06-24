from tokenizer import tokenizer

class Reader:
    def __init__(self, path:str='', file_name:str=''):
        self._path:str = path
        self._file_name:str = file_name

    def get_text(self, path:str='', file_name:str = '') -> str:
        if not path:
            path = self._path
        if not file_name:
            file_name = self._file_name

        file_path:str = f'{path}/{file_name}'

        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text:str = file.read()
        
        print("Total of char count: ", len(raw_text))
        print(raw_text[:99])

        return raw_text
    
    def get_data(self) -> list:
        txt:str = self.get_text()
        return tokenizer(txt)