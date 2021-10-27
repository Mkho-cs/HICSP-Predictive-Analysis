import pandas as pd
import numpy as np
import requests
import io

class DataLoader:
    def __init__(self, url: str) -> None:
        requested = requests.get(url).content
        requested = io.StringIO(requested.decode('utf-8'))
        self.data = pd.read_csv(requested)
        
    
    def display(self, num: int, head: bool)->None:
        disp = self.data.head(num) if head else self.data.tail(num)
        print(disp)
    
    def describe(self)->None:
        print(self.data.describe(include='all'))
        