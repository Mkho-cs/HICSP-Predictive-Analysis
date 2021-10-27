import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import requests
import io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

class DataLoader:
    def __init__(self, url: str) -> None:
        requested = requests.get(url).content
        requested = io.StringIO(requested.decode('utf-8'))
        self.data = pd.read_csv(requested)
        
    
    def display(self, num: int, head: bool)->DataFrame:
        return self.data.head(num) if head else self.data.tail(num)
        
    
    def describe(self)->DataFrame:
        return self.data.describe(include='all')
    
    def column_bars(self, subrow, size = (48, 12))-> None:
        col_count = len(self.data.columns)
        subcol = math.ceil(col_count/subrow)
        fig, axs = plt.subplots(subrow, subcol, figsize = size)
        current = 0
        for subaxes in axs:
            for axis in subaxes:
                self.data.plot.bar(x = self.data.columns[current])
                current += 1


    def drop_column(self, cols: list)->DataFrame:
        return self.data.drop(cols, inplace = True, axis = 1)
    
    def display_types(self)-> pd.Series:
        return self.data.dtypes
    
    def column_to_string(self, col: list)->None:
        self.data[col] = self.data[col].astype(str)
    
    def column_to_cat(self, col: list)->None:
        self.data[col] = self.data[col].astype('category')