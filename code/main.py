import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import scipy.optimize as opt
import datetime as dt
import requests
import json
import os
from typing import Tuple, List, Dict
import abc

def load_data():
    data = pd.read_csv('code/data/crypto.csv', index_col='dt', parse_dates=['dt'])
    data.dropna(inplace=True)
    data.sort_index(inplace=True)

    N = data.shape[1]
    ASSET_NAMES = data.columns.tolist()

    print('observations', data.shape[0])
    print('from', data.index.min())
    print('till', data.index.max())
    print('ccys', ASSET_NAMES)
    return data

def timeseries_split(df:pd.DataFrame, input_width:int, offset:int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    size = df.shape[0]
    total_width = input_width + offset
    for i in range(size - total_width):
        x_hist = df.iloc[i:i+input_width].values
        x_now = df.iloc[i + input_width].values.reshape((1, -1))
        x_test = df.iloc[i+total_width].values
        yield i, x_hist, x_now, x_test

class Evaluator:
    def __init__(self):
        self.rois = {}
        self.prices = {
            'open': [],
            'close': [],
        }

    def evaluate(self, strategy, w, x_now, x_future):
        dprice = x_future - x_now
        roi = dprice / x_now
        w = w.reshape((-1, 1))  
        roi_port = (roi @ w).item()
        if strategy not in self.rois:
            self.rois[strategy] = []
        self.rois[strategy].append(roi_port)

    def save_results(self):
        pd.concat([
            pd.DataFrame(self.rois),
            # pd.DataFrame(self.prices)
        ], axis=1).to_csv('results.csv', index=False)
        # pd.DataFrame(self.rois).to_csv('results.csv')

class Strategy:
    def __init__(self, name):
        self.name = name
        
    def predict(self, x_hist):
        self.n = x_hist.shape[1]
        w = np.zeros(self.n)
        w[0] = 1
        return w
    
    def __hash__(self):
        return hash(self.__str__())
    def __repr__(self):
        return f'{self.name}'
    def __str__(self):
        return self.__repr__()

def main():
    data = load_data()

    input_width = 10
    offset = 7

    strategies = [
        Strategy('test'),
        Strategy('other'),
    ]
    evaluator = Evaluator()

    for i, x_hist, x_now, x_future in timeseries_split(data, input_width, offset):
        evaluator.update_prices(x_now, x_future)
        for strategy in strategies:
            w_predict = strategy.predict(x_hist)
            evaluator.evaluate(strategy, w_predict, x_now, x_future)
        
    evaluator.save_results()



if __name__ == '__main__':
    main()