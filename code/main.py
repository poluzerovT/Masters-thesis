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
from strategies import *

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

def timeseries_split(df:pd.DataFrame, input_width:int, offset:int, delay:int):
    size = df.shape[0]
    total_width = input_width + offset
    for i in range(delay, size - total_width):
        x_hist = df.iloc[:i+input_width].values
        x_now = df.iloc[i + input_width].values.reshape((1, -1))
        x_test = df.iloc[i + total_width].values.reshape((1, -1))
        yield df.index[i], x_hist, x_now, x_test

class Evaluator:
    def __init__(self, data:pd.DataFrame, input_width:int, offset:int):
        self.data = data
        self.input_width = input_width
        self.offset = offset
        self.asset_names = data.columns.tolist()
        print(f'Created evaluator: {self.asset_names}')

    def evaluate(self, strategies):
        history = []
        cnt = 0
        for dt, x_hist, x_now, x_future in timeseries_split(self.data, self.input_width, self.offset, delay=2*self.input_width):
            hist = {}
            cnt += 1
            for strategy in strategies:
                strategy.fit(x_hist)
                w_predict = strategy.predict(x_now)

                hist[strategy.name] = {
                    'dt': dt.strftime(format='%Y-%m-%d'),
                    'w': pd.Series(w_predict.flatten(), index=self.asset_names).to_dict(),
                    'roi': self.port_roi_metric(w_predict, x_now, x_future),
                    'roi_mean_exp': strategy.mean_roi_est,
                    'roi_var_exp': strategy.var_roi_est
                }
                # hist[strategy.name].extend(strategy.metadata)
            history.append(hist)
            if cnt > 10:
                break
        self._history = history

    def port_roi_metric(self, w, open_price, close_price):
        diff_price = close_price - open_price
        roi = diff_price / open_price
        port_roi = (w @ roi.T).item()
        return port_roi

    def save_results(self, save_path):
        with open(save_path, 'w') as f:
            f.write(json.dumps(self._history, indent=4))


def main():
    data = load_data()

    input_width = 10
    offset = 7

    strategies = [
        MarkowitzClassic('mark_classic', input_width, offset, 1),
        SingleAsset('single', input_width, offset, 0),
        # SingleAsset('second', 1, input_width, offset),
    ]
    evaluator = Evaluator(data, input_width, offset)
    evaluator.evaluate(strategies)
    # evaluator.save_results('code/results/backtest.csv')
    evaluator.save_results('code/results/backtest.json')



if __name__ == '__main__':
    main()