# IMPORTS
import pandas as pd
import math
import os.path
import time
import ta
# from technical_indicators_lib.indicators import RSI,ATR,CHO,KST,EMA,DPO,ROCV,StochasticKAndD,TR,TSI,MACD,MOM,PO,EMA
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
import numpy as np
from ta.trend import MACD,EMAIndicator,IchimokuIndicator
from ta.momentum import RSIIndicator as RSI
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def MACD_IND(data,win_slow,win_fast,win_sign):
    MACD_IND1 = MACD(data['close'],window_slow=win_slow,window_fast=win_fast,window_sign=win_sign)
    data['MACD']         = MACD_IND1.macd()
    data['MACD_signal']  = MACD_IND1.macd_signal()
    data['MACD_Hist']    = MACD_IND1.macd_diff()
    data['MACD_ZC']      = np.where((data['MACD_Hist']*(data['MACD_Hist'].shift(1,axis=0))) < 0,1,0)
    return data

def Ichi(data,win1,win2,win3):
    Ichimoku_IND1 = IchimokuIndicator(high=data['high'], low=data['low'], window1=win1, window2=win2, window3=win3)
    data['Ichimoku_a']               = Ichimoku_IND1.ichimoku_a()
    data['Ichimoku_b']               = Ichimoku_IND1.ichimoku_b()
    data['Ichimoku_base_line']       = Ichimoku_IND1.ichimoku_base_line()
    data['Ichimoku_conversion_line'] = Ichimoku_IND1.ichimoku_conversion_line()
    return data



class DataMiner():

    def __init__(self, binance_symbols=['LTCUSDT'], time_step=['1h'], start_date='1 Oct 2020'):
        self.timestep = time_step
        self.binance_client = Client(api_key='43PXiL32cF1YFXwkeoK900wOZx8saS1T5avSRWlljStfwMrCl7lZhhJSIM1ijIzS',
                        api_secret='JjJRJ3bWQTEShF4Eu8ZigY9aEMGPnFNJMH3WoNlOQgxSgrHmLOflIavhMx0KSZFC')
        self.binance_symbols = ['LTCUSDT']
        self.start_date = '1 Oct 2020'
        self.data_steps = ['1h']

    def get_data(self):
        for symbol in self.binance_symbols:
            for step in self.timestep:


    @staticmethod
    def minutes_of_new_data(symbol, kline_size, data, start_date, source):
        if len(data) > 0:
            old = parser.parse(data["timestamp"].iloc[-1])
        elif source == "binance":
            old = datetime.strptime(start_date, '%d %b %Y')
        if source == "binance": new = pd.to_datetime(
            binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0],
            unit='ms')
        return old, new

    @staticmethod
    def get_all_binance(symbol, kline_size, start_date, save=False):
        filename = f'{symbol}-{kline_size}-data-from-{start_date}.csv'
        if os.path.isfile(filename):
            data_df = pd.read_csv(filename)
        else:
            data_df = pd.DataFrame()
        oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, start_date, source="binance")
        delta_min = (newest_point - oldest_point).total_seconds() / 60
        available_data = math.ceil(delta_min / binsizes[kline_size])
        if oldest_point == datetime.strptime(start_date, '%d %b %Y'):
            print(f'Downloading all available {kline_size} data for {symbol} from {start_date}. Be patient..!')
        else:
            print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (
                delta_min, symbol, available_data, kline_size))
        klines = binance_client.get_historical_klines(symbol, kline_size,
                                                      oldest_point.strftime("%d %b %Y %H:%M:%S"),
                                                      newest_point.strftime("%d %b %Y %H:%M:%S"))
        data = pd.DataFrame(klines,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_av',
                                     'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        if len(data_df) > 0:
            temp_df = pd.DataFrame(data)
            data_df = data_df.append(temp_df)
        else:
            data_df = data
        data_df.set_index('timestamp', inplace=True)
        if save: data_df.to_csv(filename)
        print('All caught up..!')
        return data_df


df = pd.read_csv('XABC/ADAUSDT-1h-data-from-1 Jan 2021.csv')
df2 = pd.DataFrame()
timestep = 2
print(df.iloc[:,:5])
# print(df['open'].rolling(window=1).apply(lambda x: x.iloc[0])[::2])
df2['open'] = df['open'][::2]
df2['high'] = df['high'].rolling(2).max()[::2]
df2['low'] = df['low'].rolling(2).min()[::2]
df2['close'] = df['close'][::2]
print(df2)