# region imports
import pandas as pd
import math
import os.path
import time
import ta
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
import numpy as np
from ta.trend import MACD,EMAIndicator,IchimokuIndicator
from ta.momentum import RSIIndicator as RSI
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# endregion

class DataHunter():

    def __init__(self, symbol: str, start_date: str, end_date: str, step: str,
                 binance_api_key: str= '43PXiL32cF1YFXwkeoK900wOZx8saS1T5avSRWlljStfwMrCl7lZhhJSIM1ijIzS',
                 binance_api_secret: str= 'JjJRJ3bWQTEShF4Eu8ZigY9aEMGPnFNJMH3WoNlOQgxSgrHmLOflIavhMx0KSZFC'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.step = step
        self.source = 'binance'
        self.api_key = binance_api_key
        self.api_secret = binance_api_secret
        self.binance_client = Client(api_key=self.api_key, api_secret=self.api_secret)
        self.binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440}
        self.batch_size = 750

    def _minutes_of_new_data(self, data):
        if len(data) > 0:
            old = parser.parse(data["timestamp"].iloc[-1])
        elif self.source == "binance":
            old = datetime.strptime(self.start_date, '%d %b %Y')
        if self.source == "binance": new = pd.to_datetime(
            self.binance_client.get_klines(symbol=self.symbol, interval=self.step)[-1][0], unit='ms')
        return old, new

    def _get_save_data(self, save=True):
        filename = f'{self.symbol}-{self.step}-data-from-{self.start_date}.csv'
        if os.path.isfile(filename):
            data_df = pd.read_csv(filename)
        else:
            data_df = pd.DataFrame()
        oldest_point, newest_point = self._minutes_of_new_data(data_df)
        delta_min = (newest_point - oldest_point).total_seconds() / 60
        available_data = math.ceil(delta_min / self.binsizes[self.step])
        if oldest_point == datetime.strptime(self.start_date, '%d %b %Y'):
            print(
                f'Downloading all available {self.step} data for {self.symbol} from {self.start_date}. Be patient..!')
        else:
            print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (
            delta_min, self.symbol, available_data, self.step))
        klines = self.binance_client.get_historical_klines(self.symbol, self.step,
                                                      oldest_point.strftime("%d %b %Y %H:%M:%S"),
                                                      newest_point.strftime("%d %b %Y %H:%M:%S"))
        data = pd.DataFrame(klines,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
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

    def prepare_data(self, macd_slow=6, macd_fast=24, macd_sign=12, macd2_slow=6, macd2_fast=24, macd2_sign=12):
        filename = f'{self.symbol}-{self.step}-data-from-{self.start_date}.csv'
        if os.path.isfile(filename):
            data_org = pd.read_csv(filename, index_col=0)
        else: data_org = self._get_save_data()
        data_org.index = data_org.index.map(lambda x: x if type(x) == str else str(x))
        data_org = data_org[~data_org.index.duplicated(keep='last')]
        data = data_org[:self.end_date].filter(['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                           'trades', 'tb_base_av', 'tb_quote_av'])
        data1 = copy.deepcopy(data.astype(float))
        data2 = self.Ichi_IND(data1,9,26,52)
        data3 = self.MACD_IND(data2,macd_slow,macd_fast,macd_sign,1)#6,24,12
        data4 = self.MACD_IND(data3, macd2_slow, macd2_fast, macd2_sign, 2)  # 6,24,12
        df = data4.copy(deep=True)
        df.reset_index(inplace=True)
        return df

    def download_data(self):
        filename = f'{self.symbol}-{self.step}-data-from-{self.start_date}.csv'
        if not os.path.isfile(filename):
            data_org = self._get_save_data()

    def MACD_IND(self,data,win_slow,win_fast,win_sign,number):
        MACD_IND1 = MACD(data['close'],window_slow=win_slow,window_fast=win_fast,window_sign=win_sign)
        data[f'MACD{number}']         = MACD_IND1.macd()
        data[f'MACD{number}_signal']  = MACD_IND1.macd_signal()
        data[f'MACD{number}_Hist']    = MACD_IND1.macd_diff()
        data[f'MACD{number}_ZC']      = np.where((data[f'MACD{number}_Hist']*(data[f'MACD{number}_Hist'].shift(1,axis=0))) < 0,1,0)
        return data

    def Ichi_IND(self,data,win1,win2,win3):
        Ichimoku_IND1 = IchimokuIndicator(high=data['high'], low=data['low'], window1=win1, window2=win2, window3=win3)
        data['Ichimoku_a']               = Ichimoku_IND1.ichimoku_a()
        data['Ichimoku_b']               = Ichimoku_IND1.ichimoku_b()
        data['Ichimoku_base_line']       = Ichimoku_IND1.ichimoku_base_line()
        data['Ichimoku_conversion_line'] = Ichimoku_IND1.ichimoku_conversion_line()
        return data



# a = DataHunter(symbol='BTCUSDT',start_date='1 Jan 2021', end_date='2021-03-25 00:00:00', step='1h').prepare_data()
# print(a.columns)