# IMPORTS
import pandas as pd
import numpy as np
import math
import os.path
import time
import ta
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator as RSI
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from my_geneticalgorithm import MyGeneticAlgorithm as mga

def MACD_IND(data, win_slow, win_fast, win_sign):
    MACD_IND1 = MACD(data['close'], window_slow=win_slow, window_fast=win_fast, window_sign=win_sign)
    data['MACD'] = MACD_IND1.macd()
    data['MACD_signal'] = MACD_IND1.macd_signal()
    data['MACD_Hist'] = MACD_IND1.macd_diff()
    data['MACD_ZC'] = np.where((data['MACD_Hist'] * (data['MACD_Hist'].shift(1, axis=0))) < 0, 1, 0)
    return data

def EMA_IND(data, win, i):
    EMA_IND = EMAIndicator(data['close'], window=win)
    data[f'EMA{i}'] = EMA_IND.ema_indicator()
    return data

def Ichi(data, win1, win2, win3):
    Ichimoku_IND1 = IchimokuIndicator(high=data['high'], low=data['low'], window1=win1, window2=win2, window3=win3)
    data['Ichimoku_a'] = Ichimoku_IND1.ichimoku_a()
    data['Ichimoku_b'] = Ichimoku_IND1.ichimoku_b()
    data['Ichimoku_base_line'] = Ichimoku_IND1.ichimoku_base_line()
    data['Ichimoku_conversion_line'] = Ichimoku_IND1.ichimoku_conversion_line()
    return data

def minutes_of_new_data(symbol, kline_size, data, start_date, source):
    if len(data) > 0:
        old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance":
        old = datetime.strptime(start_date, '%d %b %Y')
    if source == "binance": new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0],
                                                 unit='ms')
    return old, new

def get_all_binance(symbol, kline_size, start_date='1 Jan 2021', save=False):
    filename = f'{symbol}-{kline_size}-data-from-{start_date}.csv'
    if os.path.isfile(filename):
        data_df = pd.read_csv(filename)
    else:
        data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, start_date, source="binance")
    delta_min = (newest_point - oldest_point).total_seconds() / 60
    available_data = math.ceil(delta_min / binsizes[kline_size])
    # if oldest_point == datetime.strptime(start_date, '%d %b %Y'): print(f'Downloading all available {kline_size} data for {symbol} from {start_date}. Be patient..!')
    # else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"),
                                                  newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else:
        data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save: data_df.to_csv(filename)
    # print('All caught up..!')
    return data_df

def update_data(binance_symbols, data_steps, start_date):
    """
    this function updates all available binance data files in the directory.
    :param binance_symbols,data_steps,start_date:
    :return: data_org
    """
    for symbol_row, symbol in enumerate(binance_symbols):
        for data_step in data_steps:
            data_org = get_all_binance(symbol, data_step, start_date, save=True)
    return data_org

binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60,
            "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440}
batch_size = 750
binance_client = Client(api_key='43PXiL32cF1YFXwkeoK900wOZx8saS1T5avSRWlljStfwMrCl7lZhhJSIM1ijIzS',
                        api_secret='JjJRJ3bWQTEShF4Eu8ZigY9aEMGPnFNJMH3WoNlOQgxSgrHmLOflIavhMx0KSZFC')

binance_symbols = ['LTCUSDT']
start_date = '1 Dec 2020'
end_date = '2021-03-24 00:00:00'
data_steps = ['15m']
leverage=1
plot_width = 1500
plot_height = 1000
update_data(binance_symbols,data_steps,start_date)

def f(x):
    print(x)
    for symbol_row, symbol in enumerate(binance_symbols):
        Profit_Loss_Table_by_Year_Month_for_symbol = pd.DataFrame()
        for data_step in data_steps:
            filename = f'{symbol}-{data_step}-data-from-{start_date}.csv'
            if os.path.isfile(filename): data_org = pd.read_csv(filename, index_col=0)
            else: data_org = get_all_binance(symbol, data_step, start_date, save=True)
            data_org.index = data_org.index.map(lambda x: x if type(x) == str else str(x))
            data_org = data_org[~data_org.index.duplicated(keep='last')]
            data = data_org[:end_date].filter(['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                               'trades', 'tb_base_av', 'tb_quote_av'])
            data1 = data.astype(float).copy(deep=True)
            data2 = MACD_IND(data1, x[0], x[1], x[2])
            df = data2.copy(deep=True)
            df.reset_index(inplace=True)

            money=1
            trading_fee=.002
            buy=0
            long_price=0
            short_price=0
            for date_pointer in range(len(df)):

                if df['MACD'][date_pointer] > 0:
                    if buy==0:
                        buy = 1
                        long_price = df['close'][date_pointer]
                        long_index = df['timestamp'][date_pointer]
                        # print('long:',long_index)
                        profit_loss = ((short_price-long_price)/short_price) if short_price != 0 else 0
                        money = money + (profit_loss - trading_fee)*money
                if df['MACD'][date_pointer] < 0:
                    if buy==1:
                        buy = 0
                        short_price = df['close'][date_pointer]
                        short_index = df['timestamp'][date_pointer]
                        # print('short:',short_index)
                        profit_loss = (short_price-long_price)/long_price if long_price != 0 else 0
                        money = money + (profit_loss - trading_fee)*money
    print(money)
    return(money)


ali = {
    'slow_window':[5,6,7,12,13,26,30,40,52],
    'fast_window':[4,5,6,7,12,24,30,40,48],
    'sign_window':[4,6,8,9,10,12,14,16,18,20]
}

GA = mga(config=ali, function=f, run_iter=20, population_size=100, n_crossover=1, crossover_mode='random')
best_params=GA.run()
print(best_params)