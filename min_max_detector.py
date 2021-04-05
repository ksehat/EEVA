import math
import os.path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from binance.client import Client
from dateutil import parser
from scipy.signal import argrelextrema
import plotly.graph_objects as go

def get_max_min(prices, window_range):
    local_max = argrelextrema(prices['close'].values, np.greater)[0]
    local_min = argrelextrema(prices['close'].values, np.less)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_max_dt.append(prices.iloc[i - window_range:i + window_range]['close'].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i > window_range) and (i < len(prices) - window_range):
            price_local_min_dt.append(prices.iloc[i - window_range:i + window_range]['close'].idxmin())
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min.index.name = 'date'
    max_min = max_min.reset_index()
    max_min = max_min[~max_min.date.duplicated()]
    p = prices.reset_index()
    max_min['day_num'] = p[p['timestamp'].isin(max_min.date)].index.values
    max_min = max_min.set_index('day_num')['close']

    return max_min

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
    if oldest_point == datetime.strptime(start_date, '%d %b %Y'):
        print(f'Downloading all available {kline_size} data for {symbol} from {start_date}. Be patient..!')
    else:
        print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (
        delta_min, symbol, available_data, kline_size))
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
    print('All caught up..!')
    return data_df


binance_api_key = '[REDACTED]'  # Enter your own API-key here
binance_api_secret = '[REDACTED]'  # Enter your own API-secret here
binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
            "1d": 1440}
batch_size = 750
proxies = {
    'http': 'uk2.purepackets.com:900',
}
binance_client = Client('43PXiL32cF1YFXwkeoK900wOZx8saS1T5avSRWlljStfwMrCl7lZhhJSIM1ijIzS',
                        'JjJRJ3bWQTEShF4Eu8ZigY9aEMGPnFNJMH3WoNlOQgxSgrHmLOflIavhMx0KSZFC')#, {"timeout": 0.1, "proxies": proxies})


"""# Data"""

binance_symbols = ["ADAUSDT"]
start_date = '1 Jan 2021'
data_step = '1h'
leverage = 1
plot_width = 1000
plot_height = 600

for symbol in binance_symbols:
    data_org = pd.DataFrame()
    data_org = get_all_binance(symbol, data_step, start_date, save=True)

data_org.index = data_org.index.map(lambda x: x if type(x) == str else str(x))
data_org = data_org[~data_org.index.duplicated(keep='last')]

data = data_org.filter(['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                        'trades', 'tb_base_av', 'tb_quote_av'])

df = data.astype(float).copy(deep=True)
