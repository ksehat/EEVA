# IMPORTS
import copy
import math
import os.path
import time
from datetime import timedelta, datetime
from dateutil import parser
import pandas as pd
import numpy as np
from binance.client import Client
import ta
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator as RSI
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import smtplib


def MACD_IND(data, win_slow, win_fast, win_sign):
    MACD_IND1 = MACD(data['close'], window_slow=win_slow, window_fast=win_fast,
                     window_sign=win_sign)
    data['MACD'] = MACD_IND1.macd()
    data['MACD_signal'] = MACD_IND1.macd_signal()
    data['MACD_Hist'] = MACD_IND1.macd_diff()
    data['MACD_ZC'] = np.where((data['MACD_Hist'] * (data['MACD_Hist'].shift(1, axis=0))) < 0, 1, 0)
    return data


def Ichi(data, win1, win2, win3):
    Ichimoku_IND1 = IchimokuIndicator(high=data['high'], low=data['low'], window1=win1,
                                      window2=win2, window3=win3)
    data['Ichimoku_a'] = Ichimoku_IND1.ichimoku_a()
    data['Ichimoku_b'] = Ichimoku_IND1.ichimoku_b()
    data['Ichimoku_base_line'] = Ichimoku_IND1.ichimoku_base_line()
    data['Ichimoku_conversion_line'] = Ichimoku_IND1.ichimoku_conversion_line()
    return data


def plot_figure(df, index_X, index_A, index_B, index_C, index_buy, index_sell, X, A, B, C, width,
                height):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.05)
    fig.add_trace(go.Candlestick(x=df['timestamp'][index_X - 10:index_sell + 10],
                                 open=df['open'][index_X - 10:index_sell + 10],
                                 high=df['high'][index_X - 10:index_sell + 10],
                                 low=df['low'][index_X - 10:index_sell + 10],
                                 close=df['close'][index_X - 10:index_sell + 10]))

    fig.add_trace(go.Scatter(x=df['timestamp'][index_X - 10:index_sell + 10],
                             y=df['Ichimoku_base_line'][index_X - 10:index_sell + 10]))
    fig.add_trace(go.Scatter(x=df['timestamp'][index_X - 10:index_sell + 10],
                             y=df['Ichimoku_conversion_line'][index_X - 10:index_sell + 10]))
    fig.add_trace(go.Scatter(
        x=[df['timestamp'][index_X], df['timestamp'][index_A], df['timestamp'][index_B],
           df['timestamp'][index_C]],
        y=[X, A, B, C], mode='lines+markers',
        marker=dict(size=[10, 11, 12, 13], color=[0, 1, 2, 3])))
    fig.add_shape(type="line",
                  x0=df['timestamp'][index_buy], y0=min(df.loc[index_X:index_sell, 'low']),
                  x1=df['timestamp'][index_buy], y1=max(df.loc[index_X:index_sell, 'high']))
    fig.add_shape(type="line",
                  x0=df['timestamp'][index_sell], y0=min(df.loc[index_X:index_sell, 'low']),
                  x1=df['timestamp'][index_sell], y1=max(df.loc[index_X:index_sell, 'high']))
    fig.add_trace(go.Bar(x=df['timestamp'][index_X - 10:index_sell + 10],
                         y=df['MACD_Hist'][index_X - 10:index_sell + 10]), row=2, col=1)
    fig.update_layout(height=height, width=width, xaxis_rangeslider_visible=False)
    fig.show()


def minutes_of_new_data(symbol, kline_size, data, start_date, source):
    if len(data) > 0:
        old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance":
        old = datetime.strptime(start_date, '%d %b %Y')
    if source == "binance": new = pd.to_datetime(
        binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    return old, new


def get_all_binance(symbol, kline_size, start_date='1 Jan 2021', save=False):
    filename = f'{symbol}-{kline_size}-data-from-{start_date}.csv'
    if os.path.isfile(filename):
        data_df = pd.read_csv(filename)
    else:
        data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, start_date,
                                                     source="binance")
    delta_min = (newest_point - oldest_point).total_seconds() / 60
    available_data = math.ceil(delta_min / binsizes[kline_size])
    if oldest_point == datetime.strptime(start_date, '%d %b %Y'):
        print(
            f'Downloading all available {kline_size} data for {symbol} from {start_date}. Be patient..!')
    else:
        print(
            'Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (
                delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol, kline_size,
                                                  oldest_point.strftime("%d %b %Y %H:%M:%S"),
                                                  newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'close_time', 'quote_av', 'trades', 'tb_base_av',
                                         'tb_quote_av', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else:
        data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save: data_df.to_csv(filename)
    print(f'All caught up at {datetime.now().astimezone()}!')
    return data_df


def XABC_li2df(XABC_list, df):
    XABC_df = pd.DataFrame({
        'X': [i[0] for i in [l[0] for l in XABC_list]],
        'A': [i[1] for i in [l[0] for l in XABC_list]],
        'B': [i[2] for i in [l[0] for l in XABC_list]],
        # 'C':[i[3] for i in [l[0] for l in XABC_list]],
        'date_X': [df['timestamp'][i[0]] for i in [l[1] for l in XABC_list]],
        'date_A': [df['timestamp'][i[1]] for i in [l[1] for l in XABC_list]],
        'date_B': [df['timestamp'][i[2]] for i in [l[1] for l in XABC_list]],
        # 'date_C':[df['timestamp'][i[3]] for i in [l[1] for l in XABC_list]]
    })
    return XABC_df


def data_prep(start_date, symbol, data_step):
    data_org = get_all_binance(symbol, data_step, start_date, save=True)
    data_org.index = data_org.index.map(lambda x: x if type(x) == str else str(x))
    data_org = data_org[~data_org.index.duplicated(keep='last')]
    data = data_org[:-1].filter(['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                 'trades', 'tb_base_av', 'tb_quote_av'])
    data1 = data.astype(float).copy(deep=True)
    data2 = Ichi(data1, 9, 26, 52)
    data3 = MACD_IND(data2, 6, 30, 8)
    df = data3.copy(deep=True)
    df.reset_index(inplace=True)
    return df


def data_download(start_date, symbol, data_step):
    filename = f'{self.symbol}-{self.step}-data-from-{self.start_date}.csv'
    if not os.path.isfile(filename):
        data_org = self._get_save_data()
    return df


def XABC_hunter(df):
    ZC_Index = pd.DataFrame({'zcindex': df[df['MACD_ZC'] == 1].index.values,
                             'timestamp': df.loc[df['MACD_ZC'] == 1, 'timestamp'],
                             'MACD_Hist': df.loc[df['MACD_ZC'] == 1, 'MACD_Hist']},
                            columns=['zcindex', 'timestamp', 'MACD_Hist']).reset_index(drop=True)
    # region XABC Hunter
    XABC_list1 = []
    for row_zcindex, zcindex in ZC_Index.iterrows():
        if row_zcindex + 3 <= len(ZC_Index) - 1:
            if df['MACD_Hist'][zcindex[0]] >= 0:
                # region XABC Finder
                X = max(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['high'])
                index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['high'].idxmax()
                A = min(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['low'])
                index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['low'].idxmin()
                B = max(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['high'])
                index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['high'].idxmax()
                # C = min( df.iloc[ZC_Index.iloc[row_zcindex+3,0] : ZC_Index.iloc[row_zcindex+4,0]]['low'] )
                # index_C = df.iloc[ZC_Index.iloc[row_zcindex+3,0] : ZC_Index.iloc[row_zcindex+4,0]]['low'].idxmin()
                if A < X and B < X and B > A:  # and C<A and C<X:
                    xabc_flag = 1
                    XABC_list1.append(
                        [[X, A, B], [index_X, index_A, index_B, ZC_Index.iloc[row_zcindex + 3, 0]], xabc_flag])

                # endregion
            if df['MACD_Hist'][zcindex[0]] < 0:
                # region XABC Finder
                X = min(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'])
                index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'].idxmin()
                A = max(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['high'])
                index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['high'].idxmax()
                B = min(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['low'])
                index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['low'].idxmin()
                # C = max(df.iloc[ZC_Index.iloc[row_zcindex + 3, 0]: ZC_Index.iloc[row_zcindex + 4, 0]]['high'])
                # index_C = df.iloc[ZC_Index.iloc[row_zcindex + 3, 0]: ZC_Index.iloc[row_zcindex + 4, 0]]['high'].idxmax()
                if A > X and B > X and B < A:  # and C>A and C>X:
                    xabc_flag = 0
                    XABC_list1.append(
                        [[X, A, B], [index_X, index_A, index_B, ZC_Index.iloc[row_zcindex + 3, 0]], xabc_flag])
    return XABC_list1


binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
            "1d": 1440}
batch_size = 750
binance_client = Client(api_key='43PXiL32cF1YFXwkeoK900wOZx8saS1T5avSRWlljStfwMrCl7lZhhJSIM1ijIzS',
                        api_secret='JjJRJ3bWQTEShF4Eu8ZigY9aEMGPnFNJMH3WoNlOQgxSgrHmLOflIavhMx0KSZFC')
binance_symbols = ['ETHUSDT']
start_date = '1 Mar 2021'
# end_date = '2021-10-01 01:00:00'
data_steps = ['30m']
leverage = 1
plot_width = 1500
plot_height = 1000

XABC_dict = {}
for symbol in binance_symbols:
    for data_step in data_steps:
        XABC_dict[f'XABC_list_old_{symbol}_{data_step}'] = []

while True:
    for symbol in binance_symbols:
        for data_step in data_steps:
            df = data_prep(start_date, symbol, data_step)
            XABC_list = XABC_hunter(df)
            XABC_list_old = XABC_dict[f'XABC_list_old_{symbol}_{data_step}']
            new_XABC = [item for item in XABC_list if item not in XABC_list_old]
            warning = 0
            alarm = 0
            if new_XABC:
                flag = new_XABC[-1][2]
                A = new_XABC[-1][0][1]
                B = new_XABC[-1][0][2]
                index_B = new_XABC[-1][1][2]
                index_4 = new_XABC[-1][1][3]
                for date_pointer in range(index_4,
                                          len(df)):  # TODO: check it to see until what timestep it goes. we want it live
                    if alarm == 0:
                        if (flag == 0 and df['high'][date_pointer] >= A) or (
                                flag == 1 and df['low'][date_pointer] <= A) and warning == 0:
                            index_warning = date_pointer
                            warning = 1
                        if (flag == 0 and df['low'][date_pointer] <= B) or (
                                flag == 1 and df['high'][date_pointer] >= B) and warning == 1:
                            new_XABC_df = XABC_li2df(new_XABC, df)
                            index_alarm = date_pointer
                            s = smtplib.SMTP('smtp.gmail.com', 587)
                            s.starttls()
                            s.login("luis.figo908908@gmail.com",
                                    "vpvumdjlmzxktshi")  # "k.sehat.business2021@gmail.com", "ocpnatewbibhdqjh"
                            message = f"Subject: {'New XABC'} \n\nsalam,\n{datetime.now().astimezone()}\n{symbol} {data_step}\n{new_XABC_df.iloc[-1, :]}"
                            s.sendmail("luis.figo908908@gmail.com", ["kanan.sehat.ks@gmail.com",
                                                                     "amir_elikaee@yahoo.com",
                                                                     "saeedtrader94@gmail.com",
                                                                     "cnus.1991@yahoo.com"],
                                       message)
                            s.quit()
                            XABC_dict[f'XABC_list_old_{symbol}_{data_step}'] = XABC_list
                            print(f'email sended for {symbol},{data_step} and new_XABC is {new_XABC_df.iloc[-1, :]}')
                            warning = 0
                            alarm = 1
    print('sleeping')
    time.sleep(60 * 10)
    pass
