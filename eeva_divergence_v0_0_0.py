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
from data_prep.data_hunter import DataHunter
# from my_geneticalgorithm import MyGeneticAlgorithm as mga

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

def plot_figure(df,index_X,index_A,index_B,index_C,index_buy,index_sell,X,A,B,C,width,height):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.05)
    fig.add_trace(go.Candlestick(x=df['timestamp'][index_X-10:index_sell+10],
                                         open=df['open'][index_X-10:index_sell+10],
                                         high=df['high'][index_X-10:index_sell+10],
                                         low=df['low'][index_X-10:index_sell+10],
                                         close=df['close'][index_X-10:index_sell+10]))

    fig.add_trace(go.Scatter(x=df['timestamp'][index_X-10:index_sell+10],
                                     y=df['Ichimoku_base_line'][index_X-10:index_sell+10]))
    fig.add_trace(go.Scatter(x=df['timestamp'][index_X-10:index_sell+10],
               y=df['Ichimoku_conversion_line'][index_X-10:index_sell+10]))
    fig.add_trace(go.Scatter(x=[df['timestamp'][index_X],df['timestamp'][index_A],df['timestamp'][index_B],df['timestamp'][index_C]],
               y=[X,A,B,C],mode='lines+markers',marker=dict(size=[10,11,12,13],color=[0,1,2,3])))
    fig.add_shape(type="line",
    x0=df['timestamp'][index_buy], y0=min(df.loc[index_X:index_sell,'low']), x1=df['timestamp'][index_buy], y1=max(df.loc[index_X:index_sell,'high']))
    fig.add_shape(type="line",
    x0=df['timestamp'][index_sell], y0=min(df.loc[index_X:index_sell,'low']), x1=df['timestamp'][index_sell], y1=max(df.loc[index_X:index_sell,'high']))
    fig.add_trace(go.Bar(x=df['timestamp'][index_X - 10:index_sell + 10],
                                 y=df['MACD_Hist'][index_X - 10:index_sell + 10]), row=2, col=1)
    fig.update_layout(height=height, width=width, xaxis_rangeslider_visible=False)
    fig.show()

def minutes_of_new_data(symbol, kline_size, data, start_date, source):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance": old = datetime.strptime(start_date, '%d %b %Y')
    if source == "binance": new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    return old, new
 
def get_all_binance(symbol, kline_size, start_date='1 Jan 2021' , save = False):
    filename = f'{symbol}-{kline_size}-data-from-{start_date}.csv' 
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, start_date, source = "binance")
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    # if oldest_point == datetime.strptime(start_date, '%d %b %Y'): print(f'Downloading all available {kline_size} data for {symbol} from {start_date}. Be patient..!')
    # else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save: data_df.to_csv(filename)
    # print('All caught up..!')
    return data_df

def update_data(binance_symbols,data_steps,start_date):
    """
    this function updates all available binance data files in the directory.
    :param binance_symbols,data_steps,start_date:
    :return: data_org
    """
    for symbol_row, symbol in enumerate(binance_symbols):
        for data_step in data_steps:
            data_org = get_all_binance(symbol, data_step, start_date, save=True)
    return data_org

def MACD_phase_change(df,date_pointer):
    if df['MACD1_Hist'][date_pointer]*df['MACD1_Hist'][date_pointer-1]<0: return True
    else: return False

def max_finder(df, date_pointer, value, value_macd, value_date):
    # if new_phase:
    #     value = df['high'][date_pointer]
    #     value_macd = df['MACD1_Hist'][date_pointer]
    #     dp = date_pointer
    # elif not df['MACD1_Hist'][date_pointer-1]:
    #     value = df['high'][date_pointer]
    #     value_macd = df['MACD1_Hist'][date_pointer]
    #     dp = date_pointer
    if df['high'][date_pointer] > value:
        value = df['high'][date_pointer]
        value_macd = df['MACD1_Hist'][date_pointer]
        value_date = date_pointer
    return value, value_macd, value_date

binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60,
            "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440}
batch_size = 750
binance_client = Client(api_key= '43PXiL32cF1YFXwkeoK900wOZx8saS1T5avSRWlljStfwMrCl7lZhhJSIM1ijIzS',
                        api_secret= 'JjJRJ3bWQTEShF4Eu8ZigY9aEMGPnFNJMH3WoNlOQgxSgrHmLOflIavhMx0KSZFC')

"""Data"""
binance_symbols = ['LTCUSDT']
start_date = '1 Jan 2021'
end_date = '2021-03-28 21:00:00'
data_steps = ['1h']
leverage=1
plot_width = 1500
plot_height = 1000
update_data(binance_symbols,data_steps,start_date)

for symbol_row, symbol in enumerate(binance_symbols):
    for data_step in data_steps:
        df = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                        step=data_step).prepare_data(macd_slow=26,macd_fast=12,macd_sign=9)
        df2 = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                         step='30m').prepare_data(macd_slow=26, macd_fast=12, macd_sign=9)
        pp1 = 0
        pp2 = 0
        pp1_macd = 0
        pp2_macd = 0
        pp1_date = 0
        pp2_date = 0
        for date_pointer in range(len(df)):
            # new_macd_phase = MACD_phase_change(df, date_pointer)
            if df['MACD1_Hist'][date_pointer] >= 0:
                if df['high'][date_pointer] > pp1:
                    pp1 = df['high'][date_pointer]
                    pp1_macd = df['MACD1_Hist'][date_pointer]
                    pp1_date = date_pointer
                    if pp1 and pp2:
                        if pp1 > pp2 and pp1_macd <= pp2_macd:
                            warning=1
                            # print(df['timestamp'][pp1_date],df['timestamp'][pp2_date])
                            # print(pp1_macd,pp2_macd)
                            # print('==============')
                        if warning==1:
                            date_pointer2_str = df['timestamp'][date_pointer]
                            for date_pointer2 in range(df2[df2[\
                                    'timestamp']==date_pointer2_str].index.values[0]+1,len(df2)):
                                if df2['high'][date_pointer2] < pp1:
                                    if df2['MACD1_Hist'][date_pointer2] < 0:
                                        alarm=1
                                        print(df['timestamp'][pp2_date], df['timestamp'][
                                            pp1_date], df2['timestamp'][date_pointer2])
                                        print(pp2_macd, pp1_macd, df2['high'][date_pointer2])
                                        print('==============')
                                    else: continue
                                else:
                                    warning = 0
                                    break
                            # alarm==1
                            # alarm_point = df2['timestamp'][date_pointer2]
                            # print(alarm_point)



                        else: continue
                else: continue
            if df['MACD1_Hist'][date_pointer] < 0:
                if not pp1 == 0:
                    pp2 = pp1
                    pp2_macd = pp1_macd
                    pp2_date = pp1_date
                    pp1 = 0
                    pp1_macd = 0



                # new_macd_phase = MACD_phase_change(df,date_pointer)
                # if pp1 and pp2 > pp1 and pp1_macd >= pp2_macd:  # divergence
                #     alarm = 1
                #     print(df['timestamp'][pp1_date],df['timestamp'][pp2_date])
                #     print(df['MACD1_Hist'][pp1_date],df['MACD1_Hist'][pp2_date])
                #     print('===============================')
                #
                # elif pp2: pp1, pp1_macd, pp1_date = pp2, pp2_macd, pp2_date
                # pp2, pp2_macd, pp2_date = df['high'][date_pointer], df['MACD1_Hist'][date_pointer], date_pointer
                # if not new_macd_phase: pp2, pp2_macd, pp2_date = max_finder(df, date_pointer, pp2, pp2_macd, pp2_date)


        # long1 = None
        # long2 = None
        # long1_macd=0
        # long2_macd=0
        # dp=0
        # for date_pointer in range(1,len(df)):
        #     if df['MACD1_Hist'][date_pointer] >= 0:
        #         new_macd_phase = MACD_phase_detection(df,date_pointer)
        #         if new_macd_phase and not long1:
        #             long1, long1_macd, long1_date = next(max_finder(df,date_pointer,new_macd_phase,long1))
        #         elif new_macd_phase and long1:
        #             if long2:
        #                 long1, long1_macd, long1_date = long2, long2_macd, long2_date
        #                 long2, long2_macd, long2_date = next(max_finder(df,date_pointer,new_macd_phase,None))
        #             else: long2, long2_macd, long2_date = next(max_finder(df,date_pointer,new_macd_phase,long2))
        #         elif not new_macd_phase and long1:
        #             if long2:
        #                 long2, long2_macd, long2_date = next(max_finder(df,date_pointer,new_macd_phase,long2))
        #             else: long1, long1_macd, long1_date = next(max_finder(df,date_pointer,new_macd_phase,long1))
        #
        #         if long1 and long2:
        #             if long2 > long1 and long2_macd <= long1_macd: #divergence
        #                 alarm=1
        #                 print(long1_macd,long2_macd)
        #                 print(df['timestamp'][long1_date],df['timestamp'][long2_date])
        #                 long1, long1_macd, long1_date = long2, long2_macd, long2_date
        #                 long2, long2_macd, long2_date = None, None, None
