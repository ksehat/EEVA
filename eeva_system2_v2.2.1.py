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
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator as RSI
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_prep.data_hunter import DataHunter
import pyflowchart as pfc

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

def plot_figure(df, index_X, index_A, index_B, index_C, index_buy, index_sell, X, A, B, C, width, height):
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
        x=[df['timestamp'][index_X], df['timestamp'][index_A], df['timestamp'][index_B], df['timestamp'][index_C]],
        y=[X, A, B, C], mode='lines+markers', marker=dict(size=[10, 11, 12, 13], color=[0, 1, 2, 3])))
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

def macd_phase_change(df,date_pointer):
    if df['MACD_Hist'][date_pointer]*df['MACD_Hist'][date_pointer-1]<0: return True
    else: return False

def print_trade(df,X,A,B,xab,enter_price,exit_price,index_X,index_A,index_B,index_buy,index_sell):
    print(df['timestamp'][index_X], 'X:', X)
    print(df['timestamp'][index_A], 'A:', A)
    print(df['timestamp'][index_B], 'B:', B)
    print(df['timestamp'][xab[1][3]], 'C:', xab[0][3])
    print(df['timestamp'][index_buy], 'enter:', enter_price)
    print(df['timestamp'][index_sell], 'exit:', exit_price)

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

def xab_initializer(xab):
    X = xab[0][0]
    A = xab[0][1]
    B = xab[0][2]
    # C = xab[0][3]
    index_X = xab[1][0]
    index_A = xab[1][1]
    index_B = xab[1][2]
    # index_C = xab[1][3]
    index_4 = xab[1][4]
    flag = xab[2]
    dont_find_C = xab[5]
    return X,A,B,index_X,index_A,index_B,index_4,flag

def xab_enter_check(df,date_pointer,xab,enter):
    if xab[2] and df['close'][date_pointer] >= xab[0][2]:
        enter = 1
    if not xab[2] and df['close'][date_pointer] <= xab[0][2]:
        enter = 1
    return enter

def xab_completor(df,date_pointer,xab, XAB_del_list):
    # region Initialize XABC and flag from xabc
    X = xab[0][0]
    A = xab[0][1]
    B = xab[0][2]
    # C = xab[0][3]
    index_X = xab[1][0]
    index_A = xab[1][1]
    index_B = xab[1][2]
    # index_C = xab[1][3]
    index_4 = xab[1][4]
    flag = xab[2]
    dont_find_C = xab[5]
    # endregion

    if flag == 1:  # long
        if xab[0][3]:
            if df['MACD_Hist'][date_pointer] < 0 and xab[5] == 0:  # dont_find_C = xab[0][5]
                if df['low'][date_pointer] <= xab[0][3]:
                    xab[0][3] = df['low'][date_pointer]
                    xab[1][3] = date_pointer
            if df['MACD_Hist'][date_pointer] > 0:
                xab[5] = 1
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
        if not xab[0][3] and not xab[5]:
            if df['low'][date_pointer] <= A and df['MACD_Hist'][date_pointer] < 0 and xab[5] == 0:
                xab[0][3] = df['low'][date_pointer]
                xab[1][3] = date_pointer
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
            if df['MACD_Hist'][date_pointer] > 0:
                xab[5] = 1
                XAB_del_list.append(xab)

    if flag == 0:  # short
        if xab[0][3]:
            if df['MACD_Hist'][date_pointer] > 0 and xab[5] == 0:
                if df['high'][date_pointer] >= xab[0][3]:
                    xab[0][3] = df['high'][date_pointer]
                    xab[1][3] = date_pointer
            if df['MACD_Hist'][date_pointer] < 0:
                xab[5] = 1
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
        if not xab[0][3] and not xab[5]:
            if df['high'][date_pointer] >= A and df['MACD_Hist'][date_pointer] > 0 and xab[5] == 0:
                xab[0][3] = df['high'][date_pointer]
                xab[1][3] = date_pointer
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
            if df['MACD_Hist'][date_pointer] < 0:
                xab[5] = 1
                XAB_del_list.append(xab)

    return xab, XAB_del_list

def xab_reject_decision(df,dp,xab,XAB_del_list,XAB_check_list):
    if xab[2]==1:
        if df['low'][dp] < xab[0][3]:
            XAB_del_list.append(xab)
        elif df['close'][dp] > xab[0][2]:
            if xab not in XAB_check_list:
                XAB_check_list.append(xab)
    if xab[2]==0:
        if df['high'][dp] > xab[0][3]:
            XAB_del_list.append(xab)
        elif df['close'][dp] < xab[0][2]:
            if xab not in XAB_check_list:
                XAB_check_list.append(xab)
    return XAB_del_list, XAB_check_list

def MACD_phase_change(df,date_pointer):
    if df['MACD_Hist'][date_pointer]*df['MACD_Hist'][date_pointer-1]<0: return True
    else: return False

binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60, "2h": 120,  "4h": 240,
            "6h": 360, "12h": 720, "1d": 1440}
batch_size = 750
binance_client = Client(api_key='43PXiL32cF1YFXwkeoK900wOZx8saS1T5avSRWlljStfwMrCl7lZhhJSIM1ijIzS',
                        api_secret='JjJRJ3bWQTEShF4Eu8ZigY9aEMGPnFNJMH3WoNlOQgxSgrHmLOflIavhMx0KSZFC')

"""Data"""
binance_symbols = ['LTCUSDT']
start_date = '1 Jan 2018'
end_date = '2021-04-15 00:00:00'
data_steps = ['2h']
leverage = 1
plot_width = 1500
plot_height = 1000

for symbol_row, symbol in enumerate(binance_symbols):
    Profit_Loss_Table_by_Year_Month_for_symbol = pd.DataFrame()
    for data_step in data_steps:
        filename = f'{symbol}-{data_step}-data-from-{start_date}.csv'
        if os.path.isfile(filename):
            data_org = pd.read_csv(filename, index_col=0)
        else:
            data_org = get_all_binance(symbol, data_step, start_date, save=True)

        data_org.index = data_org.index.map(lambda x: x if type(x) == str else str(x))
        data_org = data_org[~data_org.index.duplicated(keep='last')]
        data = data_org[:end_date].filter(['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                           'trades', 'tb_base_av', 'tb_quote_av'])
        data1 = data.astype(float).copy(deep=True)
        data2 = Ichi(data1,9,26,52)
        data3 = MACD_IND(data2,7,6,18)
        df = data3.copy(deep=True)
        df.reset_index(inplace=True)
        ZC_Index = pd.DataFrame({'zcindex': df[df['MACD_ZC'] == 1].index.values,
                                 'timestamp': df.loc[df['MACD_ZC'] == 1, 'timestamp'],
                                 'MACD_Hist': df.loc[df['MACD_ZC'] == 1, 'MACD_Hist']},
                                columns=['zcindex', 'timestamp', 'MACD_Hist']).reset_index(drop=True)
        # region XAB Hunter
        # TODO: we have to change the strategy of XAB
        XAB_list = []
        for row_zcindex, zcindex in ZC_Index.iterrows():
            if row_zcindex + 3 <= len(ZC_Index) - 1:
                if df['MACD_Hist'][zcindex[0]] >= 0:
                    # region XAB Finder
                    X = max(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['high'])
                    index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['high'].idxmax()
                    A = min(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['low'])
                    index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]][
                        'low'].idxmin()
                    B = max(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['high'])
                    index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]][
                        'high'].idxmax()
                    if A < X and B < X and B > A:
                        xab_flag = 1
                        Index_4 = ZC_Index.iloc[row_zcindex + 3, 0]
                        XAB_list.append([[X, A, B, None], [index_X, index_A, index_B, None, Index_4], xab_flag, None, None, 0])
                    # endregion

                if df['MACD_Hist'][zcindex[0]] < 0:
                    # region XAB Finder
                    X = min(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'])
                    index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'].idxmin()
                    A = max(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['high'])
                    index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]][
                        'high'].idxmax()
                    B = min(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['low'])
                    index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]][
                        'low'].idxmin()
                    if A > X and B > X and B < A:
                        xab_flag = 0
                        Index_4 = ZC_Index.iloc[row_zcindex + 3, 0]
                        XAB_list.append([[X, A, B, None], [index_X, index_A, index_B, None, Index_4], xab_flag, None, None, 0])
                    # endregion
        # endregion #

        # region initializing params
        money = 1
        trade_fee = 0.002
        enter = 0
        date_pointer = 0
        date_of_trade_list = []
        profit_loss_list = []
        money_after_each_trade_list = []
        money_before_each_trade_list = []
        num_of_pos_trades = 0
        num_of_neg_trades = 0
        num_of_pos_trades_list = []
        num_of_neg_trades_list = []
        # endregion

        if not XAB_list:
            print('XAB list is empty')
            continue

        XAB_del_list = [] # This the list of XABs that are rejected
        XAB_check_list = [] # This is the list of XABs that may be entered and are valid to enter but right now the system is in trade
        for date_pointer in range(XAB_list[0][1][4], len(df)):
            XAB_valid_list = [x for x in XAB_list if date_pointer >= x[1][4]] # This is the list of XABs before the date_pointer
            for idx_xab, xab in enumerate(
                    XAB_valid_list[::-1]):  # xabc = [[X, A, B, C], [index_X, index_A, index_B, index_4, index_C], xab_flag, sl, sudo_sl, dont_find_C]
                if xab not in XAB_del_list:
                    X, A, B, index_X, index_A, index_B, index_4, flag = xab_initializer(xab)
                    if enter == 0:
                        xab, XAB_del_list = xab_completor(df, date_pointer, xab, XAB_del_list)
                        if xab[0][3]:
                            enter = xab_enter_check(df, date_pointer, xab, enter)
                        if enter==1:
                            index_buy = date_pointer
                            xab_buy = xab
                            enter_price = xab[0][2]
                            xab[3] = xab[0][3]  # C is placed in sl
                            xab[4] = xab[0][3]  # C is placed in sudo_sl
                            money_before_each_trade_list.append(money)
                        elif xab[0][3] and xab[5]:
                            XAB_del_list, XAB_check_list = xab_reject_decision(df,date_pointer,
                                                                                xab,XAB_del_list,XAB_check_list)

                    else: # If it is in trade
                        if xab != xab_buy:
                            xab, XAB_del_list = xab_completor(df, date_pointer, xab, XAB_del_list)
                            if xab[0][3]:
                                XAB_del_list, XAB_check_list = xab_reject_decision(df,
                                                                                    date_pointer,xab,XAB_del_list,XAB_check_list)
                        if xab == xab_buy:
                            if macd_phase_change(df,date_pointer): xab[3] = xab[4]
                            # This is because when the phase is changed, first you need to
                            # replace the sl with sudo_sl
                            if flag==1:
                                if df['low'][date_pointer] < xab[3]:
                                    enter = 0
                                    index_sell = date_pointer
                                    exit_price = xab[3]
                                    print_trade(df, X, A, B, xab, enter_price, exit_price, index_X, index_A, index_B,
                                                index_buy, index_sell)
                                    if exit_price > B:
                                        profit = leverage * ((exit_price - B) / B) - trade_fee
                                        money = money + profit * money
                                        profit_loss_list.append(profit)
                                        num_of_pos_trades += 1
                                        print('profit:', profit)
                                        print('money:',money)
                                    if exit_price <= B:
                                        loss = leverage * ((exit_price - B) / B) - trade_fee
                                        money = money + loss * money
                                        profit_loss_list.append(loss)
                                        num_of_neg_trades += 1
                                        print('loss:', loss)
                                        print('money:', money)
                                    # plot_figure(df, xabc[1][0], xabc[1][1], xabc[1][2], xabc[1][3], index_buy, index_sell,
                                    #             xabc[0][0], xabc[0][1], xabc[0][2], xabc[0][3], plot_width, plot_height)
                                    date_of_trade_list.append(df['timestamp'][date_pointer])
                                    num_of_neg_trades_list.append(num_of_neg_trades)
                                    num_of_pos_trades_list.append(num_of_pos_trades)
                                    money_after_each_trade_list.append(money)
                                    XAB_del_list.append(xab)

                                    if XAB_check_list:
                                        # print('==================')
                                        # print(XAB_check_list)
                                        # print('==================')
                                        enter = 1
                                        index_buy = date_pointer
                                        xab_buy = XAB_check_list[-1]
                                        enter_price = xab_buy[0][2]
                                        del XAB_check_list[-1]
                                        money_before_each_trade_list.append(money)
                                else:
                                    if XAB_check_list:
                                        XAB_del_list.extend(XAB_check_list)
                                        XAB_check_list = []
                                    if df['MACD_Hist'][date_pointer]<0:
                                        if macd_phase_change(df,date_pointer): xab[4] = df['low'][date_pointer]
                                        elif df['low'][date_pointer] <= xab[4]: xab[4] = df['low'][date_pointer]
                                    if df['MACD_Hist'][date_pointer]>0: xab[3] = xab[4]
                            if flag==0:
                                if df['high'][date_pointer] > xab[3]:
                                    enter = 0
                                    index_sell = date_pointer
                                    exit_price = xab[3]
                                    print_trade(df, X, A, B, xab, enter_price, exit_price, index_X, index_A, index_B,
                                                index_buy, index_sell)
                                    if exit_price < B:
                                        profit = leverage * ((B - exit_price) / B) - trade_fee
                                        money = money + profit * money
                                        profit_loss_list.append(profit)
                                        num_of_pos_trades += 1
                                        print('profit:', profit)
                                        print('money:',money)
                                    if exit_price >= B:
                                        loss = leverage * ((B - exit_price) / B) - trade_fee
                                        money = money + loss * money
                                        profit_loss_list.append(loss)
                                        num_of_neg_trades += 1
                                        print('loss:', loss)
                                        print('money:', money)
                                    # plot_figure(df, xabc[1][0], xabc[1][1], xabc[1][2], xabc[1][3], index_buy, index_sell,
                                    #             xabc[0][0], xabc[0][1], xabc[0][2], xabc[0][3], plot_width, plot_height)
                                    date_of_trade_list.append(df['timestamp'][date_pointer])
                                    num_of_neg_trades_list.append(num_of_neg_trades)
                                    num_of_pos_trades_list.append(num_of_pos_trades)
                                    money_after_each_trade_list.append(money)
                                    XAB_del_list.append(xab)
                                    if XAB_check_list:
                                        # print('==================')
                                        # print(XAB_check_list)
                                        # print('==================')
                                        enter = 1
                                        index_buy = date_pointer
                                        xab_buy = XAB_check_list[-1]
                                        enter_price = xab_buy[0][2]
                                        del XAB_check_list[-1]
                                        money_before_each_trade_list.append(money)
                                else:
                                    if XAB_check_list:
                                        XAB_del_list.extend(XAB_check_list)
                                        XAB_check_list = []
                                    if df['MACD_Hist'][date_pointer] > 0:
                                        if macd_phase_change(df,date_pointer): xab[4] = df['high'][date_pointer]
                                        elif df['high'][date_pointer] >= xab[4]: xab[4] = df['high'][date_pointer]
                                    if df['MACD_Hist'][date_pointer] < 0: xab[3] = xab[4]
        print(money)

        # region
        """ If there is a buy position but still the sell position doesn't
        occur it would be a problem and this problem is solved in this region
        """
        lists = [date_of_trade_list, profit_loss_list, num_of_pos_trades_list,
                 num_of_neg_trades_list, money_after_each_trade_list, money_before_each_trade_list]
        unique_len = [len(i) for i in lists]
        list_length = min(unique_len)
        for index, l in enumerate(lists):
            if len(l) > list_length:
                del lists[index][-1]
            list_length = len(l)
        # endregion

        Profit_Loss_Table = pd.DataFrame({
            'date': date_of_trade_list,
            'profit & loss': profit_loss_list,
            'num_of_pos_trades': num_of_pos_trades_list,
            'num_of_neg_trades': num_of_neg_trades_list,
            'money_after_trade': money_after_each_trade_list,
            'money_before_trade': money_before_each_trade_list
        })

        Profit_Loss_Table['date'] = pd.to_datetime(Profit_Loss_Table['date'])
        Profit_Loss_Table['num_of_all_trades'] = Profit_Loss_Table['num_of_neg_trades'] + Profit_Loss_Table[
            'num_of_pos_trades']

        Profit_Loss_Table['year'] = Profit_Loss_Table['date'].apply(lambda t: t.year)
        Profit_Loss_Table['month'] = Profit_Loss_Table['date'].apply(lambda t: t.month)
        Profit_Loss_Table['day'] = Profit_Loss_Table['date'].apply(lambda t: t.day)

        Money_each_month = Profit_Loss_Table.groupby(['year', 'month'])
        month_profit_loss_list = []
        year_month_list = []
        month_pos_trades = []
        month_neg_trades = []
        month_all_trades = []
        last_month_num_pos_trades = 0
        last_month_num_neg_trades = 0
        last_month_num_all_trades = 0
        for key, value in zip(Money_each_month.groups.keys(), Money_each_month.groups.values()):
            first_money = Profit_Loss_Table['money_before_trade'][value[0]]
            last_money = Profit_Loss_Table['money_after_trade'][value[-1]]
            month_profit = (last_money - first_money) * 100 / first_money
            month_profit_loss_list.append(month_profit)

            month_pos_trades.append(Profit_Loss_Table['num_of_pos_trades'][value[-1]] - last_month_num_pos_trades)
            month_neg_trades.append(Profit_Loss_Table['num_of_neg_trades'][value[-1]] - last_month_num_neg_trades)
            month_all_trades.append(Profit_Loss_Table['num_of_all_trades'][value[-1]] - last_month_num_all_trades)

            year_month_list.append(key)
            last_month_num_pos_trades = Profit_Loss_Table['num_of_pos_trades'][value[-1]]
            last_month_num_neg_trades = Profit_Loss_Table['num_of_neg_trades'][value[-1]]
            last_month_num_all_trades = Profit_Loss_Table['num_of_all_trades'][value[-1]]

        Profit_Loss_Table_by_Year_Month = pd.DataFrame({
            'year_month': year_month_list,
            'profit & loss': month_profit_loss_list,
            'positive trades': month_pos_trades,
            'negative trades': month_neg_trades,
            'all trades': month_all_trades,
        })
        Profit_Loss_Table_by_Year_Month = Profit_Loss_Table_by_Year_Month.add_suffix('_' + data_step)
        print(Profit_Loss_Table_by_Year_Month)
        Profit_Loss_Table_by_Year_Month_for_symbol = \
            pd.concat([Profit_Loss_Table_by_Year_Month_for_symbol, Profit_Loss_Table_by_Year_Month], axis=1)
    Profit_Loss_Table_by_Year_Month_for_symbol.to_csv(f'{symbol}-{start_date}-{data_steps}.csv', index=True)
    favorite_monthly_profit = 10
    monthly_profit_variance = np.mean(Profit_Loss_Table_by_Year_Month_for_symbol.iloc[:, 1] - favorite_monthly_profit)
    print(monthly_profit_variance)