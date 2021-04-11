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
    if df['MACD1_Hist'][date_pointer]*df['MACD1_Hist'][date_pointer-1]<0: return True
    else: return False

binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
            "1d": 1440}
batch_size = 750
binance_client = Client(api_key='43PXiL32cF1YFXwkeoK900wOZx8saS1T5avSRWlljStfwMrCl7lZhhJSIM1ijIzS',
                        api_secret='JjJRJ3bWQTEShF4Eu8ZigY9aEMGPnFNJMH3WoNlOQgxSgrHmLOflIavhMx0KSZFC')

"""Data"""
binance_symbols = ['LTCUSDT']
start_date = '1 Jan 2017'
end_date = '2021-03-23 00:00:00'
data_steps = ['30m']
leverage = 1
plot_width = 1500
plot_height = 1000

for symbol_row, symbol in enumerate(binance_symbols):
    Profit_Loss_Table_by_Year_Month_for_symbol = pd.DataFrame()
    for data_step in data_steps:
        df = DataHunter(symbol=symbol,start_date=start_date, end_date=end_date, step=data_step).prepare_data()
        ZC_Index = pd.DataFrame( {'zcindex':df[df['MACD_ZC'] == 1].index.values,
                                  'timestamp':df.loc[df['MACD_ZC']==1,'timestamp'],
                                  'MACD_Hist':df.loc[df['MACD_ZC']==1,'MACD_Hist']} ,
                                 columns=['zcindex','timestamp','MACD_Hist']).reset_index(drop=True)
        # region XAB Hunter
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
                        XAB_list.append([[X, A, B, None], [index_X, index_A, index_B, None, Index_4], xabc_flag])
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
                        xabc_flag = 0
                        Index_4 = ZC_Index.iloc[row_zcindex + 3, 0]
                        XABC_list.append([[X, A, B, None], [index_X, index_A, index_B, Index_4, None], xabc_flag])
                    # endregion
        # endregion

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
            break

        XAB_del_list = []
        for date_pointer in range(XAB_list[0][1][3], len(df)):
            XAB_valid_list = [x for x in XAB_list if date_pointer >= x[1][4]]
            for idx_xab, xab in enumerate(
                    XAB_valid_list[::-1]):  # xabc = [[X, A, B, C], [index_X, index_A, index_B, index_4, index_C], xabc_flag, warning, alarm]
                if xab not in XAB_del_list:

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
                    # endregion

                    if enter == 0:
                        # region Enter XABC
                        if flag == 1:#long
                            if xab[0][3]:
                                if df['low'][date_pointer] <= xab[0][3]:
                                    xab[0][3] = df['low'][date_pointer]
                                    xab[1][3] = date_pointer
                            elif df['low'][date_pointer] <= A:
                                xab[0][3] = df['low'][date_pointer]
                                xab[1][3] = date_pointer # TODO: clarify if you want to enter a trade at the same candle if close>B as low<C
                            if df['close'][date_pointer] >= B:
                                enter = 1
                                index_buy = date_pointer
                                xab_buy = xab
                                enter_price = B
                                sl = xab[0][3]
                                print(df['timestamp'][index_X], 'X:', X)
                                print(df['timestamp'][index_A], 'A:', A)
                                print(df['timestamp'][index_B], 'B:', B)
                                print(df['timestamp'][xab[1][3]], 'C:', xab[0][3])
                                print(df['timestamp'][date_pointer], 'enter:', B)
                                money_before_each_trade_list.append(money)
                        if flag == 0:#short
                            if xab[0][3]:
                                if df['high'][date_pointer] >= xab[0][3]:
                                    xab[0][3] = df['high'][date_pointer]
                                    xab[1][3] = date_pointer
                            elif df['high'][date_pointer] >= A:
                                xab[0][3] = df['high'][date_pointer]
                                xab[1][3] = date_pointer # TODO: clarify if you want to enter a trade at the same candle if close>B as low<C
                            if df['close'][date_pointer] <= B:
                                enter = 1
                                index_buy = date_pointer
                                xab_buy = xab
                                enter_price = B
                                sl = xab[0][3]
                                print(df['timestamp'][index_X], 'X:', X)
                                print(df['timestamp'][index_A], 'A:', A)
                                print(df['timestamp'][index_B], 'B:', B)
                                print(df['timestamp'][xab[1][3]], 'C:', xab[0][3])
                                print(df['timestamp'][date_pointer], 'enter:', B)
                                money_before_each_trade_list.append(money)
                        # endregion
                    else: # If it is in trade
                        if xab != xab_buy:
                            if xab[0][3]:
                                if flag and (df['low'][date_pointer] < xab[0][3] or df['high'][date_pointer] > B):
                                    XAB_del_list.append(xab)
                                if not flag and (df['high'][date_pointer] > xab[0][3] or df['low'][date_pointer] < B):
                                    XAB_del_list.append(xab)
                        if xab == xab_buy:


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
