"""
"""
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
from my_geneticalgorithm import MyGeneticAlgorithm as mga

def MACD_IND(data,win_slow,win_fast,win_sign):
    MACD_IND1 = MACD(data['close'],window_slow=win_slow,window_fast=win_fast,window_sign=win_sign)
    data['MACD']         = MACD_IND1.macd()
    data['MACD_signal']  = MACD_IND1.macd_signal()
    data['MACD_Hist']    = MACD_IND1.macd_diff()
    data['MACD_ZC']      = np.where((data['MACD_Hist']*(data['MACD_Hist'].shift(1,axis=0))) < 0,1,0)
    return data

def MACD_IND2(data,win_slow,win_fast,win_sign):
    MACD_IND2 = MACD(data['close'],window_slow=win_slow,window_fast=win_fast,window_sign=win_sign)
    data['MACD2']         = MACD_IND2.macd()
    data['MACD_signal2']  = MACD_IND2.macd_signal()
    data['MACD_Hist2']    = MACD_IND2.macd_diff()
    data['MACD_ZC2']      = np.where((data['MACD_Hist2']*(data['MACD_Hist2'].shift(1,axis=0))) < 0,1,0)
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
    if oldest_point == datetime.strptime(start_date, '%d %b %Y'): print(f'Downloading all available {kline_size} data for {symbol} from {start_date}. Be patient..!')
    else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save: data_df.to_csv(filename)
    print('All caught up..!')
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


binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440}
batch_size = 750
binance_client = Client(api_key= '43PXiL32cF1YFXwkeoK900wOZx8saS1T5avSRWlljStfwMrCl7lZhhJSIM1ijIzS',
                        api_secret= 'JjJRJ3bWQTEShF4Eu8ZigY9aEMGPnFNJMH3WoNlOQgxSgrHmLOflIavhMx0KSZFC')

"""Data"""

binance_symbols = [IOTAUSDT']
start_date = '1 Dec 2020'
end_date = '2020-06-01 00:00:00'
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
            if os.path.isfile(filename): data_org = pd.read_csv(filename,index_col=0)
            else: data_org = get_all_binance(symbol, data_step, start_date, save=True)

            data_org.index = data_org.index.map(lambda x: x if type(x) == str else str(x))
            data_org = data_org[~data_org.index.duplicated(keep='last')]

            data = data_org[:end_date].filter(['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                    'trades', 'tb_base_av', 'tb_quote_av'])

            data1 = data.astype(float).copy(deep=True)

            data2 = Ichi(data1,9,26,52)
            data3 = MACD_IND(data2,x[0],x[1],x[2])
            data4 = MACD_IND2(data3,x[3],x[4],x[5])
            df = data4.copy(deep=True)
            df.reset_index(inplace=True)
            ZC_Index = pd.DataFrame({'zcindex':df[df['MACD_ZC'] == 1].index.values,
                                      'timestamp':df.loc[df['MACD_ZC']==1,'timestamp'],
                                      'MACD_Hist':df.loc[df['MACD_ZC']==1,'MACD_Hist']} ,
                                     columns=['zcindex','timestamp','MACD_Hist']).reset_index(drop=True)

            # region XABC Hunter
            XABC_list = []
            for row_zcindex,zcindex in ZC_Index.iterrows():
                if row_zcindex+4 <= len(ZC_Index)-1:
                    if df['MACD_Hist'][zcindex[0]] >= 0:
                        # region XABC Finder
                        X = max( df.iloc[ zcindex[0] : ZC_Index.iloc[row_zcindex+1,0] ] ['high'] )
                        index_X = df.iloc[ zcindex[0] : ZC_Index.iloc[row_zcindex+1,0]]['high'].idxmax()
                        A = min( df.iloc[ZC_Index.iloc[row_zcindex+1,0] : ZC_Index.iloc[row_zcindex+2,0]]['low'] )
                        index_A = df.iloc[ZC_Index.iloc[row_zcindex+1,0] : ZC_Index.iloc[row_zcindex+2,0]]['low'].idxmin()
                        B = max( df.iloc[ZC_Index.iloc[row_zcindex+2,0] : ZC_Index.iloc[row_zcindex+3,0]]['high'] )
                        index_B = df.iloc[ZC_Index.iloc[row_zcindex+2,0] : ZC_Index.iloc[row_zcindex+3,0]]['high'].idxmax()
                        C = min( df.iloc[ZC_Index.iloc[row_zcindex+3,0] : ZC_Index.iloc[row_zcindex+4,0]]['low'] )
                        index_C = df.iloc[ZC_Index.iloc[row_zcindex+3,0] : ZC_Index.iloc[row_zcindex+4,0]]['low'].idxmin()
                        if A<X and B<X and C<X and B>A and C<A:
                            xabc_flag = 1
                            Index_4 = ZC_Index.iloc[row_zcindex+4,0]
                            stop_loss = C
                            sudo_stop_loss = C
                            XABC_list.append([[X,A,B,C],[index_X,index_A,index_B,index_C],xabc_flag])

                        # endregion
                    if df['MACD_Hist'][zcindex[0]] < 0:
                        # region XABC Finder
                        X = min(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'])
                        index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'].idxmin()
                        A = max(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['high'])
                        index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['high'].idxmax()
                        B = min(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['low'])
                        index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['low'].idxmin()
                        C = max(df.iloc[ZC_Index.iloc[row_zcindex + 3, 0]: ZC_Index.iloc[row_zcindex + 4, 0]]['high'])
                        index_C = df.iloc[ZC_Index.iloc[row_zcindex + 3, 0]: ZC_Index.iloc[row_zcindex + 4, 0]]['high'].idxmax()
                        if A>X and B>X and C>X and B<A and C>A:
                            xabc_flag = 0
                            Index_4 = ZC_Index.iloc[row_zcindex + 4, 0]
                            stop_loss = C
                            sudo_stop_loss = C
                            XABC_list.append([[X, A, B, C], [index_X, index_A, index_B, index_C], xabc_flag])
                        # endregion
            # endregion

            # region initializing params
            money=1
            trade_fee = 0.002
            buy=0
            date_pointer=0
            date_of_trade_list = []
            profit_loss_list = []
            money_after_each_trade_list = []
            money_before_each_trade_list=[]
            num_of_pos_trades = 0
            num_of_neg_trades = 0
            num_of_pos_trades_list = []
            num_of_neg_trades_list = []
            # endregion

            if not XABC_list:
                # print('XABC list is empty')
                return XABC_list

            XABC_del_list=[]
            for date_pointer in range(XABC_list[0][1][3], len(df)):
                XABC_valid_list = [x for x in XABC_list if date_pointer>=x[1][3]]
                for idx_xabc,xabc in enumerate(XABC_valid_list[::-1]): # xabc = [[X, A, B, C], [index_X, index_A, index_B, index_C], xabc_flag]
                    if xabc not in XABC_del_list:
                        # region XABC params initialization
                        X = xabc[0][0]
                        A = xabc[0][1]
                        B = xabc[0][2]
                        C = xabc[0][3]
                        index_X = xabc[1][0]
                        index_A = xabc[1][1]
                        index_B = xabc[1][2]
                        index_C = xabc[1][3]
                        flag = xabc[2]
                        stop_loss = xabc[3] if len(xabc)>=4 else C
                        flag_sudo_stop_loss = xabc[4] if len(xabc)>=5 else 0
                        #endregion
                        if buy == 1 and xabc != xabc_buy:
                            # region Eliminate XABC when in trade
                            if flag == 1 and (df['low'][date_pointer] < stop_loss or df['close'][date_pointer] > B): XABC_del_list.append(xabc)
                            if flag == 0 and (df['high'][date_pointer] > stop_loss or df['close'][date_pointer] < B): XABC_del_list.append(xabc)
                            # endregion
                        if buy == 1 and xabc == xabc_buy:
                            # region StopLoss
                            if flag==1 and df['MACD_Hist2'][date_pointer] < 0:
                                last_positive_MACD_index = df['low'][:date_pointer+1].loc[df['MACD_Hist2']>0].index[-1]
                                sudo_stop_loss = min(df['low'][last_positive_MACD_index+1:date_pointer+1])
                                flag_sudo_stop_loss = 1
                                xabc.append(flag_sudo_stop_loss)
                            if flag==1 and df['MACD_Hist2'][date_pointer] > 0 and flag_sudo_stop_loss == 1:
                                stop_loss = sudo_stop_loss
                                xabc[3] = stop_loss

                            if flag==0 and df['MACD_Hist2'][date_pointer] > 0:
                                last_negative_MACD_index = df['high'][:date_pointer+1].loc[df['MACD_Hist2']<0].index[-1]
                                sudo_stop_loss = max(df['high'][last_negative_MACD_index+1:date_pointer+1])
                                flag_sudo_stop_loss = 1
                                xabc.append(flag_sudo_stop_loss)
                            if flag==0 and df['MACD_Hist2'][date_pointer] < 0 and flag_sudo_stop_loss==1:
                                stop_loss = sudo_stop_loss
                                xabc[3] = stop_loss
                            # endregion
                            # region Exit XABC
                            if flag == 1 and df['low'][date_pointer] < stop_loss:
                                buy = 0
                                index_sell = date_pointer
                                XABC_del_list.append(xabc)
                                if stop_loss > B:
                                    profit = leverage * ((stop_loss - B) / B) - trade_fee
                                    money = money + profit * money
                                    profit_loss_list.append(profit)
                                    num_of_pos_trades += 1
                                    # print(df['timestamp'][date_pointer], 'sell++:', stop_loss)
                                    # print('profit:', profit)
                                if stop_loss <= B:
                                    loss = -leverage * ((B - stop_loss) / B) - trade_fee
                                    money = money + loss * money
                                    profit_loss_list.append(loss)
                                    num_of_neg_trades += 1
                                    # print(df['timestamp'][date_pointer], 'sell+:', stop_loss)
                                    # print('loss:', loss)
                                # plot_figure(df, xabc[1][0], xabc[1][1], xabc[1][2], xabc[1][3], index_buy, index_sell,
                                #             xabc[0][0], xabc[0][1], xabc[0][2], xabc[0][3], plot_width, plot_height)
                                date_of_trade_list.append(df['timestamp'][date_pointer])
                                num_of_neg_trades_list.append(num_of_neg_trades)
                                num_of_pos_trades_list.append(num_of_pos_trades)
                                money_after_each_trade_list.append(money)
                                break
                            if flag == 0 and df['high'][date_pointer] > stop_loss:
                                buy = 0
                                index_sell = date_pointer
                                XABC_del_list.append(xabc)
                                if stop_loss < B:
                                    profit = leverage * ((B - stop_loss) / B) - trade_fee
                                    money = money + profit * money
                                    profit_loss_list.append(profit)
                                    num_of_pos_trades += 1
                                    # print(df['timestamp'][date_pointer], 'sell++:', stop_loss)
                                    # print('profit:', profit)
                                if stop_loss >= B:
                                    loss = -leverage * ((stop_loss - B) / B) - trade_fee
                                    money = money + loss * money
                                    profit_loss_list.append(loss)
                                    num_of_neg_trades += 1
                                    # print(df['timestamp'][date_pointer], 'sell+:', stop_loss)
                                    # print('loss:', loss)
                                # plot_figure(df, xabc[1][0], xabc[1][1], xabc[1][2], xabc[1][3], index_buy, index_sell,
                                #             xabc[0][0], xabc[0][1], xabc[0][2], xabc[0][3], plot_width, plot_height)
                                date_of_trade_list.append(df['timestamp'][date_pointer])
                                num_of_neg_trades_list.append(num_of_neg_trades)
                                num_of_pos_trades_list.append(num_of_pos_trades)
                                money_after_each_trade_list.append(money)
                                break

                            # endregion
                        if buy==0:
                            # region Enter XABC
                            if flag==1:
                                if df['low'][date_pointer]<C: XABC_del_list.append(xabc)
                                if df['close'][date_pointer]>=B: #TODO: Ichimoku should be added
                                    buy=1
                                    index_buy = date_pointer
                                    xabc.append(stop_loss)
                                    xabc_buy = xabc
                                    # print(df['timestamp'][index_X], 'X:', X)
                                    # print(df['timestamp'][index_A], 'A:', A)
                                    # print(df['timestamp'][index_B], 'B:', B)
                                    # print(df['timestamp'][index_C], 'C:', C)
                                    # print(df['timestamp'][date_pointer], 'buy+:', B)
                                    money_before_each_trade_list.append(money)

                            if flag==0:
                                if df['high'][date_pointer]>C: XABC_del_list.append(xabc)
                                if df['close'][date_pointer]<=B: #TODO: Ichimoku should be added
                                    buy=1
                                    index_buy = date_pointer
                                    xabc.append(stop_loss)
                                    xabc_buy = xabc
                                    # print(df['timestamp'][index_X], 'X:', X)
                                    # print(df['timestamp'][index_A], 'A:', A)
                                    # print(df['timestamp'][index_B], 'B:', B)
                                    # print(df['timestamp'][index_C], 'C:', C)
                                    # print(df['timestamp'][date_pointer], 'buy+:', B)
                                    money_before_each_trade_list.append(money)

                            # endregion


            print(money)

            # region
            """ If there is a buy position but still the sell position doesn't
            occur it would be a problem and this problem is solved in this region
            """
            lists = [date_of_trade_list,profit_loss_list,num_of_pos_trades_list,
                     num_of_neg_trades_list,money_after_each_trade_list,money_before_each_trade_list]
            unique_len = [len(i) for i in lists]
            list_length = min(unique_len)
            for index,l in enumerate(lists):
                if len(l)>list_length:
                    del lists[index][-1]
                list_length = len(l)
            #endregion

            Profit_Loss_Table = pd.DataFrame({
                'date':date_of_trade_list,
                'profit & loss':profit_loss_list,
                'num_of_pos_trades':num_of_pos_trades_list,
                'num_of_neg_trades':num_of_neg_trades_list,
                'money_after_trade':money_after_each_trade_list,
                'money_before_trade':money_before_each_trade_list
            })
            Profit_Loss_Table['date'] = pd.to_datetime(Profit_Loss_Table['date'])
            Profit_Loss_Table['num_of_all_trades'] = Profit_Loss_Table['num_of_neg_trades'] + Profit_Loss_Table['num_of_pos_trades']
            Profit_Loss_Table['year']  = Profit_Loss_Table['date'].apply(lambda t: t.year)
            Profit_Loss_Table['month'] = Profit_Loss_Table['date'].apply(lambda t: t.month)
            Profit_Loss_Table['day']   = Profit_Loss_Table['date'].apply(lambda t: t.day)

            Money_each_month = Profit_Loss_Table.groupby(['year','month'])
            month_profit_loss_list=[]
            year_month_list=[]
            month_pos_trades = []
            month_neg_trades = []
            month_all_trades = []
            last_month_num_pos_trades = 0
            last_month_num_neg_trades = 0
            last_month_num_all_trades = 0
            for key,value in zip(Money_each_month.groups.keys(),Money_each_month.groups.values()):
                first_money = Profit_Loss_Table['money_before_trade'][value[0]]
                last_money = Profit_Loss_Table['money_after_trade'][value[-1]]
                month_profit=(last_money-first_money)*100/first_money
                month_profit_loss_list.append(month_profit)


                month_pos_trades.append(Profit_Loss_Table['num_of_pos_trades'][value[-1]] - last_month_num_pos_trades)
                month_neg_trades.append(Profit_Loss_Table['num_of_neg_trades'][value[-1]] - last_month_num_neg_trades)
                month_all_trades.append(Profit_Loss_Table['num_of_all_trades'][value[-1]] - last_month_num_all_trades)

                year_month_list.append(key)
                last_month_num_pos_trades = Profit_Loss_Table['num_of_pos_trades'][value[-1]]
                last_month_num_neg_trades = Profit_Loss_Table['num_of_neg_trades'][value[-1]]
                last_month_num_all_trades = Profit_Loss_Table['num_of_all_trades'][value[-1]]

            Profit_Loss_Table_by_Year_Month = pd.DataFrame({
                'year_month':year_month_list,
                'profit & loss':month_profit_loss_list,
                'positive trades':month_pos_trades,
                'negative trades':month_neg_trades,
                'all trades':month_all_trades,
            })
            Profit_Loss_Table_by_Year_Month = Profit_Loss_Table_by_Year_Month.add_suffix('_'+data_step)
            # print(Profit_Loss_Table_by_Year_Month)
            Profit_Loss_Table_by_Year_Month_for_symbol = \
                pd.concat([Profit_Loss_Table_by_Year_Month_for_symbol,Profit_Loss_Table_by_Year_Month], axis=1)
        Profit_Loss_Table_by_Year_Month_for_symbol.to_csv(f'{symbol}-{start_date}-{data_steps}.csv', index=True)
    return money

ali = {
    'fast_window':[5,6,7,12,13,26,30,40,52],
    'slow_window':[4,5,6,7,12,24,30,40,48],
    'sign_window':[4,6,8,9,10,12,14,16,18,20],
    'fast_window2':[5,6,7,12,13,26,30,40,52],
    'slow_window2':[4,5,6,7,12,24,30,40,48],
    'sign_window2':[4,6,8,9,10,12,14,16,18,20]
}

GA = mga(config=ali, function=f, run_iter=20, population_size=100, n_crossover=3, crossover_mode='random')
best_params=GA.run()
print(best_params)