# This system uses trailing stop loss with lower time step
import pandas as pd
import math
import os.path
import time
import ta
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
import numpy as np
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator as RSI
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from my_geneticalgorithm import MyGeneticAlgorithm as mga
from data_prep.data_hunter import DataHunter


def macd_phase_change(df, date_pointer):
    if df['MACD1_Hist'][date_pointer] * df['MACD1_Hist'][date_pointer - 1] < 0:
        return True
    else:
        return False


def print_trade(df, df2, X, A, B, xab, enter_price, exit_price, index_X, index_A, index_B, index_buy,
                index_sell):
    print(df['timestamp'][index_X], 'X:', X)
    print(df['timestamp'][index_A], 'A:', A)
    print(df['timestamp'][index_B], 'B:', B)
    print(df['timestamp'][xab[1][3]], 'C:', xab[0][3])
    print(df['timestamp'][index_buy], 'enter:', enter_price)
    print(df2['timestamp'][index_sell], 'exit:', exit_price)


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
    return X, A, B, index_X, index_A, index_B, index_4, flag


def xab_enter_check(df, date_pointer, xab, enter):
    if xab[2] and df['close'][date_pointer] >= xab[0][2]:
        enter = 1
    if not xab[2] and df['close'][date_pointer] <= xab[0][2]:
        enter = 1
    return enter


def xab_completor(df, date_pointer, xab, XAB_del_list):
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
            if df['MACD1_Hist'][date_pointer] < 0 and xab[5] == 0:  # dont_find_C = xab[0][5]
                if df['low'][date_pointer] <= xab[0][3]:
                    xab[0][3] = df['low'][date_pointer]
                    xab[1][3] = date_pointer
            if df['MACD1_Hist'][date_pointer] > 0:
                xab[5] = 1
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
        if not xab[0][3] and not xab[5]:
            if df['low'][date_pointer] <= A and df['MACD1_Hist'][date_pointer] < 0 and xab[5] == 0:
                xab[0][3] = df['low'][date_pointer]
                xab[1][3] = date_pointer
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
            if df['MACD1_Hist'][date_pointer] > 0:
                xab[5] = 1
                XAB_del_list.append(xab)

    if flag == 0:  # short
        if xab[0][3]:
            if df['MACD1_Hist'][date_pointer] > 0 and xab[5] == 0:
                if df['high'][date_pointer] >= xab[0][3]:
                    xab[0][3] = df['high'][date_pointer]
                    xab[1][3] = date_pointer
            if df['MACD1_Hist'][date_pointer] < 0:
                xab[5] = 1
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
        if not xab[0][3] and not xab[5]:
            if df['high'][date_pointer] >= A and df['MACD1_Hist'][date_pointer] > 0 and xab[5] == 0:
                xab[0][3] = df['high'][date_pointer]
                xab[1][3] = date_pointer
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
            if df['MACD1_Hist'][date_pointer] < 0:
                xab[5] = 1
                XAB_del_list.append(xab)

    return xab, XAB_del_list


def xab_reject_decision(df, dp, xab, XAB_del_list, XAB_check_list):
    if xab[2] == 1:
        if df['low'][dp] < xab[0][3]:
            XAB_del_list.append(xab)
        elif df['close'][dp] > xab[0][2]:
            if xab not in XAB_check_list:
                XAB_check_list.append(xab)
    if xab[2] == 0:
        if df['high'][dp] > xab[0][3]:
            XAB_del_list.append(xab)
        elif df['close'][dp] < xab[0][2]:
            if xab not in XAB_check_list:
                XAB_check_list.append(xab)
    return XAB_del_list, XAB_check_list


def equal_date_pointer(df1, df2, dp1, dp2, main_data_step):
    dp2_str = df1['timestamp'][dp1]
    if main_data_step == '30m':
        try:
            dp2 = df2[df2['timestamp'] == dp2_str].index.values[0] + 2
        except IndexError:
            dp2 = dp2 + 2
    return dp2


def stop_loss_trail(df, date_pointer, flag, xab):
    if flag == 1:
        if df['MACD1_Hist'][date_pointer] < 0:
            if macd_phase_change(df, date_pointer) or df['low'][date_pointer] <= xab[4]:
                xab[4] = df['low'][date_pointer]
        if df['MACD1_Hist'][date_pointer] > 0: xab[3] = xab[4]
    if flag == 0:
        if df['MACD1_Hist'][date_pointer] > 0:
            if macd_phase_change(df, date_pointer) or df['high'][date_pointer] >= xab[4]:
                xab[4] = df['high'][date_pointer]
        if df['MACD1_Hist'][date_pointer] < 0: xab[3] = xab[4]


def trader(x):
    print(x, symbol, data_step)
    Profit_Loss_Table_by_Year_Month_for_symbol = pd.DataFrame()
    # region Data Preparation
    df = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                    step=data_step).prepare_data(macd_slow=x[0], macd_fast=x[1],
                                                 macd_sign=x[2])
    df2 = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                     step='15m').prepare_data(macd_slow=x[0], macd_fast=x[1], macd_sign=x[2])
    ZC_Index = pd.DataFrame({'zcindex': df[df['MACD1_ZC'] == 1].index.values,
                             'timestamp': df.loc[df['MACD1_ZC'] == 1, 'timestamp'],
                             'MACD1_Hist': df.loc[df['MACD1_ZC'] == 1, 'MACD1_Hist']},
                            columns=['zcindex', 'timestamp', 'MACD1_Hist']).reset_index(
        drop=True)
    # endregion
    # region XAB Hunter
    # TODO: we have to change the strategy of XAB
    XAB_list = []
    for row_zcindex, zcindex in ZC_Index.iterrows():
        if row_zcindex + 3 <= len(ZC_Index) - 1:
            if df['MACD1_Hist'][zcindex[0]] >= 0:
                # region XAB Finder
                X = max(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['high'])
                index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]][
                    'high'].idxmax()
                A = min(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[
                    row_zcindex + 2, 0]]['low'])
                index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[
                    row_zcindex + 2, 0]][
                    'low'].idxmin()
                B = max(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[
                    row_zcindex + 3, 0]]['high'])
                index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[
                    row_zcindex + 3, 0]][
                    'high'].idxmax()
                if A < X and B < X and B > A:
                    xab_flag = 1
                    Index_4 = ZC_Index.iloc[row_zcindex + 3, 0]
                    XAB_list.append(
                        [[X, A, B, None], [index_X, index_A, index_B, None, Index_4],
                         xab_flag, None, None, 0])
                # endregion

            if df['MACD1_Hist'][zcindex[0]] < 0:
                # region XAB Finder
                X = min(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'])
                index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]][
                    'low'].idxmin()
                A = max(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[
                    row_zcindex + 2, 0]]['high'])
                index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[
                    row_zcindex + 2, 0]][
                    'high'].idxmax()
                B = min(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[
                    row_zcindex + 3, 0]]['low'])
                index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[
                    row_zcindex + 3, 0]][
                    'low'].idxmin()
                if A > X and B > X and B < A:
                    xab_flag = 0
                    Index_4 = ZC_Index.iloc[row_zcindex + 3, 0]
                    XAB_list.append(
                        [[X, A, B, None], [index_X, index_A, index_B, None, Index_4],
                         xab_flag, None, None, 0])
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
        return None
    XAB_del_list = []  # This the list of XABs that are rejected
    XAB_check_list = []  # This is the list of XABs that may be entered and are valid to enter but right now the system is in trade
    date_pointer2 = 0
    for date_pointer in range(XAB_list[0][1][4], len(df)):
        date_pointer22 = equal_date_pointer(df, df2, date_pointer, date_pointer2, data_step)
        XAB_valid_list = [x for x in XAB_list if
                          date_pointer >= x[1][4]]  # This is the list of XABs before the date_pointer
        for idx_xab, xab in enumerate(XAB_valid_list[
                                      ::-1]):  # xabc = [[X, A, B, C], [index_X, index_A, index_B, index_4, index_C], xab_flag, sl, sudo_sl, dont_find_C]
            if xab not in XAB_del_list:
                X, A, B, index_X, index_A, index_B, index_4, flag = xab_initializer(xab)
                if enter == 0:
                    xab, XAB_del_list = xab_completor(df, date_pointer, xab, XAB_del_list)
                    if xab[0][3]:
                        enter = xab_enter_check(df, date_pointer, xab, enter)
                    if enter == 1:
                        index_buy = date_pointer
                        xab_buy = xab
                        enter_price = xab[0][2]
                        xab[3] = xab[0][3]  # C is placed in sl
                        xab[4] = xab[0][3]  # C is placed in sudo_sl
                        money_before_each_trade_list.append(money)
                    elif xab[0][3] and xab[5]:
                        XAB_del_list, XAB_check_list = xab_reject_decision(df, date_pointer,
                                                                           xab,
                                                                           XAB_del_list,
                                                                           XAB_check_list)

                if enter == 1:  # If it is in trade
                    if xab != xab_buy:
                        xab, XAB_del_list = xab_completor(df, date_pointer, xab,
                                                          XAB_del_list)
                        if xab[0][3]:
                            XAB_del_list, XAB_check_list = xab_reject_decision(df,
                                                                               date_pointer,
                                                                               xab,
                                                                               XAB_del_list,
                                                                               XAB_check_list)
                    if xab == xab_buy:
                        for date_pointer2 in [date_pointer22, date_pointer22 + 1]:
                            if enter == 0 or date_pointer2 > len(df2) - 1: break
                            if macd_phase_change(df2, date_pointer2): xab[3] = xab[4]
                            # This is because when the phase is changed, first you need to
                            # replace the sl with sudo_sl
                            if flag == 1:
                                if df2['low'][date_pointer2] < xab[3]:
                                    enter = 0
                                    index_sell = date_pointer2
                                    exit_price = xab[3]
                                    # print_trade(df, df2, X, A, B, xab, enter_price, exit_price,
                                    #             index_X, index_A, index_B,
                                    #             index_buy, index_sell)
                                    if exit_price > B:
                                        profit = leverage * ((exit_price - B) / B) - trade_fee
                                        money = money + profit * money
                                        profit_loss_list.append(profit)
                                        num_of_pos_trades += 1
                                        # print('profit:', profit)
                                        # print('money:', money)
                                    if exit_price <= B:
                                        loss = leverage * ((exit_price - B) / B) - trade_fee
                                        money = money + loss * money
                                        profit_loss_list.append(loss)
                                        num_of_neg_trades += 1
                                        # print('loss:', loss)
                                        # print('money:', money)
                                    # # plot_figure(df, xabc[1][0], xabc[1][1], xabc[1][2], xabc[1][3], index_buy, index_sell,
                                    # #             xabc[0][0], xabc[0][1], xabc[0][2], xabc[0][3], plot_width, plot_height)
                                    date_of_trade_list.append(df2['timestamp'][
                                                                  date_pointer2])
                                    num_of_neg_trades_list.append(num_of_neg_trades)
                                    num_of_pos_trades_list.append(num_of_pos_trades)
                                    money_after_each_trade_list.append(money)
                                    XAB_del_list.append(xab)

                                    if XAB_check_list:
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
                                    stop_loss_trail(df2, date_pointer2, flag, xab)

                            if flag == 0:
                                if df2['high'][date_pointer2] > xab[3]:
                                    enter = 0
                                    index_sell = date_pointer2
                                    exit_price = xab[3]
                                    # print_trade(df, df2, X, A, B, xab, enter_price, \
                                    #                    exit_price,
                                    #             index_X, index_A, index_B,
                                    #             index_buy, index_sell)
                                    if exit_price < B:
                                        profit = leverage * ((B - exit_price) / B) - trade_fee
                                        money = money + profit * money
                                        profit_loss_list.append(profit)
                                        num_of_pos_trades += 1
                                        # print('profit:', profit)
                                        # print('money:', money)
                                    if exit_price >= B:
                                        loss = leverage * ((B - exit_price) / B) - trade_fee
                                        money = money + loss * money
                                        profit_loss_list.append(loss)
                                        num_of_neg_trades += 1
                                        # print('loss:', loss)
                                        # print('money:', money)
                                    # plot_figure(df, xabc[1][0], xabc[1][1], xabc[1][2], xabc[1][3], index_buy, index_sell,
                                    #             xabc[0][0], xabc[0][1], xabc[0][2], xabc[0][3], plot_width, plot_height)
                                    date_of_trade_list.append(df2['timestamp'][
                                                                  date_pointer2])
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
                                    stop_loss_trail(df2, date_pointer2, flag, xab)

    # region
    """ If there is a buy position but still the sell position doesn't
    occur it would be a problem and this problem is solved in this region
    """
    lists = [date_of_trade_list, profit_loss_list, num_of_pos_trades_list,
             num_of_neg_trades_list, money_after_each_trade_list,
             money_before_each_trade_list]
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
    Profit_Loss_Table['num_of_all_trades'] = Profit_Loss_Table['num_of_neg_trades'] + \
                                             Profit_Loss_Table[
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

        month_pos_trades.append(
            Profit_Loss_Table['num_of_pos_trades'][value[-1]] - last_month_num_pos_trades)
        month_neg_trades.append(
            Profit_Loss_Table['num_of_neg_trades'][value[-1]] - last_month_num_neg_trades)
        month_all_trades.append(
            Profit_Loss_Table['num_of_all_trades'][value[-1]] - last_month_num_all_trades)

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
    Profit_Loss_Table_by_Year_Month = Profit_Loss_Table_by_Year_Month.add_suffix(
        '_' + data_step)
    # print(Profit_Loss_Table_by_Year_Month)
    Profit_Loss_Table_by_Year_Month_for_symbol = \
        pd.concat(
            [Profit_Loss_Table_by_Year_Month_for_symbol, Profit_Loss_Table_by_Year_Month],
            axis=1)

    # print(Profit_Loss_Table_by_Year_Month_for_symbol)
    num_of_months_with_loss = Profit_Loss_Table_by_Year_Month_for_symbol[
        Profit_Loss_Table_by_Year_Month_for_symbol[
            f'profit & loss_{data_step}'] < 0].shape[0]
    num_of_months_with_profit = Profit_Loss_Table_by_Year_Month_for_symbol[
        Profit_Loss_Table_by_Year_Month_for_symbol[
            f'profit & loss_{data_step}'] > 0].shape[0]

    money_negative = Profit_Loss_Table_by_Year_Month_for_symbol[
        f'profit & loss_{data_step}'][
        Profit_Loss_Table_by_Year_Month_for_symbol[
            f'profit & loss_{data_step}'] < 0].sum()

    money_positive = Profit_Loss_Table_by_Year_Month_for_symbol[
        f'profit & loss_{data_step}'][
        Profit_Loss_Table_by_Year_Month_for_symbol[
            f'profit & loss_{data_step}'] > 0].sum()
    # print(money_positive,money_negative)
    if num_of_months_with_loss == 0:
        decision_factor = money_positive * num_of_months_with_profit * 1000
    else:
        decision_factor = (money_positive / abs(money_negative)) * (
                num_of_months_with_profit / num_of_months_with_loss)
    print(decision_factor)
    print('==========')
    return decision_factor


config = {
    'slow_window': [5, 6, 7, 12, 13, 26, 30, 40, 52],
    'fast_window': [4, 5, 6, 7, 12, 24, 30, 40, 48],
    'sign_window': [4, 6, 8, 9, 10, 12, 14, 16, 18, 20]
}

GA = mga(config=config, function=trader, run_iter=5, population_size=200, n_crossover=3,
         crossover_mode='random')

coins_datastep_list = [
    # ('LTCUSDT','1h'),
    # ('BTCUSDT','1h'),
    # ('IOTAUSDT','1h'),
    # ('ETHUSDT','1h'),
    # ('TRXUSDT', '1h'),
    # ('NEOUSDT', '1h'),
    # ('LTCUSDT', '30m'),
    # ('BTCUSDT', '30m'),
    # ('IOTAUSDT','30m'),
    # ('ETHUSDT', '30m'),
    # ('TRXUSDT', '30m'),
    ('NEOUSDT', '30m'),

    # ('LTCUSDT', '15m'),
    # ('BTCUSDT', '15m'),
    # ('IOTAUSDT', '15m'),
    # ('ETHUSDT', '15m'),
    # ('TRXUSDT', '15m'),
    # ('NEOUSDT', '15m'),
]
# region Data
start_date = '1 Apr 2020'
end_date = '2021-04-15 00:00:00'
leverage = 1
plot_width = 1500
plot_height = 1000
# endregion
# for symbol,data_step in coins_datastep_list:
#     dh = DataHunter(symbol, start_date, end_date, data_step)
#     dh.download_data()

for symbol, data_step in coins_datastep_list:
    best_params = GA.run()
    print(best_params)
    best_params.to_csv(f'Genetic-v2.2.2-{symbol}-{start_date}-{data_step}.csv', index=True)
#
#
# os.system("shutdown /s /t 1")
