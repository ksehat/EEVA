# This system uses trailing stop loss with lower time step
import pandas as pd
import math
import ast
import os.path
import concurrent.futures
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
from data_prep.data_hunter import DataHunter
from plot_XABC import plot_figure


def macd_phase_change(df, date_pointer):
    if df['MACD1_Hist'][date_pointer] * df['MACD1_Hist'][date_pointer - 1] < 0:
        return True
    else:
        return False


def print_trade(df, X, A, B, xab, enter_price, exit_price, index_X, index_A, index_B,
                index_buy,
                index_sell):
    print(df['timestamp'][index_X], 'X:', X)
    print(df['timestamp'][index_A], 'A:', A)
    print(df['timestamp'][index_B], 'B:', B)
    print(df['timestamp'][xab[1][3]], 'C:', xab[0][3])
    print(df['timestamp'][index_buy], 'enter:', enter_price)
    print(df['timestamp'][index_sell], 'exit:', exit_price)


def xab_enter_check(df, date_pointer, xab, enter, enter_price, XAB_del_list,
                    wait_candles):
    if xab[2]:
        if xab[6] >= 1:
            xab[6] += 1
        if df['close'][date_pointer] >= xab[0][2] and xab[6] == 0:
            if df['low'][date_pointer] >= xab[0][3]:
                xab[6] = 1
            else:
                XAB_del_list.append(xab)
                return enter, enter_price
        if xab[6] >= 2 and xab[6] <= wait_candles:
            if df['low'][date_pointer] <= xab[0][2]:
                enter = 1
                enter_price = xab[0][2]
        if enter == 0 and xab[6] > wait_candles:
            XAB_del_list.append(xab)
        if enter:
            xab[3] = xab[0][3]  # C is placed in sl
    if not xab[2]:
        if xab[6] >= 1:
            xab[6] += 1
        if df['close'][date_pointer] <= xab[0][2] and xab[6] == 0:
            if df['high'][date_pointer] <= xab[0][3]:
                xab[6] = 1
            else:
                XAB_del_list.append(xab)
                return enter, enter_price
        if xab[6] >= 2 and xab[6] <= wait_candles:
            if df['high'][date_pointer] >= xab[0][2]:
                enter = 1
                enter_price = xab[0][2]
        if enter == 0 and xab[6] > wait_candles:
            XAB_del_list.append(xab)
        if enter:
            xab[3] = xab[0][3]  # C is placed in sl
    return enter, enter_price


def non_xab_buy_counter(df, date_pointer, xab, XAB_del_list, wait_candles):
    if xab[2] and (df['close'][date_pointer] >= xab[0][2] or xab[6]):
        if xab[6] >= 1:
            xab[6] += 1
        if xab[6] == 0:
            xab[6] = 1
        if xab[6] >= 2 and xab[6] <= wait_candles:
            if df['low'][date_pointer] <= xab[0][2]:
                XAB_del_list.append(xab)
        if xab[6] > wait_candles:
            XAB_del_list.append(xab)
    if not xab[2] and (df['close'][date_pointer] <= xab[0][2] or xab[6]):
        if xab[6] >= 1:
            xab[6] += 1
        if xab[6] == 0:
            xab[6] = 1
        if xab[6] >= 2 and xab[6] <= wait_candles:
            if df['high'][date_pointer] >= xab[0][2]:
                XAB_del_list.append(xab)
        if xab[6] > wait_candles:
            XAB_del_list.append(xab)


def xab_completor(df, date_pointer, xab, XAB_del_list):
    C_at_this_candle = 0
    if xab[2] == 1:  # long
        if xab[0][3]:
            if df['MACD1_Hist'][date_pointer] < 0 and xab[5] == 0:  # dont_find_C = xab[5]
                if df['low'][date_pointer] <= xab[0][3]:
                    xab[0][3] = df['low'][date_pointer]
                    xab[1][3] = date_pointer
            if df['MACD1_Hist'][date_pointer] > 0:
                xab[5] = 1
        if not xab[0][3] and not xab[5]:
            if df['low'][date_pointer] <= xab[0][1] and df['MACD1_Hist'][date_pointer] < 0 and xab[
                5] == 0:
                C_at_this_candle = 1
                xab[0][3] = df['low'][date_pointer]
                xab[1][3] = date_pointer
            if df['MACD1_Hist'][date_pointer] > 0:
                xab[5] = 1
                XAB_del_list.append(xab)

    if xab[2] == 0:  # short
        if xab[0][3]:
            if df['MACD1_Hist'][date_pointer] > 0 and xab[5] == 0:
                if df['high'][date_pointer] >= xab[0][3]:
                    xab[0][3] = df['high'][date_pointer]
                    xab[1][3] = date_pointer
            if df['MACD1_Hist'][date_pointer] < 0:
                xab[5] = 1
        if not xab[0][3] and not xab[5]:
            if df['high'][date_pointer] >= xab[0][1] and df['MACD1_Hist'][date_pointer] > 0 and xab[
                5] == 0:
                C_at_this_candle = 1
                xab[0][3] = df['high'][date_pointer]
                xab[1][3] = date_pointer
            if df['MACD1_Hist'][date_pointer] < 0:
                xab[5] = 1
                XAB_del_list.append(xab)

    return xab, XAB_del_list, C_at_this_candle


def xab_reject_decision(df, dp, xab, XAB_del_list, wait_candles):
    if xab[2] == 1:
        if df['low'][dp] < xab[0][3] or (df['close'][dp] > xab[0][2] and xab[6] > wait_candles):
            XAB_del_list.append(xab)
    if xab[2] == 0:
        if df['high'][dp] > xab[0][3] or (df['close'][dp] < xab[0][2] and xab[6] > wait_candles):
            XAB_del_list.append(xab)
    return XAB_del_list


def extra_calculations(df, xab, money, enter_price,
                       exit_price, index_buy, index_sell, leverage,
                       print_outputs, trade_fee, num_of_pos_trades,
                       num_of_neg_trades, profit_loss_list,
                       num_of_neg_trades_list, num_of_pos_trades_list,
                       money_after_each_trade_list, XAB_del_list):
    if xab[2]:
        if print_outputs:
            print_trade(df, xab[0][0], xab[0][1], xab[0][2], xab,
                        enter_price,
                        exit_price,
                        xab[1][0], xab[1][1], xab[1][2],
                        index_buy, index_sell)
        if exit_price > enter_price:
            profit = leverage * ((exit_price - enter_price) / enter_price) - trade_fee
            money = money + profit * money
            profit_loss_list.append(profit)
            num_of_pos_trades += 1
            if print_outputs:
                print('profit:', profit)
                print('money:', money)
        if exit_price <= enter_price:
            loss = leverage * ((exit_price - enter_price) / enter_price) - trade_fee
            money = money + loss * money
            profit_loss_list.append(loss)
            num_of_neg_trades += 1
            if print_outputs:
                print('loss:', loss)
                print('money:', money)
        num_of_neg_trades_list.append(num_of_neg_trades)
        num_of_pos_trades_list.append(num_of_pos_trades)
        money_after_each_trade_list.append(money)
        XAB_del_list.append(xab)
    if not xab[2]:
        if print_outputs:
            print_trade(df, xab[0][0], xab[0][1], xab[0][2], xab,
                        enter_price,
                        exit_price,
                        xab[1][0], xab[1][1], xab[1][2],
                        index_buy, index_sell)
        if exit_price < enter_price:
            profit = leverage * ((enter_price - exit_price) / enter_price) - trade_fee
            money = money + profit * money
            profit_loss_list.append(profit)
            num_of_pos_trades += 1
            if print_outputs:
                print('profit:', profit)
                print('money:', money)
        if exit_price >= enter_price:
            loss = leverage * ((enter_price - exit_price) / enter_price) - trade_fee
            money = money + loss * money
            profit_loss_list.append(loss)
            num_of_neg_trades += 1
            if print_outputs:
                print('loss:', loss)
                print('money:', money)
        num_of_neg_trades_list.append(num_of_neg_trades)
        num_of_pos_trades_list.append(num_of_pos_trades)
        money_after_each_trade_list.append(money)
        XAB_del_list.append(xab)
    return money, num_of_pos_trades, num_of_neg_trades


def money_calc(df, date_pointer, xab, enter_price, exit_price, leverage, trade_fee,
               money,
               profit_loss_list,
               date_of_trade_list,
               num_of_neg_trades_list,
               num_of_pos_trades_list,
               money_after_each_trade_list,
               XAB_del_list,
               num_of_pos_trades,
               num_of_neg_trades,
               print_outputs):
    # profit_loss_list = copy.deepcopy(profit_loss_list1)
    # date_of_trade_list = copy.deepcopy(date_of_trade_list1)
    # num_of_neg_trades_list = copy.deepcopy(num_of_neg_trades_list1)
    # num_of_pos_trades_list = copy.deepcopy(num_of_pos_trades_list1)
    # money_after_each_trade_list = copy.deepcopy(money_after_each_trade_list1)
    # XAB_del_list = copy.deepcopy(XAB_del_list1)
    if xab[2] == 1:
        if exit_price > enter_price:
            profit = leverage * ((exit_price - enter_price) / enter_price) - trade_fee
            money = money + profit * money
            profit_loss_list.append(profit)
            num_of_pos_trades += 1
            if print_outputs == 1:
                print('profit:', profit)
                print('money:', money)
        if exit_price <= enter_price:
            loss = leverage * ((exit_price - enter_price) / enter_price) - trade_fee
            money = money + loss * money
            profit_loss_list.append(loss)
            num_of_neg_trades += 1
            if print_outputs == 1:
                print('loss:', loss)
                print('money:', money)
    if xab[2] == 0:
        if exit_price < enter_price:
            profit = leverage * ((enter_price - exit_price) / enter_price) - trade_fee
            money = money + profit * money
            profit_loss_list.append(profit)
            num_of_pos_trades += 1
            if print_outputs == 1:
                print('profit:', profit)
                print('money:', money)
        if exit_price >= enter_price:
            loss = leverage * ((enter_price - exit_price) / enter_price) - trade_fee
            money = money + loss * money
            profit_loss_list.append(loss)
            num_of_neg_trades += 1
            if print_outputs == 1:
                print('loss:', loss)
                print('money:', money)
    # plot_figure(df, xabc[1][0], xabc[1][1], xabc[1][2], xabc[1][3], index_buy, index_sell,
    #             xabc[0][0], xabc[0][1], xabc[0][2], xabc[0][3], plot_width, plot_height)
    date_of_trade_list.append(df['timestamp'][date_pointer])
    num_of_neg_trades_list.append(num_of_neg_trades)
    num_of_pos_trades_list.append(num_of_pos_trades)
    money_after_each_trade_list.append(money)
    XAB_del_list.append(xab)
    return (money, num_of_pos_trades, num_of_neg_trades)


def trader(args):
    print_outputs = 0
    plot_figures = 0
    symbol = args[3]
    start_date = args[4]
    end_date = args[6]
    data_step = args[5]
    leverage = args[7]
    fibo1 = 2
    fibo2 = 2.618
    wait_candles = 5
    print(args, symbol, data_step)
    # f = open("aaa.txt", "a")
    # f.write(f'\nStart: {args}')
    # f.close()
    Profit_Loss_Table_by_Year_Month_for_symbol = pd.DataFrame()
    # region Data Preparation
    df = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                    step=data_step).prepare_data(macd_slow=args[0], macd_fast=args[1],
                                                 macd_sign=args[2])
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
                         xab_flag, None, None, 0, 0])
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
                         xab_flag, None, None, 0, 0])
                # endregion
    # endregion #
    # region initializing params
    money = 1
    trade_fee = 0.002
    flag1 = 0
    money1 = None
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
    XAB_virtual_list = []
    exit_at_this_candel = 0
    enter_price = 0
    for date_pointer in range(XAB_list[0][1][4], len(df)):
        XAB_valid_list = [x for x in XAB_list if date_pointer >= x[1][4]]
        # xabc = [[X, A, B, C], [index_X, index_A, index_B, index_4, index_C], xab_flag, sl,
        # sudo_sl, dont_find_C, candle_counter]
        for idx_xab, xab in enumerate(XAB_valid_list[::-1]):
            if xab not in XAB_del_list:
                if enter == 0:
                    if xab[5] == 0 and xab[6] == 0:
                        xab, XAB_del_list, C_at_this_candle = xab_completor(df, date_pointer, xab,
                                                                            XAB_del_list)
                    if xab[0][3]:
                        enter, enter_price = xab_enter_check(df, date_pointer, xab, enter,
                                                             enter_price, XAB_del_list,
                                                             wait_candles)
                    if enter == 1:
                        index_buy = date_pointer
                        xab_buy = xab
                        money_before_each_trade_list.append(money)
                        if (xab[2] and df['low'][date_pointer] < xab[3]) or (not xab[2] and df[
                            'high'][date_pointer] > xab[3]):
                            enter = 0
                            xab_buy = None
                            index_sell = date_pointer
                            exit_price = xab[3]
                            date_of_trade_list.append(df['timestamp'][date_pointer])
                            money, num_of_pos_trades, num_of_neg_trades = extra_calculations(
                                df, xab, money, enter_price,
                                exit_price, index_buy, index_sell, leverage,
                                print_outputs, trade_fee, num_of_pos_trades,
                                num_of_neg_trades, profit_loss_list,
                                num_of_neg_trades_list, num_of_pos_trades_list,
                                money_after_each_trade_list, XAB_del_list)
                            if plot_figures:
                                plot_figure(df, xab, index_buy, date_pointer)
                            continue
                    if enter == 0 and xab[0][3] and xab[5]:
                        XAB_del_list = xab_reject_decision(df, date_pointer, xab, XAB_del_list,
                                                           wait_candles)
                    if enter == 1:
                        continue
                if enter == 1:  # If it is in trade
                    if xab != xab_buy:
                        if xab[5] == 0 and xab[6] == 0:
                            xab, XAB_del_list, C_at_this_candle = xab_completor(df, date_pointer,
                                                                                xab, XAB_del_list)
                        if xab[0][3]:
                            non_xab_buy_counter(df, date_pointer, xab, XAB_del_list, wait_candles)
                            if xab in XAB_del_list:
                                continue
                        if xab[0][3] and xab[5]:
                            XAB_del_list = xab_reject_decision(df, date_pointer, xab, XAB_del_list,
                                                               wait_candles)
                    if xab == xab_buy:
                        if date_pointer > len(df) - 1:
                            continue
                        if xab[2] == 1:
                            if df['low'][date_pointer] < xab[3]:
                                if money1 == None:
                                    enter = 0
                                    xab_buy = None
                                    index_sell = date_pointer
                                    exit_price = xab[3]
                                    if print_outputs:
                                        print_trade(df, xab[0][0], xab[0][1], xab[0][2], xab,
                                                    enter_price,
                                                    exit_price,
                                                    xab[1][0], xab[1][1], xab[1][2],
                                                    index_buy, index_sell)
                                    money, num_of_pos_trades, num_of_neg_trades = \
                                        money_calc(df, date_pointer, xab, enter_price, exit_price,
                                                   leverage, trade_fee,
                                                   money,
                                                   profit_loss_list,
                                                   date_of_trade_list,
                                                   num_of_neg_trades_list,
                                                   num_of_pos_trades_list,
                                                   money_after_each_trade_list,
                                                   XAB_del_list,
                                                   num_of_pos_trades,
                                                   num_of_neg_trades,
                                                   print_outputs)
                                    print('===================')
                                else:  # if there is something in money1
                                    enter = 0
                                    xab_buy = None
                                    index_sell = date_pointer
                                    exit_price = xab[3]
                                    if print_outputs:
                                        print_trade(df, xab[0][0], xab[0][1], xab[0][2], xab,
                                                    enter_price,
                                                    exit_price,
                                                    xab[1][0], xab[1][1], xab[1][2],
                                                    index_buy, index_sell)
                                    money_temp = money
                                    pl = (exit_price - enter_price) / enter_price - trade_fee
                                    if print_outputs:
                                        print('pl:', pl)
                                    money = money1 + money / 2 + pl * money / 2
                                    pl_tot = (money - money_temp) / money_temp
                                    if pl_tot < 0:
                                        num_of_neg_trades += 1
                                        if print_outputs:
                                            print('loss:', pl_tot)
                                    if pl_tot > 0:
                                        num_of_pos_trades += 1
                                        if print_outputs:
                                            print('profit:', pl_tot)
                                    if print_outputs:
                                        print('money:', money)
                                    profit_loss_list.append(pl_tot)
                                    date_of_trade_list.append(df['timestamp'][date_pointer])
                                    num_of_neg_trades_list.append(num_of_neg_trades)
                                    num_of_pos_trades_list.append(num_of_pos_trades)
                                    money_after_each_trade_list.append(money)
                                    XAB_del_list.append(xab)
                                    money1 = None
                                    print('===================')

                            elif df['high'][date_pointer] >= (fibo1 * abs(xab[0][2] - xab[0][
                                3]) + xab[0][3]):
                                if not money1:
                                    if print_outputs:
                                        print(f'sell in {fibo1}:')
                                    index_sell = date_pointer
                                    exit_price = (
                                            fibo1 * abs(xab[0][2] - xab[0][3]) + xab[0][3])
                                    if print_outputs:
                                        print_trade(df, xab[0][0], xab[0][1], xab[0][2], xab,
                                                    enter_price,
                                                    exit_price,
                                                    xab[1][0], xab[1][1], xab[1][2],
                                                    index_buy, index_sell)
                                    pl = (exit_price - enter_price) / enter_price - trade_fee
                                    money1 = money / 2 + pl * money / 2
                                    if print_outputs:
                                        print('pl1:', pl)
                                        print('money1:', money1)
                                    xab[3] = xab[0][2]  # risk free: trail the stop_loss to B
                                if df['high'][date_pointer] >= (fibo2 * abs(xab[0][2] - xab[0][
                                    3]) + xab[0][3]):
                                    enter = 0
                                    xab_buy = None
                                    index_sell = date_pointer
                                    exit_price = (fibo2 * abs(xab[0][2] - xab[0][3]) + xab[0][3])
                                    if print_outputs:
                                        print_trade(df, xab[0][0], xab[0][1], xab[0][2], xab,
                                                    enter_price,
                                                    exit_price,
                                                    xab[1][0], xab[1][1], xab[1][2],
                                                    index_buy, index_sell)
                                    money_temp = money
                                    pl = (exit_price - enter_price) / enter_price - trade_fee
                                    if print_outputs:
                                        print('pl:', pl)
                                    money = money1 + money / 2 + pl * money / 2
                                    pl_tot = (money - money_temp) / money_temp
                                    if pl_tot < 0:  # it will never happen
                                        num_of_neg_trades += 1
                                        if print_outputs:
                                            print('loss:', pl_tot)
                                    if pl_tot > 0:
                                        num_of_pos_trades += 1
                                        if print_outputs:
                                            print('profit:', pl_tot)
                                    if print_outputs:
                                        print('money:', money)
                                    profit_loss_list.append(pl_tot)
                                    date_of_trade_list.append(df['timestamp'][index_buy])
                                    num_of_neg_trades_list.append(num_of_neg_trades)
                                    num_of_pos_trades_list.append(num_of_pos_trades)
                                    money_after_each_trade_list.append(money)
                                    XAB_del_list.append(xab)
                                    money1 = None
                                    print('===================')
                        if xab[2] == 0:
                            if df['high'][date_pointer] > xab[3]:
                                if money1 == None:
                                    enter = 0
                                    xab_buy = None
                                    index_sell = date_pointer
                                    exit_price = xab[3]
                                    if print_outputs:
                                        print_trade(df, xab[0][0], xab[0][1], xab[0][2], xab,
                                                    enter_price,
                                                    exit_price,
                                                    xab[1][0], xab[1][1], xab[1][2],
                                                    index_buy, index_sell)
                                    money, num_of_pos_trades, num_of_neg_trades = \
                                        money_calc(df, date_pointer, xab, enter_price, exit_price,
                                                   leverage, trade_fee,
                                                   money,
                                                   profit_loss_list,
                                                   date_of_trade_list,
                                                   num_of_neg_trades_list,
                                                   num_of_pos_trades_list,
                                                   money_after_each_trade_list,
                                                   XAB_del_list,
                                                   num_of_pos_trades,
                                                   num_of_neg_trades,
                                                   print_outputs)
                                    print('===================')

                                else:  # if there is something in money1
                                    enter = 0
                                    xab_buy = None
                                    flag1 = 0
                                    index_sell = date_pointer
                                    exit_price = xab[3]
                                    if print_outputs:
                                        print_trade(df, xab[0][0], xab[0][1], xab[0][2], xab,
                                                    enter_price,
                                                    exit_price,
                                                    xab[1][0], xab[1][1], xab[1][2],
                                                    index_buy, index_sell)
                                    money_temp = money
                                    pl = (-exit_price + enter_price) / enter_price - trade_fee
                                    if print_outputs:
                                        print('pl:', pl)
                                    money = money1 + money / 2 + pl * money / 2
                                    pl_tot = (money - money_temp) / money_temp
                                    if pl_tot < 0:
                                        num_of_neg_trades += 1
                                        if print_outputs:
                                            print('loss:', pl_tot)
                                    if pl_tot > 0:
                                        num_of_pos_trades += 1
                                        if print_outputs:
                                            print('profit:', pl_tot)
                                    if print_outputs:
                                        print('money:', money)
                                    profit_loss_list.append(pl_tot)
                                    date_of_trade_list.append(df['timestamp'][index_buy])
                                    num_of_neg_trades_list.append(num_of_neg_trades)
                                    num_of_pos_trades_list.append(num_of_pos_trades)
                                    money_after_each_trade_list.append(money)
                                    XAB_del_list.append(xab)
                                    money1 = None
                                    print('===================')

                            elif df['low'][date_pointer] <= (
                                    xab[0][3] - fibo1 * abs(xab[0][2] - xab[0][3])):
                                if not money1:
                                    if print_outputs:
                                        print(f'sell in {fibo1}:')
                                    index_sell = date_pointer
                                    exit_price = (
                                            xab[0][3] - fibo1 * abs(xab[0][2] - xab[0][3]))
                                    if print_outputs:
                                        print_trade(df, xab[0][0], xab[0][1], xab[0][2], xab,
                                                    enter_price,
                                                    exit_price,
                                                    xab[1][0], xab[1][1], xab[1][2],
                                                    index_buy, index_sell)
                                    pl = (-exit_price + enter_price) / enter_price - trade_fee
                                    money1 = money / 2 + pl * money / 2
                                    if print_outputs:
                                        print('pl1:', pl)
                                        print('money1:', money1)
                                    xab[3] = xab[0][2]  # risk free: trail the stop_loss to B
                                if df['low'][date_pointer] <= (
                                        xab[0][3] - fibo2 * abs(xab[0][2] - xab[0][3])):
                                    enter = 0
                                    xab_buy = None
                                    index_sell = date_pointer
                                    exit_price = (xab[0][3] - fibo2 * abs(xab[0][2] - xab[0][
                                        3]))
                                    if print_outputs:
                                        print_trade(df, xab[0][0], xab[0][1], xab[0][2], xab,
                                                    enter_price,
                                                    exit_price,
                                                    xab[1][0], xab[1][1], xab[1][2],
                                                    index_buy, index_sell)
                                    money_temp = money
                                    pl = (-exit_price + enter_price) / enter_price - trade_fee
                                    if print_outputs:
                                        print('pl:', pl)
                                    money = money1 + money / 2 + pl * money / 2
                                    pl_tot = (money - money_temp) / money_temp
                                    if pl_tot < 0:  # it will never happen
                                        num_of_neg_trades += 1
                                        if print_outputs:
                                            print('loss:', pl_tot)
                                    if pl_tot > 0:
                                        num_of_pos_trades += 1
                                        if print_outputs:
                                            print('profit:', pl_tot)
                                    if print_outputs:
                                        print('money:', money)
                                    profit_loss_list.append(pl_tot)
                                    date_of_trade_list.append(df['timestamp'][date_pointer])
                                    num_of_neg_trades_list.append(num_of_neg_trades)
                                    num_of_pos_trades_list.append(num_of_pos_trades)
                                    money_after_each_trade_list.append(money)
                                    XAB_del_list.append(xab)
                                    money1 = None
                                    print('===================')
    print(money)

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
        'money_before_trade': money_before_each_trade_list[:len(money_after_each_trade_list)]
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
    print(Profit_Loss_Table_by_Year_Month)
    Profit_Loss_Table_by_Year_Month_for_symbol = \
        pd.concat(
            [Profit_Loss_Table_by_Year_Month_for_symbol, Profit_Loss_Table_by_Year_Month],
            axis=1)
    f = open("aaa.txt", "a")
    f.write(f'\nFinish: {args},{money}')
    f.close()
    return Profit_Loss_Table_by_Year_Month_for_symbol, money


binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240,
            "6h": 360, "12h": 720, "1d": 1440}
batch_size = 750


def main():
    run_mode = 1
    if run_mode == 1:
        """Data"""
        symbol = 'ETHUSDT'
        start_date = '1 Mar 2021'
        end_date = '2021-10-01 00:00:00'
        data_step = '30m'
        leverage = 1
        plot_width = 1500
        plot_height = 1000
        macd_list = [
            [26, 12, 9, symbol, start_date, data_step, end_date, leverage]
        ]
        for macd_value in macd_list:
            trader(macd_value)

    else:
        # NOTE: if you want to give the macd_list manually, please change this part.
        os.chdir('D:/Python projects/EEVA/trader/Genetic Results/Sys1.2.0/Genetic/weighted '
                 'variance')
        csv_files = os.listdir()
        this_sys_related_csv_files = [x for x in csv_files if 'Genetic' in x]  # and 'ETHUSDT' in x]

        for f in this_sys_related_csv_files:
            df_csv = pd.read_csv(f)
            macd_list = [ast.literal_eval(x) for x in df_csv['members'].tolist()]
            file_name_list = f.split('-')
            symbol = [file_name_list[4]][0]
            start_date = '1 Mar 2018'
            end_date = '2021-12-24 14:00:00'
            data_step = [file_name_list[-1].split('.')[0]][0]
            leverage = 1
            trader_required_variables = [symbol, start_date, end_date, data_step, leverage]
            DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                       step=data_step).download_data()
            DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                       step='15m').download_data()
            # for macd_value in macd_list:
            #     trader(*macd_value)
            for i in macd_list:
                i.extend(trader_required_variables)
            with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
                for macd_value, trader_output in zip(macd_list, executor.map(trader, macd_list)):
                    Profit_Loss_Table_by_Year_Month_for_symbol, money = trader_output
                    Profit_Loss_Table_by_Year_Month_for_symbol.to_csv(f'{macd_value[3]}'
                                                                      f'-{macd_value[4]}'
                                                                      f'-{macd_value[-2]}'
                                                                      f'-{macd_value[:3]}'
                                                                      f'-{money}.csv',
                                                                      index=True)
                    print(macd_value, trader_output)


if __name__ == '__main__':
    main()
