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
from eeva_1_4_1 import print_trade, extra_calculations
from utils.utils import create_pl_table


def macd_phase_change(df, date_pointer):
    if df['MACD1_Hist'][date_pointer] * df['MACD1_Hist'][date_pointer - 1] < 0:
        return True
    else:
        return False


def xab_enter_check(df, date_pointer, df2, date_pointer2, xab, enter, enter_price, wait_candles):
    if xab[2] and (df['high'][date_pointer] >= xab[0][2] or xab[6]):
        if xab[6] >= 1:
            xab[6] += 1
        if not xab[7] and xab[6] == 0:
            if df['low'][date_pointer] > xab[0][3]:
                enter = 1
                enter_price = xab[0][2]
            else:
                return enter, enter_price, 1
        if xab[7] and xab[6] == 0:
            if df['close'][date_pointer] <= xab[0][2]:
                enter = 1
                enter_price = df['close'][date_pointer]
            if df['close'][date_pointer] > xab[0][2]:
                xab[6] = 1
        if xab[6] >= 2 and xab[6] <= wait_candles:
            if df['low'][date_pointer] <= xab[0][2]:
                enter = 1
                enter_price = xab[0][2]
        if enter == 0 and xab[6] > wait_candles:
            return enter, enter_price, 1
        if enter:
            xab[3] = xab[0][3]  # C is placed in sl
            xab[4] = xab[0][3]  # C is placed in sl
            for dp2 in [date_pointer2, date_pointer2 + 1]:
                if dp2 >= len(df2):
                    break
                if df2['MACD1_Hist'][dp2] < 0:
                    xab[4] = df2['low'][dp2]  # C is placed in sudo_sl
                    i = 0
                    while df2['MACD1_Hist'][dp2 - i] < 0:
                        if df2['low'][dp2 - i] <= xab[4] and df2['low'][dp2 - i] >= xab[0][3]:
                            xab[4] = df2['low'][dp2 - i]
                        i += 1
    if not xab[2] and df['low'][date_pointer] <= xab[0][2]:
        if xab[6] >= 1:
            xab[6] += 1
        if not xab[7] and xab[6] == 0:
            if df['high'][date_pointer] < xab[0][3]:
                enter = 1
                enter_price = xab[0][2]
            else:
                return enter, enter_price, 1
        if xab[7] and xab[6] == 0:
            if df['close'][date_pointer] >= xab[0][2]:
                enter = 1
                enter_price = df['close'][date_pointer]
            if df['close'][date_pointer] < xab[0][2]:
                xab[6] = 1
        if xab[6] >= 2 and xab[6] <= wait_candles:
            if df['high'][date_pointer] >= xab[0][2]:
                enter = 1
                enter_price = xab[0][2]
        if enter == 0 and xab[6] > wait_candles:
            return enter, enter_price, 1
        if enter:
            xab[3] = xab[0][3]  # C is placed in sl
            xab[4] = xab[0][3]  # C is placed in sl
            for dp2 in [date_pointer2, date_pointer2 + 1]:
                if dp2 >= len(df2):
                    break
                if df2['MACD1_Hist'][dp2] > 0:
                    xab[4] = df2['high'][dp2]  # C is placed in sudo_sl
                    i = 0
                    while df2['MACD1_Hist'][dp2 - i] > 0:
                        if df2['high'][dp2 - i] >= xab[4] and df2['high'][dp2 - i] <= xab[0][3]:
                            xab[4] = df2['high'][dp2 - i]
                        i += 1
    return enter, enter_price, 0


def xab_completor(df, date_pointer, xab):
    xab[7] = 0
    if xab[2] == 1:  # long
        if xab[0][3]:
            if df['MACD1_Hist'][date_pointer] < 0 and xab[5] == 0:  # dont_find_C = xab[0][5]
                if df['low'][date_pointer] <= xab[0][3]:
                    xab[0][3] = df['low'][date_pointer]
                    xab[1][3] = date_pointer
            if df['MACD1_Hist'][date_pointer] > 0:
                xab[5] = 1
        if not xab[0][3] and not xab[5]:
            if df['low'][date_pointer] <= xab[0][1] and df['MACD1_Hist'][date_pointer] < 0 and xab[
                5] == 0:
                xab[7] = 1
                xab[0][3] = df['low'][date_pointer]
                xab[1][3] = date_pointer
            if df['MACD1_Hist'][date_pointer] > 0:
                xab[5] = 1
                return 1

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
                xab[7] = 1
                xab[0][3] = df['high'][date_pointer]
                xab[1][3] = date_pointer
            if df['MACD1_Hist'][date_pointer] < 0:
                xab[5] = 1
                return 1


def xab_reject_decision(df, dp, xab, wait_candles):
    if xab[2] == 1:
        if df['low'][dp] < xab[0][3] or (df['close'][dp] > xab[0][2] and xab[6] > wait_candles):
            return 1
    if xab[2] == 0:
        if df['high'][dp] > xab[0][3] or (df['close'][dp] < xab[0][2] and xab[6] > wait_candles):
            return 1


def stop_loss_trail(df, date_pointer, xab):
    if xab[2] == 1:
        if df['MACD1_Hist'][date_pointer] < 0:
            if macd_phase_change(df, date_pointer) or df['low'][date_pointer] <= xab[4]:
                xab[4] = df['low'][date_pointer]
        if df['MACD1_Hist'][date_pointer] > 0: xab[3] = xab[4]
    if xab[2] == 0:
        if df['MACD1_Hist'][date_pointer] > 0:
            if macd_phase_change(df, date_pointer) or df['high'][date_pointer] >= xab[4]:
                xab[4] = df['high'][date_pointer]
        if df['MACD1_Hist'][date_pointer] < 0: xab[3] = xab[4]


def equal_date_pointer(df1, df2, dp1, dp2):
    dp2_str = df1['timestamp'][dp1]
    try:
        dp2 = df2[df2['timestamp'] == dp2_str].index.values[0] + 2
    except IndexError:
        print(f"there occurs an error in {df1['timestamp'][dp1]}")  # This is for debug of
        dp2 = dp2 + 2
    return dp2


def trader(args):
    print_outputs = 1
    plot_figures = 0
    symbol = args[3]
    start_date = args[4]
    data_step = args[5]
    end_date = args[6]
    leverage = args[7]
    wait_candles = 5
    print(args, symbol, data_step)
    Profit_Loss_Table_by_Year_Month_for_symbol = pd.DataFrame()
    # region Data Preparation
    df = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                    step=data_step).prepare_data(macd_slow=args[0], macd_fast=args[1],
                                                 macd_sign=args[2])
    df2 = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                     step='15m').prepare_data(macd_slow=args[0], macd_fast=args[1],
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
                         xab_flag, None, None, 0, 0, 0])
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
                         xab_flag, None, None, 0, 0, 0])
                # endregion
    # endregion #
    # region initializing params
    money = 1
    trade_fee = 0.002
    enter = 0
    date_pointer = 0
    date_of_trade_list = []
    profit_loss_list = []
    # endregion
    if not XAB_list:
        print('XAB list is empty')
        return None
    date_pointer2 = 0
    enter_price = 0
    for xab in XAB_list:
        enter = 0
        break_flag = 0
        enter_price = None
        exit_price = None
        for date_pointer in range(xab[1][4], len(df)):
            if break_flag == 1:
                break
            date_pointer22 = equal_date_pointer(df, df2, date_pointer, date_pointer2)
            if enter == 0:
                if xab[5] == 0 and xab[6] == 0:
                    break_flag = xab_completor(df, date_pointer, xab)
                    if break_flag == 1:
                        break
                if xab[0][3]:
                    enter, enter_price, break_flag = xab_enter_check(df, date_pointer,
                                                                     df2, date_pointer22 - 2,
                                                                     xab, enter,
                                                                     enter_price,
                                                                     wait_candles)
                    if break_flag == 1:
                        break
                if enter == 1:
                    index_buy = date_pointer
                    if (xab[2] and df['low'][date_pointer] < xab[3]) or (not xab[2] and df[
                        'high'][date_pointer] > xab[3]):
                        enter = 0
                        index_sell = date_pointer
                        exit_price = xab[3]
                        date_of_trade_list.append(df['timestamp'][date_pointer])
                        money = extra_calculations(df, df2, xab, money, enter_price,
                                           exit_price, index_buy, index_sell, leverage,
                                           print_outputs, trade_fee, profit_loss_list, 0)
                        if plot_figures:
                            plot_figure(df, xab, index_buy, date_pointer)
                if enter == 0 and xab[0][3] and (xab[5] or xab[6]):
                    break_flag = xab_reject_decision(df, date_pointer, xab, wait_candles)
                    if break_flag == 1:
                        break
            if enter == 1:  # If it is in trade
                for date_pointer2 in [date_pointer22, date_pointer22 + 1]:
                    sl_at_this_candle = 0
                    if date_pointer > len(df) - 1 or date_pointer2 > len(df2) - 1:
                        break_flag = 1
                        break
                    if xab[2] == 1:
                        if macd_phase_change(df2, date_pointer2) and df2['MACD1_Hist'][
                            date_pointer2] > 0:
                            xab[3] = xab[4]
                            if xab[4] != xab[0][3]:
                                sl_at_this_candle = 1
                        if df2['low'][date_pointer2] < xab[3]:
                            enter = 0
                            index_sell = date_pointer2
                            if sl_at_this_candle == 0:
                                exit_price = xab[3]
                            else:
                                exit_price = df2['close'][date_pointer2]
                            sl_at_this_candle = 0  # very important
                            date_of_trade_list.append(df2['timestamp'][
                                                          date_pointer2])
                            money = extra_calculations(df, df2, xab, money, enter_price,
                                                       exit_price, index_buy, index_sell, leverage,
                                                       print_outputs, trade_fee,
                                                       profit_loss_list, 1)
                            if plot_figures:
                                plot_figure(df, xab, index_buy, date_pointer + 1)
                            break_flag = 1
                            break
                        else:
                            stop_loss_trail(df2, date_pointer2, xab)

                    if xab[2] == 0:
                        if macd_phase_change(df2, date_pointer2) and df2['MACD1_Hist'][
                            date_pointer2] < 0:
                            xab[3] = xab[4]
                            if xab[4] != xab[0][3]:
                                sl_at_this_candle = 1
                        if df2['high'][date_pointer2] > xab[3]:
                            enter = 0
                            index_sell = date_pointer2
                            if sl_at_this_candle == 0:
                                exit_price = xab[3]
                            else:
                                exit_price = df2['close'][date_pointer2]
                            sl_at_this_candle = 0  # very important
                            date_of_trade_list.append(df2['timestamp'][
                                                          date_pointer2])
                            money = extra_calculations(df, df2, xab, money, enter_price,
                                                       exit_price, index_buy, index_sell, leverage,
                                                       print_outputs, trade_fee,
                                                       profit_loss_list, 1)
                            if plot_figures:
                                plot_figure(df, xab, index_buy, date_pointer + 1)
                            break_flag = 1
                            break
                        else:
                            stop_loss_trail(df2, date_pointer2, xab)
    print(money)
    return create_pl_table(date_of_trade_list, profit_loss_list, data_step), money


binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240,
            "6h": 360, "12h": 720, "1d": 1440}
batch_size = 750


def main():
    run_mode = 1
    if run_mode == 1:
        """Data"""
        symbol = 'ETHUSDT'
        start_date = '1 Dec 2021'
        end_date = '2022-02-30 00:00:00'
        data_step = '30m'
        leverage = 1
        plot_width = 1500
        plot_height = 1000
        macd_list = [
            [5, 6, 4, symbol, start_date, data_step, end_date, leverage]
        ]
        for macd_value in macd_list:
            trader(macd_value)

    else:
        # NOTE: if you want to give the macd_list manually, please change this part.
        os.chdir('D:/Python projects/EEVA/trader/Genetic Results/Sys1.5.0/Genetic/weighted '
                 'money')
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
