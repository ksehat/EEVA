# This system uses trailing stop loss with lower time step
import pandas as pd
import numpy as np
import ast
import os.path
import concurrent.futures
from my_geneticalgorithm_multiprocess import MyGeneticAlgorithm as mga
from data_prep.data_hunter import DataHunter


def macd_phase_change(df, date_pointer):
    if df['MACD1_Hist'][date_pointer] * df['MACD1_Hist'][date_pointer - 1] < 0:
        return True
    else:
        return False


def print_trade(df, df2, X, A, B, xab, enter_price, exit_price, index_X, index_A, index_B,
                index_buy,
                index_sell):
    print(df['timestamp'][index_X], 'X:', X)
    print(df['timestamp'][index_A], 'A:', A)
    print(df['timestamp'][index_B], 'B:', B)
    print(df['timestamp'][xab[1][3]], 'C:', xab[0][3])
    print(df['timestamp'][index_buy], 'enter:', enter_price)
    print(df['timestamp'][index_sell], 'exit:', exit_price)


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


def xab_enter_check(df, date_pointer, xab, enter, enter_price, fibo_enter, C_at_this_candle,
                    XAB_virtual_list):
    if xab[2] and df['high'][date_pointer] >= xab[0][2]:
        if not C_at_this_candle:
            enter = 1
            enter_price = xab[0][2]
        if C_at_this_candle:
            if df['close'][date_pointer] <= xab[0][2]:
                enter = 1
                enter_price = df['close'][date_pointer]
            if df['close'][date_pointer] > xab[0][2]:
                enter = 1
                enter_price = xab[0][2]
                XAB_virtual_list.append(xab)
    if not xab[2] and df['low'][date_pointer] <= xab[0][2]:
        if not C_at_this_candle:
            enter = 1
            enter_price = xab[0][2]
        if C_at_this_candle:
            if df['close'][date_pointer] >= xab[0][2]:
                enter = 1
                enter_price = df['close'][date_pointer]
            if df['close'][date_pointer] < xab[0][2]:
                enter = 1
                enter_price = xab[0][2]
                XAB_virtual_list.append(xab)
    return enter, enter_price, XAB_virtual_list


def xab_completor(df, date_pointer, xab, XAB_del_list):
    C_at_this_candle = 0
    if xab[2] == 1:  # long
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
            if df['low'][date_pointer] <= xab[0][1] and df['MACD1_Hist'][date_pointer] < 0 and xab[
                5] == 0:
                C_at_this_candle = 1
                xab[0][3] = df['low'][date_pointer]
                xab[1][3] = date_pointer
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
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
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
        if not xab[0][3] and not xab[5]:
            if df['high'][date_pointer] >= xab[0][1] and df['MACD1_Hist'][date_pointer] > 0 and xab[
                5] == 0:
                C_at_this_candle = 1
                xab[0][3] = df['high'][date_pointer]
                xab[1][3] = date_pointer
                xab[3] = xab[0][3]
                xab[4] = xab[0][3]
            if df['MACD1_Hist'][date_pointer] < 0:
                xab[5] = 1
                XAB_del_list.append(xab)

    return xab, XAB_del_list, C_at_this_candle


def xab_reject_decision(df, dp, xab, XAB_del_list, XAB_check_list):
    if xab[2] == 1:
        if df['low'][dp] < xab[0][3] or df['close'][dp] > xab[0][2]:
            XAB_del_list.append(xab)
    if xab[2] == 0:
        if df['high'][dp] > xab[0][3] or df['close'][dp] < xab[0][2]:
            XAB_del_list.append(xab)
    return XAB_del_list, XAB_check_list


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


def base_coefficient_calc(tot_n_month, static_weight):
    A = 0
    for i in range(tot_n_month):
        A = A + pow(static_weight, i)
    return 1 / A


def trader(args, symbol, data_step):
    print_outputs = 0
    # symbol = args[3]
    # start_date = args[4]
    # end_date = args[5]
    # data_step = args[6]
    # leverage = args[7]
    fibo1 = args[3]
    # print(args, symbol, data_step)
    # f = open("aaa.txt", "a")
    # f.write(f'\nStart: {args}')
    # f.close()
    Profit_Loss_Table_by_Year_Month_for_symbol = pd.DataFrame()
    # region Data Preparation
    df = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                    step=data_step).prepare_data(macd_slow=args[0], macd_fast=args[1],
                                                 macd_sign=args[2])
    # df2 = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
    #                  step='15m').prepare_data(macd_slow=args[0], macd_fast=args[1],
    #                                           macd_sign=args[2])
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
    XAB_virtual_list = []
    exit_at_this_candel = 0
    fibo_enter = 1.618
    enter_price = 0
    for date_pointer in range(XAB_list[0][1][4], len(df)):
        exit_at_this_candel = 0
        # This is the list of XABs before the date_pointer
        XAB_valid_list = [x for x in XAB_list if date_pointer >= x[1][4]]
        # xabc = [[X, A, B, C], [index_X, index_A, index_B, index_4, index_C], xab_flag, sl, sudo_sl, dont_find_C]
        for idx_xab, xab in enumerate(XAB_valid_list[::-1]):
            if xab not in XAB_del_list:
                if enter == 0:
                    if xab[5] == 0:
                        xab, XAB_del_list, C_at_this_candle = xab_completor(df, date_pointer, xab,
                                                                            XAB_del_list)
                    if xab[0][3]:  # and not exit_at_this_candel:
                        enter, enter_price, XAB_virtual_list = xab_enter_check(df, date_pointer,
                                                                               xab, enter,
                                                                               enter_price,
                                                                               fibo_enter,
                                                                               C_at_this_candle,
                                                                               XAB_virtual_list)
                    if enter == 1:
                        index_buy = date_pointer
                        xab_buy = xab
                        money_before_each_trade_list.append(money)
                    if enter == 0 and xab[0][3] and xab[5]:
                        XAB_del_list, XAB_check_list = xab_reject_decision(df, date_pointer,
                                                                           xab,
                                                                           XAB_del_list,
                                                                           XAB_check_list)
                    if enter == 1:
                        continue
                if enter == 1:  # If it is in trade
                    if xab != xab_buy:
                        if xab[5] == 0:
                            xab, XAB_del_list, C_at_this_candle = xab_completor(df, date_pointer,
                                                                                xab, XAB_del_list)
                        if xab[0][3] and xab[5]:
                            XAB_del_list, XAB_check_list = xab_reject_decision(df,
                                                                               date_pointer,
                                                                               xab,
                                                                               XAB_del_list,
                                                                               XAB_check_list)
                    if xab == xab_buy:
                        if date_pointer > len(df) - 1:
                            continue
                        if xab[2] == 1:
                            if df['low'][date_pointer] <= xab[0][3] or (df['high'][date_pointer] >= \
                                                                        xab[0][3] + fibo1 * (abs(
                                        xab[0][2] - xab[0][3]))):
                                enter = 0
                                xab_buy = None
                                index_sell = date_pointer
                                exit_price = xab[0][3] if df['low'][date_pointer] <= xab[0][3] \
                                    else (xab[0][3] + fibo1 * (abs(xab[0][2] - xab[0][3])))
                                if print_outputs:
                                    print_trade(df, df2, xab[0][0], xab[0][1], xab[0][2], xab,
                                                enter_price,
                                                exit_price,
                                                xab[1][0], xab[1][1], xab[1][2],
                                                index_buy, index_sell)
                                if exit_price > enter_price:
                                    profit = leverage * ((
                                                                     exit_price - enter_price) / enter_price) - trade_fee
                                    money = money + profit * money
                                    profit_loss_list.append(profit)
                                    num_of_pos_trades += 1
                                    if print_outputs:
                                        print('profit:', profit)
                                        print('money:', money)
                                if exit_price <= enter_price:
                                    loss = leverage * ((
                                                                   exit_price - enter_price) / enter_price) - trade_fee
                                    money = money + loss * money
                                    profit_loss_list.append(loss)
                                    num_of_neg_trades += 1
                                    if print_outputs:
                                        print('loss:', loss)
                                        print('money:', money)
                                date_of_trade_list.append(df['timestamp'][date_pointer])
                                num_of_neg_trades_list.append(num_of_neg_trades)
                                num_of_pos_trades_list.append(num_of_pos_trades)
                                money_after_each_trade_list.append(money)
                                XAB_del_list.append(xab)

                        if xab[2] == 0:
                            if df['high'][date_pointer] >= xab[0][3] or df['low'][date_pointer] <= \
                                    xab[0][3] - fibo1 * abs(xab[0][2] - xab[0][3]):
                                enter = 0
                                xab_buy = None
                                index_sell = date_pointer
                                exit_price = xab[0][3] if df['high'][date_pointer] >= xab[0][3] \
                                    else (xab[0][3] - fibo1 * abs(xab[0][2] - xab[0][3]))
                                if print_outputs:
                                    print_trade(df, df2, xab[0][0], xab[0][1], xab[0][2], xab,
                                                enter_price, \
                                                exit_price,
                                                xab[1][0], xab[1][1], xab[1][2],
                                                index_buy, index_sell)
                                if exit_price < enter_price:
                                    profit = leverage * ((
                                                                     enter_price - exit_price) / enter_price) - trade_fee
                                    money = money + profit * money
                                    profit_loss_list.append(profit)
                                    num_of_pos_trades += 1
                                    if print_outputs:
                                        print('profit:', profit)
                                        print('money:', money)
                                if exit_price >= enter_price:
                                    loss = leverage * ((
                                                                   enter_price - exit_price) / enter_price) - trade_fee
                                    money = money + loss * money
                                    profit_loss_list.append(loss)
                                    num_of_neg_trades += 1
                                    if print_outputs:
                                        print('loss:', loss)
                                        print('money:', money)
                                date_of_trade_list.append(df['timestamp'][date_pointer])
                                num_of_neg_trades_list.append(num_of_neg_trades)
                                num_of_pos_trades_list.append(num_of_pos_trades)
                                money_after_each_trade_list.append(money)
                                XAB_del_list.append(xab)
    if print_outputs:
        print(XAB_virtual_list)
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
    print(Profit_Loss_Table_by_Year_Month)
    Profit_Loss_Table_by_Year_Month_for_symbol = \
        pd.concat(
            [Profit_Loss_Table_by_Year_Month_for_symbol, Profit_Loss_Table_by_Year_Month],
            axis=1)
    Profit_Loss_Table_by_Year_Month_for_symbol[f'weighted profit & loss_{data_step}'] = np.nan
    Profit_Loss_Table_by_Year_Month_for_symbol[f'weighted positive trades_{data_step}'] = np.nan
    Profit_Loss_Table_by_Year_Month_for_symbol[f'weighted negative trades_{data_step}'] = np.nan
    static_weight = 1.1
    tot_n_month = len(Profit_Loss_Table_by_Year_Month_for_symbol)
    base_coefficient = base_coefficient_calc(tot_n_month, static_weight)
    for i in range(len(Profit_Loss_Table_by_Year_Month_for_symbol)):
        Profit_Loss_Table_by_Year_Month_for_symbol[f'weighted profit & loss_{data_step}'][i] = \
            pow(static_weight, i) * base_coefficient * Profit_Loss_Table_by_Year_Month_for_symbol[
                f'profit & loss_{data_step}'][
                i]

        Profit_Loss_Table_by_Year_Month_for_symbol[f'weighted positive trades_{data_step}'][i] = \
            pow(static_weight, i) * base_coefficient * Profit_Loss_Table_by_Year_Month_for_symbol[
                f'positive trades_{data_step}'][
                i]

        Profit_Loss_Table_by_Year_Month_for_symbol[f'weighted negative trades_{data_step}'][i] = \
            pow(static_weight, i) * base_coefficient * Profit_Loss_Table_by_Year_Month_for_symbol[
                f'negative trades_{data_step}'][
                i]
    # endregion

    num_of_months_with_loss = Profit_Loss_Table_by_Year_Month_for_symbol[
        f'weighted negative trades_{data_step}'].sum()

    num_of_months_with_profit = Profit_Loss_Table_by_Year_Month_for_symbol[
        f'weighted positive trades_{data_step}'].sum()

    money_negative = Profit_Loss_Table_by_Year_Month_for_symbol[
        f'weighted profit & loss_{data_step}'][
        Profit_Loss_Table_by_Year_Month_for_symbol[
            f'weighted profit & loss_{data_step}'] < 0].sum()

    money_positive = Profit_Loss_Table_by_Year_Month_for_symbol[
        f'weighted profit & loss_{data_step}'][
        Profit_Loss_Table_by_Year_Month_for_symbol[
            f'weighted profit & loss_{data_step}'] > 0].sum()

    if num_of_months_with_loss == 0 or money_negative == 0:
        decision_factor = money_positive * num_of_months_with_profit * 1000
    else:
        decision_factor = (money_positive / abs(money_negative)) * (
                num_of_months_with_profit / num_of_months_with_loss)
    print(decision_factor)
    print('==========')
    # f = open("aaa.txt", "a")
    # f.write(f'\nFinish {args}'
    #         f' {data_step}')
    # f.close()
    return decision_factor

def main():
    for symbol, data_step in coins_datastep_list:
        GA = mga(config=config, function=trader, symbol=symbol, data_step=data_step, run_iter=5,
                 population_size=350,
                 n_crossover=4,
                 crossover_mode='random')
        best_params = GA.run(keep_frac=20, mutate_frac=10, crossover_frac=10)
        print(best_params)
        best_params.to_csv(f'Genetic-weighted-variance-v0.3.0-{symbol}-{start_date}'
                           f'-{data_step}.csv',
                           index=True)

config = {
    'slow_window': [3, 5, 6, 7, 12, 13, 26, 30, 40, 52],
    'fast_window': [4, 5, 6, 7, 12, 24, 30, 40, 48],
    'sign_window': [4, 6, 8, 9, 10, 12, 14, 16, 18, 20],
    'fibo1': [2, 2.618, 3, 3.618, 4]
}

coins_datastep_list = [
    # ('LTCUSDT','1h'),
    # ('BTCUSDT','1h'),
    # ('IOTAUSDT','1h'),
    # ('ETHUSDT','1h'),
    # ('TRXUSDT', '1h'),
    # ('NEOUSDT', '1h'),
    # ('ETHUSDT', '30m'),
    ('NEOUSDT', '30m'),
    # ('IOTAUSDT', '30m'),
    # ('TRXUSDT', '30m'),
    # ('LTCUSDT', '30m'),
    # ('BTCUSDT', '30m'),

    # ('LTCUSDT', '15m'),
    # ('BTCUSDT', '15m'),
    # ('IOTAUSDT', '15m'),
    # ('ETHUSDT', '15m'),
    # ('TRXUSDT', '15m'),
    # ('NEOUSDT', '15m'),
]
# region Data
start_date = '1 Apr 2020'
end_date = '2021-04-30 00:00'
leverage = 1
plot_width = 1500
plot_height = 1000

# for symbol, data_step in coins_datastep_list:
#     dh = DataHunter(symbol, start_date, end_date, data_step)
#     dh.download_data()

if __name__ == '__main__':
    main()
