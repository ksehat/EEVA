import itertools
import os
import glob
import shutil
import pandas as pd
import numpy as np


def op_results(mode, csv_files, num_of_macds, measurement_date):
    static_weight = 1.1
    money_list = []
    macd_list = []
    weighted_money_list = []
    variance_list = []
    weighted_variance_list = []
    mean_value_from_date_list = []
    mean_value_all_list = []
    var_value_from_date_list = []
    var_value_all_list = []
    min_value_from_date_list = []
    min_value_all_list = []
    max_value_from_date_list = []
    max_value_all_list = []
    win_rate_mean_list = []
    sum_of_all_pl_list = []
    num_all_trades_list = []
    for file in csv_files:
        if 'e-' not in file:
            info = file.split('-')
        else:
            continue
        if mode == 'best combo':
            macd_list.append(info[:num_of_macds])
            data_step = info[-1].split('.csv')[0]
            money_list.append(None)
        if mode == 'optimization':
            macd_list.append(info[:num_of_macds][0])
            data_step = info[-2]
            money_list.append(float(info[-1].split('.csv')[0]))
        Profit_Loss_Table_by_Year_Month_for_symbol = pd.read_csv(file)
        tot_n_month = len(Profit_Loss_Table_by_Year_Month_for_symbol)
        base_coefficient = base_coefficient_calc(tot_n_month, static_weight)
        for i in range(len(Profit_Loss_Table_by_Year_Month_for_symbol)):
            Profit_Loss_Table_by_Year_Month_for_symbol.loc[i, f'weighted profit & loss_{data_step}'] \
                = pow(static_weight, i) * base_coefficient * \
                  Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'][i]

            Profit_Loss_Table_by_Year_Month_for_symbol.loc[
                i, f'weighted positive trades_{data_step}'] \
                = pow(static_weight, i) * base_coefficient * \
                  Profit_Loss_Table_by_Year_Month_for_symbol[f'positive trades_{data_step}'][i]

            Profit_Loss_Table_by_Year_Month_for_symbol.loc[
                i, f'weighted negative trades_{data_step}'] \
                = pow(static_weight, i) * base_coefficient * \
                  Profit_Loss_Table_by_Year_Month_for_symbol[f'negative trades_{data_step}'][i]

        Profit_Loss_Table_by_Year_Month_for_symbol['win rate month'] = \
            Profit_Loss_Table_by_Year_Month_for_symbol[f'positive trades_{data_step}'] / \
            Profit_Loss_Table_by_Year_Month_for_symbol[f'all trades_{data_step}']

        weighted_num_of_months_with_loss = Profit_Loss_Table_by_Year_Month_for_symbol[
            f'weighted negative trades_{data_step}'].sum()

        weighted_num_of_months_with_profit = Profit_Loss_Table_by_Year_Month_for_symbol[
            f'weighted positive trades_{data_step}'].sum()

        weighted_money_negative = Profit_Loss_Table_by_Year_Month_for_symbol[
            f'weighted profit & loss_{data_step}'][
            Profit_Loss_Table_by_Year_Month_for_symbol[
                f'weighted profit & loss_{data_step}'] < 0].sum()

        weighted_money_positive = Profit_Loss_Table_by_Year_Month_for_symbol[
            f'weighted profit & loss_{data_step}'][
            Profit_Loss_Table_by_Year_Month_for_symbol[
                f'weighted profit & loss_{data_step}'] > 0].sum()

        weighted_money = Profit_Loss_Table_by_Year_Month_for_symbol[f'weighted profit & loss_' \
                                                                    f'{data_step}'].sum()

        num_of_months_with_loss = Profit_Loss_Table_by_Year_Month_for_symbol[
            Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'] < 0].shape[0]

        num_of_months_with_profit = Profit_Loss_Table_by_Year_Month_for_symbol[
            Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'] > 0].shape[0]

        money_negative = Profit_Loss_Table_by_Year_Month_for_symbol[
            f'profit & loss_{data_step}'][Profit_Loss_Table_by_Year_Month_for_symbol[
                                              f'profit & loss_{data_step}'] < 0].sum()

        money_positive = Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'][
            Profit_Loss_Table_by_Year_Month_for_symbol[
                f'profit & loss_{data_step}'] > 0].sum()

        # sum_of_all_pl = Profit_Loss_Table_by_Year_Month_for_symbol[
        #     f'sum of all pl_{data_step}'].sum()
        num_all_trades = Profit_Loss_Table_by_Year_Month_for_symbol[f'all trades_{data_step}'].sum()
        try:
            mean_value_from_date_list.append(
                Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'][
                Profit_Loss_Table_by_Year_Month_for_symbol[
                    Profit_Loss_Table_by_Year_Month_for_symbol[
                        f'year_month_{data_step}'] == measurement_date].index[0]:].mean())
        except:
            mean_value_from_date_list.append(None)

        mean_value_all_list.append(
            Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'].mean())

        try:
            var_value_from_date_list.append(
                Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'][
                Profit_Loss_Table_by_Year_Month_for_symbol[
                    Profit_Loss_Table_by_Year_Month_for_symbol[
                        f'year_month_{data_step}'] == measurement_date].index[0]:].var())
        except:
            var_value_from_date_list.append(None)

        var_value_all_list.append(
            Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'].var())

        try:
            min_value_from_date_list.append(
                Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'][
                Profit_Loss_Table_by_Year_Month_for_symbol[
                    Profit_Loss_Table_by_Year_Month_for_symbol[
                        f'year_month_{data_step}'] == measurement_date].index[0]:].min())
        except:
            min_value_from_date_list.append(None)

        min_value_all_list.append(
            Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'].min())

        try:
            max_value_from_date_list.append(
                Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'][
                Profit_Loss_Table_by_Year_Month_for_symbol[
                    Profit_Loss_Table_by_Year_Month_for_symbol[
                        f'year_month_{data_step}'] == measurement_date].index[0]:].max())
        except:
            max_value_from_date_list.append(None)

        win_rate_mean_list.append(Profit_Loss_Table_by_Year_Month_for_symbol['win rate '
                                                                             'month'].mean() * 100)

        max_value_all_list.append(
            Profit_Loss_Table_by_Year_Month_for_symbol[f'profit & loss_{data_step}'].max())

        if num_of_months_with_loss == 0 or money_negative == 0:
            variance = money_positive * num_of_months_with_profit * 1000
        else:
            variance = (money_positive / abs(money_negative)) * (
                    num_of_months_with_profit / num_of_months_with_loss)

        if weighted_num_of_months_with_loss == 0 or weighted_money_negative == 0:
            weighted_variance = weighted_money_positive * weighted_num_of_months_with_profit * 1000
        else:
            weighted_variance = (weighted_money_positive / abs(weighted_money_negative)) * (
                    weighted_num_of_months_with_profit / weighted_num_of_months_with_loss)

        variance_list.append(variance)
        weighted_variance_list.append(weighted_variance)
        weighted_money_list.append(weighted_money)
        # sum_of_all_pl_list.append(sum_of_all_pl)
        num_all_trades_list.append(num_all_trades)
    try:
        df1 = pd.DataFrame({
            'macd': macd_list,
            'weighted_money': weighted_money_list,
            'weighted_variance': weighted_variance_list,
            'variance': variance_list,
            'money': money_list,
            # 'sum of pl': sum_of_all_pl_list,
            'all trades': num_all_trades_list,
            f'mean_value_from_{measurement_date}': mean_value_from_date_list,
            'mean_value_all': mean_value_all_list,
            f'var_value_from_{measurement_date}': var_value_from_date_list,
            'var_value_all': var_value_all_list,
            f'min_value_from_{measurement_date}': min_value_from_date_list,
            'min_value_all': min_value_all_list,
            f'max_value_from_{measurement_date}': max_value_from_date_list,
            'max_value_all': max_value_all_list,
            'win rate avg': win_rate_mean_list
        })
        # df1['sum pl / all trades'] = df1['sum of pl'] / df1['all trades']
    except:
        print('DataFrame could not be completed!!!')
    df1.sort_values(by='min_value_all', inplace=True, ignore_index=True, ascending=False)
    return df1


def best_combination_macd_finder(csv_files, num_of_macds, save_directory,
                                 selected_macd_files_dir, f, system):
    macd_combinations = itertools.combinations(csv_files, num_of_macds)
    for combination in macd_combinations:
        os.chdir(selected_macd_files_dir)
        df_sum = pd.read_csv(combination[-1], index_col=1)
        df_sum.drop(df_sum.columns[0], inplace=True, axis=1)
        macd_list = []
        for file in combination:
            info = file.split('-')
            macd_list.append(info[0])
        joined_macd_list = '-'.join(macd_list)
        joined_info = '-'.join(info[1:-1])
        combination_file_name = '-'.join([joined_macd_list, joined_info]) + '.csv'
        for i in range(len(combination) - 1):
            df1 = pd.read_csv(combination[i], index_col=1)
            df1.drop(df1.columns[0], inplace=True, axis=1)
            df_sum = df_sum.add(df1, fill_value=0)
        if not os.path.exists(save_directory + f'/MACD combinations/{system}' + f'/{f}/'):
            os.makedirs(save_directory + f'/MACD combinations/{system}' + f'/{f}/')
        os.chdir(save_directory + f'/MACD combinations/{system}' + f'/{f}/')
        df_sum.to_csv(save_directory + f'MACD combinations/{system}' + f'/{f}/' +
                      combination_file_name)
    combo_csv_files = [x for x in
                       os.listdir(save_directory + f'MACD combinations/{system}' + f'/{f}')]
    os.chdir(save_directory + f'/MACD combinations/{system}' + f'/{f}')
    df = op_results(mode, combo_csv_files, num_of_macds, '(2021, 1)')
    if not os.path.exists(save_directory + f'Best MACD combination selection/{system}' + f'/{f}'):
        os.makedirs(save_directory + f'Best MACD combination selection/{system}' + f'/{f}')
    os.chdir(save_directory + f'Best MACD combination selection/{system}' + f'/{f}')
    df.to_csv(f'{f}.csv')


def macd_files_selector(system, num_tops, sort_by, save_directory, directory_data, f):
    # os.chdir(save_directory + f'/{f}/')
    csv_file = os.listdir(save_directory + f'{system}' + f'/{f}/')
    csv_files = [x for x in os.listdir(directory_data + f'{f}/') if f.split(' ')[0] in x and
                 'data' not in x]
    df = pd.read_csv(save_directory + f'{system}' + f'/{f}/' + csv_file[0])
    df.sort_values(by=sort_by, inplace=True, ignore_index=True, ascending=False)
    best_macds = list(df.iloc[:num_tops, 1])
    selected_csv_file_names = []
    for x in best_macds:
        for y in csv_files:
            if x == y.split('-')[0]:
                selected_csv_file_names.append(y)
                break
    save_dir_best_macd_run_files = f'D:/Python projects/EEVA/trader/Optimization ' \
                                   f'results/best macd runs/{system}'
    if not os.path.exists(save_dir_best_macd_run_files + f'/{f}/'):
        os.makedirs(save_dir_best_macd_run_files + f'/{f}/')
    files_in_dest_folder = os.listdir(save_dir_best_macd_run_files + f'/{f}/')
    for i in files_in_dest_folder:
        os.remove(save_dir_best_macd_run_files + f'/{f}/' + i)
    for x in selected_csv_file_names:
        shutil.copy(directory_data + f'/{f}/' + x, save_dir_best_macd_run_files + f'/{f}/')
    return save_dir_best_macd_run_files + f'/{f}/'


def base_coefficient_calc(tot_n_month, static_weight):
    A = 0
    for i in range(tot_n_month):
        A = A + pow(static_weight, i)
    return 1 / A


systems = [
    # 'eeva_0_1_3',
    # 'eeva_0_3_3',
    # 'eeva_0_2_4',
    # 'eeva_1_4_0',
    # 'eeva_1_4_1',
    # 'eeva_1_4_3',
    # 'eeva_1_4_4',
    # 'eeva_1_4_6',
    # 'eeva_1_4_5',
    # 'eeva_1_5_0',
    # 'eeva_1_5_2',
    # 'eeva_1_5_4',
    # 'eeva_1_8_0',
    # 'eeva_0_2_1',
    # 'eeva_0_3_1',
    # 'eeva_0_3_3',
    # 'eeva_0_4_0',
    # 'eeva_0_4_3',
    # 'eeva_0_4_5',
    # 'eeva_0_4_6',
    # 'eeva_1_3_0',
    # 'eeva_0_1_2',
    # 'eeva_0_2_2',
    # 'eeva_1_2_2',
    # 'ichi_0_0_0_test',
    # 'eeva_0_3_2'
    # 'ichi_0_0_8_trail'
    # 'ichi_0_0_2',
    # 'ichi_0_0_2_with_laggingspan'
    # 'eeva_0_3_4_rr_closepullback_futures'
    '5ema'
]
mode = 'optimization'
for system in systems:
    directory_data = f'D:/Python projects/EEVA/trader/Optimization results/All combinations ' \
                     f'run/{system}/'
    save_directory = f'D:/Python projects/EEVA/trader/Optimization results/Cost fun ' \
                     f'results/'
    each_coin_folders = os.listdir(directory_data)
    for f in each_coin_folders:
        if mode == 'best combo':
            selected_macd_files_dir = macd_files_selector(system, 10, 'min_value_all',
                                                          save_directory,
                                                          directory_data, f)
        coin_and_datastep = f.split(' ')
        if mode == 'best combo':
            csv_files = [x for x in os.listdir(selected_macd_files_dir)]
            best_combination_macd_finder(csv_files, 4, 'D:/Python '
                                                       'projects/EEVA/trader/Optimization results/',
                                         selected_macd_files_dir, f, system)

        if mode == 'optimization':
            csv_files = [x for x in os.listdir(directory_data + f'/{f}') if
                         coin_and_datastep[0] in x and 'from' not in x]
            os.chdir(directory_data + f'{f}')
            df = op_results(mode, csv_files, 1, '(2022, 1)')
            if not os.path.exists(save_directory + system + f'/{f}'):
                os.makedirs(save_directory + system + f'/{f}')
            os.chdir(save_directory + system + f'/{f}')
            df.to_csv(f'{f}.csv')
