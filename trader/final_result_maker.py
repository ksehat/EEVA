import os
import pandas as pd

general_folder_path = 'D:/Python projects/EEVA/trader/Genetic Results/Sys1.2.0/Genetic/weighted ' \
                      'variance'
os.chdir(general_folder_path)
folders = os.listdir()
for folder in folders:
    path = f'{general_folder_path}/{folder}'
    csv_files = [x for x in os.listdir(f'{general_folder_path}/{folder}') if 'Genetic' not in x]
    os.chdir(f'{general_folder_path}/{folder}')
    this_sys_related_csv_files = [x for x in csv_files if '1 Mar 2018' in x and 'Genetic' not in x]
    macd_value = []
    money = []
    mean_value_from_date = []
    mean_value_all = []
    var_value_from_date = []
    var_value_all = []
    min_value_from_date = []
    min_value_all = []
    max_value_from_date = []
    max_value_all = []
    for f in this_sys_related_csv_files:
        df_csv = pd.read_csv(f)
        file_name_list = f.split('-')
        macd_value.append(file_name_list[3])
        money.append(float(file_name_list[-1].replace('.csv', '')))
        mean_value_from_date.append(df_csv['profit & loss_30m'][df_csv[df_csv['year_month_30m'] == '(' \
                                                                                                   '2021, ' \
                                                                                                   '1)'].index[
                                                                    0]:].mean())
        mean_value_all.append(df_csv['profit & loss_30m'].mean())
        var_value_from_date.append(df_csv['profit & loss_30m'][df_csv[df_csv['year_month_30m'] == '(' \
                                                                                                  '2021, ' \
                                                                                                  '1)'].index[
                                                                   0]:].var())
        var_value_all.append(df_csv['profit & loss_30m'].var())
        min_value_from_date.append(df_csv['profit & loss_30m'][df_csv[df_csv['year_month_30m'] == '(' \
                                                                                                  '2021, ' \
                                                                                                  '1)'].index[
                                                                   0]:].min())
        min_value_all.append(df_csv['profit & loss_30m'].min())
        max_value_from_date.append(df_csv['profit & loss_30m'][df_csv[df_csv['year_month_30m'] == '(' \
                                                                                                  '2021, ' \
                                                                                                  '1)'].index[
                                                                   0]:].max())
        max_value_all.append(df_csv['profit & loss_30m'].max())

    final_result = pd.DataFrame({
        'macd_value': macd_value,
        'money': money,
        'mean_value_from_date': mean_value_from_date,
        'mean_value_all': mean_value_all,
        'var_value_from_date': var_value_from_date,
        'var_value_all': var_value_all,
        'min_value_from_date': min_value_from_date,
        'min_value_all': min_value_all,
        'max_value_from_date': max_value_from_date,
        'max_value_all': max_value_all
    })
    final_result.sort_values(['money'], ascending=False, inplace=True)
    final_result.to_csv(f'final_results_{folder}.csv')
