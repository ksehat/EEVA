import pandas as pd
from data_prep.data_hunter import DataHunter
from utils.utils import create_pl_table, macd_phase_change, equal_date_pointer


def trader(args):
    symbol = args[5]
    start_date = args[6]
    end_date = args[8]
    data_step = args[7]
    leverage = args[9]
    print(args, symbol, data_step)
    time_step_list = ['5m', '15m', '30m', '1h', '2h', '4h']
    lower_data_step = [idx for idx, elem in enumerate(time_step_list) if elem == data_step]
    df1 = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                     step=data_step).prepare_data(macd_slow=26, macd_fast=12, macd_sign=9)

    df2 = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                     step=time_step_list[lower_data_step[0] - 1]).prepare_data(macd_slow=26,
                                                                               macd_fast=12,
                                                                               macd_sign=9)
    pp1 = None
    pp2 = None
    pp1_macd = None
    pp2_macd = None
    pp1_date = None
    pp2_date = None
    np1 = None
    np2 = None
    np1_macd = None
    np2_macd = None
    np1_date = None
    np2_date = None
    dp2 = 0
    for dp in range(len(df1)):
        if df1['MACD1_Hist'][dp] >= 0:
            if macd_phase_change(df1, dp):
                np1 = np2
                np1_macd = np2_macd
                np1_date = np2_date
                np2 = None
                np2_macd = None
            if not pp2 or df1['high'][dp] > pp2:
                pp2 = df1['high'][dp]
            if not pp2_macd or df1['MACD1_Hist'][dp] > pp2_macd:
                pp2_macd = df1['MACD1_Hist'][dp]
                pp2_date = dp
            if pp1 and pp2:
                if pp2 > pp1 and pp2_macd <= pp1_macd:
                    short_alarm = 1
                    print(df1['timestamp'][pp1_date])
                    print(df1['timestamp'][pp2_date])
                    print(df2['timestamp'][dp2])
                    print(pp1_macd, pp2_macd, df2['high'][dp2])
                    print('==============')
        if df1['MACD1_Hist'][dp] < 0:
            if macd_phase_change(df1, dp):
                pp1 = pp2
                pp1_macd = pp2_macd
                pp1_date = pp2_date
                pp2 = None
                pp2_macd = None
            if not np2 or df1['low'][dp] < np2:
                np2 = df1['low'][dp]
            if not np2_macd or df1['MACD1_Hist'][dp] < np2_macd:
                np2_macd = df1['MACD1_Hist'][dp]
                np2_date = dp
            if np1 and np2:
                if np2 < np1 and np2_macd >= np1_macd:
                    long_alarm = 1
                    print(df1['timestamp'][np1_date])
                    print(df1['timestamp'][np2_date])
                    print(df2['timestamp'][dp2])
                    print(np1_macd, np2_macd, df2['high'][dp2])
                    print('==============')


def main():
    symbol = 'ETHUSDT'
    start_date = '1 Jan 2022'
    end_date = '2022-08-01 00:00:00'
    data_step = '1h'
    leverage = 1
    input_list = [
        [20, 34, 20, 40, 2, symbol, start_date, data_step, end_date, 2]
    ]
    for input_value in input_list:
        trader(input_value)


if __name__ == '__main__':
    main()
