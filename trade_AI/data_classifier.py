import numpy as np
import pandas as pd
from data_prep.data_hunter_futures import DataHunterFutures
from utils.utils import create_pl_table


def trader(args):
    symbol = args[0]
    start_date = args[1]
    end_date = args[2]
    data_step = args[3]
    leverage = args[4]
    print(args, symbol, data_step)
    df = DataHunterFutures(symbol=symbol, start_date=start_date, end_date=end_date,
                           step=data_step).prepare_data(macd_slow=args[5], macd_fast=args[6],
                                                        macd_sign=args[7],
                                                        bb_win=args[8], bb_win_dev=args[9],
                                                        ichi1=args[10], ichi2=args[11],
                                                        ichi3=args[12], ichi4=args[13])
    rr = 3
    sl_percent = 0.005
    tp_percent = rr * sl_percent
    df['close_long_index'] = np.nan
    df['close_short_index'] = np.nan
    df['class_long'] = np.nan
    df['class_short'] = np.nan
    for i in range(len(df)):
        tp = df['close'][i] * (1+tp_percent)
        sl = df['close'][i] * (1-sl_percent)
        for j in range(i+1, len(df)-1):
            if df['low'][j] < sl:
                df['close_long_index'][i] = j
                df['class_long'][i] = 0
                break
            if df['high'][j] >= tp:
                df['close_long_index'][i] = j
                df['class_long'][i] = 1
                break
    for i in range(len(df)):
        tp = df['close'][i] * (1-tp_percent)
        sl = df['close'][i] * (1+sl_percent)
        for j in range(i+1, len(df)-1):
            if df['high'][j] > sl:
                df['close_short_index'][i] = j
                df['class_short'][i] = 0
                break
            if df['low'][j] <= tp:
                df['close_short_index'][i] = j
                df['class_short'][i] = 1
                break
    df.to_csv(f'df_{data_step}.csv')



def main():
    """Data"""
    symbol = 'ETHUSDT'
    start_date = '1 Jan 2020'
    end_date = '2022-08-01 00:00:00'
    data_step = '15m'
    leverage = 1
    input_list = [
        [symbol, start_date, end_date, data_step, leverage, 26, 12, 9, 20, 2, 9, 26, 52, 26]
    ]
    for value in input_list:
        trader(value)


if __name__ == '__main__':
    main()
