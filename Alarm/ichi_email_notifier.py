import copy
import math
import os.path
import time
import random
from quote import quote
from datetime import timedelta, datetime
from dateutil import parser
import pandas as pd
import numpy as np
from binance.client import Client
import smtplib
from data_prep.data_hunter_futures import DataHunterFutures


def XABC_li2df(XABC_list, df):
    XABC_df = pd.DataFrame({
        'X': [i[0] for i in [l[0] for l in XABC_list]],
        'A': [i[1] for i in [l[0] for l in XABC_list]],
        'B': [i[2] for i in [l[0] for l in XABC_list]],
        # 'C':[i[3] for i in [l[0] for l in XABC_list]],
        'date_X': [df['timestamp'][i[0]] for i in [l[1] for l in XABC_list]],
        'date_A': [df['timestamp'][i[1]] for i in [l[1] for l in XABC_list]],
        'date_B': [df['timestamp'][i[2]] for i in [l[1] for l in XABC_list]],
        # 'date_C':[df['timestamp'][i[3]] for i in [l[1] for l in XABC_list]]
    })
    return XABC_df


def email_sender(message, emails_list):
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("luis.figo908908@gmail.com",
            "vpvumdjlmzxktshi")  # "k.sehat.business2021@gmail.com", "ocpnatewbibhdqjh"
    s.sendmail("luis.figo908908@gmail.com", emails_list, message)
    s.quit()


def macd_phase_change(df, date_pointer):
    if df['MACD1_Hist'][date_pointer] * df['MACD1_Hist'][date_pointer - 1] < 0:
        return True
    else:
        return False


def enter_check_ichi(df, ts_list, ls, ichi4, enter_price):
    dp = -2
    if df.iloc[dp]['timestamp'] not in ts_list:
        if df.iloc[dp]['base_line'] <= df.iloc[dp]['conversion_line']:
            if df.iloc[dp]['close'] >= df.iloc[dp]['lead_a_shift'] and df.iloc[dp]['close'] >= \
                    df.iloc[dp]['lead_b_shift']:
                if df.iloc[dp]['lead_a'] >= df.iloc[dp]['lead_b']:
                    # if df.iloc[dp]['close'] > max(df.iloc[dp - ichi4 + 1:dp]['high']):
                    enter_price = df.iloc[dp]['close']
                    ts_list.append(df.iloc[dp]['timestamp'])
                    return 1, 0, ts_list, enter_price
        if df.iloc[dp]['base_line'] >= df.iloc[dp]['conversion_line']:
            if df.iloc[dp]['close'] <= df.iloc[dp]['lead_a_shift'] and df.iloc[dp]['close'] <= \
                    df.iloc[dp]['lead_b_shift']:
                if df.iloc[dp]['lead_a'] <= df.iloc[dp]['lead_b']:
                    # if df.iloc[dp]['close'] < min(df.iloc[dp - ichi4 + 1:dp]['low']):
                    enter_price = df.iloc[dp]['close']
                    ts_list.append(df.iloc[dp]['timestamp'])
                    return 1, 1, ts_list, enter_price
    return 0, ls, ts_list, enter_price


def exit_check(df, ls, enter, enter_price, exit_price, target_flag, target_percent,
               tp1_percent, ts_list):
    target_flag_at_this_candle = 0
    trade_fee = 0
    dp = -1
    if df.iloc[dp]['timestamp'] not in ts_list:
        if ls == 0:
            tp1 = (1 + tp1_percent) * enter_price
            # if df['low'][dp] < (1-0.01)*enter_price:
            #     exit_price = (1-0.01)*enter_price
            #     return 0, exit_price, 0
            if df.iloc[dp]['high'] >= tp1:
                exit_price = tp1
                return 0, exit_price, 0
            if (df.iloc[dp]['high'] >= (1 + target_percent) * enter_price) and target_flag == 0:
                target_flag = 1
                target_flag_at_this_candle = 1
            if target_flag == 0:
                if df.iloc[dp]['low'] <= df.iloc[dp-1]['base_line']:
                    exit_price = df.iloc[dp-1]['base_line']
                    return 0, exit_price, 0
                if df.iloc[dp]['close'] <= df.iloc[dp]['base_line']:
                    exit_price = df.iloc[dp]['close']
                    return 0, exit_price, 0
            if target_flag == 1 and target_flag_at_this_candle == 1:
                if df.iloc[dp]['close'] <= enter_price * (1 + trade_fee):
                    exit_price = df.iloc[dp]['close']
                    return 0, exit_price, 0
            if target_flag == 1 and target_flag_at_this_candle == 0:
                if df.iloc[dp]['low'] <= enter_price * (1 + trade_fee):
                    exit_price = enter_price * (1 + trade_fee)
                    return 0, exit_price, 0
                if df.iloc[dp]['close'] <= df.iloc[dp]['base_line']:
                    exit_price = df.iloc[dp]['close']
                    return 0, exit_price, 0
        if ls == 1:
            tp1 = (1 - tp1_percent) * enter_price
            # if df['high'][dp] > (1+0.01)*enter_price:
            #     exit_price = (1+0.01)*enter_price
            #     return 0, exit_price, 0
            if df.iloc[dp]['low'] <= tp1:
                exit_price = tp1
                return 0, exit_price, 0
            if (df.iloc[dp]['low'] <= (1 - target_percent) * enter_price) and target_flag == 0:
                target_flag = 1
                target_flag_at_this_candle = 1
            if target_flag == 0:
                if df.iloc[dp]['high'] >= df.iloc[dp-1]['base_line']:
                    exit_price = df.iloc[dp-1]['base_line']
                    return 0, exit_price, 0
                if df.iloc[dp]['close'] >= df.iloc[dp]['base_line']:
                    exit_price = df.iloc[dp]['close']
                    return 0, exit_price, 0
            if target_flag == 1 and target_flag_at_this_candle == 1:
                if df.iloc[dp]['close'] >= enter_price * (1 - trade_fee):
                    exit_price = df.iloc[dp]['close']
                    return 0, exit_price, 0
            if target_flag == 1 and target_flag_at_this_candle == 0:
                if df.iloc[dp]['high'] >= enter_price * (1 - trade_fee):
                    exit_price = enter_price * (1 - trade_fee)
                    return 0, exit_price, 0
                if df.iloc[dp]['close'] >= df.iloc[dp]['base_line']:
                    exit_price = df.iloc[dp]['close']
                    return 0, exit_price, 0

    return enter, exit_price, target_flag


def ichi_alarm_sender(symbol, data_step, ichi_value, emails_list, ichi_list):
    try:
        time.sleep(5 * 1)
        df = DataHunterFutures(symbol=symbol, start_date='1 Feb 2022', end_date='2025-08-01 00:00:00',
                        step=data_step, print_output=False).prepare_data_online(ichi1=ichi_value[0],
                                                            ichi2=ichi_value[1],
                                                            ichi3=ichi_value[2],
                                                            ichi4=ichi_value[3])
        ts_list, enter, ls, enter_price, exit_price, target_flag, target_percent, tp1_percent = \
            ichi_list
        if enter == 0:
            enter, ls, ts_list, enter_price = enter_check_ichi(df, ts_list, ls,
                                                               ichi_value[3], enter_price)
            if enter == 1:
                message = f"Subject: {'Ichi alarm'} \n\nHi dear traders, \n{symbol} " \
                          f"{data_step} {ichi_value}\nAlarm is received at UTC time " \
                          f"{datetime.datetime.utcnow()}.\n Please open a " \
                          f"{'long' if ls == 0 else 'short'} position."
                email_sender(message, emails_list)
                print(
                    f"email sended for {symbol}, {data_step}, {ichi_value} at {datetime.datetime.utcnow()}.")
        if enter == 1:
            enter, exit_price, target_flag = exit_check(df, ls, enter, enter_price, exit_price,
                                                        target_flag, target_percent, tp1_percent,
                                                        ts_list)
            if enter == 0:
                message = f"Subject: {'Ichi alarm'} \n\nHi dear traders, \n{symbol} " \
                          f"{data_step} {ichi_value}\nAlarm is received at UTC time " \
                          f"{datetime.datetime.utcnow()}.\n Please exit from the position which " \
                          f"you entered at {ts_list[-1]}."
                email_sender(message, emails_list)
                print(
                    f"exit email sended for {symbol}, {data_step}, {ichi_value} at"
                    f" {datetime.datetime.utcnow()}.")
        return [ts_list, enter, ls, enter_price, exit_price, target_flag, target_percent, tp1_percent]
    except Exception as e:
        print(f'Error is: {e}'
              f'\nerror is occured for {symbol} {data_step} with ichi values:{ichi_value}!!!')
        message = f"Subject: {'ERROR in ichi alarm'} \n\nHi dear traders,\nIt seems we are co" \
                  f"nfronting some problems for {symbol} {data_step} with ichi values" \
                  f" {ichi_value}. " \
                  f"\nThis error occured at {datetime.datetime.utcnow()}." \
                  f"\nOur back-end team is working on it and certainly it will be solved ASAP, " \
                  f"because nothing is impossible for our team. " \
                  f"\nAlso it is worth mentioning that " \
                  f"we have the most handsome guys in our team." \
                  f"\nWe really appreciate for your patience."
        return ichi_list


emails_list = ["kanan.sehat.ks@gmail.com",
               # "amir_elikaee@yahoo.com",
               "saeedtrader94@gmail.com",
               # "mohammad.mehmanchi@gmail.com",
               # "sarah.arab@protonmail.com",
               # "sephedoo@gmail.com",
               # "joe.goodwill02@gmail.com",
               # "sarehpeyman@gmail.com",
               # "rezajdadi@gmail.com"
               ]

inputs_dict = {
    'ETHUSDT': {
        '1h': [[18, 30, 26, 13, 0.01, 0.08]]#, [20, 18, 40, 10, 0.01, 0.08], [14, 34, 26, 13, 0.01,
                                             #                              0.08],
               # [20, 34, 26, 13, 0.01, 0.08]]
    },
    # 'TRXUSDT': {
    #     '30m': [[26, 3, 3], [40, 6, 6], [30, 3, 3], [40, 4, 3], [40, 3, 4]],
    # },
    # 'NEOUSDT': {
    #     '30m': [[40, 3, 3], [52, 4, 3], [52, 3, 4], [52, 3, 3], [40, 4, 3], [40, 3, 4], [3, 4, 4]],
    # },
    # 'IOTAUSDT': {
    #     '30m': [[52, 3, 14]]
    # },
}

ts_dict = {}
for symbol, v in inputs_dict.items():
    for data_step, ichi_list in v.items():
        for ichi_value in ichi_list:
            ts_dict[f'{symbol}_{data_step}_{ichi_value}'] = [[], 0, 0, 0, 0, 0,
                                                             ichi_value[4], ichi_value[5]]

while True:
    for symbol, v in inputs_dict.items():
        for data_step, ichi_list in v.items():
            for ichi_value in ichi_list:
                ts_dict[f'{symbol}_{data_step}_{ichi_value}'] = ichi_alarm_sender(symbol, data_step,
                                                                                  ichi_value,
                                                                                  emails_list,
                                                                                  ts_dict[
                                                                                      f'{symbol}_{data_step}_{ichi_value}'])
                time.sleep(1)
