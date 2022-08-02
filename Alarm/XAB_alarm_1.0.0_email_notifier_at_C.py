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
from data_prep.data_hunter import DataHunter


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


def XABC_hunter(df):
    ZC_Index = pd.DataFrame({'zcindex': df[df['MACD1_ZC'] == 1].index.values,
                             'timestamp': df.loc[df['MACD1_ZC'] == 1, 'timestamp'],
                             'MACD1_Hist': df.loc[df['MACD1_ZC'] == 1, 'MACD1_Hist']},
                            columns=['zcindex', 'timestamp', 'MACD1_Hist']).reset_index(drop=True)
    XABC_list1 = []
    for row_zcindex, zcindex in ZC_Index.iterrows():
        if row_zcindex + 3 <= len(ZC_Index) - 1:
            if df['MACD1_Hist'][zcindex[0]] >= 0:
                # region XABC Finder
                X = max(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['high'])
                index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['high'].idxmax()
                A = min(
                    df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]][
                        'low'])
                index_A = \
                    df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]][
                        'low'].idxmin()
                B = max(
                    df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]][
                        'high'])
                index_B = \
                    df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]][
                        'high'].idxmax()
                index_B_phase_change = ZC_Index.iloc[row_zcindex + 3, 0]
                # C = min( df.iloc[ZC_Index.iloc[row_zcindex+3,0] : ZC_Index.iloc[row_zcindex+4,0]]['low'] )
                # index_C = df.iloc[ZC_Index.iloc[row_zcindex+3,0] : ZC_Index.iloc[row_zcindex+4,0]]['low'].idxmin()
                if A < X and B < X and B > A:  # and C<A and C<X:
                    xabc_flag = 1
                    XABC_list1.append(
                        [[X, A, B], [index_X, index_A, index_B, index_B_phase_change], xabc_flag])

                # endregion
            if df['MACD1_Hist'][zcindex[0]] < 0:
                # region XABC Finder
                X = min(df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'])
                index_X = df.iloc[zcindex[0]: ZC_Index.iloc[row_zcindex + 1, 0]]['low'].idxmin()
                A = max(
                    df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]][
                        'high'])
                index_A = \
                    df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]][
                        'high'].idxmax()
                B = min(
                    df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]][
                        'low'])
                index_B = \
                    df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]][
                        'low'].idxmin()
                index_B_phase_change = ZC_Index.iloc[row_zcindex + 3, 0]
                # C = max(df.iloc[ZC_Index.iloc[row_zcindex + 3, 0]: ZC_Index.iloc[row_zcindex + 4, 0]]['high'])
                # index_C = df.iloc[ZC_Index.iloc[row_zcindex + 3, 0]: ZC_Index.iloc[row_zcindex + 4, 0]]['high'].idxmax()
                if A > X and B > X and B < A:  # and C>A and C>X:
                    xabc_flag = 0
                    XABC_list1.append(
                        [[X, A, B], [index_X, index_A, index_B, index_B_phase_change], xabc_flag])
    return XABC_list1


def email_sender(message, emails_list):
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("luis.figo908908@gmail.com",
            "vpvumdjlmzxktshi")  # "k.sehat.business2021@gmail.com", "ocpnatewbibhdqjh"
    s.sendmail("luis.figo908908@gmail.com", emails_list, message)
    s.quit()


def xab_completor(df, date_pointer, xab, XABC_del_list, C, index_C):
    if xab[2] == 1:
        if df['low'][date_pointer] <= xab[0][1]:
            XABC_del_list.append(xab)
            C = df['low'][date_pointer]
            index_C = date_pointer
            return C, index_C, 1
        if df['MACD1_Hist'][date_pointer] > 0 and macd_phase_change(df, date_pointer):
            XABC_del_list.append(xab)
            return C, index_C, 1
        return C, index_C, 0

    if xab[2] == 0:
        if df['high'][date_pointer] >= xab[0][1]:
            XABC_del_list.append(xab)
            C = df['high'][date_pointer]
            index_C = date_pointer
            return C, index_C, 1
        if df['MACD1_Hist'][date_pointer] < 0 and macd_phase_change(df, date_pointer):
            XABC_del_list.append(xab)
            return C, index_C, 1
        return C, index_C, 0


def macd_phase_change(df, date_pointer):
    if df['MACD1_Hist'][date_pointer] * df['MACD1_Hist'][date_pointer - 1] < 0:
        return True
    else:
        return False


def XAB_completion_alarm_sender(symbol, data_step, macd_value, XABC_and_counter_del_list,
                                emails_list):
    try:
        XABC_del_list, counter = XABC_and_counter_del_list
        counter += 1
        df = DataHunter(symbol=symbol, start_date='1 Jul 2021', end_date='2022-12-01 01:00:00',
                        step=data_step).prepare_data_online(macd_slow=macd_value[0],
                                                            macd_fast=macd_value[1],
                                                            macd_sign=macd_value[2])
        XABC_list = XABC_hunter(df)
        for xab in XABC_list:
            if xab not in XABC_del_list:
                break_flag = 0
                C = None
                index_C = None
                for dp in range(xab[1][3], len(df) - 1):
                    C, index_C, break_flag = xab_completor(df, dp, xab, XABC_del_list, C, index_C)
                    if C and index_C and counter > 1:
                        message = f"Subject: {'Alarm'} \n\nHi dear traders, \n{symbol} " \
                                  f"{data_step} {macd_value}\nAlarm is received at UTC time " \
                                  f"{datetime.now()} and it should be received at UTC time " \
                                  f"{df['timestamp'][index_C]}. \n\n\n X:{xab[0][0]} and X_date" \
                                  f":{df['timestamp'][xab[1][0]]} \nA:{xab[0][1]} and A_date" \
                                  f":{df['timestamp'][xab[1][1]]} \nB:{xab[0][2]} and B_date" \
                                  f":{df['timestamp'][xab[1][2]]} \nC:{C} and C_date" \
                                  f":{df['timestamp'][index_C]}"
                        email_sender(message, emails_list)
                        print(
                            f"email sended for {symbol}, {data_step}, {macd_value} and xab is "
                            f"\nX:{xab[0][0]} and X_date:{df['timestamp'][xab[1][0]]}"
                            f"\nA:{xab[0][1]} and A_date:{df['timestamp'][xab[1][1]]}"
                            f"\nB:{xab[0][2]} and B_date:{df['timestamp'][xab[1][2]]}"
                            f"\nC:{C} and C_date:{df['timestamp'][index_C]}")
                    if break_flag == 1:
                        break
        return [XABC_del_list, counter]
    except Exception as e:
        print(f'Error is: {e}'
              f'\nerror is occured for {symbol} {data_step} with macd values:{macd_value}!!!')
        message = f"Subject: {'ERROR in Alarm'} \n\nHi dear traders,\nIt seems we are co" \
                  f"nfronting some problems for {symbol} {data_step} with macd values" \
                  f" {macd_value}. " \
                  f"\nThis error occured at {datetime.now().astimezone()}." \
                  f"\nOur back-end team is working on it and certainly it will be solved ASAP, " \
                  f"because nothing is impossible for our team. " \
                  f"\nAlso it is worth mentioning that " \
                  f"we have the most handsome guys in our team." \
                  f"\nWe really appreciate for your patience."
        # email_sender(message, emails_list)
        time.sleep(5 * 1)
        return [XABC_del_list, counter]


emails_list = ["kanan.sehat.ks@gmail.com",
               "amir_elikaee@yahoo.com",
               "saeedtrader94@gmail.com",
               "mohammad.mehmanchi@gmail.com",
               "sarah.arab@protonmail.com",
               "sephedoo@gmail.com",
               "joe.goodwill02@gmail.com",
               "sarehpeyman@gmail.com",
               "rezajdadi@gmail.com"
               ]

inputs_dict = {
    'ETHUSDT': {
        '30m': [[3, 4, 18], [3, 24, 3], [3, 4, 12], [3, 12, 4], [3, 4, 20], [3, 6, 6]]
    },
    'TRXUSDT': {
        '30m': [[26, 3, 3], [40, 6, 6], [30, 3, 3], [40, 4, 3], [40, 3, 4]],
    },
    'NEOUSDT': {
        '30m': [[40, 3, 3], [52, 4, 3], [52, 3, 4], [52, 3, 3], [40, 4, 3], [40, 3, 4], [3, 4, 4]],
    },
    # 'IOTAUSDT': {
    #     '30m': [[52, 3, 14]]
    # },
}

XABC_dict = {}
for symbol, v in inputs_dict.items():
    for data_step, macd_list in v.items():
        for macd_value in macd_list:
            XABC_dict[f'XABC_list_old_{symbol}_{data_step}_{macd_value}'] = [[], 0]
        df = DataHunter(symbol=symbol, start_date='1 Jul 2021', end_date='2022-12-01 01:00:00',
                        step=data_step).prepare_data_online(macd_slow=26,
                                                            macd_fast=12,
                                                            macd_sign=9)

while True:
    for symbol, v in inputs_dict.items():
        for data_step, macd_list in v.items():
            for macd_value in macd_list:
                XABC_dict[f'XABC_list_old_{symbol}_{data_step}_{macd_value}'] = \
                    XAB_completion_alarm_sender(
                        symbol,
                        data_step,
                        macd_value,
                        XABC_dict[f'XABC_list_old_{symbol}_{data_step}_{macd_value}'],
                        emails_list)
    print('sleeping!!!')
    # time.sleep(60 * 8)
