# IMPORTS
import copy
import math
import os.path
import time
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
                A = min(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['low'])
                index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['low'].idxmin()
                B = max(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['high'])
                index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['high'].idxmax()
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
                A = max(df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['high'])
                index_A = df.iloc[ZC_Index.iloc[row_zcindex + 1, 0]: ZC_Index.iloc[row_zcindex + 2, 0]]['high'].idxmax()
                B = min(df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['low'])
                index_B = df.iloc[ZC_Index.iloc[row_zcindex + 2, 0]: ZC_Index.iloc[row_zcindex + 3, 0]]['low'].idxmin()
                index_B_phase_change = ZC_Index.iloc[row_zcindex + 3, 0]
                # C = max(df.iloc[ZC_Index.iloc[row_zcindex + 3, 0]: ZC_Index.iloc[row_zcindex + 4, 0]]['high'])
                # index_C = df.iloc[ZC_Index.iloc[row_zcindex + 3, 0]: ZC_Index.iloc[row_zcindex + 4, 0]]['high'].idxmax()
                if A > X and B > X and B < A:  # and C>A and C>X:
                    xabc_flag = 0
                    XABC_list1.append(
                        [[X, A, B], [index_X, index_A, index_B, index_B_phase_change], xabc_flag])
    return XABC_list1


binsizes = {"1m": 1, "5m": 5, "8m": 8, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
            "1d": 1440}
batch_size = 750
binance_client = Client(api_key='43PXiL32cF1YFXwkeoK900wOZx8saS1T5avSRWlljStfwMrCl7lZhhJSIM1ijIzS',
                        api_secret='JjJRJ3bWQTEShF4Eu8ZigY9aEMGPnFNJMH3WoNlOQgxSgrHmLOflIavhMx0KSZFC')
binance_symbols = ['ETHUSDT']
start_date = '1 Jul 2021'
end_date = '2022-12-01 01:00:00'
data_steps = ['30m']
leverage = 1
plot_width = 1500
plot_height = 1000
args = [5, 7, 4]

XABC_dict = {}
for symbol in binance_symbols:
    for data_step in data_steps:
        XABC_dict[f'XABC_list_old_{symbol}_{data_step}'] = []

while True:
    try:
        for symbol in binance_symbols:
            for data_step in data_steps:
                df = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                                step=data_step).prepare_data_online(macd_slow=args[0], macd_fast=args[1],
                                                             macd_sign=args[2])
                XABC_list = XABC_hunter(df)
                XABC_list_old = XABC_dict[f'XABC_list_old_{symbol}_{data_step}']
                new_XABC = [item for item in XABC_list if item not in XABC_list_old]
                if new_XABC:
                    # if we find a new X and A and B, it means the MACD phase of B is
                    # changed and then we can send the email.
                    flag = new_XABC[-1][2]
                    A = new_XABC[-1][0][1]
                    B = new_XABC[-1][0][2]
                    index_B = new_XABC[-1][1][2]
                    index_B_phase_change = new_XABC[-1][1][3]
                    # for date_pointer in range(index_B, len(df)):  # TODO: check it to see until
                    #     # what timestep it goes. we want it live
                    #     if alarm == 0:
                    #         if (flag==1 and df['MACD_Hist1']<0) or (flag==0 and df[
                    #             'MACD_Hist1']>0):
                    #         if (flag == 0 and df['high'][date_pointer] >= A) or (
                    #                 flag == 1 and df['low'][date_pointer] <= A) and warning == 1:
                    new_XABC_df = XABC_li2df(new_XABC, df)
                    s = smtplib.SMTP('smtp.gmail.com', 587)
                    s.starttls()
                    s.login("luis.figo908908@gmail.com",
                            "vpvumdjlmzxktshi")  # "k.sehat.business2021@gmail.com", "ocpnatewbibhdqjh"
                    message = f"Subject: {'New XABC'} \n\nsalam,\n{symbol} {data_step} {args}" \
                              f"Alarm is received at {datetime.now().astimezone()} \n" \
                              f"and it should be received at UTC time" \
                              f" {df['timestamp'][index_B_phase_change]}" \
                              f"\n{new_XABC_df.iloc[-1, :]}"
                    s.sendmail("luis.figo908908@gmail.com", ["kanan.sehat.ks@gmail.com",
                                                             "amir_elikaee@yahoo.com",
                                                             "saeedtrader94@gmail.com",
                                                             "mohammad.mehmanchi@gmail.com",
                                                             "sarah.arab@protonmail.com",
                                                             "sephedoo@gmail.com"
                                                             ],
                               message)
                    s.quit()
                    XABC_dict[f'XABC_list_old_{symbol}_{data_step}'] = XABC_list
                    print(f'email sended for {symbol},{data_step} and new_XABC is {new_XABC_df.iloc[-1, :]}')
                    alarm = 1
        print('sleeping')
        time.sleep(60 * 10)
        pass
    except:
        print ('error is occured!!!')
        time.sleep(60 * 1)
        pass
