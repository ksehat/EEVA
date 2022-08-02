from data_prep.data_hunter import DataHunter
from utils.utils import create_pl_table


def sl_tp_definer(enter_price, ls):
    if ls:
        sl = enter_price - 0.01 * enter_price
        tp = enter_price + 0.01 * enter_price
    if not ls:
        sl = enter_price + 0.01 * enter_price
        tp = enter_price - 0.01 * enter_price
    return sl, tp


df = DataHunter(symbol='ETHUSDT', start_date='1 Jan 2022', end_date='2022-03-28 00:00:00',
                step='30m').prepare_data(macd_slow=26, macd_fast=12,
                                         macd_sign=9, bb_win=20, bb_win_dev=2)

trade_fee = 0.002
money = 1
leverage = 1
enter = 0
date_of_trade_list = []
profit_loss_list = []
for dp in range(len(df)):
    if enter == 0:
        if df['close'][dp] > df['BB_high'][dp]:
            enter = 1
            index_buy = dp
            enter_price = df['close'][dp]
            ls = 0
            sl, tp = sl_tp_definer(enter_price, ls)
            continue
        if df['close'][dp] < df['BB_low'][dp] and enter == 0:
            enter = 1
            index_buy = dp
            enter_price = df['close'][dp]
            ls = 1
            sl, tp = sl_tp_definer(enter_price, ls)
            continue
    if enter == 1:
        if ls:
            if df['low'][dp] < df['BB_low'][dp]:
                exit_price = df['close'][dp]
                index_sell = dp
                enter = 0
                continue
            if df['high'][dp] >= df['BB_high'][dp]:
                exit_price = df['close'][dp]
                index_sell = dp
                enter = 0
                continue
        if not ls:
            if df['high'][dp] > df['BB_high'][dp]:
                exit_price = df['close'][dp]
                index_sell = dp
                enter = 0
            if df['low'][dp] <= df['BB_low'][dp]:
                exit_price = df['close'][dp]
                index_sell = dp
                enter = 0
            if enter == 0:
                if ls:
                    pl = leverage * ((exit_price - enter_price) / enter_price) - trade_fee
                if not ls:
                    pl = leverage * ((enter_price - exit_price) / enter_price) - trade_fee
                money = money + pl * money
                date_of_trade_list.append(dp)
                profit_loss_list.append(pl)
                enter_date = df['timestamp'][index_buy]
                exit_date = df['timestamp'][index_sell]
                print(f'=============\nenter:{enter_date}\nexit:{exit_date}\n'
                      f'pl:{pl},money:{money}')
