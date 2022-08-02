from data_prep.data_hunter import DataHunter
from utils.utils import create_pl_table


def enter_check_ichi(df, dp, ichi4):
    if df['base_line'][dp] <= df['conversion_line'][dp]:
        if df['close'][dp] >= df['lead_a_shift'][dp] and df['close'][dp] >= df['lead_b_shift'][dp]:
            if df['lead_a'][dp] >= df['lead_b'][dp]:
                if df['close'][dp] > max(df['high'][dp - ichi4 + 1:dp]):
                    return 1, 0
    if df['base_line'][dp] >= df['conversion_line'][dp]:
        if df['close'][dp] <= df['lead_a_shift'][dp] and df['close'][dp] <= df['lead_b_shift'][dp]:
            if df['lead_a'][dp] <= df['lead_b'][dp]:
                if df['close'][dp] < min(df['low'][dp - ichi4 + 1:dp]):
                    return 1, 1
    return 0, None


def exit_check(df, dp, ls, enter, exit_price):
    if ls == 0:
        if df['close'][dp] <= df['base_line'][dp]:
            exit_price = df['close'][dp]
            return 0, exit_price
    if ls == 1:
        if df['close'][dp] >= df['base_line'][dp]:
            exit_price = df['close'][dp]
            return 0, exit_price

    return enter, exit_price


def money_calc(df, money, enter_price, index_buy, index_sell, ls, trade_fee, pl_list):
    if ls == 0:  # long
        pl = (max(df['high'][index_buy + 1:index_sell + 1]) - enter_price) / enter_price - trade_fee
        pl_list.append(pl)
        money = money + pl * money
    if ls == 1:  # short
        pl = (enter_price - min(df['low'][index_buy + 1:index_sell + 1])) / enter_price - trade_fee
        pl_list.append(pl)
        money = money + pl * money
    return money, pl


def trader(args):
    symbol = args[5]
    start_date = args[6]
    end_date = args[8]
    data_step = args[7]
    leverage = args[9]
    print(args, symbol, data_step)
    df = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                    step=data_step).prepare_data(macd_slow=26, macd_fast=12, macd_sign=9, bb_win=20,
                                                 bb_win_dev=2, ichi1=args[0], ichi2=args[1],
                                                 ichi3=args[2],
                                                 ichi4=args[3])

    trade_fee = 0.002
    money = 1
    pl = 0
    leverage = 1
    enter = 0
    enter_price = 0
    exit_price = 0
    target_flag = 0
    target_flag_at_this_candle = 0
    target_percent = 0
    target_percent = args[4]
    date_of_trade_list = []
    pl_list = []

    for dp in range(len(df)):
        if enter == 0:
            enter, ls = enter_check_ichi(df, dp, args[3])  # ls==0 means long
            if enter == 1:
                enter_price = df['close'][dp]
                index_buy = dp
                continue
        if enter == 1:
            enter, exit_price = exit_check(df, dp, ls, enter, exit_price)
            if enter == 0:
                index_sell = dp
                exit_best = (max(df['high'][index_buy + 1:index_sell + 1]) if ls == 0 else
                              min(df['low'][index_buy + 1:index_sell + 1]))
                index_best = (df['high'][index_buy+1:index_sell+1].idxmax() if ls==0 else
                              df['low'][index_buy+1:index_sell+1].idxmin())
                date_of_trade_list.append(df['timestamp'][dp])

                money, pl = money_calc(df, money, enter_price, index_buy, index_sell, ls, trade_fee,
                                       pl_list)
                # if pl <= 0.005:
                #     print('enter at:', df['timestamp'][index_buy], enter_price)
                #     print('exit at:', df['timestamp'][index_sell], exit_price)
                #     print('best at:', df['timestamp'][index_best], exit_best)
                #     print('pl:', pl)
                #     print('money:', money)
                #     print('================')
    print(money)
    return create_pl_table(date_of_trade_list, pl_list, data_step), money

symbol = 'ETHUSDT'
start_date = '1 Mar 2021'
end_date = '2022-08-01 00:00:00'
data_step = '1h'
leverage = 1
input_list = [
    [9, 10, 10, 5, 1, symbol, start_date, data_step, end_date, leverage]
]
for input_value in input_list:
    trader(input_value)