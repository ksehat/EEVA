from data_prep.data_hunter import DataHunter
from utils.utils import create_pl_table, macd_phase_change


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


def exit_check(df, dp, ls, enter, enter_price, exit_price, sl, sl_at_this_candle):
    if ls == 0:
        if df['low'][dp] <= (1 - 0.01) * enter_price:
            exit_price = (1 - 0.01) * enter_price
            return 0, exit_price
        if df['low'][dp] <= sl:
            if sl_at_this_candle:
                exit_price = df['close'][dp]
            else:
                exit_price = sl
            return 0, exit_price
        if df['high'][dp] >= (1 + 0.10) * enter_price:
            exit_price = (1 + 0.10) * enter_price
            return 0, exit_price

    if ls == 1:
        if df['high'][dp] >= (1 + 0.01) * enter_price:
            exit_price = (1 + 0.01) * enter_price
            return 0, exit_price
        if df['high'][dp] >= sl:
            if sl_at_this_candle:
                exit_price = df['close'][dp]
            else:
                exit_price = sl
            return 0, exit_price
        if df['low'][dp] <= (1 - 0.10) * enter_price:
            exit_price = (1 - 0.10) * enter_price
            return 0, exit_price

    return enter, exit_price


def money_calc(money, enter_price, exit_price, ls, trade_fee, pl_list):
    if ls == 0:  # long
        pl = (exit_price - enter_price) / enter_price - trade_fee
        pl_list.append(pl)
        money = money + pl * money
    if ls == 1:  # short
        pl = (enter_price - exit_price) / enter_price - trade_fee
        pl_list.append(pl)
        money = money + pl * money
    return money, pl


def sl_at_enter(df, dp, enter_price, ls):
    if not ls:  # long
        sl = enter_price * (1 - 0.01)
        sudo_sl = enter_price * (1 - 0.01)
        if df['MACD1_Hist'][dp] < 0:
            sudo_sl = df['low'][dp]  # C is placed in sudo_sl
            i = 0
            while df['MACD1_Hist'][dp - i] < 0:
                if df['low'][dp - i] <= sudo_sl and df['low'][dp - i] >= sl:
                    sudo_sl = df['low'][dp - i]
                i += 1
    if ls:  # short
        sl = enter_price * (1 + 0.01)
        sudo_sl = enter_price * (1 + 0.01)
        if df['MACD1_Hist'][dp] > 0:
            sudo_sl = df['high'][dp]  # C is placed in sudo_sl
            i = 0
            while df['MACD1_Hist'][dp - i] > 0:
                if df['high'][dp - i] >= sudo_sl and df['high'][dp - i] <= sl:
                    sudo_sl = df['high'][dp - i]
                i += 1
    return sl, sudo_sl


def sl_at_exit(df, dp, ls, sl, sudo_sl, enter_price):
    if ls == 0:
        sl_at_this_candle = 0
        if macd_phase_change(df, dp) and df['MACD1_Hist'][dp] > 0:
            sl = sudo_sl
            if sudo_sl != enter_price * (1 - 0.01):
                sl_at_this_candle = 1
    if ls == 1:
        sl_at_this_candle = 0
        if macd_phase_change(df, dp) and df['MACD1_Hist'][dp] < 0:
            sl = sudo_sl
            if sudo_sl != enter_price * (1 + 0.01):
                sl_at_this_candle = 1
    return sl, sl_at_this_candle


def sudo_sl_trail(df, dp, ls, sudo_sl):
    if ls == 0:
        if df['MACD1_Hist'][dp] < 0:
            if macd_phase_change(df, dp) or df['low'][dp] <= sudo_sl:
                sudo_sl = df['low'][dp]
    if ls == 1:
        if df['MACD1_Hist'][dp] > 0:
            if macd_phase_change(df, dp) or df['high'][dp] >= sudo_sl:
                sudo_sl = df['high'][dp]
    return sudo_sl


def trader(args):
    symbol = args[5]
    start_date = args[6]
    end_date = args[8]
    data_step = args[7]
    leverage = args[9]
    print(args, symbol, data_step)
    df = DataHunter(symbol=symbol, start_date=start_date, end_date=end_date,
                    step=data_step).prepare_data(macd_slow=args[0],
                                                 macd_fast=args[1],
                                                 macd_sign=args[2],
                                                 bb_win=20,
                                                 bb_win_dev=2,
                                                 ichi1=3,
                                                 ichi2=3,
                                                 ichi3=3,
                                                 ichi4=2)
    trade_fee = 0.002
    money = 1
    pl = 0
    leverage = 1
    enter = 0
    enter_price = 0
    exit_price = 0
    target_flag = 0
    target_flag_at_this_candle = 0
    date_of_trade_list = []
    pl_list = []

    for dp in range(len(df)):
        if enter == 0:
            enter, ls = enter_check_ichi(df, dp, args[3])  # ls==0 means long
            if enter == 1:
                enter_price = df['close'][dp]
                index_buy = dp
                sl, sudo_sl = sl_at_enter(df, dp, enter_price, ls)
                continue
        if enter == 1:
            sudo_sl = sudo_sl_trail(df, dp, ls, sudo_sl)
            sl, sl_at_this_candle = sl_at_exit(df, dp, ls, sl, sudo_sl, enter_price)
            enter, exit_price = exit_check(df, dp, ls, enter, enter_price, exit_price, sl, sl_at_this_candle)
            if enter == 0:
                index_sell = dp
                date_of_trade_list.append(df['timestamp'][dp])
                money, pl = money_calc(money, enter_price, exit_price, ls, trade_fee, pl_list)
                # print('enter at:', df['timestamp'][index_buy], enter_price)
                # print('exit at:', df['timestamp'][index_sell], exit_price)
                # print('pl:', pl)
                # print('money:', money)
                # print('================')
    print(money)
    return create_pl_table(date_of_trade_list, pl_list, data_step), money


# symbol = 'ETHUSDT'
# start_date = '1 Mar 2021'
# end_date = '2022-08-01 00:00:00'
# data_step = '1h'
# leverage = 1
# input_list = [
#     [9, 26, 52, 26, 1, symbol, start_date, data_step, end_date, leverage]
# ]
# for input_value in input_list:
#     trader(input_value)
