from data_prep.data_hunter_futures import DataHunterFutures
from utils.utils import create_pl_table, candlestick_patterns


def enter_check_ichi(df, dp, ichi4):
    if df['base_line'][dp] <= df['conversion_line'][dp] and df['close'][dp] >= df['base_line'][
        dp] and df['close'][dp] >= df['conversion_line'][dp]:
        if df['close'][dp] >= df['lead_a_shift'][dp] and df['close'][dp] >= df['lead_b_shift'][dp]:
            if df['lead_a'][dp] >= df['lead_b'][dp]:
                # if df['close'][dp] > max(df['high'][dp - ichi4 + 1:dp]):
                return 1, 0
    if df['base_line'][dp] >= df['conversion_line'][dp] and df['close'][dp] <= df['base_line'][
        dp] and df['close'][dp] <= df['conversion_line'][dp]:
        if df['close'][dp] <= df['lead_a_shift'][dp] and df['close'][dp] <= df['lead_b_shift'][dp]:
            if df['lead_a'][dp] <= df['lead_b'][dp]:
                # if df['close'][dp] < min(df['low'][dp - ichi4 + 1:dp]):
                return 1, 1
    return 0, None


def exit_check(df, dp, ls, enter, enter_price, exit_price, index_buy, target_flag, target_percent,
               tp1_percent, trade_fee):
    target_flag_at_this_candle = 0
    if ls == 0:
        tp1 = (1 + tp1_percent) * enter_price
        # if df['low'][dp] < (1-0.01)*enter_price:
        #     exit_price = (1-0.01)*enter_price
        #     return 0, exit_price, 0
        if df['high'][dp] >= tp1:
            exit_price = tp1
            return 0, exit_price, 0
        if (df['high'][dp] >= (1 + target_percent) * enter_price) and target_flag == 0:
            target_flag = 1
            target_flag_at_this_candle = 1
        if target_flag == 0:
            if df['low'][dp] <= df['base_line'][dp - 1]:
                exit_price = df['base_line'][dp - 1]
                return 0, exit_price, 0
            if df['close'][dp] <= df['base_line'][dp]:
                exit_price = df['close'][dp]
                return 0, exit_price, 0
        if target_flag == 1 and target_flag_at_this_candle == 1:
            if df['close'][dp] <= enter_price * (1 + trade_fee):
                exit_price = df['close'][dp]
                return 0, exit_price, 0
        if target_flag == 1 and target_flag_at_this_candle == 0:
            if df['low'][dp] <= enter_price * (1 + trade_fee):
                exit_price = enter_price * (1 + trade_fee)
                return 0, exit_price, 0
            if df['close'][dp] <= df['base_line'][dp]:
                exit_price = df['close'][dp]
                return 0, exit_price, 0
            return candlestick_patterns(df, dp, ls, 'engulf', index_buy, target_flag)
    if ls == 1:
        tp1 = (1 - tp1_percent) * enter_price
        # if df['high'][dp] > (1+0.01)*enter_price:
        #     exit_price = (1+0.01)*enter_price
        #     return 0, exit_price, 0
        if df['low'][dp] <= tp1:
            exit_price = tp1
            return 0, exit_price, 0
        if (df['low'][dp] <= (1 - target_percent) * enter_price) and target_flag == 0:
            target_flag = 1
            target_flag_at_this_candle = 1
        if target_flag == 0:
            if df['high'][dp] >= df['base_line'][dp - 1]:
                exit_price = df['base_line'][dp - 1]
                return 0, exit_price, 0
            if df['close'][dp] >= df['base_line'][dp]:
                exit_price = df['close'][dp]
                return 0, exit_price, 0
        if target_flag == 1 and target_flag_at_this_candle == 1:
            if df['close'][dp] >= enter_price * (1 - trade_fee):
                exit_price = df['close'][dp]
                return 0, exit_price, 0
        if target_flag == 1 and target_flag_at_this_candle == 0:
            if df['high'][dp] >= enter_price * (1 - trade_fee):
                exit_price = enter_price * (1 - trade_fee)
                return 0, exit_price, 0
            if df['close'][dp] >= df['base_line'][dp]:
                exit_price = df['close'][dp]
                return 0, exit_price, 0
            return candlestick_patterns(df, dp, ls, 'engulf', index_buy, target_flag)


    return enter, exit_price, target_flag


def money_calc(money, enter_price, exit_price, ls, trade_fee, pl_list, leverage):
    if ls == 0:  # long
        pl = leverage * ((exit_price - enter_price) / enter_price - trade_fee)
        pl_list.append(pl)
        money = money + pl * money
    if ls == 1:  # short
        pl = leverage * ((enter_price - exit_price) / enter_price - trade_fee)
        pl_list.append(pl)
        money = money + pl * money
    return money, pl


def trader(args):
    symbol = args[6]
    start_date = args[7]
    end_date = args[9]
    data_step = args[8]
    leverage = args[10]
    print(args, symbol, data_step)
    df = DataHunterFutures(symbol=symbol, start_date=start_date, end_date=end_date,
                           step=data_step).prepare_data(macd_slow=26, macd_fast=12, macd_sign=9,
                                                        bb_win=20,
                                                        bb_win_dev=2,
                                                        ichi1=args[0],
                                                        ichi2=args[1],
                                                        ichi3=args[2],
                                                        ichi4=args[3])

    trade_fee = 0.0012
    money = 1
    pl = 0
    enter = 0
    enter_price = 0
    exit_price = 0
    target_flag = 0
    target_flag_at_this_candle = 0
    target_percent = args[4]
    tp1_percent = args[5]
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
            enter, exit_price, target_flag = exit_check(df, dp, ls, enter,
                                                        enter_price,
                                                        exit_price,
                                                        index_buy,
                                                        target_flag,
                                                        target_percent,
                                                        tp1_percent,
                                                        0)
            if enter == 0:
                index_sell = dp
                date_of_trade_list.append(df['timestamp'][dp])
                money, pl = money_calc(money, enter_price, exit_price, ls, trade_fee, pl_list,
                                       leverage)
                print('enter at:', df['timestamp'][index_buy], enter_price)
                print('exit at:', df['timestamp'][index_sell], exit_price)
                print('pl:', pl)
                print('money:', money)
                print('================')
    print(money)
    return create_pl_table(date_of_trade_list, pl_list, data_step), money


symbol = 'ETHUSDT'
start_date = '1 Jan 2022'
end_date = '2022-08-01 00:00:00'
data_step = '1h'
leverage = 1
input_list = [
    [4, 34, 10, 16, 0.01, 0.08, symbol, start_date, data_step, end_date, 1]
]
for input_value in input_list:
    trader(input_value)
