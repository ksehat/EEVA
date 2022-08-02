from data_prep.data_hunter_futures import DataHunterFutures
from utils.utils import create_pl_table
from ta.trend import EMAIndicator


def EMA_IND(data, win):
    ema_ind = EMAIndicator(close=data['close'], window=win)
    data[f'ema_{win}'] = ema_ind.ema_indicator()
    return data


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
                           step=data_step).prepare_data()
    df = EMA_IND(df, args[0])
    df = EMA_IND(df, args[1])
    df = EMA_IND(df, args[2])
    df = EMA_IND(df, args[3])
    df = EMA_IND(df, args[4])

    trade_fee = 0.0012
    money = 1
    pl = 0
    enter = 0
    enter_price = 0
    exit_price = 0
    date_of_trade_list = []
    pl_list = []

    for dp in range(len(df)):
        if enter == 0:
            if df.iloc[dp][f'ema_{args[4]}'] < df.iloc[dp][f'ema_{args[3]}'] and \
                df.iloc[dp][f'ema_{args[3]}'] < df.iloc[dp][f'ema_{args[2]}']:
                enter = 1
                ls = 0
            if df.iloc[dp][f'ema_{args[4]}'] > df.iloc[dp][f'ema_{args[3]}'] and \
                df.iloc[dp][f'ema_{args[3]}'] > df.iloc[dp][f'ema_{args[2]}']:
                enter = 1
                ls = 1
            if enter == 1:
                enter_price = df['close'][dp]
                index_buy = dp
                continue
        if enter == 1:
            if ls == 0 and df.iloc[dp][f'ema_{args[3]}'] > df.iloc[dp][f'ema_{args[2]}'] :#and
                                                # df.iloc[dp][f'ema_{args[2]}']):
                enter = 0
            if ls == 1 and df.iloc[dp][f'ema_{args[3]}'] < df.iloc[dp][f'ema_{args[2]}']:# and
                                                            # df.iloc[dp][f'ema_{args[2]}']):
                enter = 0
            if enter==0:
                exit_price = df['close'][dp]
                index_sell = dp
                date_of_trade_list.append(df['timestamp'][dp])
                money, pl = money_calc(money, enter_price, exit_price, ls, trade_fee, pl_list,
                                       leverage)
                # print('enter at:', df['timestamp'][index_buy], enter_price)
                # print('exit at:', df['timestamp'][index_sell], exit_price)
                # print('pl:', pl)
                # print('money:', money)
                # print('================')
    print(money)
    return create_pl_table(date_of_trade_list, pl_list, data_step), money


def main():
    """Data"""
    symbol = 'ETHUSDT'
    start_date = '1 Feb 2022'
    end_date = '2022-03-01 00:00:00'
    data_step = '1h'
    leverage = 1
    ema_list = [
        [5, 10, 5, 70, 200, 0, symbol, start_date, data_step, end_date, leverage]
    ]
    for ema_value in ema_list:
        trader(ema_value)


if __name__ == '__main__':
    main()