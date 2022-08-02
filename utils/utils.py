import pandas as pd

def create_pl_table(date_of_trade_list, profit_loss_list, data_step):
    Profit_Loss_Table = pd.DataFrame({
        'date': date_of_trade_list,
        'profit & loss': profit_loss_list
    })

    Profit_Loss_Table['date'] = pd.to_datetime(Profit_Loss_Table['date'])
    Profit_Loss_Table['year'] = Profit_Loss_Table['date'].apply(lambda t: t.year)
    Profit_Loss_Table['month'] = Profit_Loss_Table['date'].apply(lambda t: t.month)
    Profit_Loss_Table['day'] = Profit_Loss_Table['date'].apply(lambda t: t.day)

    Money_each_month = Profit_Loss_Table.groupby(['year', 'month'])
    month_profit_loss_list = []
    year_month_list = []
    month_pos_trades = []
    month_neg_trades = []
    month_all_trades = []
    sum_pl_list = []
    for key, value in zip(Money_each_month.groups.keys(), Money_each_month.groups.values()):
        money_after_trade = 1
        sum_pl = 0
        for i in value:
            money_after_trade = money_after_trade + Profit_Loss_Table['profit & loss'][
                i] * money_after_trade
            sum_pl = sum_pl + Profit_Loss_Table['profit & loss'][i]
        month_pos_trades.append(len(Profit_Loss_Table['profit & loss'][value][Profit_Loss_Table[
                                                                                  'profit & loss'] > 0]))
        month_neg_trades.append(len(Profit_Loss_Table['profit & loss'][value][Profit_Loss_Table[
                                                                                  'profit & loss'] < 0]))
        month_all_trades.append(len(value))
        month_profit = (money_after_trade - 1) * 100
        month_profit_loss_list.append(month_profit)
        sum_pl_list.append(sum_pl)
        year_month_list.append(key)

    Profit_Loss_Table_by_Year_Month = pd.DataFrame({
        'year_month': year_month_list,
        'profit & loss': month_profit_loss_list,
        'sum of all pl': sum_pl_list,
        'positive trades': month_pos_trades,
        'negative trades': month_neg_trades,
        'all trades': month_all_trades,
    })
    Profit_Loss_Table_by_Year_Month = Profit_Loss_Table_by_Year_Month.add_suffix(
        '_' + data_step)
    print(Profit_Loss_Table_by_Year_Month)
    return Profit_Loss_Table_by_Year_Month


def max_num_of_loss_trades(pl_list):
    counter = 0
    old_counter = 0
    for pl in pl_list:
        if pl <= 0:
            counter += 1
        else:
            if counter >= old_counter:
                old_counter = counter
            counter = 0
    return old_counter


def macd_phase_change(df, date_pointer):
    if df['MACD1_Hist'][date_pointer] * df['MACD1_Hist'][date_pointer - 1] < 0:
        return True
    else:
        return False


def equal_date_pointer(df1, df2, dp1, dp2):
    dp2_str = df1['timestamp'][dp1]
    try:
        dp2 = df2[df2['timestamp'] == dp2_str].index.values[0] + 2
    except IndexError:
        print(f"there occurs an error in {df1['timestamp'][dp1]}")
        dp2 = dp2 + 2
    return dp2


def candlestick_patterns(df, dp, ls, pattern, index_buy, target_flag):
    if pattern == 'engulf':
        if dp-1 > index_buy:
            if ls == 0:
                if df['close'][dp] <= df['open'][dp]:
                    if df['close'][dp] <= df['low'][dp-1]:
                        if df['close'][dp] <= df['open'][dp] - 2/3 * abs(df['open'][dp]-df['low'][dp]):
                            exit_price = df['close'][dp]
                            print('Exit with engulfing.')
                            return 0, exit_price, 0
            if ls == 1:
                if df['close'][dp] >= df['open'][dp]:
                    if df['close'][dp] >= df['high'][dp-1]:
                        if df['close'][dp] >= df['open'][dp] + 2/3 * abs(df['open'][dp]-df['high'][dp]):
                            exit_price = df['close'][dp]
                            print('Exit with engulfing.')
                            return 0, exit_price, 0
        return 1, 0, target_flag
