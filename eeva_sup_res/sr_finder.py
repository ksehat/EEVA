from data_prep.data_hunter_futures import DataHunterFutures


def sr_finder(df, start_date, end_date, win):
    if start_date not in list(df['timestamp']):
        print('start_date is not in the DataFrame.')
        return []
    start_date_index = df[df['timestamp'] == start_date].index[0]
    end_date_index = df[df['timestamp'] == end_date].index[0]
    df = df[start_date_index:end_date_index]
    res_list = []
    sup_list = []
    for i in range(win, len(df) - win):
        if df.iloc[i]['high'] >= max(df.iloc[i - win:i + win]['high']):
            higher_band = df.iloc[i]['high']
            lower_band = max(df.iloc[i - win:i + win]['close'])
            if not ((df.iloc[i+1:]['close'] > higher_band).sum() >= 1):
                res_list.append([df.iloc[i]['high'], max(df.iloc[i - win:i + win]['close'])])
            else:
                first_index = df.iloc[i+1:][df['close'] > higher_band].first_valid_index()
                if not (df.iloc[first_index+1:]['close'] < lower_band).sum() >= 1:
                    res_list.append([df.iloc[i]['high'], max(df.iloc[i - win:i + win]['close'])])
        if df.iloc[i]['low'] <= min(df.iloc[i - win:i + win]['low']):
            lower_band = df.iloc[i]['low']
            higher_band = min(df.iloc[i - win:i + win]['close'])
            if not ((df.iloc[i+1:]['close'] < lower_band).sum() >= 1):
                sup_list.append([df.iloc[i]['low'], min(df.iloc[i - win:i + win]['close'])])
            else:
                first_index = df.iloc[i+1:][df['close'] < lower_band].first_valid_index()
                if not (df.iloc[first_index+1:]['close'] > higher_band).sum() >= 1:
                    sup_list.append([df.iloc[i]['low'], min(df.iloc[i - win:i + win]['close'])])
    return (res_list, sup_list)


df = DataHunterFutures(symbol='ETHUSDT', start_date='12 May 2022', end_date='2024-06-08 00:00:00',
                       step='1h').prepare_data_online()
res, sup = sr_finder(df, '2022-05-12 00:00:00', '2022-05-28 00:00:00', 6)
print('res:',res)
print('====================')
print('sup:',sup)
