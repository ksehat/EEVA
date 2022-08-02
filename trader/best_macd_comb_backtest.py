from sys_runner import sys_runner

systems = [
    'eeva_0_3_3',
    'eeva_0_1_3'
]

coins_datastep_list = [
    # ('LTCUSDT', '1h'),
    # ('BTCUSDT', '1h'),
    # ('IOTAUSDT', '1h'),
    # ('ETHUSDT', '1h'),
    # ('TRXUSDT', '1h'),
    # ('NEOUSDT', '1h'),
    # ('BNBUSDT', '1h'),
    ('ETHUSDT', '30m'),
    # ('NEOUSDT', '30m'),
    # ('IOTAUSDT', '30m'),
    # ('TRXUSDT', '30m'),
    # ('LTCUSDT', '30m'),
    # ('BTCUSDT', '30m'),
    # ('BNBUSDT', '30m'),
    # ('MATICUSDT', '30m'),
    # ('LTCUSDT', '15m'),
    # ('BTCUSDT', '15m'),
    # ('IOTAUSDT', '15m'),
    # ('ETHUSDT', '15m'),
    # ('TRXUSDT', '15m'),
    # ('NEOUSDT', '15m'),
    # ('BNBUSDT', '15m'),
    # ('MATICUSDT', '15m'),
]

config = {
    'slow_window': [3, 5, 6, 7, 12, 13, 26, 30, 40, 52],
    'fast_window': [3, 4, 5, 6, 7, 12, 24, 30, 40, 48],
    'sign_window': [3, 4, 6, 8, 9, 10, 12, 14, 16, 18, 20]
}


def main():
    start_date_list = ['1 Jan 2020', '1 Feb 2020', '1 Mar 2020', '1 Apr 2020', '1 May 2020',
                       '1 Jun 2020']
    end_date_list = ['2020-02-01 00:00:00', '2020-03-01 00:00:00', '2020-04-01 00:00:00',
                     '2020-05-01 00:00:00', '2020-06-01 00:00:00', '2020-07-01 00:00:00']
    sys_runner(systems, config, coins_datastep_list,
               start_date='1 Mar 2021',
               end_date='2022-03-01 00:00:00',
               save_dir=f'D:/Python projects/EEVA/trader/Optimization results/Best MACD comb '
                        f'backtest month by month/')


if __name__ == '__main__':
    main()
