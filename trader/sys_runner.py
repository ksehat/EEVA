import importlib
from Optimization.all_combinations_evaluator import AllCombinationsEvaluator as ace
from data_prep.data_hunter import DataHunter

systems = [
    # 'eeva_0_3_3',
    # 'eeva_0_2_1',
    # 'eeva_0_2_4',
    'eeva_1_2_2',
    # 'ichi_0_0_0'
    # 'ichi_0_0_2'
    # 'ichi_0_0_8_trail'
]


def sys_runner(systems, config, coins_datastep_list, start_date, end_date, save_dir):
    for symbol, data_step in coins_datastep_list:
        dh = DataHunter(symbol, start_date, end_date, data_step)
        dh.download_data()
        dh = DataHunter(symbol, start_date, end_date, '15m')
        dh.download_data()
        for trader_sys in systems:
            imported_sys = importlib.import_module(trader_sys)
            all_comb_runner = ace(symbol, start_date, data_step, end_date, 1,
                                  config=config, function=imported_sys.trader,
                                  trader_name=trader_sys,
                                  save_dir=save_dir + f'{trader_sys}/{symbol} {data_step}')

        all_comb_runner.run()


# config = {
#     'slow_window': [3, 5, 8, 13, 21, 26, 30, 34, 40, 55],
#     'fast_window': [3, 5, 8, 12, 13, 21, 34, 40, 48, 55],
#     'sign_window': [2, 3, 4, 5, 6, 8, 9, 10, 13, 21]
# }

config = {
    'slow_window': [3, 5, 6, 7, 12, 13, 26, 30, 40, 52],
    'fast_window': [3, 4, 5, 6, 7, 12, 24, 30, 40, 48],
    'sign_window': [3, 4, 6, 8, 9, 10, 12, 14, 16, 18, 20]
}

# config = {
#     'ichi1': [3, 4, 5, 7, 8, 9, 10, 14, 18, 20],
#     'ichi2': [8, 10, 13, 16, 18, 21, 26, 30, 34, 40],
#     'ichi3': [5, 10, 20, 26, 30, 40, 52, 60, 65, 70],
#     'ichi4': [5, 8, 10, 13, 16, 40],  # 20, 26, 30,
#     'target_percent': [.005, .01, .015, .02],
#     'tp1_percent': [.03, .05, .08]
# }

coins_datastep_list = [
    # ('ETHUSDT', '4h'),
    # ('ETHUSDT', '2h'),
    # ('LTCUSDT', '4h'),
    # ('LTCUSDT', '2h'),
    # ('NEOUSDT', '4h'),
    # ('NEOUSDT', '2h'),
    # ('BTCUSDT', '4h'),
    # ('BTCUSDT', '2h'),
    # ('LTCUSDT', '1h'),
    # ('BTCUSDT', '1h'),
    # ('IOTAUSDT', '1h'),
    # ('ETHUSDT', '1h'),
    # ('TRXUSDT', '1h'),
    # ('NEOUSDT', '1h'),
    # ('BNBUSDT', '1h'),
    ('ETHUSDT', '30m'),
    ('NEOUSDT', '30m'),
    ('IOTAUSDT', '30m'),
    ('TRXUSDT', '30m'),
    ('LTCUSDT', '30m'),
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

if __name__ == '__main__':
    save_dir = f'D:/Python projects/EEVA/trader/Optimization results/All combinations run/'
    start_date = '1 May 2021'
    end_date = '2022-05-1 00:00:00'
    sys_runner(systems, config, coins_datastep_list, start_date, end_date, save_dir)
