import concurrent.futures
import os
import random
import math
import pandas as pd
import numpy as np
import copy
import itertools
from tqdm import tqdm


class AllCombinationsEvaluator():

    def __init__(self, *args, config: dict, function, trader_name, save_dir):
        self.args = args
        self.config = config
        self.function = function
        self.trader_name = trader_name
        self.save_dir = save_dir

    def generate_all_combinations(self):
        return list(itertools.product(*list(self.config.values())))

    def evaluate(self, member):
        output = self.function(member + list(self.args))
        return output

    def run(self):
        all_comb = self.generate_all_combinations()
        # region Evaluate each member and make member/score dataframe
        member_list = []
        score_list = []
        for member in all_comb:
            member_list.append(list(member))
        with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
            for member, output in zip(member_list, executor.map(self.evaluate, member_list)):
                # directory_save_runs = f'D:/Python projects/EEVA/trader/Optimization results/All ' \
                #                       f'combinations run/{self.trader_name}/{self.args[0]} {self.args[-3]}'
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                os.chdir(self.save_dir)
                filename_first_part = str(member) + '-' + '-'.join(self.args[:-2])
                try:
                    if len(output[0]) != 0:
                        output[0].to_csv(f'{filename_first_part}-{output[1]}.csv', index=True)
                except:
                    if output[0]:
                        output[0].to_csv(f'{filename_first_part}-{output[1]}.csv', index=True)
        os.chdir('D:/Python projects/EEVA/trader/')

# ali = {
#     'fast_window': [3, 4, 5, 6, 7, 8, 9],
#     'slow_window': [10, 20, 30, 40, 50, 60],
#     'sign_window': [130, 140, 150, 160],
# }


# def f(X, symbol, data_step):
#     if (X[0] + X[1] + X[2]) == 205:
#         return None
#     else:
#         return (X[0] + X[1] + X[2])
# from eeva_1_5_0 import trader
#
#
# def main():
#     best_params = ga.run()
#
#
# ali = {
#     'slow_window': [4, 5],
#     'fast_window': [3],
#     'sign_window': [18]
# }
# ga = AllCombinationsEvaluator('ETHUSDT', '1 Aug 2021', '30m', '2021-10-01 00:00:00', 1,
#                               config=ali, function=trader)
#
# if __name__ == '__main__':
#     main()
