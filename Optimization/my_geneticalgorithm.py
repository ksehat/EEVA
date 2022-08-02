import random
import math
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm


class MyGeneticAlgorithm():

    def __init__(self, config: dict, function, n_crossover: int = None,
                 crossover_mode: str = 'rigid', population_size: int = 100,
                 run_iter=100, mutation_prob=.8, maximize: bool = True):
        self.config = config
        self.variables_list = [k for k in config]
        self.params_list = [self.config[k] for k in self.config]
        self.n_crossover = n_crossover
        self.crossover_mode = crossover_mode
        self.function = function
        self.max_population_size = math.prod([len(self.config[k]) for k in self.config])
        self.population_size = population_size if population_size <= self.max_population_size else self.max_population_size
        self.run_iter = run_iter
        self.mutation_prob = mutation_prob
        self.maximize = maximize
        self.all_populations = pd.DataFrame(columns=['members', 'score'])

    def generate_random_member(self):
        return [random.choice(self.params_list[i]) for i in range(len(self.params_list))]

    def evaluate(self, member):
        if not self.all_populations.empty:
            all_populations = list(self.all_populations['members'].values)
            if member in all_populations:
                index = all_populations.index(member)
                return self.all_populations['score'][index]
        output = self.function(member)
        if not output:
            if self.maximize:
                return 1e-20
            else:
                return 1e20
        try:
            if output < 0:
                return 1e-20
            else:
                return output
        except:
            if self.maximize:
                return 1e-20
            else:
                return 1e20

    def crossover(self, first_parent, second_parent, n_crossover=None, crossover_mode=None):
        if n_crossover is None:
            n_crossover = self.n_crossover

        if crossover_mode is None:
            crossover_mode = self.crossover_mode

        if random.random() > 0.5:
            first_parent, second_parent = second_parent, first_parent
        parents = [first_parent, second_parent]
        chromosome_len = len(first_parent)
        nucleotides = list(range(chromosome_len))

        crossover_indices = []
        if crossover_mode == 'random':
            crossover_indices = sorted(random.sample(nucleotides, k=n_crossover))

        if crossover_mode == 'equal':
            crossover_indices = list(range(0, chromosome_len, (chromosome_len // n_crossover)))[1:]

        nucleotide_selection = [
            i % 2 for i in np.cumsum(
                [1 if idx in crossover_indices else 0
                 for idx in range(chromosome_len)]
            )
        ]
        child = [parents[nucleotide_selection[i]][i] for i in range(chromosome_len)]
        return child

    def mutate(self, member):
        # Decide whether to mutate or not
        if random.random() > self.mutation_prob:
            return None
        mutation_point = random.randrange(len(member))
        params_list = copy.deepcopy(self.params_list[mutation_point])
        params_list.remove(member[mutation_point])
        member[mutation_point] = random.sample(params_list, k=1)[0]
        return member

    def run(self, keep_frac: float = 3, crossover_frac: float = 3, mutate_frac: float = 3):
        n_keep = math.floor((keep_frac / 100) * self.population_size)
        n_crossover = math.floor((crossover_frac / 100) * self.population_size)
        n_mutate = math.floor((mutate_frac / 100) * self.population_size)
        n_random = self.population_size - n_keep - n_mutate - n_crossover
        pbar = tqdm(total=self.run_iter)
        # region generate random population
        population = []
        iter = 1
        while iter <= self.population_size:
            member = self.generate_random_member()
            if member not in population:
                population.append(member)
                iter += 1
        # endregion
        # region Evaluate each member and make member/score dataframe
        member_list = []
        score_list = []
        for member in population:
            member_list.append(member)
            score_list.append(self.evaluate(member))
        population_score_df = pd.DataFrame({
            'members': member_list,
            'score': score_list
        })
        population_score_df.sort_values('score', inplace=True, ignore_index=True, ascending=False)
        self.all_populations = copy.deepcopy(population_score_df)
        new_population_score_df = pd.DataFrame(
            {'members': [list(np.full(len(self.variables_list), np.nan))] *
                        self.population_size,
             'score': list(np.full([self.population_size],
                                   np.nan))
             })
        new_population_score_df[:n_keep] = copy.deepcopy(population_score_df[:n_keep])
        # endregion
        # region iter until required score is satisfied
        run_iter = 1
        while run_iter < self.run_iter:
            # region crossover
            iter_cross = 0
            while iter_cross < n_crossover:
                first_parent = random.choices(population_score_df['members'],
                                              weights=population_score_df['score'])
                second_parent = random.choices(population_score_df['members'],
                                               weights=population_score_df['score'])
                child = copy.deepcopy(self.crossover(first_parent[0], second_parent[0]))
                if child not in list(new_population_score_df['members'][:n_keep + iter_cross]):
                    new_population_score_df.at[n_keep + iter_cross, 'members'] = copy.deepcopy(
                        child)
                    new_population_score_df.at[n_keep + iter_cross, 'score'] = copy.deepcopy(
                        self.evaluate(child))
                    iter_cross += 1
            # endregion
            # region mutation
            iter_mut = 0
            while iter_mut < n_mutate:
                mut_member = copy.deepcopy(random.choices(population_score_df['members'],
                                                          weights=population_score_df['score'])[0])
                child = copy.deepcopy(self.mutate(mut_member))
                if child == None: continue
                if child not in list(
                        new_population_score_df['members'][:n_keep + n_crossover + iter_mut]):
                    new_population_score_df.at[
                        n_keep + n_crossover + iter_mut, 'members'] = copy.deepcopy(child)
                    new_population_score_df.at[
                        n_keep + n_crossover + iter_mut, 'score'] = copy.deepcopy(
                        self.evaluate(child))
                    iter_mut += 1
            # endregion
            # region random
            iter_rand = 0
            while iter_rand < n_random:
                random_member = copy.deepcopy(self.generate_random_member())
                if random_member not in list(new_population_score_df['members'][
                                             :n_keep + n_crossover + n_mutate + iter_rand]):
                    new_population_score_df.at[
                        n_keep + n_crossover + n_mutate + iter_rand, 'members'] = copy.deepcopy(
                        random_member)
                    new_population_score_df.at[
                        n_keep + n_crossover + n_mutate + iter_rand, 'score'] = copy.deepcopy(
                        self.evaluate(random_member))
                    iter_rand += 1
            # endregion
            new_population_score_df.sort_values(['score'], inplace=True, ignore_index=True,
                                                ascending=not self.maximize)
            population_score_df = copy.deepcopy(new_population_score_df)
            self.all_populations = pd.concat([self.all_populations, population_score_df],
                                             ignore_index=True)
            run_iter += 1
            pbar.update(run_iter)
        # endregion
        return new_population_score_df[:20]


# ali = {
#     'fast_window': [3, 4, 5, 6, 7, 8, 9],
#     'slow_window': [10, 20, 30, 40, 50, 60],
#     'sign_window': [130, 140, 150, 160],
# }
#
#
# def f(X):
#     return (X[0] + X[1] + X[2])
#
#
# ga = MyGeneticAlgorithm(config=ali, function=f, run_iter=5, population_size=20, n_crossover=3,
#                         crossover_mode='random')
# best_params = ga.run()
# print(best_params)
