import os
import re
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, DotProduct
from scipy.stats import ecdf, lognorm
import numpy as np
import requests
import json
import random
from optuna.samplers import BaseSampler
from openai import OpenAI



class Sampler(BaseSampler):
    def __init__(self, max_space_size, n_batches, batch_size):
        self.instruction_candidates = None
        self.demo_candidates = None
        self.max_space_size = max_space_size
        self.search_space = []
#        self.relative_search_space = {}
        self.embeddings = {}
        self.n_batches = n_batches
        self.min_cold_start = 4
        self.next_batch = []
        self.values = []
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.distributions = None
        self.labels = []

    def create_search_space(self, instruction_candidates, demo_candidates):

        self.search_space = {}
        self.instruction_candidates = instruction_candidates
        self.demo_candidates = demo_candidates
        for _ in range(self.max_space_size):
            sample = []
            text = []
            for i in self.instruction_candidates.keys():
                sample.append(random.randint(0, len(self.instruction_candidates[i])-1))
                text.append(self.instruction_candidates[i][sample[-1]])
                if self.demo_candidates:
                    sample.append(random.randint(0, len(self.demo_candidates[i])-1))
                    text.append( "\n".join([x['question'] + "\n" + x['answer'] for x in self.demo_candidates[i][sample[-1]]]))
            self.search_space[tuple(sample)] = "\n".join(text)

        client = OpenAI(api_key=os.environ['OPEN_AI_KEY'])
        embedding_search_space = [e.embedding for e in client.embeddings.create(
            input=[self.search_space[k] for k in self.search_space.keys()],
            model="text-embedding-3-small"
        ).data]

        for i in instruction_candidates.keys():
            self.labels.append(f'{i}_predictor_instruction')
            if demo_candidates:
                self.labels.append(f'{i}_predictor_demos')

    
        self.distributions = self._get_param_distributions(instruction_candidates, demo_candidates)


        
        
        for j, e in zip(self.search_space.keys(), embedding_search_space):
            self.embeddings[j] = e

    def _get_param_distributions(self, instruction_candidates, demo_candidates):
        from optuna.distributions import CategoricalDistribution

        param_distributions = {}

        for i in range(len(instruction_candidates)):
            param_distributions[f"{i}_predictor_instruction"] = CategoricalDistribution(
                range(len(instruction_candidates[i]))
            )
            if demo_candidates:
                param_distributions[f"{i}_predictor_demos"] = CategoricalDistribution(range(len(demo_candidates[i])))

        return param_distributions

    def before_trial(self, study, trial):

        # everything is together because when you sample, everything is together
        # that's why each instruction_candidates is not passed to sample independent
        # so, why are there multiple values?
        # this will have to be fixed, Sensai: go back to a different gpr for each predictor


        self.scored  = []
        self.scores = []
        for t in study.get_trials():
            if t.values:
                idx = []
                for k in t.params.keys():
                    idx.append(t.params[k])
                if tuple(idx) in self.search_space.keys():
                    self.scored.append(tuple(idx))
                    self.scores.append(t.values[0])

        if len(self.next_batch) == 0:
            for q in range(self.batch_size):
                available_space = [s for s in self.search_space.keys() if s not in self.scored]
                if len(self.search_space) - len(available_space) >= 3:
                    self.fit()
                    self.create_batches()
                    best_idx = self.get_best_batch()
                choice = random.choice(available_space)
#                choice['scored'] = True
                Q = {}
                for l, c in zip(self.labels, choice):
                    Q[l] = c
                self.next_batch.append(Q.copy())


        return 1

    def after_trial(self, study, trial, state, values):
        
        return len(self.next_batch)

    def sample_relative(self, study, trial, search_space):
        self.next_value = self.next_batch.pop()
        return self.next_value

    def sample_independent(self, study, trial, param_name,  distribution):

        return self.next_value[param_name]

    def infer_relative_search_space(self, study, trial):
            return self.distributions

    def fit(self):

        self.unscored = [q for q in self.search_space if q not in self.scored]
        self.gpr = GaussianProcessRegressor(kernel = Matern() + WhiteKernel())
        scores_ecdf = ecdf(self.scores)
        transformed_scores = np.log(lognorm.ppf(scores_ecdf.cdf.evaluate(self.scores) * .999 + .0005, 1))
        self.gpr.fit([self.embeddings[s] for s in self.scored], transformed_scores)
        self.y_best = max(transformed_scores)
        
        self.mu, self.sigma = self.gpr.predict([self.embeddings[u] for u in self.unscored], return_cov=True)
#        return mu, sigma


    def create_batches(self):
        
        self.batch_mu = []
        self.batch_sigma = []
        batches = []
        self.batch_idx = []
        n_to_choose_from = len(self.unscored)
        for z in range(self.n_batches):
            batch = random.sample(range(0, n_to_choose_from-1), self.batch_size)
            self.batch_idx.append(batch)
            m, s = self.gpr.predict([self.embeddings[self.unscored[i]] for i in batch], return_cov=True)
            self.batch_mu.append(','.join([str(x) for x in m]))
            sigma = [','.join([str(y) for y in x]) for x in s]
            self.batch_sigma.append(';'.join(sigma))


        
    def get_best_batch(self, gpu=False):
        
        boaz = 'SHOULD NEVER SEE'
        try:
            if gpu:
                url = f'http://34.130.49.1:5000/gpu_qei?y_best={self.y_best}&n={self.batch_size}'
            else:
                url = f'https://boaz.onrender.com/qei'
            data = {'k': ';'.join(self.batch_mu),
                    'sigma': '|'.join(self.batch_sigma),
                    'n': str(self.batch_size),
                    'y_best': '.2'}
            headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-RapidAPI-Key": os.environ['X_RapidAPI_Key']
                    }
            response = requests.post(url, json.dumps(data), headers=headers)
            boaz = eval(response.content.decode('utf-8'))
        except Exception as e:
            print('Bayesian Issues:', e)
            return random.randint(0, len(self.batch_mu))
        fboaz = [float(x) for x in boaz['scores'].split(',')]
        best = -1
        for i, mx in enumerate(fboaz):
            if mx > best:
                best = float(mx)
                best_idx = i
        return best_idx
            

        