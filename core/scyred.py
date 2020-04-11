"""
Created on Fri Apr 10 14:58:29 2020

@author: caiocarneloz
"""
import sys
sys.path.append('../')
from inspyred import ec
import inspyred
from random import Random
from algorithms.pso_opt import pso_opt

class optimizer:

    def __init__(self, model, input_params, silent=False):

        self.dimensions  = len(input_params)
    
        self.model       = model
        self.params      = input_params
        self.param_names = list(input_params.keys())
        self.maximize    = False
        self.silent      = silent
        self.fit_func    = None
        self.X_train     = None
        self.X_test      = None
        self.y_train     = None 
        self.y_test      = None
    
        mins = []
        maxs = []
    
        for interval in input_params.values():
            mins.append(interval[0])
            maxs.append(interval[1])
            
        self.bounder     = ec.Bounder(lower_bound=mins, upper_bound=maxs)
        
            
    def fit(self, X_train, X_test, y_train, y_test, fit_func):
        
        self.fit_func = fit_func
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        
    def run_pso(self, pop_size, social_rate, max_evaluations, maximize):
        
        problem = pso_opt(self.model, self.params, self.X_train, self.X_test, 
                          self.y_train, self.y_test, self.fit_func, self.silent)
        
        pso = inspyred.swarm.PSO(Random())

        pso.terminator = [inspyred.ec.terminators.evaluation_termination,
                          inspyred.ec.terminators.diversity_termination]

        final_pop = pso.evolve(generator=problem.generator,
                    evaluator=problem.evaluator,pop_size=pop_size,
                    bounder=problem.bounder,maximize=False,
                    max_evaluations=max_evaluations,social_rate=social_rate)

        dict_opt = {}
        for key, j in zip(self.params.keys(), range(len(self.params))):
            dict_opt[key] = type(self.bounder.lower_bound[j])(final_pop[0].candidate[j])
            
        return dict_opt