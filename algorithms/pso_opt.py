"""
Created on Fri Apr 10 14:58:29 2020

@author: caiocarneloz
"""
from inspyred import ec

class pso_opt:

    def __init__(self, model, input_params, X_train, X_test, y_train, y_test, fit_func, silent):

        self.dimensions     = len(input_params)

        self.model          = model
        self.param_names    = list(input_params.keys())
        self.X_train        = X_train
        self.X_test         = X_test
        self.y_train        = y_train
        self.y_test         = y_test
        self.maximize       = False
        self.fit_func       = fit_func
        self.silent         = silent

        mins                = []
        maxs                = []

        for interval in input_params.values():
            mins.append(interval[0])
            maxs.append(interval[1])
            
        self.bounder        = ec.Bounder(lower_bound=mins, upper_bound=maxs)


    def generator(self, random, args):

        particle = []
    
        for i in range(0,self.dimensions):
            particle.append(random.uniform(self.bounder.lower_bound[i], self.bounder.upper_bound[i]))
    
        return particle


    def evaluator(self, candidates, args):

        fitness = []

        for c in candidates:
            
            params = dict((self.param_names[i], type(self.bounder.lower_bound[i])(c[i])) for i in range(self.dimensions))

            model = self.model
            model_instance = model(**params)
            model_instance.fit(self.X_train, self.y_train)

            prediction = model_instance.predict(self.X_test)
            metric = self.fit_func(self.y_test, prediction)

            if not self.silent:
                print(params)
                print(metric)

            fitness.append(metric)

        return fitness