import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

#test 1
data = pd.DataFrame(data={'A': [0.0, 0.0, 1.0], 'B': [0.0, 1.0, 0.0], 'C': [1.0, 1.0, 0.0]})
#data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})

print(data)
model = BayesianModel([('A', 'C'), ('B', 'C')])
estimator = BayesianEstimator(model, data)
cpd_C = estimator.estimate_cpd('C', prior_type="dirichlet", pseudo_counts=[1, 2])
print(cpd_C)

#test 2
values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
                       columns=['A', 'B', 'C', 'D'])
model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D')])
estimator = BayesianEstimator(model, values)
a = estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=5)
for i in a:
    print(i)
#print(a)
#print(type(a))
#print(len(a))
#print(a[0])
