import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
                       columns=['A', 'B', 'C', 'D'])
model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D')])
estimator = BayesianEstimator(model, values)
estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=5)