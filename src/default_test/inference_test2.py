'''
Created on Sep 21, 2017

@author: Adele
'''
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation

G = BayesianModel([('diff', 'grade'), ('intel', 'grade'),
                    ('intel', 'SAT'), ('grade', 'letter')])
diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
grade_cpd = TabularCPD('grade', 3,
                        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
                        evidence=['diff', 'intel'],
                        evidence_card=[2, 3])
sat_cpd = TabularCPD('SAT', 2,
                      [[0.1, 0.2, 0.7],
                       [0.9, 0.8, 0.3]],
                      evidence=['intel'], evidence_card=[3])
letter_cpd = TabularCPD('letter', 2,
                         [[0.1, 0.4, 0.8],
                          [0.9, 0.6, 0.2]],
                         evidence=['grade'], evidence_card=[3])
G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
bp = BeliefPropagation(G)
bp.calibrate()
