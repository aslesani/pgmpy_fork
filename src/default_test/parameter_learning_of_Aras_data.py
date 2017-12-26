'''
Created on July 13, 2017

@author: Adele
'''
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BdeuScore, K2Score, BicScore
from pgmpy.estimators import HillClimbSearch
import csv
import time
import dataPreparation# import get_work_lists

#print(dataPreparation.get_work_lists())
#feature_names = dataPreparation.get_work_lists()
#feature_names.append("Person")
#print(feature_names)
#mydata = np.random.randint(low=0, high=2,size=(100, 6))
mydata = np.genfromtxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Aras\House A\CSV_Summery\Sequential\Day\occur\Whole_data.csv', delimiter=",")
#pd.read_csv(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\data.csv')
#print(mydata)
feature_names = [str(i) for i in range (1,41)]
feature_names.append("Person")
feature_names.append("activity")
print(feature_names)
data = pd.DataFrame(mydata, columns= feature_names)#['X', 'Y'])
print(data)

# structure by <pgmpy.estimators.BicScore.BicScore object at 0x000001A73A66C080>
start_time = time.time()
structures = [#BicScore 
              [('5', '29'), ('5', '10'), ('1', 'Person'), ('1', '4'), ('36', '28'),
               ('36', '25'), ('36', '35'), ('17', '18'), ('17', '38'), ('17', '37'), 
               ('6', '5'), ('6', '9'), ('14', '13'), ('23', '13'), ('40', '1'), 
               ('Person', '29'), ('Person', '12'), ('24', '23'), ('28', '27'), 
               ('2', '1'), ('2', '40'), ('20', '19'), ('25', '35'), ('8', '9'), 
               ('9', '10'), ('27', '35'), ('27', '17'), ('39', '40'), ('39', '4'), 
               ('15', '16'), ('34', '33'), ('26', '28'), ('26', '36'), ('26', '25'), 
               ('activity', '39'), ('activity', '2'), ('activity', '6'), ('activity', '14'), 
               ('activity', '15'), ('activity', '11'), ('activity', 'Person'), 
               ('activity', '7'), ('activity', '24'), ('activity', '20'), ('activity', '22'), 
               ('activity', '26'), ('38', '37'), ('18', '32'), ('32', '29'), ('32', '31'), 
               ('16', '32'), ('16', '17'), ('16', '37'), ('33', 'activity'), ('33', '36'), 
               ('33', '26'), ('7', '8'), ('22', '28'), ('22', '21'), ('29', '9'), ('29', '30'), 
               ('13', '29'), ('11', '38'), ('11', '12'), ('4', '3')],
               
               #BdeuScore 
               [('5', '10'), ('1', 'Person'), ('1', '31'), ('1', '2'), ('1', '7'), ('36', '35'), ('17', '3'), 
                ('17', '31'), ('17', '18'), ('17', '19'), ('17', '40'), ('17', '37'), ('6', '5'), ('6', '30'), 
                ('6', '9'), ('6', '10'), ('14', '11'), ('23', '14'), ('23', '30'), ('21', '14'), ('40', '18'), 
                ('40', '5'), ('40', '3'), ('40', '2'), ('Person', '20'), ('Person', '12'), ('Person', '8'), 
                ('Person', '30'), ('Person', '22'), ('3', '5'), ('3', '1'), ('3', '23'), ('30', '29'), 
                ('30', '2'), ('24', '23'), ('28', '27'), ('28', '23'), ('20', '19'), ('12', '38'), 
                ('25', '36'), ('25', '33'), ('25', '14'), ('25', '35'), ('25', '39'), ('25', '21'), ('8', '9'), 
                ('9', '10'), ('27', '25'), ('27', '36'), ('27', '35'), ('39', '32'), ('39', '40'), ('31', '29'), 
                ('31', '23'), ('15', '16'), ('34', '25'), ('34', '40'), ('34', '33'), ('34', '21'), 
                ('26', '27'), ('26', '36'), ('26', '33'), ('26', '28'), ('26', 'activity'), ('26', '22'), 
                ('26', '25'), ('26', '34'), ('26', '35'), ('activity', '39'), ('activity', '1'), 
                ('activity', '17'), ('activity', '6'), ('activity', '13'), ('activity', '34'), 
                ('activity', '11'), ('activity', 'Person'), ('activity', '37'), ('activity', '7'), 
                ('activity', '28'), ('activity', '24'), ('activity', '20'), ('activity', '22'), 
                ('activity', '4'), ('37', '38'), ('18', '32'), ('18', '3'), ('18', '31'), ('18', '19'), 
                ('32', '29'), ('32', '31'), ('32', '30'), ('16', '38'), ('16', 'activity'), ('16', '32'), 
                ('16', '17'), ('19', '24'), ('33', '21'), ('33', '32'), ('33', '36'), ('33', '40'), 
                ('33', '4'), ('7', '8'), ('7', '6'), ('22', '14'), ('22', '21'), ('29', '9'), ('13', '14'), 
                ('13', '38'), ('13', '19'), ('13', '12'), ('13', '30'), ('11', '12'), ('4', '18'), 
                ('4', '5'), ('4', '3'), ('4', '13'), ('4', '23')],
              
              #K2Score
              [('20', '19'), ('20', '9'), ('20', '1'), ('15', '28'), ('15', '16'), ('4', '3'), ('4', '14'), 
               ('23', '13'), ('3', '12'), ('3', '1'), ('38', '17'), ('38', '15'), ('38', '35'), ('38', '37'), 
               ('38', 'activity'), ('17', '15'), ('17', '18'), ('21', '34'), ('21', '22'), ('32', '31'), 
               ('32', '29'), ('14', '12'), ('14', '13'), ('19', '9'), ('19', '24'), ('19', '1'), 
               ('19', 'activity'), ('19', 'Person'), ('19', '35'), ('28', '27'), ('9', '10'), ('1', '2'), 
               ('1', '7'), ('1', 'Person'), ('16', '35'), ('16', '32'), ('16', 'activity'), ('26', '34'), 
               ('26', '21'), ('26', '4'), ('26', '25'), ('26', 'activity'), ('12', '11'), ('35', '25'), 
               ('35', '36'), ('35', '26'), ('8', '9'), ('22', '14'), ('22', 'Person'), ('22', '34'), 
               ('27', '25'), ('27', '26'), ('27', 'activity'), ('24', '9'), ('24', '23'), ('36', '25'), 
               ('36', '28'), ('36', '26'), ('7', '8'), ('7', '5'), ('30', '2'), ('40', '2'), 
               ('activity', '21'), ('activity', '4'), ('activity', '24'), ('activity', '12'), 
               ('activity', '7'), ('activity', '14'), ('activity', '34'), ('activity', '9'), 
               ('activity', '5'), ('activity', 'Person'), ('activity', '39'), ('activity', '1'), 
               ('18', '20'), ('18', '15'), ('18', '32'), ('29', '30'), ('34', '33'), ('25', '14'), 
               ('25', '34'), ('25', '5'), ('25', '33'), ('25', '39'), ('33', '32'), ('5', '10'), ('5', '6'), 
               ('Person', '7'), ('Person', '29'), ('Person', '11'), ('Person', '12'), ('39', '40'), 
               ('39', '32'), ('13', '29'), ('6', '9'), ('6', '29'), ('37', '15'), ('37', '17'), 
               ('37', 'activity')]
                ]

prior_type = ["dirichlet", "BDeu", "K2"]

for i in range(1,3):
    f = open(r'C:\Users\Adele\Desktop\PhD thesis tests\parameterLearning_' + prior_type[i], 'w')
    print(prior_type[i])
    model = BayesianModel(structures[i])
    estimator = BayesianEstimator(model, data)
    
    end_time = time.time()
    print("parameter learning in seconds:{}".format(end_time-start_time))
    
    all_cpds = estimator.get_parameters(prior_type=prior_type[i])#BDeu")#"dirichlet")
    for c in all_cpds:
        print(c)
        f.write(c.tostring())  # python will convert \n to os.linesep

    f.close() 

#for my_node in feature_names:
 #   the_cpd = estimator.estimate_cpd(my_node, prior_type="dirichlet")
  #  print(the_cpd)




#casas7_model = BayesianModel()
#casas7_model.fit(data, estimator=BayesianEstimator)#MaximumLikelihoodEstimator)
#print(casas7_model.get_cpds())
#casas7_model.get_n