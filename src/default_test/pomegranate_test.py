'''
Created on April 1, 2018

@author: Adele
'''
import numpy as np

from pomegranate import HiddenMarkovModel
from pomegranate import NormalDistribution, State, DiscreteDistribution, MultivariateDistribution, MultivariateGaussianDistribution
from pomegranate import MarkovChain
from read_write import read_data_from_CSV_file
from dataPreparation import create_sequence_of_sensor_events_based_on_activity
from read_write import read_sequence_based_CSV_file_with_activity , read_sequence_based_CSV_file_without_activity, read_sequence_of_bags_CSV_file_with_activity, repaet_person_tags_as_much_as_seq_length
#from snowballstemmer import algorithms

from Validation import calculate_different_metrics
from plot_results import plot_results
import sys
import inspect

from shuffle_data import unison_shuffled_copies

def test_sample_from_site():
    
    dists = [NormalDistribution(5, 1), NormalDistribution(1, 7), NormalDistribution(8,2)]
    trans_mat = np.array([[0.7, 0.3, 0.0],
                             [0.0, 0.8, 0.2],
                             [0.0, 0.0, 0.9]])
    starts = np.array([1.0, 0.0, 0.0])
    ends = np.array([0.0, 0.0, 0.1])
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)
    model.plot()


def build_the_same_model_in_test_sample_from_site_line_by_line():
    
    # State olds emission distribution, but not
    #transition distribution, because that's stored in the graph edges.
    s1 = State(NormalDistribution(5, 1))
    s2 = State(NormalDistribution(1, 7))
    s3 = State(NormalDistribution(8, 2))
    model = HiddenMarkovModel()
    model.add_states(s1, s2, s3)
    model.add_transition(model.start, s1, 1.0)
    model.add_transition(s1, s1, 0.7)
    model.add_transition(s1, s2, 0.3)
    model.add_transition(s2, s2, 0.8)
    model.add_transition(s2, s3, 0.2)
    model.add_transition(s3, s3, 0.9)
    model.add_transition(s3, model.end, 0.1)
    model.bake()
    
    model.plot()

def build_an_hmm_example():
    # i think the characters in each DiscreteDistribution definition, means the emission matrix for each state
    # because it says the probability of seeing each character when the system is in that state
    d1 = DiscreteDistribution({'A' : 0.35, 'C' : 0.20, 'G' : 0.05, 'T' : 0.40})
    d2 = DiscreteDistribution({'A' : 0.25, 'C' : 0.25, 'G' : 0.25, 'T' : 0.25})
    d3 = DiscreteDistribution({'A' : 0.10, 'C' : 0.40, 'G' : 0.40, 'T' : 0.10})

    s1 = State(d1, name="s1")
    s2 = State(d2, name="s2")
    s3 = State(d3, name="s3")

    model = HiddenMarkovModel('example')
    model.add_states([s1, s2, s3])
    model.add_transition(model.start, s1, 0.90)
    model.add_transition(model.start, s2, 0.10)
    model.add_transition(s1, s1, 0.80)
    model.add_transition(s1, s2, 0.20)
    model.add_transition(s2, s2, 0.90)
    model.add_transition(s2, s3, 0.10)
    model.add_transition(s3, s3, 0.70)
    model.add_transition(s3, model.end, 0.30)
    model.bake()
    
    for i in range(len(model.states)):
        print(model.states[i].name )
    model.plot()
    #print(model.log_probability(list('ACGACTATTCGAT')))
    
    #print(", ".join(state.name for i, state in model.viterbi(list('ACGACTATTCGAT'))[1]))

    print("forward:" , model.forward( list('ACG') ))
    #print("backward:" , model.backward( list('ACG') ))
    #print("forward_backward:" , model.forward_backward( list('ACG') ))


    #print("Viterbi:" , model.viterbi( list('ACG') ))

def create_hmm_from_sample(file_address):

    #data, _ , _ = read_sequence_based_CSV_file_with_activity(file_address = file_address, has_header = True , separate_data_based_on_persons = False )
    #data = read_data_from_CSV_file(dest_file = file_address, data_type = np.int ,  has_header = True , return_as_pandas_data_frame = False )
    '''
    data = np.delete(data , 2, 1)
    data = np.delete(data , 2, 1)
    data = np.delete(data , 0, 1)
    data = np.delete(data , 0, 1)
    data = np.delete(data , 0, 1)
    print(np.shape(data))
    '''
    #print(data)
    data = np.array([['a' , 'b'] , ['a' , 'b']])
    data = np.array([[np.array([1,2,3]) , np.array([1,1,1])] , [np.array([1,1,2]) , np.array([1,2,2])]])
    data = [np.array([[1, 2, 3], [1, 2, 3]], np.int32) , np.array([[1, 2, 3], [1, 2, 3]], np.int32) , np.array([[1, 2, 3], [1, 2, 3]], np.int32)]
    print(data)
    #data = np.array([[['a' , 'b'] , ['a' , 'a']] , [['a' , 'b'] , ['b' , 'b']]])

    
    #data = create_sequence_of_sensor_events_based_on_activity(address_to_read = file_address, has_header = False, address_for_save = " ", isSave = False)#read_data_from_CSV_file(dest_file = file_address, data_type = numpy.int ,  has_header = False , return_as_pandas_data_frame = False )
    model = HiddenMarkovModel.from_samples(MultivariateDistribution, n_components=3, X=data)# according to my tests :D n_components is number of hidden states
    
    #print(model)
    #print(model._baum_welch_summarize())
    #model.plot()
    '''
    print("dense_transition_matrix:" , model.dense_transition_matrix())
    print("edge_count:" , model.edge_count())
    print("edges:" , model.edges)
    print("name:" , model.name)
    print("state_count:" , model.state_count())
    '''
    print(model)
    #print(model.)
    #print("summarize:" , model.summarize())
    #print(model.thaw())
    
def get_set_of_Markov_chain_nodes(model):
    
    '''
    return set of valid nodes of the model
    
    Parameters:
    ==========
    model: the Markov chain model
    
    '''
    #model.distributions[0] is the initial value of each parameter and 
    #model.distributions[1] is the value of p(i | j)
    list_of_parameters = model.distributions[1].parameters[0]
    #print((list_of_parameters[0][2]))
    
    set_of_nodes = set()
    
    for i in range(len(list_of_parameters)):
        if list_of_parameters[i][2] != 0:
            set_of_nodes.add(list_of_parameters[i][0])
            set_of_nodes.add(list_of_parameters[i][1])
    
    #print((set_of_nodes))
    return set_of_nodes   
    
    
    #print(dist[1].__class__.__bases__)

    #propnames = [name for (name, value) in inspect.getmembers(type(dist[1]), isinstance('values', property))]
    #print(dist[1].parameters[0][0])

def remove_extra_test_columns(final_test_set , list_of_markov_chain_models):
    
    '''
    remove the test columns which are not in the final model
    
    the final_test_set is created for each model in list_of_markov_chain_models separately
    '''    
    number_of_persons = len(list_of_markov_chain_models)
    
    new_final_test_set = np.zeros(number_of_persons, dtype = np.ndarray)
    for i in range(number_of_persons):
        new_final_test_set[i] = np.ndarray(shape = (len(final_test_set) , ), dtype = np.ndarray )
    
    for person in range(number_of_persons):
        list_of_set_of_nodes = list(get_set_of_Markov_chain_nodes(list_of_markov_chain_models[person]))
        #print(set_of_nodes)
        for seq, index in zip(final_test_set, range(len(final_test_set))):
            mask = np.isin(seq, list_of_set_of_nodes)
            new_final_test_set[person][index] = seq[mask]
    
    counter = 0
    for i in range(len(final_test_set)):
        #print( len(final_test_set[i]) , len(new_final_test_set[0][i]) , len(new_final_test_set[1][i]))
        
        for per in range(number_of_persons):
            if len(final_test_set[i]) != len(new_final_test_set[per][i]):
                counter +=1
    
    #print("number of edited sequences:" , counter)       
    
    return new_final_test_set


def create_casas7_hmm(file_address, has_activity):
     
    if has_activity:
        list_of_data , list_of_persons , _ = read_sequence_based_CSV_file_with_activity(file_address = file_address, has_header = True , separate_data_based_on_persons = False)
    else:
        list_of_data , list_of_persons = read_sequence_based_CSV_file_without_activity(file_address = file_address, has_header = True , separate_data_based_on_persons = False)

    
    
    model = ""
    
    try:
        model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=5, X=list_of_data , algorithm = 'baum-welch' )
        #model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=2, X=list_of_data , labels = list_of_persons , algorithm = 'labeled' )
    except KeyError:
        print('there is an exception')
    print(model)
    
    #print((list_of_persons[0]))
    print("np.shape(list_of_data):" , np.shape(list_of_data))
    
    #print(model._baum_welch_summarize())
    model.plot()
    print("dense_transition_matrix:" , model.dense_transition_matrix())
    print("edge_count:" , model.edge_count())
    print("edges:" , model.edges)
    print("name:" , model.name)
    print("state_count:" , model.state_count())
    #print("summarize:" , model.summarize())
    print(model.thaw())
    
    
def create_casas7_markov_chain(file_address , has_activity):
    '''
    create markov chain for each person separately
    train_set = 80% of data
    test_set = 20% of data
    '''
    
    if has_activity:
        list_of_data , list_of_persons , _ = read_sequence_based_CSV_file_with_activity(file_address = file_address, 
                                                                      has_header = True, 
                                                                      separate_data_based_on_persons = True)
    else:
        list_of_data , list_of_persons = read_sequence_based_CSV_file_without_activity(file_address = file_address, 
                                                                      has_header = True, 
                                                                      separate_data_based_on_persons = True)

    
    
    number_of_persons = len(list_of_data)
    
    #create list of Person IDs
    person_IDs = np.zeros(number_of_persons , dtype = int)
    for i in range(number_of_persons):
        person_IDs[i] = list_of_persons[i][0]
    
    #split dataset to 80% for train and 20% for test
    train_set = np.zeros(number_of_persons , dtype = np.ndarray)
    test_set = np.zeros(number_of_persons , dtype = np.ndarray)
    
    
    for per in range(number_of_persons):
        number_of_train_set_data = int( 0.8 * len(list_of_data[per]))
        train_set[per] = list_of_data[per][0:number_of_train_set_data]
        test_set[per] = list_of_data[per][number_of_train_set_data:]
    
    
    #train models
    list_of_markov_chain_models = np.zeros(number_of_persons , dtype = MarkovChain)
    for per in range(number_of_persons):
        list_of_markov_chain_models[per] = MarkovChain.from_samples(X = train_set[per])
        #print("Person:" , per)
        #print(list_of_markov_chain_models[per].distributions)
     
    #create actual labels and concatenate 
    actual_labels = np.zeros(0 , dtype = int)
    final_test_set = np.zeros(0 , dtype = np.ndarray)
    #print(len(actual_labels))
    for per in range(number_of_persons):
        l = list( [ person_IDs[per] ] * len(test_set[per]))
        #print(l)
        actual_labels = np.concatenate((actual_labels , l) , axis = 0)
        final_test_set = np.concatenate((final_test_set , test_set[per]) , axis = 0)
   
    #test
    predicted_labels = np.zeros_like(actual_labels)
    logp = np.zeros(number_of_persons)
    number_of_exceptions = 0
    for seq, index in zip(final_test_set , range(len(final_test_set))):
        #print("seq:" , seq)
        for model , index_of_logp in zip(list_of_markov_chain_models , range(number_of_persons)):
            try:
                logp[index_of_logp] = model.log_probability(seq)
            except KeyError:
                number_of_exceptions +=1
                #print("number of model:" , index_of_logp)
                #print("seq:" , seq)
                logp[index_of_logp] = -sys.maxsize + 1
        arg_max_logp = np.argmax(logp)# return the max index
        predicted_labels[index] = person_IDs[arg_max_logp]
    
    ind = np.where(np.not_equal(predicted_labels , -sys.maxsize + 1 ))
    print("len(final_test_set):",len(final_test_set) , "number_of_exceptions:" , number_of_exceptions)
    #print(len(predicted_labels) , len(ind[0]))
    predicted_labels = predicted_labels[ind]
    actual_labels = actual_labels[ind]
    scores = calculate_different_metrics(actual_labels , predicted_labels)
    
    #print(scores)
    return scores
    '''
    model.plot()
    print("dense_transition_matrix:" , model.dense_transition_matrix())
    print("edge_count:" , model.edge_count())
    print("edges:" , model.edges)
    print("name:" , model.name)
    print("state_count:" , model.state_count())
    #print("summarize:" , model.summarize())
    print(model.thaw())
    '''
def create_casas7_markov_chain_with_prepared_train_and_test(train_set, list_of_persons_in_train , test_set , list_of_persons_in_test, return_predicted_lables = False, remove_test_columns_which_are_not_in_final_model = False):
    '''
    create markov chain for each person separately
    train_set = an ndarray that has train_set for each person separately
    test_set = 
    '''
        
    number_of_persons = len(train_set)
    #print("list_of_persons_in_train:", list_of_persons_in_train)

    #create list of Person IDs
    person_IDs = np.zeros(number_of_persons , dtype = int)
    for i in range(number_of_persons):
        person_IDs[i] = list_of_persons_in_train[i][0]
    
    #train models
    list_of_markov_chain_models = np.zeros(number_of_persons , dtype = MarkovChain)
    for per in range(number_of_persons):
        list_of_markov_chain_models[per] = MarkovChain.from_samples(X = train_set[per])
        #print(list_of_markov_chain_models[per].distributions)
     
    #create actual labels and concatenate 
    actual_labels = np.zeros(0 , dtype = int)
    final_test_set = np.zeros(0 , dtype = np.ndarray)
    #print(len(actual_labels))
    for per in range(number_of_persons):
        l = list( [ person_IDs[per] ] * len(test_set[per]))
        #print(l)
        actual_labels = np.concatenate((actual_labels , l) , axis = 0)
        final_test_set = np.concatenate((final_test_set , test_set[per]) , axis = 0)
   
    #test
    predicted_labels = np.zeros_like(actual_labels)
    logp = np.zeros(number_of_persons)
    number_of_exceptions = 0
    
    
    if remove_test_columns_which_are_not_in_final_model:
        new_final_test_set = remove_extra_test_columns(final_test_set , list_of_markov_chain_models)
        
        #get_set_of_Markov_chain_nodes(list_of_markov_chain_models[0])
    else:
        new_final_test_set = np.zeros(number_of_persons, dtype = np.ndarray)
        for i in range(number_of_persons):
            new_final_test_set[i] = np.ndarray(shape = (len(final_test_set) , ), dtype = np.ndarray )
            new_final_test_set[i] = final_test_set
    
    #print("len(predicted_labels):", len(predicted_labels))
# bayad predicted_labels jadid shavad bar asase in ke alan new_final_test_set daram 
    #for seq, index in zip(new_final_test_set[index_of_logp] , range(len(new_final_test_set[index_of_logp]))):
    #for each model, a spareated test_set is created
   
   # for i in range(len(new_final_test_set[0])):
    #    print(len(new_final_test_set[0]))
    
    for index in range(len(new_final_test_set[0])):
       
        for model , index_of_logp in zip(list_of_markov_chain_models , range(number_of_persons)):
            #if index == 0:
            #print("seq:" , seq)
            seq = new_final_test_set[index_of_logp][index]
            try:
                logp[index_of_logp] = model.log_probability(seq)
            except KeyError:
                number_of_exceptions +=1
                logp[index_of_logp] = -sys.maxsize + 1
        arg_max_logp = np.argmax(logp)# return the max index
        predicted_labels[index] = person_IDs[arg_max_logp]
    
    ind = np.where(np.not_equal(predicted_labels , -sys.maxsize + 1 ))
    #print("len(final_test_set):",len(final_test_set) , "number_of_exceptions:" , number_of_exceptions)
    #print(len(predicted_labels) , len(ind[0]))
    predicted_labels = predicted_labels[ind]
    actual_labels = actual_labels[ind]
    scores = calculate_different_metrics(actual_labels , predicted_labels)
    #for i in range(len(actual_labels)):
     #   print(actual_labels[i] , predicted_labels[i])
    #print("scores:" , scores)
    if return_predicted_lables:
        return scores, predicted_labels
    else:
        return scores

def create_casas7_HMM_with_prepared_train_and_test_based_on_seq_of_activities(train_set, list_of_persons_in_train , test_set , list_of_persons_in_test):
    '''
    create a single HMM for all of persons
    train_set = an ndarray that has train_set for each person separately
    test_set = 
    '''
    
    #concatinate train_sets and test_sets of all of people    
    number_of_persons = len(train_set)
    final_train_set = train_set[0]
    final_test_set = test_set[0]
    final_train_set_labels = list_of_persons_in_train[0]
    final_test_set_labels = list_of_persons_in_test[0]
    #print(type(final_train_set) , type(train_set) , type(train_set[1]))
    for per in range(1, number_of_persons):
        final_train_set = np.concatenate((final_train_set , train_set[per]) , axis = 0)
        final_test_set = np.concatenate((final_test_set , test_set[per]), axis = 0)
        final_train_set_labels = np.concatenate((final_train_set_labels , list_of_persons_in_train[per]), axis = 0)
        final_test_set_labels = np.concatenate((final_test_set_labels , list_of_persons_in_test[per]), axis = 0)
    
    #r = np.shape(final_train_set)
    #for i in range(r[0]):
    #    print(np.shape(final_train_set[i]))
    #final_train_set = np.array([[1,2,3,0,0] , [1,2,0,0,0]], dtype = np.ndarray)
    #final_train_set_labels = np.array([1,2] , dtype= np.ndarray)
    print(type(final_train_set[11]) , np.shape(final_train_set[11]))
    print(final_train_set[0:2])
    model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=2, X=final_train_set , labels = final_train_set_labels , algorithm = 'labeled')# according to my tests :D n_components is number of hidden states
    print(model)
    #return 0
    #test
    '''predicted_labels = np.zeros_like(actual_labels)
    logp = np.zeros(number_of_persons)
    number_of_exceptions = 0
    for seq, index in zip(final_test_set , range(len(final_test_set))):
        #print("seq:" , seq)
        for model , index_of_logp in zip(list_of_markov_chain_models , range(number_of_persons)):
            try:
                logp[index_of_logp] = model.log_probability(seq)
            except KeyError:
                number_of_exceptions +=1
                #print("number of model:" , index_of_logp)
                #print("seq:" , seq)
                logp[index_of_logp] = -sys.maxsize + 1
        arg_max_logp = np.argmax(logp)# return the max index
        predicted_labels[index] = person_IDs[arg_max_logp]
    
    ind = np.where(np.not_equal(predicted_labels , -sys.maxsize + 1 ))
    print("len(final_test_set):",len(final_test_set) , "number_of_exceptions:" , number_of_exceptions)
    #print(len(predicted_labels) , len(ind[0]))
    predicted_labels = predicted_labels[ind]
    actual_labels = actual_labels[ind]
    scores = calculate_different_metrics(actual_labels , predicted_labels)
    
    return scores
    '''

def prepare_train_and_test_based_on_a_list_of_k_data(list_of_data ,list_of_labels ,which_index_for_test):
    '''
    get a list_of_data with k group of data
    return the 'which_index_for_test' th group of data as test_set 
    and concat others as train_set
    
    '''
    test_set = list_of_data[which_index_for_test]
    test_set_labels = list_of_labels[which_index_for_test]
    
    list_of_data = np.delete(list_of_data , which_index_for_test , axis = 0)
    list_of_labels = np.delete(list_of_labels , which_index_for_test , axis = 0)
    
    train_set = list_of_data[0]
    train_set_labels = list_of_labels[0]
    
    for i in range(1 , len(list_of_data)):
        train_set = np.concatenate((train_set , list_of_data[i]) , axis = 0)
        train_set_labels = np.concatenate((train_set_labels , list_of_labels[i]) , axis = 0)
       
        
    return train_set , train_set_labels , test_set , test_set_labels
    
    
def test_prepare_train_and_test_based_on_a_list_of_k_data():
    
    data = np.zeros(10 , dtype = np.ndarray)
    labels = np.zeros(10 , dtype = np.ndarray)
    for i in range(10):
        data[i] = np.full(shape = (2,2) , fill_value = i , dtype = int)
        labels[i] = np.full(shape = (2,1) , fill_value = i , dtype = int)
    #print(data)
    #print(labels)
    
    train_set , train_set_labels , test_set , test_set_labels = prepare_train_and_test_based_on_a_list_of_k_data(data , labels , 5)
    
def calculate_f1_scoreaverage(scores , k):
    avg = 0
    for i in range(k):
        avg += scores[i]['f1_score_micro']
    
    avg /= k
    
    return avg
    
    
def select_the_best_delta_using_the_best_strategy_markov_chain(k=10 , shuffle = True , type_of_feature_vector = 0, string_of_dataset = 'Twor2009'):
    '''
	Parameters:
	==========
	type_of_feature_vector: the index of types
	'''

    address_to_read = r"E:\pgmpy\{dataset}\{t}\delta_{delta}min.csv"
    types = ['Seq of sensor events_no overlap_based on different deltas' , 
             'Seq of sensor events_based_on_activity_and_no_overlap_delta', 
             'Seq of Bag of sensor events_based_on_activity_and_no_overlap_delta',
			 'Seq of sensor events_based on activities']
			 
    #when use 'Seq of sensor events_based on activities', you should use this address_to_read
    if type_of_feature_vector == 3:
        address_to_read = r"E:\pgmpy\{dataset}\{t}\based_on_activities.csv"
    
    deltas = list(range(1,15))+[15,30,45,60,75,90,100, 120,150, 180,200,240,300,400,500,600,700,800,900,1000]
    #deltas = [150, 180,200,240,300,400,500,600,700,800,900,1000]
    #deltas = range(1100,5001,100) 
    t = types[type_of_feature_vector]
    print(t)
    print(string_of_dataset)
    best_score = 0
    best_delta = 0
    best_train_set = 0
    best_test_set = 0
    best_train_set_person_labels = 0
    best_test_set_person_labels = 0
    
    list_of_deltas = []
    list_of_f1_micros = []
    
    for d in deltas:
        if type_of_feature_vector != 3:
            print("delta:" , d)
        if t == types[0]:
            list_of_data , list_of_persons = read_sequence_based_CSV_file_without_activity(
                                        file_address = address_to_read.format(dataset = string_of_dataset, t = t, delta= d), 
                                        has_header = True, 
                                        separate_data_based_on_persons = True)
        
        elif t == types[1]:
            list_of_data , list_of_persons , _ = read_sequence_based_CSV_file_with_activity(file_address = address_to_read.format(dataset = string_of_dataset, t = t , delta= d), 
                                        has_header = True, separate_data_based_on_persons = True)
        
        elif t == types[2]:
            list_of_data , list_of_persons , _ = read_sequence_of_bags_CSV_file_with_activity(file_address = address_to_read.format(dataset = string_of_dataset, t = t , delta= d), 
                                        has_header = True, separate_data_based_on_persons = True)
           
        elif t == types[3]:
            if d != deltas[0]:
                print("break")
                break
            list_of_data , list_of_persons , _ = read_sequence_based_CSV_file_with_activity(file_address = address_to_read.format(dataset = string_of_dataset, t = t), 
                                        has_header = True, separate_data_based_on_persons = True)
          
        number_of_persons = len(list_of_data)
        train_set = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set = np.zeros(number_of_persons , dtype = np.ndarray)
        train_set_person_labels = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set_person_labels = np.zeros(number_of_persons , dtype = np.ndarray)

        #I think 2 is number of persons 
        k_splitted_train_set = np.ndarray(shape = (number_of_persons , k) , dtype = np.ndarray)
        k_splitted_train_set_person_labels = np.ndarray(shape = (number_of_persons , k) , dtype = np.ndarray)

        for per in range(number_of_persons):
            
            if shuffle:
                list_of_data[per] , list_of_persons[per] = unison_shuffled_copies(list_of_data[per] , list_of_persons[per])
                
            number_of_train_set_data = int( 0.8 * len(list_of_data[per]))
            #print("number_of_train_set_data:" , number_of_train_set_data)
            train_set[per] = list_of_data[per][0:number_of_train_set_data]
            train_set_person_labels[per] = list_of_persons[per][0:number_of_train_set_data]
            test_set[per] = list_of_data[per][number_of_train_set_data:]
            test_set_person_labels[per] = list_of_persons[per][number_of_train_set_data:]

            
            #split both train_set and test_set to k=10 groups
            #print("len(train_set[per]):" , len(train_set[per]) , "number_of_train_set_data:" , number_of_train_set_data)
            number_of_each_group_of_data = int(len(train_set[per]) / k)
            
            start = 0
            for i in range(k-1):
                end = (i+1) * number_of_each_group_of_data
                k_splitted_train_set[per][i] = train_set[per][start:end]
                k_splitted_train_set_person_labels[per][i] =  train_set_person_labels [per][start:end]
                start = end
            k_splitted_train_set[per][k-1] = train_set[per][start:]
            k_splitted_train_set_person_labels[per][k-1] = train_set_person_labels [per][start:]
               

        train_set_k = np.zeros(number_of_persons , dtype = np.ndarray)
        train_set_labels_k = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set_k = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set_labels_k = np.zeros(number_of_persons , dtype = np.ndarray)
        scores = np.zeros(k , dtype = dict)
        for i in range(k):
            for per in range(number_of_persons):
                train_set_k[per] , train_set_labels_k[per] , test_set_k[per] , test_set_labels_k[per] = prepare_train_and_test_based_on_a_list_of_k_data(k_splitted_train_set[per] , k_splitted_train_set_person_labels[per] , i)
            
            #print("#######i:" , i)
            #print("train_set_labels_k:", len(train_set_labels_k[0]))#.shape)#tolist())
            #print("test_set_labels_k:", set(test_set_labels_k.tolist()))
            scores[i] = create_casas7_markov_chain_with_prepared_train_and_test(train_set = train_set_k, 
                                                                                list_of_persons_in_train=train_set_labels_k, 
                                                                                test_set=test_set_k, 
                                                                                list_of_persons_in_test=test_set_labels_k)
        scores_avg = calculate_f1_scoreaverage(scores , k)
        print("scores_avg" , scores_avg)
        
        list_of_deltas.append(d)
        list_of_f1_micros.append(scores_avg)
        
        if scores_avg > best_score:
            best_score = scores_avg
            best_delta = d
            best_train_set = train_set
            best_test_set = test_set
            best_train_set_person_labels = train_set_person_labels
            best_test_set_person_labels = test_set_person_labels
    
    print("Validation Scores:")
    if type_of_feature_vector == 3:
        print("best_validation_score:" , best_score)

    else:
        print("best_delta:" , best_delta , "best_validation_score:" , best_score)
    
    test_score =  create_casas7_markov_chain_with_prepared_train_and_test(train_set = best_train_set , list_of_persons_in_train= best_train_set_person_labels, test_set= best_test_set , list_of_persons_in_test= best_test_set_person_labels)
    print("test_score:" , test_score)
    
    #plot_results(list_of_deltas, list_of_f1_micros, t , y_label = "f1 score micro")


def select_the_best_delta_using_the_best_strategy_HMM_cherknevis(k=10 , shuffle = True):
    '''
    please read the code later.
    I combined multiple strategies on multiple dataset. it is dangerous to modify it :D
    '''

    address_to_read = r"E:\pgmpy\Seq of Bag of sensor events_based_on_activity_and_no_overlap_delta\delta_{delta}min.csv"
    
    deltas = [15,30,45,60,75,90,100, 120,150, 180,200,240,300,400,500,600,700,800,900,1000]
    
    best_score = 0
    best_delta = 0
    best_train_set = 0
    best_test_set = 0
    best_train_set_person_labels = 0
    best_test_set_person_labels = 0
    
    list_of_deltas = []
    list_of_f1_micros = []
    
    for d in deltas:
        print("delta:" , d)
       
        list_of_data , list_of_persons , _ = read_sequence_of_bags_CSV_file_with_activity(file_address = address_to_read.format(delta= d), 
                                        has_header = True, separate_data_based_on_persons = True)
           
        
        
        number_of_persons = len(list_of_data)
        train_set = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set = np.zeros(number_of_persons , dtype = np.ndarray)
        train_set_person_labels = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set_person_labels = np.zeros(number_of_persons , dtype = np.ndarray)


        k_splitted_train_set = np.ndarray(shape = (2 , k) , dtype = np.ndarray)
        k_splitted_train_set_person_labels = np.ndarray(shape = (2 , k) , dtype = np.ndarray)

        for per in range(number_of_persons):
            
            if shuffle:
                list_of_data[per] , list_of_persons[per] = unison_shuffled_copies(list_of_data[per] , list_of_persons[per])
            
            # repeat person tags
            #is_one_person = true because i just send the data of one person in each iteration
            list_of_persons[per] = repaet_person_tags_as_much_as_seq_length(list_of_data[per] , list_of_persons[per] , is_one_person=True)
            
            number_of_train_set_data = int( 0.8 * len(list_of_data[per]))
            train_set[per] = list_of_data[per][0:number_of_train_set_data]
            train_set_person_labels[per] = list_of_persons[per][0:number_of_train_set_data]
            test_set[per] = list_of_data[per][number_of_train_set_data:]
            test_set_person_labels[per] = list_of_persons[per][number_of_train_set_data:]

            
            #split both train_set and test_set to k=10 groups
            print("len(train_set[per]):" , len(train_set[per]) , "number_of_train_set_data:" , number_of_train_set_data)
            number_of_each_group_of_data = int(len(train_set[per]) / k)
            
            start = 0
            for i in range(k-1):
                end = (i+1) * number_of_each_group_of_data
                k_splitted_train_set[per][i] = train_set[per][start:end]
                k_splitted_train_set_person_labels[per][i] =  train_set_person_labels [per][start:end]
                start = end
            k_splitted_train_set[per][k-1] = train_set[per][start:]
            k_splitted_train_set_person_labels[per][k-1] = train_set_person_labels [per][start:]
               

        train_set_k = np.zeros(number_of_persons , dtype = np.ndarray)
        train_set_labels_k = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set_k = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set_labels_k = np.zeros(number_of_persons , dtype = np.ndarray)
        scores = np.zeros(k , dtype = dict)
        
        for i in range(k):
            for per in range(number_of_persons):
                train_set_k[per] , train_set_labels_k[per] , test_set_k[per] , test_set_labels_k[per] = prepare_train_and_test_based_on_a_list_of_k_data(k_splitted_train_set[per] , k_splitted_train_set_person_labels[per] , i)
            
            scores[i] = create_casas7_HMM_with_prepared_train_and_test_based_on_seq_of_activities(train_set = train_set_k , list_of_persons_in_train=train_set_labels_k , test_set=test_set_k , list_of_persons_in_test=test_set_labels_k)
       
        scores_avg = calculate_f1_scoreaverage(scores , k)
        print("scores_avg" , scores_avg)
        
        list_of_deltas.append(d)
        list_of_f1_micros.append(scores_avg)
        
        if scores_avg > best_score:
            best_score = scores_avg
            best_delta = d
            best_train_set = train_set
            best_test_set = test_set
            best_train_set_person_labels = train_set_person_labels
            best_test_set_person_labels = test_set_person_labels
    
    print("Validation Scores:")
    print("best_delta:" , best_delta , "best_validation_score:" , best_score)
    
    #test_score =  create_casas7_HMM_with_prepared_train_and_test_based_on_seq_of_activities(train_set = best_train_set , list_of_persons_in_train= best_train_set_person_labels, test_set= best_test_set , list_of_persons_in_test= best_test_set_person_labels)
    #print("test_score:" , test_score)
    
    plot_results(list_of_deltas, list_of_f1_micros, 'Seq of Bag of events based on activities', y_label = "f1 score micro")


def select_the_best_delta_using_the_best_strategy_HMM(k=10 , shuffle = True):
    '''
    '''

    address_to_read = r"E:\pgmpy\Seq of sensor events_based_on_activity_and_no_overlap_delta\delta_{delta}min.csv"
    
    deltas = [15,30,45,60,75,90,100, 120,150, 180,200,240,300,400,500,600,700,800,900,1000]
    
    best_score = 0
    best_delta = 0
    best_train_set = 0
    best_test_set = 0
    best_train_set_person_labels = 0
    best_test_set_person_labels = 0
    
    list_of_deltas = []
    list_of_f1_micros = []
    
    for d in deltas:
        print("delta:" , d)
       
        list_of_data , list_of_persons , _ = read_sequence_based_CSV_file_with_activity(file_address = address_to_read.format(delta= d) , has_header = True, separate_data_based_on_persons = True)
        
        number_of_persons = len(list_of_data)
        train_set = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set = np.zeros(number_of_persons , dtype = np.ndarray)
        train_set_person_labels = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set_person_labels = np.zeros(number_of_persons , dtype = np.ndarray)


        k_splitted_train_set = np.ndarray(shape = (2 , k) , dtype = np.ndarray)
        k_splitted_train_set_person_labels = np.ndarray(shape = (2 , k) , dtype = np.ndarray)

        for per in range(number_of_persons):
            
            if shuffle:
                list_of_data[per] , list_of_persons[per] = unison_shuffled_copies(list_of_data[per] , list_of_persons[per])
            
            # repeat person tags
            #is_one_person = true because i just send the data of one person in each iteration
            list_of_persons[per] = repaet_person_tags_as_much_as_seq_length(list_of_data[per] , list_of_persons[per] , is_one_person=True)
            
            number_of_train_set_data = int( 0.8 * len(list_of_data[per]))
            train_set[per] = list_of_data[per][0:number_of_train_set_data]
            train_set_person_labels[per] = list_of_persons[per][0:number_of_train_set_data]
            test_set[per] = list_of_data[per][number_of_train_set_data:]
            test_set_person_labels[per] = list_of_persons[per][number_of_train_set_data:]

            
            #split both train_set and test_set to k=10 groups
            print("len(train_set[per]):" , len(train_set[per]) , "number_of_train_set_data:" , number_of_train_set_data)
            number_of_each_group_of_data = int(len(train_set[per]) / k)
            
            start = 0
            for i in range(k-1):
                end = (i+1) * number_of_each_group_of_data
                k_splitted_train_set[per][i] = train_set[per][start:end]
                k_splitted_train_set_person_labels[per][i] =  train_set_person_labels [per][start:end]
                start = end
            k_splitted_train_set[per][k-1] = train_set[per][start:]
            k_splitted_train_set_person_labels[per][k-1] = train_set_person_labels [per][start:]
               

        train_set_k = np.zeros(number_of_persons , dtype = np.ndarray)
        train_set_labels_k = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set_k = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set_labels_k = np.zeros(number_of_persons , dtype = np.ndarray)
        scores = np.zeros(k , dtype = dict)
        
        for i in range(k):
            for per in range(number_of_persons):
                train_set_k[per] , train_set_labels_k[per] , test_set_k[per] , test_set_labels_k[per] = prepare_train_and_test_based_on_a_list_of_k_data(k_splitted_train_set[per] , k_splitted_train_set_person_labels[per] , i)
            
            scores[i] = create_casas7_HMM_with_prepared_train_and_test_based_on_seq_of_activities(train_set = train_set_k , list_of_persons_in_train=train_set_labels_k , test_set=test_set_k , list_of_persons_in_test=test_set_labels_k)
       
        print("**************************\nscores:", scores)
        scores_avg = calculate_f1_scoreaverage(scores , k)
        print("scores_avg" , scores_avg)
        
        list_of_deltas.append(d)
        list_of_f1_micros.append(scores_avg)
        
        if scores_avg > best_score:
            best_score = scores_avg
            best_delta = d
            best_train_set = train_set
            best_test_set = test_set
            best_train_set_person_labels = train_set_person_labels
            best_test_set_person_labels = test_set_person_labels
    
    print("Validation Scores:")
    print("best_delta:" , best_delta , "best_validation_score:" , best_score)
    
    #test_score =  create_casas7_HMM_with_prepared_train_and_test_based_on_seq_of_activities(train_set = best_train_set , list_of_persons_in_train= best_train_set_person_labels, test_set= best_test_set , list_of_persons_in_test= best_test_set_person_labels)
    #print("test_score:" , test_score)
    
    plot_results(list_of_deltas, list_of_f1_micros, 'Seq of events based on activities and no overlap delta', y_label = "f1 score micro")



def create_and_test_model_based_on_activities(shuffle = True , add_str_to_path = ''):
    

    address_to_read = r"E:\pgmpy\{path}\Seq of sensor events_based on activities\based_on_activities.csv".format(path = add_str_to_path)
    print("Seq of sensor events_based on activities")

    list_of_data , list_of_persons , _ = read_sequence_based_CSV_file_with_activity(file_address = address_to_read, 
                                        has_header = True, separate_data_based_on_persons = True)
        
           
        
    number_of_persons = len(list_of_data)
    train_set = np.zeros(number_of_persons , dtype = np.ndarray)
    test_set = np.zeros(number_of_persons , dtype = np.ndarray)
    train_set_person_labels = np.zeros(number_of_persons , dtype = np.ndarray)
    test_set_person_labels = np.zeros(number_of_persons , dtype = np.ndarray)


    for per in range(number_of_persons):
        
        if shuffle:
            list_of_data[per] , list_of_persons[per] = unison_shuffled_copies(list_of_data[per] , list_of_persons[per])
     
        number_of_train_set_data = int( 0.8 * len(list_of_data[per]))
        train_set[per] = list_of_data[per][0:number_of_train_set_data]
        train_set_person_labels[per] = list_of_persons[per][0:number_of_train_set_data]
        test_set[per] = list_of_data[per][number_of_train_set_data:]
        test_set_person_labels[per] = list_of_persons[per][number_of_train_set_data:]
 
     
    test_score =  create_casas7_markov_chain_with_prepared_train_and_test(train_set = train_set , list_of_persons_in_train= train_set_person_labels, test_set= test_set , list_of_persons_in_test= test_set_person_labels)
    print("test_score:" , test_score)
   


if __name__ == "__main__":
    
    address_to_read= r"E:\pgmpy\Seq of sensor events_based on activities\based_on_activities.csv"

    #address_to_read = r"E:\pgmpy\separation of train and test\31_3\{type}\train\delta_{delta}min.csv"
    types = ['Seq of sensor events_no overlap_based on different deltas' , 'Seq of sensor events_based_on_activity_and_no_overlap_delta']
    #create_casas7_markov_chain(file_address=address_to_read.format(delta = 15) , has_activity=True)
    #create_casas7_markov_chain(file_address=address_to_read , has_activity=True)
    
    select_the_best_delta_using_the_best_strategy_markov_chain(k = 10, 
                                                               shuffle=True, 
                                                               type_of_feature_vector = 0, 
                                                               string_of_dataset= 'OpenSHS3_30days')
    
	#create_and_test_model_based_on_activities(True , 'Tulum2009')
	
	#select_the_best_delta_using_the_best_strategy_HMM(k= 10, shuffle = True)