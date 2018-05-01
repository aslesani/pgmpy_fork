'''
Created on April 1, 2018

@author: Adele
'''
from pomegranate import HiddenMarkovModel
from pomegranate import NormalDistribution, State, DiscreteDistribution
from pomegranate import MarkovChain
import numpy as np
from DimensionReductionandBNStructureLearning import read_data_from_CSV_file
from dataPreparation import create_sequence_of_sensor_events_based_on_activity
from read_write import read_sequence_based_CSV_file_with_activity , read_sequence_based_CSV_file_without_activity
from snowballstemmer import algorithms

from Validation import calculate_different_metrics
import sys

def test_sample_from_site():
    
    dists = [NormalDistribution(5, 1), NormalDistribution(1, 7), NormalDistribution(8,2)]
    trans_mat = numpy.array([[0.7, 0.3, 0.0],
                             [0.0, 0.8, 0.2],
                             [0.0, 0.0, 0.9]])
    starts = numpy.array([1.0, 0.0, 0.0])
    ends = numpy.array([0.0, 0.0, 0.1])
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

    data = read_data_from_CSV_file(dest_file = file_address, data_type = np.int ,  has_header = False , return_as_pandas_data_frame = False )
    data = np.delete(data , 2, 1)
    data = np.delete(data , 2, 1)
    data = np.delete(data , 0, 1)
    data = np.delete(data , 0, 1)
    data = np.delete(data , 0, 1)
    print(np.shape(data))


    #data = create_sequence_of_sensor_events_based_on_activity(address_to_read = file_address, has_header = False, address_for_save = " ", isSave = False)#read_data_from_CSV_file(dest_file = file_address, data_type = numpy.int ,  has_header = False , return_as_pandas_data_frame = False )
    model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components=3, X=data)# according to my tests :D n_components is number of hidden states
    
    #print(model)
    #print(model._baum_welch_summarize())
    #model.plot()
    print("dense_transition_matrix:" , model.dense_transition_matrix())
    print("edge_count:" , model.edge_count())
    print("edges:" , model.edges)
    print("name:" , model.name)
    print("state_count:" , model.state_count())
    print(model)
    #print(model.)
    #print("summarize:" , model.summarize())
    #print(model.thaw())
    
    
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
        pass
    print(model)
    
    #print((list_of_persons[0]))
    print("np.shape(list_of_data):" , numpy.shape(list_of_data))
    
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
    print("number_of_exceptions:" , number_of_exceptions)
    #print(len(predicted_labels) , len(ind[0]))
    predicted_labels = predicted_labels[ind]
    actual_labels = actual_labels[ind]
    scores = calculate_different_metrics(actual_labels , predicted_labels)
    print("len(final_test_set):",len(final_test_set))
    print(scores)
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
def create_casas7_markov_chain_with_prepared_train_and_test(train_set, list_of_persons_in_train , test_set , list_of_persons_in_test):
    '''
    create markov chain for each person separately
    train_set = an ndarray that has train_set for each person separately
    test_set = 
    '''
        
    number_of_persons = len(train_set)
    
    #create list of Person IDs
    person_IDs = np.zeros(number_of_persons , dtype = int)
    for i in range(number_of_persons):
        person_IDs[i] = list_of_persons_in_train[i][0]
    
    
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
    print("number_of_exceptions:" , number_of_exceptions)
    #print(len(predicted_labels) , len(ind[0]))
    predicted_labels = predicted_labels[ind]
    actual_labels = actual_labels[ind]
    scores = calculate_different_metrics(actual_labels , predicted_labels)
    print("len(final_test_set):",len(final_test_set))
    print(scores)
    return scores

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
        
    '''
    print(train_set)
    print(train_set_labels)
    print(test_set)
    print(test_set_labels)
    '''
    
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
    
    
def select_the_best_delta_using_the_best_strategy(k=10):
    

    address_to_read = r"E:\pgmpy\separation of train and test\31_3\{t}\train\delta_{delta}min.csv"
    types = ['Seq of sensor events_no overlap_based on different deltas' , 'Seq of sensor events_based_on_activity_and_no_overlap_delta']

    deltas = [15,30,45,60,75,90,100, 120,150, 180,200,240,300,400,500,600,700,800,900,1000]
    
    t = types[0]
    print(t)
    best_score = 0
    best_delta = 0
    best_train_set = 0
    best_test_set = 0
    best_train_set_person_labels = 0
    best_test_set_person_labels = 0
    
    for d in deltas:
        print("delta:" , d)
        if t == types[0]:
            list_of_data , list_of_persons = read_sequence_based_CSV_file_without_activity(file_address = address_to_read.format(t = t , delta= d), 
                                        has_header = True, separate_data_based_on_persons = True)
        
        elif t == types[1]:
            list_of_data , list_of_persons , _ = read_sequence_based_CSV_file_with_activity(file_address = address_to_read.format(t = t , delta= d), 
                                        has_header = True, separate_data_based_on_persons = True)
        
            
        number_of_persons = len(list_of_data)
        train_set = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set = np.zeros(number_of_persons , dtype = np.ndarray)
        train_set_person_labels = np.zeros(number_of_persons , dtype = np.ndarray)
        test_set_person_labels = np.zeros(number_of_persons , dtype = np.ndarray)


        k_splitted_train_set = np.ndarray(shape = (2 , k) , dtype = np.ndarray)
        k_splitted_train_set_person_labels = np.ndarray(shape = (2 , k) , dtype = np.ndarray)

        for per in range(number_of_persons):
            number_of_train_set_data = int( 0.8 * len(list_of_data[per]))
            train_set[per] = list_of_data[per][0:number_of_train_set_data]
            train_set_person_labels[per] = list_of_persons[per][0:number_of_train_set_data]
            test_set[per] = list_of_data[per][number_of_train_set_data:]
            test_set_person_labels[per] = list_of_persons[per][number_of_train_set_data:]

            
            #split both train_set and test_set to k=10 groups
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
            
            scores[i] = create_casas7_markov_chain_with_prepared_train_and_test(train_set = train_set_k , list_of_persons_in_train=train_set_labels_k , test_set=test_set_k , list_of_persons_in_test=test_set_labels_k)
        scores_avg = calculate_f1_scoreaverage(scores , k)
        print("scores_avg" , scores_avg)
        if scores_avg > best_score:
            best_score = scores_avg
            best_delta = d
            best_train_set = train_set
            best_test_set = test_set
            best_train_set_person_labels = train_set_person_labels
            best_test_set_person_labels = test_set_person_labels
    
    print("Validation Scores:")
    print("best_delta:" , best_delta , "best_validation_score:" , best_score)
    
    test_score =  create_casas7_markov_chain_with_prepared_train_and_test(train_set = best_train_set , list_of_persons_in_train= best_train_set_person_labels, test_set= best_test_set , list_of_persons_in_test= best_test_set_person_labels)
    print("test_score:" , test_score)


if __name__ == "__main__":
    #test_sample_from_site()
    #build_the_same_model_line_by_line()
    #build_an_hmm_example()
    dest_file = r"E:\a1.csv"#"C:\f5_0_10_no_col.csv"
    #create_hmm_from_sample(dest_file)
    #create_casas7_hmm(file_address = dest_file)
    address_to_read= r"E:\pgmpy\separation of train and test\31_3\Seq of sensor events_based on activities\train\based_on_activities.csv"

    #address_to_read = r"E:\pgmpy\separation of train and test\31_3\{type}\train\delta_{delta}min.csv"
    types = ['Seq of sensor events_no overlap_based on different deltas' , 'Seq of sensor events_based_on_activity_and_no_overlap_delta']
    #create_casas7_markov_chain(file_address=address_to_read.format(delta = 15) , has_activity=True)
    #create_casas7_markov_chain(file_address=address_to_read , has_activity=True)
    select_the_best_delta_using_the_best_strategy()
    #test_prepare_train_and_test_based_on_a_list_of_k_data()