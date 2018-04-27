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
from read_write import read_sequence_based_CSV_file
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
    
    
def create_casas7_hmm(file_address):
     
    list_of_data , list_of_persons , _ = read_sequence_based_CSV_file(file_address = file_address, has_header = True , separate_data_based_on_persons = False)
    #print(list_of_persons[0])
    #print(numpy.shape((list_of_persons)))
    #list_of_persons2 = [i for i in list_of_persons if i ==1]
    #print((list_of_persons2))
    
    
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
    
    
def create_casas7_markov_chain(file_address):
    '''
    create markov chain for each person separately
    '''
    
    list_of_data , list_of_persons , _ = read_sequence_based_CSV_file(file_address = file_address, 
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
        print("Person:" , per)
        print(list_of_markov_chain_models[per].distributions)
     
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
                print("number of model:" , index_of_logp)
                print("seq:" , seq)
                logp[index_of_logp] = -sys.maxsize + 1
        arg_max_logp = np.argmax(logp)# return the max index
        predicted_labels[index] = person_IDs[arg_max_logp]
    
    ind = np.where(np.not_equal(predicted_labels , -sys.maxsize + 1 ))
    print("number_of_exceptions:" , number_of_exceptions)
    print(len(predicted_labels) , len(ind[0]))
    predicted_labels = predicted_labels[ind]
    actual_labels = actual_labels[ind]
    scores = calculate_different_metrics(actual_labels , predicted_labels)
    print(scores)
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
    
if __name__ == "__main__":
    #test_sample_from_site()
    #build_the_same_model_line_by_line()
    #build_an_hmm_example()
    dest_file = r"E:\a1.csv"#"C:\f5_0_10_no_col.csv"
    #create_hmm_from_sample(dest_file)
    #create_casas7_hmm(file_address = dest_file)
    create_casas7_markov_chain(file_address=dest_file)

    