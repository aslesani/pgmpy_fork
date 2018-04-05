'''
Created on April 1, 2018

@author: Adele
'''
from pomegranate import HiddenMarkovModel
from pomegranate import NormalDistribution, State, DiscreteDistribution
import numpy
from DimensionReductionandBNStructureLearning import read_data_from_CSV_file

def test_sample_from_site():
    
    dists = [NormalDistribution(5, 1), NormalDistribution(1, 7), NormalDistribution(8,2)]
    trans_mat = numpy.array([[0.7, 0.3, 0.0],
                             [0.0, 0.8, 0.2],
                             [0.0, 0.0, 0.9]])
    starts = numpy.array([1.0, 0.0, 0.0])
    ends = numpy.array([0.0, 0.0, 0.1])
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)
    m = HiddenMarkovModel.from_matrix()
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
    #model.plot()
    #print(model.log_probability(list('ACGACTATTCGAT')))
    
    #print(", ".join(state.name for i, state in model.viterbi(list('ACGACTATTCGAT'))[1]))

    print("forward:" , model.forward( list('ACG') ))
    #print("backward:" , model.backward( list('ACG') ))
    #print("forward_backward:" , model.forward_backward( list('ACG') ))


    #print("Viterbi:" , model.viterbi( list('ACG') ))

def create_hmm_from_sample(file_address):
    
    data = read_data_from_CSV_file(dest_file = file_address, data_type = numpy.int ,  has_header = False , return_as_pandas_data_frame = False )
    model = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=data)
    print(model)
    #print(model._baum_welch_summarize())
    model.plot()
    print("dense_transition_matrix:" , model.dense_transition_matrix())
    print("edge_count:" , model.edge_count())
    print("edges:" , model.edges)
    print("name:" , model.name)
    print("state_count:" , model.state_count())
    #print("summarize:" , model.summarize())
    print(model.thaw())
    
    
if __name__ == "__main__":
    #test_sample_from_site()
    #build_the_same_model_line_by_line()
    build_an_hmm_example()
    dest_file = r"C:\f5_0_10_no_col.csv"
    #create_hmm_from_sample(dest_file)

    