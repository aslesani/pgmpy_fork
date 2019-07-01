'''
Created on May 1, 2019

@author: Adele
'''

from read_write import read_data_from_CSV_file, separate_dataset_based_on_persons
from read_write import data_preparation_for_sequences_based_deep_models
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from plot_paper_results import plot_some_plots


def create_fixed_size_seq_from_bag_of_events(address_to_read, len_of_each_seq, has_activity):

    data = read_data_from_CSV_file(dest_file = address_to_read, data_type=int , has_header=True)
    rows , cols = data.shape
    #print("all of data shape:" , rows, cols)
    
    if has_activity:
        list_of_persons = data[:, -2]
        list_of_activities=data[:,-1]
        data = data[:,0:cols-2] 
    else:
        list_of_persons = data[:, -1]
        list_of_activities=[]
        data = data[:,0:cols-1]

    # it is important to say that if I use Seq of Bag of sensor events_based_on_activity_and_no_overlap_delta features
    #the length of feature vectors is not constant
    #print(list_of_persons.shape)
    if has_activity:
        data , list_of_persons, _ = separate_dataset_based_on_persons(list_of_data = data, 
                                                              list_of_persons = list_of_persons, 
                                                              list_of_activities = list_of_activities,
                                                              has_activity=has_activity)
    else:
        data , list_of_persons = separate_dataset_based_on_persons(list_of_data = data, 
                                                              list_of_persons = list_of_persons, 
                                                              list_of_activities = list_of_activities,
                                                              has_activity=has_activity)
    
    number_of_persons = list_of_persons.shape[0]
    
    for each_person in range(number_of_persons):
        rows, cols = data[each_person].shape
        #print("rows of each person before reshape", rows)
        #rows - rows % len_of_each_seq ignore extra rows of samples to create fixed lenghs sequences
        ignored_rows = rows % len_of_each_seq
        new_rows = int((rows - ignored_rows )/len_of_each_seq)
        
        #print("#####",list_of_persons[each_person].shape)

        #print("new_rows of each person:", new_rows)
        data[each_person] = data[each_person][:rows - ignored_rows].reshape(new_rows , len_of_each_seq, cols)
        list_of_persons[each_person] = list_of_persons[each_person][0:new_rows]#keep personIDs as much as samples
        #print("data[each_person].shape", data[each_person].shape)
        #print(data[each_person].shape)
        #print(list_of_persons[each_person])
        #print("************")

    for person in range(1, number_of_persons):
        #print(list_of_persons[0].shape , list_of_persons[person].shape)
        data[0] = np.concatenate((data[0], data[person]), axis=0)
        list_of_persons[0] = np.concatenate((list_of_persons[0], list_of_persons[person]), axis=0)

    print("create_fixed_size_seq_from_bag_of_events completed!")
    return data[0], list_of_persons[0]

def create_autoencoder_based_on_sequences_of_bag_of_events(list_of_data, list_of_persons, val_data, val_persons, LSTM_len = 120, embedding_vector_dim = 20):

    (number_of_samples, seq_len, number_of_features ) = list_of_data.shape
    print("number_of_samples:" , number_of_samples)
    print("LSTM_len:", LSTM_len)
    model = Sequential()
    embedding_output_dim = (5,10)
    input_dim = (seq_len, number_of_features)
    #model.add(Embedding(input_dim = number_of_features+1 , output_dim = embedding_output_dim))#(number_of_features+1, embedding_vector_dim))
    model.add(LSTM(LSTM_len, activation='relu', input_shape = (seq_len, number_of_features)))
    model.add(RepeatVector(seq_len))
    model.add(LSTM(LSTM_len, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(number_of_features)))
    
    model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
    model.summary()
    # fit model
    history = model.fit(list_of_data, list_of_data, 
                        epochs=10, batch_size=32, verbose=1, 
                        validation_data=(val_data, val_data))
    print(history.history)
    print("#######################")
    #plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
    # demonstrate recreation
    yhat = model.predict(list_of_data, verbose=1)
    print("predict results", yhat)#[0,:,0])

    return history

def create_autoencoder_based_on_sequences_of_events(list_of_data, list_of_persons, val_data, val_persons, LSTM_len = 120, embedding_vector_dim = 20, epoch = 10, batch_size = 32):

    (number_of_samples, seq_len) = list_of_data.shape
    list_of_data = list_of_data.reshape(number_of_samples, seq_len , 1)
    val_samples , val_data_seq_len = val_data.shape
    val_data = val_data.reshape(val_samples, val_data_seq_len, 1)

    print("number_of_samples:" , number_of_samples)
    print("LSTM_len:", LSTM_len)
    model = Sequential()
    #model.add(Embedding(input_dim = number_of_features+1 , output_dim = embedding_output_dim))#(number_of_features+1, embedding_vector_dim))
    model.add(LSTM(LSTM_len, activation='relu', input_shape = (seq_len,1)))
    model.add(RepeatVector(seq_len))
    model.add(LSTM(LSTM_len, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    
    model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
    model.summary()
    # fit model
    history = model.fit(list_of_data, list_of_data, 
                        epochs=epoch, batch_size=batch_size, verbose=1, 
                        validation_data=(val_data, val_data))
    print(history.history)
    print("#######################")
    #plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
    # demonstrate recreation
    #yhat = model.predict(list_of_data, verbose=1)
    #print("predict results", yhat)#[0,:,0])

    return history


def ready_data_for_modeling_seq_of_bag_of_events(dataset_name, delta, has_activity, len_of_each_seq):

    '''
    it split the tran as validation sets with the ratio of 80:20
    '''
    address_to_read = r"E:\pgmpy\{dataset_name}\Bag of sensor events_no overlap_based on different deltas\delta_{delta}min.csv"
    address_to_read = address_to_read.format(dataset_name = dataset_name , delta = delta)
    list_of_data, list_of_persons  = create_fixed_size_seq_from_bag_of_events(address_to_read = address_to_read , 
                                             len_of_each_seq = len_of_each_seq, 
                                             has_activity = has_activity)    
    
    data_shape= list_of_data.shape
    train_size = int(.8 * data_shape[0])

    history = create_autoencoder_based_on_sequences_of_bag_of_events(list_of_data = list_of_data[0:train_size],
                                                                     list_of_persons = list_of_persons[0:train_size],
                                                                     val_data = list_of_data[train_size:],
                                                                     val_persons = list_of_persons[train_size:])
    
    return history
                    

def ready_data_for_modeling_seq_of_events(dataset_name, delta, has_activity, len_of_each_seq, epoch , batch_size, LSTM_hidden_states ):

    '''
    '''
    address_to_read = r"E:\pgmpy\{dataset_name}\Seq of sensor events_no overlap_based on different deltas\delta_{delta}min.csv"
    address_to_read = address_to_read.format(dataset_name = dataset_name , delta = delta)
    x_train, x_test, y_train, y_test, number_of_words, max_seq_len = data_preparation_for_sequences_based_deep_models(address_to_read, 122 , len_of_each_seq)  
    
    history = create_autoencoder_based_on_sequences_of_events(list_of_data = x_train,
                                                              list_of_persons = y_train,
                                                              val_data = x_test,
                                                              val_persons = y_test,
                                                              epoch = epoch , 
                                                              batch_size = batch_size,
                                                              LSTM_len = LSTM_hidden_states)
    
    return history



def select_seq_len_hyperparameter_for_seq_of_bag_of_events(dataset_name, delta, has_activity):
    seq = []
    train_acc = []
    val_acc = []
    for seq_len in range(1,10):
        history = ready_data_for_modeling_seq_of_bag_of_events(dataset_name, delta, has_activity, seq_len)
        best_epoch_index = np.argmax(history.history['acc'])
        seq.append(seq_len)
        train_acc.append(history.history['acc'][best_epoch_index])
        val_acc.append(history.history['val_acc'][best_epoch_index])

    plot_some_plots("Sequence length" , "accuracy" , seq, [train_acc, val_acc] , ["train acc" , "val acc"])

def select_seq_len_hyperparameter_for_seq_of_events(dataset_name, delta, has_activity, epoch , batch_size):
    seq = []
    train_acc = []
    val_acc = []
    for seq_len in range(1,11):
        history = ready_data_for_modeling_seq_of_events(dataset_name = dataset_name , 
                                                        delta = delta, 
                                                        has_activity = has_activity,
                                                        len_of_each_seq = seq_len,
                                                        epoch = epoch , 
                                                        batch_size = batch_size)        
        best_epoch_index = np.argmax(history.history['acc'])
        seq.append(seq_len)
        train_acc.append(history.history['acc'][best_epoch_index])
        val_acc.append(history.history['val_acc'][best_epoch_index])

    print("================================")
    print("delta:", delta)
    print("{:15s}{:15s}{:15s}".format("seq_len", "train acc" , "val acc"))
    template = "{:15.2f}"
    for i in range(len(seq)):
        print(template.format(seq[i]), template.format(train_acc[i]), template.format(val_acc[i]))


    best_train_acc_for_this_delta = np.argmax(train_acc)
    print("best train accyracy is for seq_len:" , seq[best_train_acc_for_this_delta])
    plot_some_plots("Sequence length" , "accuracy" , seq, [train_acc, val_acc] , ["train acc" , "val acc"], dataset_name + " (delta={})".format(delta))
    

def select_LSTM_hidden_state_hyperparameter_for_autoencoder(dataset_name, delta, has_activity, epoch , batch_size, seq_len):
    number_of_states = []
    train_acc = []
    val_acc = []
    print("seq_len:" , seq_len)

    for hidden_state in range(10,111,10):
        history = ready_data_for_modeling_seq_of_events(dataset_name = dataset_name , 
                                                        delta = delta, 
                                                        has_activity = has_activity,
                                                        len_of_each_seq = seq_len,
                                                        epoch = epoch , 
                                                        batch_size = batch_size,
                                                        LSTM_hidden_states = hidden_state)        
        best_epoch_index = np.argmax(history.history['acc'])
        number_of_states.append(hidden_state)
        train_acc.append(history.history['acc'][best_epoch_index])
        val_acc.append(history.history['val_acc'][best_epoch_index])

    print("================================")
    print("delta:", delta)
    print("{:15s}{:15s}{:15s}".format("#hidden_states", "train acc" , "val acc"))
    template = "{:15.2f}"
    for i in range(len(number_of_states)):
        print(template.format(number_of_states[i]), template.format(train_acc[i]), template.format(val_acc[i]))


    best_train_acc_for_this_delta = np.argmax(train_acc)
    print("best train accyracy is for number of hidden states:" , number_of_states[best_train_acc_for_this_delta])
    plot_some_plots("#Hidden states" , "accuracy" , number_of_states, [train_acc, val_acc] , ["train acc" , "val acc"], dataset_name + " (delta={})".format(delta) + ", seq_len={}".format(seq_len))
    


if __name__ == "__main__":
    #ready_data_for_model(len_of_each_seq = 1)
    '''
    select_seq_len_hyperparameter_for_seq_of_bag_of_events(dataset_name = '' , 
                                                           delta = 0, 
                                                           has_activity = False)
           
                                                           '''
    for delta in [0.1]:#, 0.2, 0.3, .3, .4, .6 , .7 , .8 , .9]:
        select_LSTM_hidden_state_hyperparameter_for_autoencoder(dataset_name = 'Twor2009' , 
                                          delta = delta, 
                                          has_activity = False,
                                          epoch = 10 , 
                                          batch_size =32, 
                                          seq_len = 8)
