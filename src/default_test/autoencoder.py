'''
Created on May 1, 2019

@author: Adele
'''

from read_write import read_data_from_CSV_file, separate_dataset_based_on_persons
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from plot_results import plot_results


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

def create_autoencoder(list_of_data, list_of_persons, val_data, val_persons, LSTM_len = 120, embedding_vector_dim = 20):

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

def ready_data_for_model(dataset_name, delta, has_activity, len_of_each_seq):

    address_to_read = r"E:\pgmpy\Twor2009\Bag of sensor events_no overlap_based on different deltas\delta_0.1min.csv"
    list_of_data, list_of_persons  = create_fixed_size_seq_from_bag_of_events(address_to_read = address_to_read , 
                                             len_of_each_seq = len_of_each_seq, 
                                             has_activity = False)    
    
    data_shape= list_of_data.shape
    train_size = int(.8 * data_shape[0])

    #a = list_of_data[0:train_size]
    #print(a.shape)
    
    history = create_autoencoder(list_of_data = list_of_data[0:train_size], 
                       list_of_persons = list_of_persons[0:train_size],
                       val_data = list_of_data[train_size:],
                       val_persons = list_of_persons[train_size:])
    
    return history
                    

def select_seq_len_hyperparameter(dataset_name, delta, has_activity,):
    seq = []
    train_acc = []
    val_acc = []
    for seq_len in range(1,10):
        history = ready_data_for_model(dataset_name, delta, has_activity, seq_len)
        best_epoch_index = np.argmax(history.history['acc'])
        seq.append(seq_len)
        train_acc.append(history.history['acc'])
        val_acc.append(history.history['val_acc'])




if __name__ == "__main__":
    #ready_data_for_model(len_of_each_seq = 1)
    select_seq_len_hyperparameter(dataset_name = '' , delta = 0, has_activity = False)