'''
Created on Jul 11, 2018

@author: Adele
'''

from read_write import read_data_from_CSV_file
import numpy as np
from custom_classifiers import test_different_classifiers
from read_write import read_sequence_based_CSV_file_without_activity
from pomegranate_test import unison_shuffled_copies, create_casas7_markov_chain_with_prepared_train_and_test
from pomegranate_test import prepare_train_and_test_based_on_a_list_of_k_data, calculate_f1_scoreaverage
from Validation import plot_results

def apply_NB_on_sensor_data_with_hour_of_day_feature( k , shuffle, file_to_read = ' ' , add_str_to_path = ' '):
    
    if file_to_read == ' ':
        file_to_read = r'E:\pgmpy\{path}\sensor_data_each_row_one_features_is_one_on_and_off+hour_of_day.csv'.format(path = add_str_to_path)
        print(add_str_to_path)
    data = read_data_from_CSV_file(dest_file = file_to_read, 
                                   data_type = int, 
                                   has_header = False, 
                                   return_as_pandas_data_frame = False, 
                                   remove_date_and_time = False, 
                                   return_header_separately = False, 
                                   convert_int_columns_to_int = False)
    
    data_target = data[: , -2] # the last col is hour of day and the -2 col is person

    _ , cols = np.shape(data)
    data_features = np.delete(data, cols - 2, axis = 1)
    
    print(data_target)
    print(data_features)
    names, avg_f_score = test_different_classifiers(data_features = data_features, 
                               data_target = data_target, 
                               k = k, 
                               shuffle = shuffle, 
                               selected_classifiers = [16])# NB is 17
    
    print(names, avg_f_score)

def select_the_best_number_of_events_using_the_best_strategy_markov_chain(k=10 , shuffle = True, add_str_to_path = ' '):
    

    address_to_read = r"E:\pgmpy\{path}\Seq of sensor events_based_on_number_of_events\number_of_events={ne}.csv"
    number_of_events = range(3,10)
    print(add_str_to_path)

    best_score = 0
    best_number_of_events = 0
    best_train_set = 0
    best_test_set = 0
    best_train_set_person_labels = 0
    best_test_set_person_labels = 0
    
    list_of_number_of_events = []
    list_of_f1_micros = []
    
    for n in number_of_events:
        print("number of events:" , n)
        list_of_data , list_of_persons = read_sequence_based_CSV_file_without_activity(file_address = address_to_read.format(path = add_str_to_path, ne = n), 
                                                                                       has_header = True, 
                                                                                       separate_data_based_on_persons = True)
        
        
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
            
            scores[i] = create_casas7_markov_chain_with_prepared_train_and_test(train_set = train_set_k , list_of_persons_in_train=train_set_labels_k , test_set=test_set_k , list_of_persons_in_test=test_set_labels_k)
        scores_avg = calculate_f1_scoreaverage(scores , k)
        print("scores_avg" , scores_avg)
        
        list_of_number_of_events.append(n)
        list_of_f1_micros.append(scores_avg)
        
        if scores_avg > best_score:
            best_score = scores_avg
            best_number_of_events = n
            best_train_set = train_set
            best_test_set = test_set
            best_train_set_person_labels = train_set_person_labels
            best_test_set_person_labels = test_set_person_labels
    
    print("Validation Scores:")
    print("best_number_of_events:" , best_number_of_events , "best_validation_score:" , best_score)
    
    test_score =  create_casas7_markov_chain_with_prepared_train_and_test(train_set = best_train_set , list_of_persons_in_train= best_train_set_person_labels, test_set= best_test_set , list_of_persons_in_test= best_test_set_person_labels)
    print("test_score:" , test_score)
    
    #plot_results(list_of_number_of_events, list_of_f1_micros, "number of events" , y_label = "f1 score micro")



if __name__ == '__main__':
    
    apply_NB_on_sensor_data_with_hour_of_day_feature(k = 10, shuffle = False , file_to_read=' ' , add_str_to_path='Tulum2010' )
    #select_the_best_number_of_events_using_the_best_strategy_markov_chain(k = 10, shuffle = False, add_str_to_path='Tulum2010')