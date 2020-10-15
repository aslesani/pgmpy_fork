'''
Created on Apr 13, 2018

@author: Adele
'''
import csv 
import numpy as np
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing import sequence
from shuffle_data import unison_shuffled_copies
import pickle

def read_data_from_file(dest_file, data_type , remove_date_and_time = True , has_header = False ):
    '''
    this function is used when there is data and time columns in dataset
    and the user want to remove them
    the output type is int
    
    Parameters:
    ==========
    dest_file:
    data_type:
    remove_date_and_time: if is true, the date and time columns(two last columns) are removed
    '''
    with open(dest_file,'r') as dest_f:
        data_iter = csv.reader(dest_f, 
                               delimiter = ',')#quotechar = '"')
        
        if has_header:
            next(data_iter)#skip first line
        
        data = [data for data in data_iter]
    
    if remove_date_and_time == True:
        data = np.delete(np.delete(data, -1, 1), -1 , 1)
    
    #rows = len(data)
    #print(rows)
    
    return_value= np.asarray(data, dtype = data_type)#np.int)
    #print(return_value)
    return return_value

def read_data_from_CSV_file(dest_file, data_type, has_header = False, return_as_pandas_data_frame = False, remove_date_and_time = False, return_header_separately = False, convert_int_columns_to_int = False):
    '''
    this function is a replacement for read_data_from_PCA_output_file and read_data_from_PCA_digitized_file
    with more capabalities.
    
    Parameters:
    ==========
    dest_file: 
    data_type: type of data that should be read  
    has_header = if the file has header, it is set to True. The header is the first line that starts whit '#' character 
    return_as_pandas_data_frame = if True, the return_value is pandas Dataframe, else numpy ndaaray
    
    convert_int_columns_to_int: if the user wants to keep date and time columns, then she should 
                                specify data_type as object and then set convert_int_columns_to_int to True
    
    Returns:
    ========
    return_value: type of it is pandas Dataframe or numpy ndaaray
    
    '''
    header = ""
    with open(dest_file,'r') as dest_f:
        data_iter = csv.reader(dest_f, 
                               delimiter = ',')#quotechar = '"')
    
        if has_header:
            header = next(data_iter)
            header[0] = header[0].split('# ')[1] # remove # from first element
        
        
        data = [data for data in data_iter]
    
    selected_cols = len(data[0]) -2
    #print(selected_cols)
    if remove_date_and_time:
        #print(data[0:3])
        #print('remove_date_and_time')
        #print(selected_cols)
        #print(data[0][0:5])
        data = [data[0:selected_cols] for data in data]#data[:][0:selected_cols]#np.delete(np.delete(data, -1, 1), -1 , 1)

    #print(len(data[0]))
    
    return_value= np.asarray(data, dtype = data_type)
    
    if convert_int_columns_to_int:
        rows , cols_to_convert = np.shape(return_value)
        
        if remove_date_and_time == False:
            cols_to_convert -= 2
        
        for r in range(rows):
            for c in range(cols_to_convert):
                return_value[r,c] = int(return_value[r,c])
        
    
    if return_as_pandas_data_frame:
        return_value = pd.DataFrame(return_value , columns = header)
        
    if return_header_separately:
        return header , return_value
    
    else:   
        return return_value


def separate_dataset_based_on_persons(list_of_data, list_of_persons, list_of_activities, has_activity):
    
    '''
    Parameters:
    ===========
    list_of_data:
    list_of_persons:
    list_of_activities:if has_activity== False, ignore this argument
    has_activity: if True, consider the list_of_activities as well
    '''

    number_of_persons = len(set(list_of_persons))
    new_list_of_data = np.zeros(number_of_persons , dtype = np.ndarray)
    new_list_of_persons = np.zeros(number_of_persons ,dtype = np.ndarray)
    
    if has_activity:
        new_list_of_activities = np.zeros(number_of_persons ,dtype = np.ndarray)

    
    list_of_persons = np.array(list_of_persons)
    list_of_data = np.array(list_of_data)
    if has_activity:
        list_of_activities = np.array(list_of_activities)

    new_list_index = 0
    
    for per in list(sorted(set(list_of_persons))):
        indexes = np.where(np.equal(list_of_persons , per))
        new_list_of_persons[new_list_index] = list_of_persons[indexes]
        new_list_of_data[new_list_index] = list_of_data[indexes]
        
        if has_activity:
            new_list_of_activities[new_list_index] = list_of_activities[indexes]
       
        new_list_index += 1
    
    if has_activity:
        return new_list_of_data , new_list_of_persons , new_list_of_activities
    else:
        return new_list_of_data , new_list_of_persons


def read_sequence_based_CSV_file_with_activity(file_address , has_header, separate_data_based_on_persons, separate_words = True):
    '''
    Parameters:
    ==========
    file_address:
    has_header:
    separate_data_based_on_persons: if True, retrun return values as a numpy array with len of number of persons
    
    Retrun:
    =======
    list_of_data: list of sequenced data
    list_of_persons: list of corresponding person numbers
    list_of_activities: list of corresponding activity numbers

    '''
    header = ""
    list_of_data = []
    with open(file_address,'r') as dest_f:
        list_of_data = list(dest_f)
        number_of_rows = len(list_of_data)
        if has_header:
            number_of_rows -= 1
            header = list_of_data[0].split('# ')[1] 
            list_of_data = list_of_data[1:]#remove the first line(header)
        
    list_of_persons = list([0] * number_of_rows) 
    list_of_activities = list([0] * number_of_rows)            
       
    for line in range(0 , number_of_rows):
        d = list_of_data[line].split('[')[1]
        d = d.split(']')
        if separate_words:
            seq = d[0].replace("'" , '').split(', ')
            list_of_data[line] = np.array(seq)
        else:
            seq = d[0].replace("'" , '')
            list_of_data[line] = seq
            
        other = d[1].split(',')
        
        try:
           list_of_persons[line] = int(other[1])# * len(seq)
           list_of_activities[line] = int(other[2])
        except Exception as e:
            print("Exception!")
            print('len(list_of_persons):' , len(list_of_persons))
            print('len(list_of_activities):' , len(list_of_activities))
            print('line:' , line)
            print('len(other):' , len(other))

   
    if separate_data_based_on_persons:
        
        return separate_dataset_based_on_persons(list_of_data , list_of_persons , list_of_activities ,True)

    else:
        return list_of_data , list_of_persons , list_of_activities


def read_sequence_based_CSV_file_with_activity_as_strings(file_address , has_header, separate_data_based_on_persons):
    '''
    return the string of each sequence (not separated)
    
    Parameters:
    ==========
    file_address:
    has_header:
    separate_data_based_on_persons: if True, retrun return values as a numpy array with len of number of persons
    
    Retrun:
    =======
    list_of_data: list of strings
    list_of_persons: list of corresponding person numbers
    list_of_activities: list of corresponding activity numbers

    '''
    header = ""
    list_of_data = []
    with open(file_address,'r') as dest_f:
        list_of_data = list(dest_f)
        number_of_rows = len(list_of_data)
        if has_header:
            number_of_rows -= 1
            header = list_of_data[0].split('# ')[1] 
            list_of_data = list_of_data[1:]#remove the first line(header)
        
    list_of_persons = list([0] * number_of_rows) 
    list_of_activities = list([0] * number_of_rows)            
       
    for line in range(0 , number_of_rows):
        d = list_of_data[line].split('[')[1]
        d = d.split(']')
        seq = d[0].replace("'" , '').split(', ')
        other = d[1].split(',')
        list_of_persons[line] = int(other[1])# * len(seq)
        list_of_activities[line] = int(other[2])
        list_of_data[line] = np.array(seq)
   
    if separate_data_based_on_persons:
        
        return separate_dataset_based_on_persons(list_of_data , list_of_persons , list_of_activities ,True)

    else:
        return list_of_data , list_of_persons , list_of_activities



def read_sequence_based_CSV_file_without_activity(file_address , has_header, separate_data_based_on_persons, separate_words = True):
    '''
    Parameters:
    ==========
    file_address:
    has_header:
    separate_data_based_on_persons: if True, retrun return values as a numpy array with len of number of persons
    
    Retrun:
    =======
    list_of_data: list of sequenced data
    list_of_persons: list of corresponding person numbers
    list_of_activities: list of corresponding activity numbers

    '''
    header = ""
    list_of_data = []
    with open(file_address,'r') as dest_f:
        list_of_data = list(dest_f)
        number_of_rows = len(list_of_data)
        if has_header:
            number_of_rows -= 1
            header = list_of_data[0].split('# ')[1] 
            list_of_data = list_of_data[1:]#remove the first line(header)
        
    list_of_persons = list([0] * number_of_rows) 
       
    for line in range(0 , number_of_rows):
        d = list_of_data[line].split('[')[1]
        d = d.split(']')
       
        if separate_words:
            seq = d[0].replace("'" , '').split(', ')
            list_of_data[line] = np.array(seq)
        else:
            seq = d[0].replace("'" , '')
            list_of_data[line] = seq
          
        other = d[1].split(',')
        list_of_persons[line] = int(other[1])# * len(seq)
      
    if separate_data_based_on_persons:
        
        new_list_of_data , new_list_of_persons = separate_dataset_based_on_persons(list_of_data= list_of_data, 
                                              list_of_persons = list_of_persons, 
                                              list_of_activities = 0, 
                                              has_activity = False)
    
        return new_list_of_data , new_list_of_persons
    
    else:
        return list_of_data , list_of_persons


def read_sequence_of_bags_CSV_file_with_activity(file_address , has_header, separate_data_based_on_persons):
    '''
    Parameters:
    ==========
    file_address:
    has_header:
    separate_data_based_on_persons: if True, retrun return values as a numpy array with len of number of persons
    
    Retrun:
    =======
    list_of_data: list of sequenced data
    list_of_persons: list of corresponding person numbers
    list_of_activities: list of corresponding activity numbers

    '''
    header = ""
    list_of_data = []
    with open(file_address,'r') as dest_f:
        list_of_data = list(dest_f)
        number_of_rows = len(list_of_data)
        if has_header:
            number_of_rows -= 1
            header = list_of_data[0].split('# ')[1] 
            list_of_data = list_of_data[1:]#remove the first line(header)
        
    list_of_persons = list([0] * number_of_rows) 
    list_of_activities = list([0] * number_of_rows)            
       
    for line in range(0 , number_of_rows):
        #print('list_of_data[line]:' , list_of_data[line])
        d = list_of_data[line].split('[[')[1]
        #print('d1:' , d)
        #print('type(d):' , type(d))

        d = d.split(']]')
        #print('len(d2):' , len(d))
        #d = d.split(']')
        d[0] = d[0].split('], [')
        #print('len(d[0]):' , len(d[0]))
        seq = list([0] * len(d[0]))
        for i in range(len(d[0])):
            seq[i] = d[0][i].split(', ')
        #print('seq:' , len(seq) , type(seq) , type(seq[0]) , len(seq[0]) , type(seq[0][0]))
        other = d[1].split(',')
        #print('other:' , other)
        list_of_persons[line] = int(other[1])# * len(seq)
        #print('list_of_persons[line]:' , list_of_persons[line])
        list_of_activities[line] = int(other[2])
        #print('list_of_activities[line]:' , list_of_activities[line])
        for i in range(len(seq)):
            for j in range(len(seq[i])):
                seq[i][j] = int(seq[i][j])
        
        list_of_data[line] = np.array(seq)
        
    
    if separate_data_based_on_persons:
        
        return separate_dataset_based_on_persons(list_of_data , list_of_persons , list_of_activities ,has_activity = True)
       
    else:
        return list_of_data , list_of_persons , list_of_activities


def repaet_person_tags_as_much_as_seq_length(list_of_data , list_of_persons, is_one_person):
   
    if is_one_person:
        for i in range(len(list_of_data)):
            list_of_persons[i] = list([list_of_persons[i]] * len(list_of_data[i]))
        
    else:
        number_of_persons = len(list_of_persons)
        print("number_of_persons:" , number_of_persons)
        for per in range(number_of_persons):
            for i in range(len(list_of_data[per])):
                a = np.array(list([list_of_persons[per][i]] * len(list_of_data[per][i])))
                
                list_of_persons[per][i] = a
   
    return list_of_persons.copy()


def convert_binary_classes_to_zero_and_one(data):
  
    values = sorted(list(set(data)))
    for i in range(len(data)):
        data[i] = values.index(data[i])

    
    return data

def convert_binary_classes_to_zero_and_one_for_real_person_tags_and_prediction_tags(train_tags,\
                                                                    sequence_of_real_person_tags, predicted_tags):
  
    '''
    Parameters:
    ===========
    train_tags: 
    sequence_of_real_person_tags:
    predicted_tags:
    '''
    
    #create set of all person tags
    print(train_tags.shape)
    train_tags_set = set(train_tags)
    predicted_tags_set = set(predicted_tags)
    sequence_of_real_person_tags_set = set()
    for i in sequence_of_real_person_tags:
        sequence_of_real_person_tags_set = sequence_of_real_person_tags_set.union(set(i)) 
        
    values = train_tags_set.union(predicted_tags_set).union(sequence_of_real_person_tags_set)    
    values = sorted(list(values))
 
    for i in range(len(train_tags)):
        train_tags[i] = values.index(train_tags[i])
    
    for i in range(len(predicted_tags)):
        predicted_tags[i] = values.index(predicted_tags[i])
        
    for i in range(len(sequence_of_real_person_tags)):
        sequence_of_real_person_tags[i] = [values.index(sequence_of_real_person_tags[i][j]) \
                                           for j in range(len(sequence_of_real_person_tags[i]))]

    
    return train_tags, sequence_of_real_person_tags, predicted_tags



def test_convert_binary_classes_to_zero_and_one_for_real_person_tags_and_prediction_tags(pickle_file_address):

    with open(pickle_file_address, 'rb') as f:
        data = pickle.load(f)
    
    train = data[0]
    non_train = data[1]
    
    train_tags = train[:,1]
    sequence_of_real_person_tags = non_train[:,1]
    predicted_tags = non_train[:,2]
    
    print(train_tags)
    print('----------------------------------')
    print(sequence_of_real_person_tags)
    print('----------------------------------')
    print(predicted_tags)
    print('**********************************')
    
    train_tags, sequence_of_real_person_tags, predicted_tags = \
                            convert_binary_classes_to_zero_and_one_for_real_person_tags_and_prediction_tags(\
                            train_tags,
                            sequence_of_real_person_tags, 
                            predicted_tags)
  
    print(train_tags)
    print('----------------------------------')
    print(sequence_of_real_person_tags)
    print('----------------------------------')
    print(predicted_tags)



def test_convert_binary_classes_to_zero_and_one():
    data = [2,1,1,1,2]
    data = convert_binary_classes_to_zero_and_one(data)
    print(data)



def data_preparation_for_sequences_based_deep_models(address_to_read, number_of_words, max_seq_len, hasActivitycol, shuffle):
    '''
    this module read a sequence based data file and tokenize it before using in deep models like LSTM.
    In addition it splits the data as train and test samples
    '''
    if hasActivitycol:
	    list_of_data , list_of_persons, _ = read_sequence_based_CSV_file_with_activity(file_address = address_to_read, 
                                                                                 has_header = True , 
                                                                                 separate_data_based_on_persons = False, 
                                                                                 separate_words= False)
    else:
        list_of_data , list_of_persons = read_sequence_based_CSV_file_without_activity(file_address = address_to_read, 
                                                                                 has_header = True , 
                                                                                 separate_data_based_on_persons = False, 
                                                                                 separate_words= False)
    
    if shuffle:
        print("$$$$$$$$before shuffle")
        list_of_data , list_of_persons = unison_shuffled_copies(list_of_data , list_of_persons)
        print("$$$$$$$after shuffle")
    	
	#sensor_events , number_of_events = get_set_of_sensor_events(sequences)
    
    list_of_persons = convert_binary_classes_to_zero_and_one(list_of_persons)
  
    tokenizer = Tokenizer(num_words = number_of_words + 1, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~')
    #list_of_data = [r'salam man', r"'M38_off' , 'M38_on'"]
    tokenizer.fit_on_texts(list_of_data)
    sequences = tokenizer.texts_to_sequences(list_of_data)
   
    #max_features = 121#number_of_events
    # cut texts after this number of words (among top max_features most common words)
    #maxlen = 20#80#max_seq_len

        
    #80% of data for train and 20% for test
    train_numbers = int(0.8 * len(sequences))
    x_train, y_train = sequences[0: train_numbers] , list_of_persons[0:train_numbers]
    x_test, y_test = sequences[train_numbers+1:] , list_of_persons[train_numbers+1:]

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_seq_len)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    
    return x_train, x_test, y_train, y_test, number_of_words, max_seq_len



def data_preparation_for_sequences_based_deep_models_with_prediction(train_sequence, non_train_sequence,\
                                        number_of_words, max_seq_len):
    
    '''
    this module recives two sequences, one for train sequences 
    and one for non-train sequences. then tokenize it before using in deep models like LSTM.
    Parameters:
    ===========
    train_sequence: a ndarray in which each row contains a list of sequences and the person tag. 
    non_train_sequence: a ndarray in which each row contains a list of sensor events seperated based on predicted persons,
                        a list of person tags which its length is as equal as the sensor events sequences and
                        demonstrates the real person tags of each sensor event,
                        and a person number which can be considered as predicted person tag, which is more frequent 
                        in the list of real person tags.
                        
                        
    '''
    
    x_train = train_sequence[:,0]
    y_train = train_sequence[:,1]
    x_nontrain = non_train_sequence[:,0]
    y_nontrain_sequence_of_real_person_tags = non_train_sequence[:,1]
    y_nontrain_predicted = non_train_sequence[:,2]
    
    y_train, y_nontrain_sequence_of_real_person_tags, y_nontrain_predicted = \
                            convert_binary_classes_to_zero_and_one_for_real_person_tags_and_prediction_tags(\
                            train_tags = y_train,
                            sequence_of_real_person_tags = y_nontrain_sequence_of_real_person_tags, 
                            predicted_tags = y_nontrain_predicted)
                            
  
    tokenizer = Tokenizer(num_words = number_of_words + 1, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~')
    tokenizer.fit_on_texts(np.concatenate((x_train, x_nontrain), axis = 0))#fit on all sensor events
    
    x_train = tokenizer.texts_to_sequences(x_train)
    x_nontrain = tokenizer.texts_to_sequences(x_nontrain)

    #lens = []
    #for i in range(len(x_train)):
    #    lens.append(len(x_train[i]))
    
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
    x_nontrain = sequence.pad_sequences(x_nontrain, maxlen=max_seq_len)
    print('x_train shape after pad:', x_train.shape)
    print('x_nontrain shape after pad:', x_nontrain.shape)
    
    return x_train, x_nontrain, \
           y_train, y_nontrain_sequence_of_real_person_tags, y_nontrain_predicted

def test_data_preparation_for_sequences_based_deep_models_with_prediction(pickle_file):
    
    number_of_words = 31
    max_seq_len = 20
    
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        
    train_sequence = data[0]
    non_train_sequence = data[1]
    
    data_preparation_for_sequences_based_deep_models_with_prediction(train_sequence, non_train_sequence,\
                                        number_of_words, max_seq_len)
 
if __name__ == "__main__":
    print("I Commented keras. Do not forget!")
    #separate_data_based_on_persons = True
    #a , b , c = read_sequence_of_bags_CSV_file_with_activity(file_address = r'C:\b.csv' , has_header= True, separate_data_based_on_persons = separate_data_based_on_persons)
    #repeated_per = repaet_person_tags_as_much_as_seq_length(a , b , separate_data_based_on_persons=separate_data_based_on_persons)
    #address_to_read= r"E:/pgmpy/Twor2009/Seq of sensor events_based on activities/based_on_activities.csv"
    #list_of_data , list_of_persons , _ = read_sequence_based_CSV_file_with_activity(file_address = address_to_read, has_header = True , separate_data_based_on_persons = False)
    
    pickle_file = r"E:\pgmpy\Twor2009\train_and_nontrain_sequences_train_percent_0.8.pkl"
    test_data_preparation_for_sequences_based_deep_models_with_prediction(pickle_file)
    #test_convert_binary_classes_to_zero_and_one_for_real_person_tags_and_prediction_tags(pickle_file)
    #data_preparation_for_sequences_based_deep_models(address_to_read,  10,20, True, True)