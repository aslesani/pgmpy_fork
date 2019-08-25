'''
Created on July 14, 2019

@author: Adele
'''
from read_write import read_sequence_based_CSV_file_without_activity
import numpy as np
from dataPreparation import convert_string_to_datetime
from datetime import timedelta
from read_write import read_data_from_CSV_file, separate_dataset_based_on_persons
from scipy.spatial.distance import cityblock #manhatan distance
import warnings
from multiprocessing import Pool
from itertools import product
from functools import reduce

from sklearn.cluster import KMeans

dataset_rows = {'Twor2009':130337, 'Tulum2009':277157, 'Tulum2010':1014507}
each_row_one_feature = r"E:\pgmpy\{}\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv"

def cluster_samples_using_KMeans(data, n_clusters):
    #X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    print("kmeans.labels_:", kmeans.labels_)
    print("kmeans.cluster_centers_:", kmeans.cluster_centers_)
    return kmeans.cluster_centers_

def create_a_single_table(unique, counts):
    
    number_of_people = len(unique)
    new_uniques = unique[0].tolist()
    
    for i in range(1, number_of_people):
        for j in range(len(unique[i])):
            if unique[i][j].tolist() not in new_uniques:
                new_uniques.append(unique[i][j].tolist())

    new_counts = np.zeros(shape = (number_of_people, len(new_uniques)), dtype = int) 
    
    for i in range(number_of_people):
        for j in range(len(unique[i])):
            new_counts[i][new_uniques.index(unique[i][j].tolist())] = counts[i][j]

    for i in range(len(new_uniques)):
        p = [new_uniques[i]]
        for person in range(number_of_people):
            p.append(new_counts[person][i])
        print(p)

    return new_uniques, new_counts        


def return_sequences_of_desired_length(data_address, seq_len):

    list_of_data , _ = read_sequence_based_CSV_file_without_activity(file_address = data_address,
                                        has_header = True, 
                                        separate_data_based_on_persons = True)
    
    number_of_people = len(list_of_data)
    
    restricted_length_sequences = np.zeros(number_of_people ,dtype = np.ndarray)
    
    for p in range(number_of_people):
        data = []
        number_of_samples = len(list_of_data[p])
        for s in range(number_of_samples):
            if len(list_of_data[p][s]) == seq_len:
                data.append(list_of_data[p][s])
        restricted_length_sequences[p] = np.asarray(data)
    
    unique =  np.zeros(number_of_people, dtype = np.ndarray)
    counts =  np.zeros(number_of_people, dtype = np.ndarray)
    for p in range(number_of_people):
        unique[p], counts[p] = np.unique(restricted_length_sequences[p], return_counts=True, axis = 0)# axis = 0 indicates to count the unique rows
    
    create_a_single_table(unique,counts)
    
    return restricted_length_sequences
    
def return_list_of_activity_durations(number_of_entire_rows, address_to_read):
    '''
    
    '''
    f= open(address_to_read , "r")
        
    counter = -1
    
    first = True
    for line in f:
        #print(line)
        cells = line.split(',')
       
        if first:
            number_of_columns = 3#Person_ID, Activity_ID, +1 is for datetime column
            all_features = np.zeros((number_of_entire_rows, number_of_columns), dtype= object )#np.str)1003 +1
        
        converted_cells = []
        converted_cells.append(int(cells[-4]))#Person_ID
        converted_cells.append(int(cells[-3]))#Activity_ID
        converted_cells.append(convert_string_to_datetime(cells[-2], cells[-1].split('\n')[0])) #DateTime. the time has '\n' at end taht should be removed 
      
        counter+=1
               
        if first == True:
            first  = False
           

        if counter < number_of_entire_rows:
            all_features[counter] = converted_cells
        else:
            all_features = np.vstack([all_features,converted_cells])
        
    #seperate each person data in a list (-4 is the index of person column)
    person_IDs = list(set(all_features[: , 0]))
    #print(person_IDs)
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    #print(type(person_data))
    for i in range(number_of_residents):
        person_data[i] = all_features[np.where(all_features[:,0] == person_IDs[i])]
        #print("*****************\n {}".format(i))
        #print(type(person_data[i]))
        #print(person_data[i].shape)
     
    #save bag of features in deltaT for each person   
    person_durations = np.zeros(number_of_residents, dtype = np.ndarray)

    for each_person in range(number_of_residents):
        #print("each_line:{}".format(each_line+1))
        new_counter = 0
        
        #initialize 
        person_data_number_of_rows , _ = person_data[each_person].shape
        # create a ndarray with size of all data of the person 
        person_durations[each_person] = np.ndarray(shape = (person_data_number_of_rows , 2), dtype = int)#just save activityID and its duration
        
        last_activity = person_data[each_person][0][1]
        last_start_datetime = person_data[each_person][0][2]
        
        for offset in range(1, len(person_data[each_person]), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                
            # compare delta time in minutes
            if person_data[each_person][offset][1] == last_activity:
                continue
            else:  
                # save activity column 
                person_durations[each_person][new_counter][0] = last_activity
                #add person column value
                person_durations[each_person][new_counter][1] = timedelta.total_seconds(person_data[each_person][offset-1][2] - last_start_datetime) / 60
                last_start_datetime = person_data[each_person][offset][2]
                last_activity = person_data[each_person][offset][1]
                
                new_counter += 1
                
        #update last row column and activity number
        person_durations[each_person][new_counter][0] = last_activity
        person_durations[each_person][new_counter][1] = timedelta.total_seconds(person_data[each_person][offset][2] - last_start_datetime)/60
                
        #remove additional items (because when i created the person_bag[each_person], the size was number of rows )
        person_durations[each_person] = np.delete(person_durations[each_person] , range(new_counter + 1 , person_data_number_of_rows) , axis = 0)    
         
    for i in range(number_of_residents):
        print("resident number:", person_IDs[i] , "\n---------------------------\n")
        #print(sorted(list(set(person_durations[i][:,1]))))
        print(person_durations[i][:,1])
        print("average activity duration:" , np.average(person_durations[i][:,1]))
        #for j in range(len(person_durations[i])):
        #    print(person_durations[i])

    return person_durations

def activity_durations_histogram(dataset_name):

    activity_duration = return_list_of_activity_durations(dataset_rows[dataset_name] , each_row_one_feature.format(dataset_name))
    u,c  = np.unique(activity_duration[1][:,1], return_counts=True, axis = 0)# just for person 0 
    print(u)
    print(c)
	
	
    sum = 0
    for i in range(0,2):
	    sum = sum + c[i]
    print("0-1:" , sum)
	
	
    sum = 0
    for i in range(2,11):
	    sum = sum + c[i]
    print("2-10:" , sum)
	
    sum = 0
    for i in range(11,50):
	    sum = sum + c[i]
    print("11-50:" , sum)
	  
    sum = 0
    for i in range(50,len(c)):
	    sum = sum + c[i]
    print(">50:" , sum)
	  	  
    #for i in range(len(u)):
	#    print(u[i], c[i])


	
def calculate_inner_class_diff_parallel(data):
    
    warnings.filterwarnings('error')

    pool = Pool(processes=4) # start 4 worker processes
    
    number_of_samples = len(data)
    
    args_indexs = []
    for i in range(number_of_samples-1):#because the last sample does not have a data after it 
        for j in range(i+1,number_of_samples):
            args_indexs.append([i,j])
    args_indexs = np.array(args_indexs)
    
    #print(args_indexs, "@@@@" , args_indexs[:,0], "@@@@" , args_indexs[:,1])
    #print([data[i] for i in args_indexs[:,0]])
    #distances = pool.map(cityblock, [data[i] for i in args_indexs[:,0]], [data[j] for j in args_indexs[:,1]] )
    distances = pool.starmap(cityblock, zip([data[i] for i in args_indexs[:,0]], [data[j] for j in args_indexs[:,1]]))#product(data, repeat = 2) )
    #distances = distances/2# because it repeat 
    #print(distances)
    sumation = reduce((lambda x, y: x + y), distances)
    return sumation
    '''
    manhatan_dist = 0
    previous_manhatan_dist = 0
    list_of_manhatan_dist = []
    for i in range(number_of_samples-1):#because the last sample does not have a data after it 
        for j in range(i+1,number_of_samples):
            cb = cityblock(data[i], data[j])
            try:
                previous_manhatan_dist = manhatan_dist
                manhatan_dist = manhatan_dist + cb
            except RuntimeWarning:# if there is overflow
                list_of_manhatan_dist.append(previous_manhatan_dist)
                manhatan_dist = cb
        #print("i:" , i , "j:" , j, "manhatan_dist:" , manhatan_dist)
    list_of_manhatan_dist.append(manhatan_dist)
    print("list_of_manhatan_dist:", list_of_manhatan_dist)
    return int(manhatan_dist)
    '''
def calculate_inner_class_diff(data):
    
    warnings.filterwarnings('error')

    number_of_samples = len(data)
    manhatan_dist = 0
    previous_manhatan_dist = 0
    list_of_manhatan_dist = []
    for i in range(number_of_samples-1):#because the last sample does not have a data after it 
        for j in range(i+1,number_of_samples):
            cb = cityblock(data[i], data[j])
            print(cb)
            try:
                previous_manhatan_dist = manhatan_dist
                manhatan_dist = manhatan_dist + cb
            except RuntimeWarning:# if there is overflow
                list_of_manhatan_dist.append(previous_manhatan_dist)
                manhatan_dist = cb
        #print("i:" , i , "j:" , j, "manhatan_dist:" , manhatan_dist)
    list_of_manhatan_dist.append(manhatan_dist)
    print("list_of_manhatan_dist:", list_of_manhatan_dist)
    return int(manhatan_dist)

def calculate_between_class_diff(data1, data2):
    
    manhatan_dist = 0

    for i in range(len(data1)):
        for j in range(len(data2)):
            manhatan_dist = manhatan_dist + cityblock(data1[i], data2[j])

    return int(manhatan_dist)


def calculate_inner_and_outer_differece_for_two_residetns(file_address, hasActivity):
    
    data = read_data_from_CSV_file(dest_file = file_address , 
                            data_type = int ,  
                            has_header = True , 
                            return_as_pandas_data_frame = False , 
                            remove_date_and_time = False , 
                            return_header_separately = False , 
                            convert_int_columns_to_int = True)
    
    if hasActivity:
        data = np.delete(data, -1 , 1)
    
    _, cols = np.shape(data)

    data, persons = separate_dataset_based_on_persons(list_of_data = data[:,0:cols - 1], 
                                      list_of_persons = data[:,-1] , 
                                      list_of_activities = 0, 
                                      has_activity = False)# because we remove the activity if does
    #print(data[0], "\n******************\n", data[1])
    #print(persons)
    
    #data = np.arange(18).reshape((2,3,3))
    #persons = [1,2]

    number_of_persons = len(persons)
    inner_class_diff = 0
    sigma_denominator = 0#c(n1,2) + c(n2,2) in which ni is the number of feature vectors for person i
    
    for per in range(number_of_persons):
        print("person:", per)
        inner_class_diff = inner_class_diff + calculate_inner_class_diff(data[per])
        print("inner_class_diff:", inner_class_diff)
        person_number_of_samples = len(data[per])
        sigma_denominator = sigma_denominator + (person_number_of_samples * (person_number_of_samples-1) / 2) #c(ni,2)
        print("sigma_denominator:", sigma_denominator)
  
    inner_class_diff = inner_class_diff / sigma_denominator
    
    between_class_diff = calculate_between_class_diff(data[0], data[1])
    between_class_diff = between_class_diff /  (len(data[0]) * len(data[1]) )

    #print(inner_class_diff, between_class_diff)
    return inner_class_diff, between_class_diff


def calculate_inner_and_outer_differece_for_two_residetns_parallel(file_address, hasActivity):
    
    data = read_data_from_CSV_file(dest_file = file_address , 
                            data_type = int ,  
                            has_header = True , 
                            return_as_pandas_data_frame = False , 
                            remove_date_and_time = False , 
                            return_header_separately = False , 
                            convert_int_columns_to_int = True)
    
    if hasActivity:
        data = np.delete(data, -1 , 1)
    
    _, cols = np.shape(data)

    data, persons = separate_dataset_based_on_persons(list_of_data = data[:,0:cols - 1], 
                                      list_of_persons = data[:,-1] , 
                                      list_of_activities = 0, 
                                      has_activity = False)# because we remove the activity if does
  
    number_of_persons = len(persons)
    inner_class_diff = 0
    sigma_denominator = 0#c(n1,2) + c(n2,2) in which ni is the number of feature vectors for person i
    
    for per in range(number_of_persons):
        print("person:", per)
        inner_class_diff = inner_class_diff + calculate_inner_class_diff_parallel(data[per])
        print("inner_class_diff:", inner_class_diff)
        person_number_of_samples = len(data[per])
        sigma_denominator = sigma_denominator + (person_number_of_samples * (person_number_of_samples-1) / 2) #c(ni,2)
        print("sigma_denominator:", sigma_denominator)
  
    inner_class_diff = inner_class_diff / sigma_denominator
    print("final inner calss diff:" , inner_class_diff)
    between_class_diff = 0
    #between_class_diff = calculate_between_class_diff(data[0], data[1])
    #between_class_diff = between_class_diff /  (len(data[0]) * len(data[1]) )

    #print(inner_class_diff, between_class_diff)
    return inner_class_diff, between_class_diff


def claculate_inner_and_outer_differences_for_all_dataset_and_feature_strategies():

    TBWF_address = r"E:\pgmpy\{dataset}\Bag of sensor events_no overlap_based on different deltas\delta_{delta}min.csv"
    ABWF_address = r"E:\pgmpy\{dataset}\Bag of sensor events_based on activities\based_on_activities.csv"
    HWF_address = r"E:\pgmpy\{dataset}\Bag of sensor events_based_on_activity_and_no_overlap_delta\delta_{delta}min.csv"
    
    TBWF_best_delta = [900,15,15]#for Twor2009, Tulum2009 and Tulum2010 datasets
    HWF_best_delta = [1000,15,15]#for Twor2009, Tulum2009 and Tulum2010 datasets
    
    for index, dataset in zip(range(3), ["Twor2009", "Tulum2009", "Tulum2010"]):
        if index < 2 :
            continue 
        print(dataset, "############")
        
        
        print("TBWF:")
        inner_diff, between_diff = calculate_inner_and_outer_differece_for_two_residetns(file_address = TBWF_address.format(dataset= dataset, delta = TBWF_best_delta[index]),
                                                                                         hasActivity = False)
        print("inner_diff:", inner_diff)
        print("between_diff:", between_diff)
        print("inner/between:" , inner_diff/ between_diff)
       
        print("ABWF:")
        inner_diff, between_diff = calculate_inner_and_outer_differece_for_two_residetns(file_address = ABWF_address.format(dataset= dataset),
                                                                                         hasActivity = True)
        print("inner_diff:", inner_diff)
        print("between_diff:", between_diff)
        print("inner/between:" , inner_diff/ between_diff)
        
        print("HWF:")
        inner_diff, between_diff = calculate_inner_and_outer_differece_for_two_residetns(file_address = HWF_address.format(dataset= dataset, delta = HWF_best_delta[index]),
                                                                                         hasActivity = True)
        print("inner_diff:", inner_diff)
        print("inner/between:" , inner_diff/ between_diff)
        print("between_diff:", between_diff)
        

def claculate_inner_and_outer_differences_for_all_dataset_and_feature_strategies_parallel():

    TBWF_address = r"E:\pgmpy\{dataset}\Bag of sensor events_no overlap_based on different deltas\delta_{delta}min.csv"
    ABWF_address = r"E:\pgmpy\{dataset}\Bag of sensor events_based on activities\based_on_activities.csv"
    HWF_address = r"E:\pgmpy\{dataset}\Bag of sensor events_based_on_activity_and_no_overlap_delta\delta_{delta}min.csv"
    
    TBWF_best_delta = [900,15,15]#for Twor2009, Tulum2009 and Tulum2010 datasets
    HWF_best_delta = [1000,15,15]#for Twor2009, Tulum2009 and Tulum2010 datasets
    
    for index, dataset in zip(range(3), ["Twor2009", "Tulum2009", "Tulum2010"]):
        if index <2 :
            continue 
        print(dataset, "############")
        
        
        print("TBWF:")
        inner_diff, between_diff = calculate_inner_and_outer_differece_for_two_residetns_parallel(file_address = TBWF_address.format(dataset= dataset, delta = TBWF_best_delta[index]),
                                                                                         hasActivity = False)
        print("inner_diff:", inner_diff)

        '''
        print("between_diff:", between_diff)
        print("inner/between:" , inner_diff/ between_diff)
       
        print("ABWF:")
        inner_diff, between_diff = calculate_inner_and_outer_differece_for_two_residetns_parallel(file_address = ABWF_address.format(dataset= dataset),
                                                                                         hasActivity = True)
        print("inner_diff:", inner_diff)
        print("between_diff:", between_diff)
        print("inner/between:" , inner_diff/ between_diff)
        
        print("HWF:")
        inner_diff, between_diff = calculate_inner_and_outer_differece_for_two_residetns_parallel(file_address = HWF_address.format(dataset= dataset, delta = HWF_best_delta[index]),
                                                                                         hasActivity = True)
        print("inner_diff:", inner_diff)
        print("inner/between:" , inner_diff/ between_diff)
        print("between_diff:", between_diff)
        '''

def calculate_inner_and_outer_differece_for_two_residetns_after_clustering(file_address, hasActivity):
    
    data = read_data_from_CSV_file(dest_file = file_address , 
                            data_type = int ,  
                            has_header = True , 
                            return_as_pandas_data_frame = False , 
                            remove_date_and_time = False , 
                            return_header_separately = False , 
                            convert_int_columns_to_int = True)
    
    if hasActivity:
        data = np.delete(data, -1 , 1)
    
    _, cols = np.shape(data)

    data, persons = separate_dataset_based_on_persons(list_of_data = data[:,0:cols - 1], 
                                      list_of_persons = data[:,-1] , 
                                      list_of_activities = 0, 
                                      has_activity = False)# because we remove the activity if does
   
    number_of_persons = len(persons)
    inner_class_diff = 0
    sigma_denominator = 0#c(n1,2) + c(n2,2) in which ni is the number of feature vectors for person i
    
    for per in range(number_of_persons):
        print("person:", per)
        print("len(data[per]):", len(data[per]))
       
        if len(data[per]) > 1000:
            data[per] = cluster_samples_using_KMeans(data[per], 1000)
        
        inner_class_diff = inner_class_diff + calculate_inner_class_diff(data[per])
        print("inner_class_diff:", inner_class_diff)
        person_number_of_samples = len(data[per])
        sigma_denominator = sigma_denominator + (person_number_of_samples * (person_number_of_samples-1) / 2) #c(ni,2)
        print("sigma_denominator:", sigma_denominator)
  
    inner_class_diff = inner_class_diff / sigma_denominator
    
    between_class_diff = calculate_between_class_diff(data[0], data[1])
    between_class_diff = between_class_diff /  (len(data[0]) * len(data[1]) )

    #print(inner_class_diff, between_class_diff)
    return inner_class_diff, between_class_diff


def claculate_inner_and_outer_differences_for_Tulum2010_feature_strategies():
    
    dataset = "Tulum2010"
    TBWF_address = r"E:\pgmpy\{dataset}\Bag of sensor events_no overlap_based on different deltas\delta_{delta}min.csv"
    ABWF_address = r"E:\pgmpy\{dataset}\Bag of sensor events_based on activities\based_on_activities.csv"
    HWF_address = r"E:\pgmpy\{dataset}\Bag of sensor events_based_on_activity_and_no_overlap_delta\delta_{delta}min.csv"
    
    TBWF_best_delta = 15#for Twor2009, Tulum2009 and Tulum2010 datasets
    HWF_best_delta = 15#for Twor2009, Tulum2009 and Tulum2010 datasets
    '''   
    print("TBWF:")
    inner_diff, between_diff = calculate_inner_and_outer_differece_for_two_residetns_after_clustering(file_address = TBWF_address.format(dataset= dataset, delta = TBWF_best_delta),
                                                                                         hasActivity = False)
    print("inner_diff:", inner_diff)
    print("between_diff:", between_diff)
    print("inner/between:" , inner_diff/ between_diff)
    ''' 
    print("ABWF:")
    inner_diff, between_diff = calculate_inner_and_outer_differece_for_two_residetns_after_clustering(file_address = ABWF_address.format(dataset= dataset),
                                                                                         hasActivity = True)
    print("inner_diff:", inner_diff)
    print("between_diff:", between_diff)
    print("inner/between:" , inner_diff/ between_diff)
        
    print("HWF:")
    inner_diff, between_diff = calculate_inner_and_outer_differece_for_two_residetns_after_clustering(file_address = HWF_address.format(dataset= dataset, delta = HWF_best_delta),
                                                                                         hasActivity = True)
    print("inner_diff:", inner_diff)
    print("inner/between:" , inner_diff/ between_diff)
    print("between_diff:", between_diff)
        

def test_inner_and_outer_difference():
    
    data = np.arange(18).reshape((2,3,3))
    #print(data)
    #print("###########################")
    print("inner class:" , calculate_inner_class_diff(data[1]))
    print("parallel inner class:" , calculate_inner_class_diff_parallel(data[1]))
    
    #print("between class:" , calculate_between_class_diff(data[0], data[1]))
    


if __name__ == "__main__":
    data_address = r"E:\pgmpy\Twor2009\Seq of sensor events_no overlap_based on different deltas\delta_900min.csv"
    
    dataset_name = 'Twor2009'
    #activity_duration = return_list_of_activity_durations(dataset_rows(dataset_name) , each_row_one_feature.format(dataset_name))
    #activity_durations_histogram('Twor2009')
    Twor2009_ABWF = r"E:\pgmpy\Twor2009\Bag of sensor events_based on activities\based_on_activities.csv"
    #calculate_inner_and_outer_differece_for_two_residetns(Twor2009_ABWF, True)
    
    #claculate_inner_and_outer_differences_for_all_dataset_and_feature_strategies()
    
    #claculate_inner_and_outer_differences_for_all_dataset_and_feature_strategies_parallel()
    claculate_inner_and_outer_differences_for_Tulum2010_feature_strategies() 
    
    #test_inner_and_outer_difference()
    
    
    '''
    for length in range(2,10):
        print("########################################################")
        print("Sequence Length:", length)
        return_sequences_of_desired_length(data_address = data_address, seq_len = length)  
    '''
    