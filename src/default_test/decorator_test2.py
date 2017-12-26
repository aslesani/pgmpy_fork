'''
Created on Oct 20, 2017

@author: Adele
'''

def time_this(original_function):      
    def new_function(*args,**kwargs):
        import datetime                 
        before = datetime.datetime.now()                     
        x = original_function(*args,**kwargs)                
        after = datetime.datetime.now()                      
        print("Elapsed Time = {0}".format(after-before))      
        return x                                             
    return new_function                                   
    
@time_this
def func_a(stuff):
    import time
    time.sleep(3)
    #print("hello")

if __name__ == '__main__':
    func_a(1)
