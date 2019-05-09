# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:35:57 2019

@author: Adele
"""

class PersonWorkDict:
    
    def __init__(self, PID, WID):
        self.PID = PID
        self.WID = WID
        
def test_PersonWorkDict():
    myobj = PersonWorkDict(12,4)
    print(myobj.PID)
    print(myobj.WID)

def test_list_of_dicts():
    mylist = []
    myobj1 = [12,4]
    myobj2 = [1,2]
    myobj3 = [1,2]
    
    mylist.append([12,4])
    mylist.append([1,2])
    print(mylist)

    mylist.remove([1,2])
    print(mylist)           
if __name__ == "__main__":
    #test_PersonWorkDict()
    test_list_of_dicts()