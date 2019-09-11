

import numpy as np


def unison_shuffled_copies(a, b):

    changed_type = False
    if type(a) == list:
        changed_type = True
        a = np.array(a)
        b = np.array(b)
        
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    
    a , b = a[p], b[p]
    
    if changed_type:
        return a.tolist(), b.tolist()
    else:
        return a,b