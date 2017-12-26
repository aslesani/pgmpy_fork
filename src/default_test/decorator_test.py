'''
Created on Oct 20, 2017

@author: Adele
'''

from functools import wraps

def print_on_call(func):
    @wraps(func)
    def wrapper(*args, **kw):
        print('{} called'.format(func.__name__))
        try:
            res = func(*args, **kw)
        finally:
            print('{} finished'.format(func.__name__))
        return res
    return wrapper

#@decorate_all_functions(print_on_call)
class Foo:
    def func1(self):
        print('1')

    def func2(self):
        print('2')
        
        
c = Foo()
c.func1()
c.func2()
        