'''
Created on July 31, 2019

@author: Adele
'''
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
import numpy as np

def train_generator():
    while True:
        sequence_length = np.random.randint(10, 100)
        number_of_samples = np.random.randint(10,100)
        x_train = np.random.random((number_of_samples, sequence_length, 5))
        # y_train will depend on past 5 timesteps of x
        y_train = x_train[:, :, 0]
        for i in range(1, 5):
            y_train[:, i:] += x_train[:, :-i, i]
        y_train = to_categorical(y_train > 2.5)
        #print(x_train)#, y_train)
        yield x_train, y_train


def LSTM_variable_length():
    
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(None, 5)))
    model.add(LSTM(8, return_sequences=True))
    model.add(TimeDistributed(Dense(2, activation='sigmoid')))

    print(model.summary(90))

    model.compile(loss ='categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])


    model.fit_generator(train_generator(), steps_per_epoch=30, epochs=10, verbose=1)

if __name__ == "__main__":
    LSTM_variable_length()