import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, SimpleRNN, Dropout
from keras.callbacks import LambdaCallback

import wandb
from wandb.keras import WandbCallback

import plotutil
from plotutil import PlotCallback

wandb.init()
config = wandb.config

# forecasting using predictions as input
config.repeated_predictions = True
# lookback should be far enough to capture anything in past that will matter in future
config.look_back = 4

def load_data(data_type="airline"):
    if data_type == "flu":
        df = pd.read_csv('flusearches.csv')
        data = df.flu.astype('float32').values
    elif data_type == "airline":
        df = pd.read_csv('international-airline-passengers.csv')
        data = df.passengers.astype('float32').values
    elif data_type == "sin":
        df = pd.read_csv('sin.csv')
        data = df.sin.astype('float32').values
    return data

# convert an array of values into a dataset matrix
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-config.look_back-1):
        a = dataset[i:(i+config.look_back)]
        dataX.append(a)
        dataY.append(dataset[i + config.look_back])
    return np.array(dataX), np.array(dataY)

data = load_data("sin")
    
# normalize data to between 0 and 1
# should probably normalize test based on training data
# can leave room for potential large value in test - normalize 0 and .9
max_val = max(data)
min_val = min(data)
data=(data-min_val)/(max_val-min_val)

# split into train and test sets, no shuffle b/c data matters
split = int(len(data) * 0.70)
train = data[:split]
test = data[split:]

# adds new extra dimension
trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)

trainX = trainX[:, :, np.newaxis]
testX = testX[:, :, np.newaxis]

# create and fit the RNN
model = Sequential()
# 1 => number of inputs/dimensions
# config.look_back, # = input shape = 4 previous numbers and 1 number in each case
# state size (state vector size) is first 1
# cannot change state size without adding final perceptron
model.add(SimpleRNN(2, input_shape=(config.look_back,1 )))
# dense layer makes so that output returns 1 number
model.add(Dense(1))
# rmsprop traditional for LSTM, adam usually used for RNN
model.compile(loss='mae', optimizer='rmsprop')
# more epochs - know if loss keeps going down
# learning rate - higher (0.01, 0.1) to make model converge
# if loss going up then learning rate is might be too high
model.fit(trainX, trainY, epochs=1000, batch_size=20, validation_data=(testX, testY),  callbacks=[WandbCallback(), PlotCallback(trainX, trainY, testX, testY, config.look_back)])





