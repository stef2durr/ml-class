# A very simple perceptron for classifying american sign language letters
# keras = wrapper on tensorflow
import signdata
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
# setting loss, optimizer and epochs
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 10

# load data
# loading into tensors - x=pixels, y=labels
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

# 28 by 28 grayscale image
img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
# takes numbers in y_train and transforms to vectors
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

# you may want to normalize the data here..

# create model
# sequential - each layer feeds into next
model=Sequential()
# flattens 28 by 28 into single long vector
model.add(Flatten(input_shape=(img_width, img_height)))
# adds 100 layers - convention to use powers of 2
model.add(Dense(100))
# dense builds perceptron
model.add(Dense(num_classes))
# loss = categorical cross entropy
model.compile(loss=config.loss, optimizer=config.optimizer,
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test), callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
