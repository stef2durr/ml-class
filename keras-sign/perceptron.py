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

# you may want to normalize the data here..
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# 28 by 28 grayscale image
img_width = X_test.shape[1]
img_height = X_test.shape[2]

#reshape input data - needs extra dimension (2d convolution = 3 dimensions)
X_train = X_train.reshape(X_train.shape[0], config.img_width, config.img_height, 1)
X_test = X_test.reshape(X_test.shape[0], config.img_width, config.img_height, 1)

# one hot encode outputs
# takes numbers in y_train and transforms to vectors
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]


# create model
# sequential - each layer feeds into next
model=Sequential()
# add convolutional network
model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(28, 28,1),
    activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# flattens 28 by 28 into single long vector
model.add(Flatten(input_shape=(img_width, img_height)))
# adds 100 layers = multi-layer - convention to use powers of 2
# probably needs relu activation
model.add(Dense(100), activation="relu")
# randomly sets 40% activations to 0
model.add(Dropout(0.4))
# dense builds perceptron, adds activation to be between 0 and 1 instead of linear
# dense builds set of perceptrons - multiplies weights by inputs
# if overfitting - add dropout
model.add(Dense(num_classes), activation="softmax")
# loss = categorical cross entropy
model.compile(loss=config.loss, optimizer=config.optimizer,
                metrics=['accuracy'])

# 3 layer convolutional network makes sense?
# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test), callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])
