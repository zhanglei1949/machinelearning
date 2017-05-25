import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import regularizers
from keras.utils import np_utils
import csv
import h5py

import scipy.io as scio

def preprocess(filename, train_ratio):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # 45 groups of data
    # only use one
    data = scio.loadmat(filename)
    #data['x_train'] = 1x45
    X_Train_raw = data['X_Train']
    Y_Train_raw = data['Y_Train']
    #shape (45,)
    print 'x train raw', X_Train_raw.shape
    groups = X_Train_raw.shape[1]
    length = X_Train_raw[0][0].shape[0]
    print 'groups', groups
    print 'length', length
    X = np.zeros((groups*length, 310))
    Y = np.zeros((groups*length, 1))
    for i in range(X_Train_raw.shape[1]):
    #for i in range(3):
        X[i*length: (i+1)*length] = X_Train_raw[0][i]
        Y[i*length: (i+1)*length] = Y_Train_raw[0][i]
    print len(Y), 'samples'
    train_part = int(train_ratio*len(Y))
    x_train = X[:train_part]
    y_train = Y[:train_part]
    x_test = X[train_part:]
    y_test = Y[train_part:]
    print 'train'
    print x_train.shape
    print y_train.shape
    print 'test'
    print x_test.shape
    print y_test.shape

    return x_train, y_train, x_test, y_test
def train_and_evaluate(filename, model_name,train_ratio = 0.7, batch_size = 320,epochs = 6):
    model = Sequential()

    model.add(Dense(420, activation='relu', input_shape=(310, )))
    model.add(Dropout(0.5))
    #model.add(Dense(32,kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001)))
    #model.add(Dropout(0.25))
    #model.add(Dense(100,activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
    #model.add(Dense(48, activation = 'relu', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001)))
    #model.add(Dropout(0.25))
    model.add(Dense(800, activation = 'sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(200,activation = 'sigmoid'))
    model.add(Dropout(0.5))
    #model.add(Dense(240,activation = 'relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #a batch for a file
    #use the last one as test set
    x_train, y_train, x_test, y_test = preprocess(filename, train_ratio)
    x_train = np.reshape(x_train, (len(x_train), 310))
    x_test = np.reshape(x_test, (len(x_test), 310))
    y_train = np_utils.to_categorical(y_train,3)
    y_test = np_utils.to_categorical(y_test, 3)
    model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, shuffle = True, validation_split = 0.2)
    loss = model.evaluate(x_test, y_test, batch_size = 64)
    print loss
    model.save(model_name)

def main():
    filename = '../EEG.mat'
    train_and_evaluate(filename =filename, model_name = 'dense_5_25_1.h5',train_ratio = 0.7, epochs = 4,batch_size = 128)

main()
# Generate dummy data
