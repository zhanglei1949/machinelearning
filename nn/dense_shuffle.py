# accuray 70%
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
from sklearn import preprocessing

from keras.backend.tensorflow_backend import set_session
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

np.random.seed(1234)
def preprocess(filename):
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
    X_Test_raw = data['X_Test']
    Y_Test_raw = data['Y_Test']
    #shape (45,)
    print 'x train raw', X_Train_raw.shape
    print 'test raw', X_Test_raw.shape
    # train
    train_groups = X_Train_raw.shape[1]
    train_group_length = X_Train_raw[0][0].shape[0]
    print 'groups', train_groups
    print 'length', train_group_length
    #test
    test_groups = X_Test_raw.shape[1]
    test_group_length = X_Test_raw[0][0].shape[0]
    print 'groups', test_groups
    print 'length', test_group_length

    x_train = np.zeros((train_groups*train_group_length, 310))
    y_train = np.zeros((train_groups*train_group_length, 1))
    for i in range(X_Train_raw.shape[1]):
    #for i in range(3):
        x_train[i*train_group_length: (i+1)*train_group_length] = X_Train_raw[0][i]
        y_train[i*train_group_length: (i+1)*train_group_length] = Y_Train_raw[0][i]
    #print len(y_train), 'train samples'
    x_test = np.zeros((test_groups*test_group_length, 310))
    y_test = np.zeros((test_groups*test_group_length, 1))
    for i in range(X_Test_raw.shape[1]):
    #for i in range(3):
        x_test[i*test_group_length: (i+1)*test_group_length] = X_Test_raw[0][i]
        y_test[i*test_group_length: (i+1)*test_group_length] = Y_Test_raw[0][i]
    print 'train'
    print x_train.shape
    print y_train.shape
    print 'test'
    print x_test.shape
    print y_test.shape
    #change data
    
    #x = np.concatenate((x_train,x_test),axis=0)
    #y = np.concatenate((y_train,y_test),axis=0)
    train_data = np.hstack((x_train,y_train))
    print train_data.shape
    np.random.shuffle(train_data)

    x_train_new = train_data[:,:-1]
    y_train_new = train_data[:,-1:]
    print 'train'
    print x_train_new.shape
    print y_train_new.shape
    print 'test'
    print x_test.shape
    print y_test.shape
    return x_train_new, y_train_new, x_test, y_test
def train_and_evaluate(filename, model_name,batch_size = 320,epochs = 6):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    KTF.set_session(tf.Session(config=config))
    model = Sequential()
    model.add(Dropout(0.3, input_shape=(310, )))
    model.add(Dense(310, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    
    model.add(Dense(620,activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(310,activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='sigmoid'))
    #0.002
    sgd = SGD(lr=0.002, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #a batch for a file
    #use the last one as test set
    x_train, y_train, x_test, y_test = preprocess(filename)
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    x_train = np.reshape(x_train, (len(x_train), 310))
    x_test = np.reshape(x_test, (len(x_test), 310))
    y_train = np_utils.to_categorical(y_train,3)
    y_test = np_utils.to_categorical(y_test, 3)
    model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, shuffle = False, validation_split = 0.2)
    loss = model.evaluate(x_test, y_test, batch_size = 32)
    print loss
    #classes = model.predict(x_test,batch_size =32)
    classes = model.predict_classes(x_test,batch_size =32)
    res = [0,0,0]
    for i in range(len(classes)):
        if(classes[i]==0):
            res[1]+=1
        elif (classes[i]==1):
            res[2]+=1
        elif (classes[i]==2):
            res[0]+=1
    print res
    model.save(model_name)

def main():
    filename = '../EEG.mat'
    train_and_evaluate(filename =filename, model_name = '6_14_1.h5', epochs = 12,batch_size =140 )

main()