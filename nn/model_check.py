import csv
import h5py
from  keras.models import load_model
import scipy.io as scio
import numpy as np
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

    return x_train, y_train, x_test, y_test
def check(filename, model_name):
    model = load_model(model_name)
    x_train, y_train, x_test, y_test = preprocess(filename)
    #x_train = np.reshape(x_train, (len(x_train), 310))
    x_test = np.reshape(x_test, (len(x_test), 310))
    #y_train = np_utils.to_categorical(y_train,3)
    #y_test = np_utils.to_categorical(y_test, 3)
    #model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, shuffle = True, validation_split = 0.2)
    #loss = model.evaluate(x_test, y_test, batch_size = 32)
    #print loss
    classes = model.predict_classes(x_test,batch_size =32)
    res = [0,0,0]
    print classes[:100]
    for i in range(len(classes)):
        if(classes[i]==0):
            res[1]+=1
        elif (classes[i]==1):
            res[2]+=1
        else:
            res[0]+=1
    print res
    res = [0,0,0]
    for i in range(len(y_test)):
        if(y_test[i]==0):
            res[1]+=1
        elif (y_test[i]==1):
            res[2]+=1
        elif (y_test[i]==-1):
            res[0]+=1
    print res
    #model.save(model_name)

def main():
    filename = '../EEG.mat'
    check(filename =filename, model_name = 'dense_5_26_69.h5')

main()
