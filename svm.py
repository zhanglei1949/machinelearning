# svm model for machine learning project
import numpy as np
import scipy.io as scio

from sklearn import svm
def preprocess(filename, train_ratio):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # 45 groups of data
    # only use one
    data = scio.loadmat(filename)
    X_Train = data['X_Train'][0][0]
    Y_Train = data['Y_Train'][0][0]
    print X_Train.shape
    print Y_Train.shape
    x_train = X_Train[:train_ratio*(X_Train.shape[0])]
    y_train = Y_Train[:train_ratio*(Y_Train.shape[0])]
    x_test = X_Train[train_ratio*(X_Train.shape[0]):]
    y_test = Y_Train[train_ratio*(X_Train.shape[0]):]
    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape
    return x_train, y_train, x_test, y_test

def main():
    filename = 'EEG.mat'
    x_train, y_train, x_test, y_test = preprocess(filename, 0.7)
    clf = svm.NuSVC()
    clf.fit(x_train, y_train)
    #
    predicted = clf.predict(x_test)
#    print predicted
    cnt = 0
    for i in range(len(predicted)):
        if (predicted[i]==y_test[i][0]):
            cnt+=1;
    print 'accuracy', float(cnt)/len(x_test)
main()
