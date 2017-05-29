import math
import numpy as np
import scipy.io as scio

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

def distance(x,center):#caculate distance from center of the group to sample
    L=0
    for i in range(len(x)):
        L+=(x[i]-center[i])*(x[i]-center[i])
    return math.sqrt(L)

def new_center(label,data):#renew the center for new group of samples
    cnt1=0
    cnt2=0
    cnt3=0
    center1=data[0]-data[0]
    center2 = data[0] - data[0]
    center3 = data[0] - data[0]
    for i in range (len(data)):
        if label[i]==-1.0:
            cnt1+=1
            center1+=data[i]
        if label[i]==0.0:
            cnt2+=1
            center2 += data[i]
        if label[i]==1.0:
            cnt3+=1
            center3 += data[i]
    print cnt1,cnt2,cnt3
    return center1/cnt1,center2/cnt2,center3/cnt3

def k_mean(data,y,x_test,iteration):
    data = np.array(data,float)#data should be arrange in samples*features(n*310)
    label=y
    center1,center2,center3=new_center(y,data)

    for j in range(iteration):
        tmp1 = center1
        tmp2 = center2
        tmp3 = center3
        print j
        for i in range(len(x_test)):
            L1=distance(x_test[i],center1)
            L2 = distance(x_test[i], center2)
            L3 = distance(x_test[i], center3)
            if L1<=L2 and L1<=L3:
                label[i]=-1.0
            if L2<L1 and L2<=L3:
                label[i]=0.0
            if L3<L1 and L3<L2:
                label[i]=1.0
        center1, center2, center3 = new_center(label, x_test)
        #if center2==tmp2 and center1==tmp1 and center3==tmp3:
            #break

    return label

def test(label,y_test):
    cnt=0
    for i in range (len(y_test)):

        if (label[i]==y_test[i][0]):
                cnt+=1
    print cnt
    return float(cnt)/len(x_test)

filename = 'EEG.mat'
x_train,y_train,x_test,y_test=preprocess(filename)
label=k_mean(x_train,y_train,x_test,10)
print test(label,y_test)
