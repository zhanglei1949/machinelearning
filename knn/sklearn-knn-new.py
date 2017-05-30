import math
import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier

def read_data(file_name):
    x = []
    y = []
    with open(file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
                row = rows
                #print row
                tmp = row
                tmp = [float(i) for i in tmp]
                y.append(tmp[0])
                x.append(tmp[1:])

    return np.array(x),np.array(y)

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
    for i in range (len[data]):
        if label[i]==0:
            cnt1+=1
            center1+=data[i]
        if label[i]==1:
            cnt2+=1
            center2 += data[i]
        if label[i]==2:
            cnt3+=1
            center3 += data[i]
    return center1/cnt1,center2/cnt2,center3/cnt3

def k_nn(x_train,y_train,x_test,y_test,k):
    label=y_test-y_test


    for i in range(len(x_test)):
        Distance = np.zeros(len(x_train))
        for j in range(len(x_train)):
            Distance[j]=distance(x_test[i],x_train[j])
        seq=np.argsort(Distance)
        cnt1=0
        cnt2=0
        cnt3=0
        print i
        for j in range (k):

            if(y_train[seq[j]]==1.0):
                cnt1+=1
            if (y_train[seq[j]] == 0.0):
                cnt2 += 1
            if (y_train[seq[j]] == -1.0):
                cnt3 += 1
        if cnt1>=cnt2 and cnt1>=cnt3:
            label[i]=1.0
        if cnt2>cnt1 and cnt2>=cnt3:
            label[i]=0.0
        if cnt3>cnt2 and cnt3>cnt1:
            label[i]=-1.0
        print cnt1,cnt2,cnt3
        print label[i], y_test[i]
    acc=0.0
    for i in range(len(x_test)):

        if(label[i]==y_test[i]):
            acc+=1.0
    return acc/len(x_test)



x_train,y_train=read_data('selec_Training.csv')
x_test,y_test=read_data('selec_Testing.csv')
#print k_nn(x_train[:5000],y_train[:5000],x_test[:2000],y_test[:2000],200)
neighbors = KNeighborsClassifier(n_neighbors=200)
neighbors.fit(x_train[:5000], y_train[:5000])
pre = neighbors.predict(x_test[:1500])

acc = float((pre == y_test[:1500]).sum()) / 1500
print  acc
