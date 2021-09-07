# The code creates the datasets and performs isolation forest to detect outliers
#functions

# to generate datapoints

def generate_datapoint(r,bias):
    data_point=[]
    for i in range(128):
        z=random.randint(1,r)
        if z>=bias:
            bit=1
        else:
            bit=0
        b=[bit]
        data_point.append(b)
    return(data_point)

# to get inputs for X_test
def get_test_samples():
    sample = input('\nenter the sample:    ')
    count=0
    for _ in sample:
        if _ =="1" or "0":
            count+=1
        else:
            print ("\nplease enter a valid sample")
            get_test_samples()
    if count == 128:
        return(sample)
    else:
        print ("\nplease enter a valid sample")
        get_test_samples()




#any 128 bit number
# input is an array of 128 arrays with each sub array containing 0 or 1 for the ip
# output is an array of 32 arrays with float values between 0.0 and 1.5
def convolute_ip(a):
    y=[]
    count=0
    z=""
    for _ in a:
        _=_.strip('[')
        _=_.strip(']')
        count+=1
        if count == 4:
            z+=str(_[0])
            y.append(int(z,2)/10)
            z=""
            count=0
        else:
            z+=_

    return(y)



#to predict if an ip is outlier
def pedict_outlier(predictor,X):
    z=predictor.predict(X)
    if z==1:
        return('null')
    else:
        return(X)


#main Structure

#---Imoprting Libraries---

import numpy as np
import pandas as pd
import re
import csv

from sklearn.ensemble import IsolationForest

#---Creating Datasets---

with open('ip_data_train.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for _ in range(10000):
        writer.writerow(generate_datapoint(4,2))

with open('ip_data_test.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for _ in range(100):
        writer.writerow(generate_datapoint(4,4))
    
    for _ in range(100):
        writer.writerow(generate_datapoint(3,3))
    
    for _ in range(100):
        writer.writerow(generate_datapoint(2,2))

    for _ in range(100):
        writer.writerow(generate_datapoint(3,2))



#---Training Phase---


rng = np.random.RandomState(42) # to get replicable results
dataset = pd.read_csv("ip_data_train.csv")
X_train=dataset.iloc[:, :].values

dataset_test = pd.read_csv("ip_data_test.csv")
X_test=dataset_test.iloc[:, :].values

outliers=[]
X_train_convoluted=[]
X_test_convoluted=[]
compare_test=[]


for a in X_train:
    X_train_convoluted.append(convolute_ip(a))

for a in X_test:
    X_test_convoluted.append(convolute_ip(a))


# contamination = 0 - 0.5 -- expected outlier to total ratio

predictor = IsolationForest(max_samples=100, contamination = 0.01, random_state=rng)

predictor.fit(X_train_convoluted)

#---Test Phase---


for _ in X_test_convoluted:
    X=[]
    Y=[]
    for i in _:
        X.append(i)
    Y.append(X)
    z = pedict_outlier(predictor,Y)
    if type(z)==str:
        outliers.append(z)
    else:
        outliers.append('anomaly')

for _ in range(len(X_test_convoluted)):
    compare_test.append((_+1,outliers[_]))

print(compare_test)


