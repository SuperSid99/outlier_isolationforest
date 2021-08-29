#functions


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
# input is a string of 128 char containing only 0 or 1
# output is an array of 32 arrays with float values between 0.0 and 1.5
def convolute_ip_test(a):
    b=[]
    count=0
    y=[]
    z=""
    for _ in a:
        count+=1
        if count == 4:
            z+=_
            y.append(int(z,2)/10)
            b.append(y)
            y=[]
            z=""
            count=0
        else:
            z+=_

    return(b)



#any 128 bit number
# input is an array of 128 arrays with each sub array containing 0 or 1 for the ip
# output is an array of 32 arrays with float values between 0.0 and 1.5
def convolute_ip_train(a):
    b=[]
    y=[]
    count=0
    z=""
    for _ in a:
        count+=1
        if count == 4:
            z+=str(_[0])
            y.append(int(z,2)/10)
            b.append(y)
            y=[]
            z=""
            count=0
        else:
            z+=_

    return(b)

# to un convolute to the original 128 bit ip
#input is an array of 32 arrays with float values between 0.0 and 1.5
#returns a string
def un_convolute_ip_test(a):
    b=''
    for _ in a:
        c=(bin(10*_[0]))
        b+=c[2:]
        
    return(b)

#to predict if an ip is outlier
def pedict_outlier(predictor,X):
    z=predictor.predict(X)
    if z==1:
        return('null')
    else:
        return(X)


#main Structure

#---Training Phase---

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

d_set=input("\n Enter the name of the dataset")
rng = np.random.RandomState(42) # to get replicable results
dataset = pd.read_csv(d_set)
X_train=dataset.iloc[:, :].values

X_test=[]
outliers=[]
X_train_convoluted=[]
compare_test=[]

for a in X_train:
    X_train_convoluted.append(convolute_ip_train(a))

# contamination = 0 - 0.5 -- expected outlier to total ratio

predictor = IsolationForest(max_samples=100, contamination = 0.1, random_state=rng)
predictor.fit(X_train_convoluted)

#---Test Phase---


while(True):

    m_samples=input('\nDo you want to add samples? (yes or no)')
    if m_samples.lower()[0]=='y':
        X_test.append(get_test_samples()
    elif m_samples.lower()[0]=='n':
        break
    else:
        print("\nenter a valid responce")


for _ in X_test:
    X=convolute_ip_test(_)
    z = pedict_outlier(predictor,X)
    if type(z)==str:
        outliers.append(z)
    else:
        outliers.append(un_convolute_ip_test(z))

for _ in range(len(X_test)):
    compare_test.append((X_test[_],outliers[_]))

print(compare_test)
