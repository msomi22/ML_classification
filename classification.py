
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import time

import numpy as np
import pandas as pd


data = pd.read_table('data/sample.csv', sep='|')  
#netSalary	lifeComfortable	hasCar	houseType	isBrokeEndMonth
data["LC_cleaned"]=np.where(data["lifeComfortable"]=="yes",0,1) 
data["HC_cleaned"]=np.where(data["hasCar"]=="yes",0,1) 
data["BEM_cleaned"]=np.where(data["isBrokeEndMonth"]=="yes",0,1)  

# Cleaning dataset of NaN
data=data[[
    "netSalary",
    "LC_cleaned",
    "HC_cleaned",
    "BEM_cleaned",
    "houseType"  
]].dropna(axis=0, how='any')

#print data

X_train, X_test = train_test_split(data, test_size=0.5, random_state=0)  

gnb = GaussianNB() 
used_features =[
    "LC_cleaned",
    "HC_cleaned",
    "BEM_cleaned"
] 

X = X_train[used_features].values
Y = X_train["netSalary"]
gnb.fit(X,Y) 


#print X_test[used_features]


'''
0=yes,1=no 
LC_cleaned * HC_cleaned * BEM_cleaned  
lifeComfortable	* hasCar *	isBrokeEndMonth

'''
person = [1,1,0]   
person = np.array(person).reshape((-1, 1))

pred = gnb.predict(person)  

print pred   


'''
data.drop('houseType', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), 
                                        title='Box Plot for each input variable')
plt.show()
'''