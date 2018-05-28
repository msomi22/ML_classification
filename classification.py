
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm 
import time

import numpy as np
import pandas as pd


data = pd.read_table('data/sample.csv', sep='|')  
#netSalary	lifeComfortable	hasCar	houseType	isBrokeEndMonth
data["LifeComfort_cleaned"]=np.where(data["lifeComfortable"]=="yes",0,1) 
data["HasCar_cleaned"]=np.where(data["hasCar"]=="yes",0,1) 
data["BrokeEndMonth_cleaned"]=np.where(data["isBrokeEndMonth"]=="yes",0,1)  

data["HouseType_cleaned"]=np.where(data["houseType"]=="1 room",0,
                                    np.where(data["houseType"]=="2 room",1,
                                    	np.where(data["houseType"]=="3 room",2,
                                    		np.where(data["houseType"]=="4 room",3,
                                    			np.where(data["houseType"]=="5 room",4,
                                    				np.where(data["houseType"]=="6 room",5,6)  
                                    				) 
                                    			)
                                    		)
                                        ) 
                                    )




# Cleaning dataset 
data=data[[
    "netSalary",
    "LifeComfort_cleaned",
    "HasCar_cleaned",
    "BrokeEndMonth_cleaned",
    "HouseType_cleaned"  
]].dropna(axis=0, how='any')

#print data

x_features =[
    "LifeComfort_cleaned",
    "HasCar_cleaned",
    "BrokeEndMonth_cleaned",
    "HouseType_cleaned"  
] 

X = data[x_features].values
Y = data["netSalary"]

#print X
#print Y 


clf = svm.SVC(gamma=0.001, C=100) 
clf.fit(X,Y)

#toPred = np.array([0, 1, 0, 1]).reshape(-1, 1)
toPred = np.array([0, 1, 0, 1]).reshape(1, -1) 
#print toPred   

predOut = clf.predict(toPred) 

print predOut 












































'''

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


0=yes,1=no 
LC_cleaned * HC_cleaned * BEM_cleaned  
lifeComfortable	* hasCar *	isBrokeEndMonth


person = [1,1,0]   
person = np.array(person).reshape((-1, 1))

pred = gnb.predict(person)  

print pred   


data.drop('houseType', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), 
                                        title='Box Plot for each input variable')
plt.show()


'''