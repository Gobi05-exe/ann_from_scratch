#importing libraries
import pandas as pd
import numpy as np
from model import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#preprocessing data
dataset=pd.read_excel('Folds5x2_pp.xlsx')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#training the model
ann=Model()
ann.add(Dense(6))
ann.add(Activation(6,'relu'))
ann.add(Dense(6))
ann.add(Activation(6,'relu'))
ann.add(Dense(1))
ann.train(X_train,y_train,20,0.1)

#prediction
y_pred=ann.predict(X_test)
print(ann.get_accuracy(y_pred,y_test))