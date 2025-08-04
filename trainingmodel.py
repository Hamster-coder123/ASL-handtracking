# 3.9.1 venv

import mediapipe as mp
import cv2
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


alphabet = ['a', 'b','c', 'd', 'e', 'f']

cwd = os.getcwd()

datapath = os.path.join(cwd,'data')


X = []
Y = []


for i in alphabet:
    alphapath = os.path.join(datapath,i)
    for file in os.listdir(alphapath):
        data = np.load(os.path.join(alphapath, file))
        # if(data.shape):
        datashape = len(data)
        X.append(data.flatten())
        Y.append(i)

print(X,Y)

X = np.array(X) 
Y = np.array(Y)
xtrain,xtest,ytrain,ytest  = train_test_split(X,Y, test_size = 0.2)

Knn = KNeighborsClassifier(n_neighbors = 5 )
Knn.fit(xtrain,ytrain)
ypredict = Knn.predict(xtest)
accuracy = accuracy_score(ytest, ypredict)

print(accuracy)


joblib.dump(Knn, "knn2_model.pkl")
