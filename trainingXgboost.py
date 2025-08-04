import xgboost as xgb
import mediapipe as mp
import cv2
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder






def train_xgboost(X, y):
    """
    XGBoost often gives the best results for structured data
        
    Excellent performance on landmark coordinates
    Handles feature interactions well
    Built-in regularization
    """
    # Encode labels to numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    # Convert back to original labels for reporting
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    print(f"XGBoost Accuracy: {accuracy_score(y_test_labels, y_pred_labels):.4f}")

    return xgb_model, le

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




model, label_encoder = train_xgboost(X,Y)



joblib.dump(model, "knn3_model.pkl")