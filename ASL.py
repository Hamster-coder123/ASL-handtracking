
import mediapipe as mp
import cv2
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



# ASL sign language 

# Goal: Have a functional ASL hand tracker that can detect the ASL signs

# - Download dataset to train the model

# - Mediapipe gives us the coordinates of each landmark

# - Train model for the ASL signs
#     - Inputs - the coordinates of the landmarks in the image
#     - Outputs - ASL signs 





# def processframe(frame, hands):
    
#     imagergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(imagergb)
#     mhl = results.multi_hand_landmarks

#     pos = []

#     if(mhl):
#         for hand in mhl:
        
#             for id, lm in enumerate(hand[0].landmark):
#                 pos.append([lm.x, lm.y, lm.z])
    
#         pos = np.array(pos).flatten()
#         print(pos)
#         return pos
def processframe(image,hands): 
    frame_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    ## Store the inforamtion in a numpy array, if nothing, then store 0's --> but we do not need it anyways, so just skip it
    if results.multi_hand_landmarks: 
        mhl = results.multi_hand_landmarks
        lmList = []
        for _,lm in enumerate(mhl[0].landmark): 
            lmList.extend([lm.x,lm.y,lm.z])
        lmList = np.array(lmList)
        print(lmList.shape)
        return lmList
    else:
        return None

cam = cv2.VideoCapture(0)


camW = 1280

camH = 720

cam.set(3,camW)
cam.set(4,camH)

framesperL = 400

capturerate = 5

mphands = mp.solutions.hands

hands = mphands.Hands(max_num_hands = 1)


# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
#  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

alphabet = ['d', 'e', 'f']

cwd = os.getcwd()

datapath = os.path.join(cwd,'data')








if not os.path.exists(datapath):  
    os.mkdir(datapath)
for i in alphabet: 
    path = os.path.join(datapath,i)
    if not os.path.exists(path): 
        os.mkdir(path)


for i in alphabet:
    collected = 0
    framenumber = 0
    while collected <= framesperL:
        success, frame = cam.read()
        if(framenumber % capturerate == 0):
            coords = processframe(frame, hands)
            # print(coords)
            # 
            if (coords is not None)  :
                collected += 1
                # print(f"{collected}-100 for {i}")
                np.save(os.path.join(datapath,i, f"{collected}.npy"),coords)
        framenumber += 1
        cv2.putText(frame,f"{collected}-100 for {i}", (10,370), cv2.FONT_HERSHEY_DUPLEX, 3, (255,0,0), 3)
        cv2.putText(frame,f"Collecting letter  {i}", (10,70), cv2.FONT_HERSHEY_DUPLEX, 3, (255,0,0), 3)    
        cv2.imshow("Webcam", frame)
        cv2.waitKey(1)
    cv2.waitKey(3000)










# X = []
# Y = []
# print("hahafkdjlsf")

# for i in alphabet:
#     alphapath = os.path.join(datapath,i)
#     for file in os.listdir(alphapath):
#         data = np.load(os.path.join(alphapath, file))
#         X.append(data)
#         Y.append(i)

# print("hahafkdjlsf")

# X = np.array(X) 
# Y = np.array(Y)

# xtrain,xtest,ytrain,ytest  = train_test_split(X,Y, test_size = 0.2)

# Knn = KNeighborsClassifier(n_neighbors = 3 )

# Knn.fit(xtrain,ytrain)

# ypredict = Knn.predict(xtest)

# accuracy = accuracy_score(ytest, ypredict)

# print(accuracy, "model accuracy")