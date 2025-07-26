
import mediapipe as mp
import cv2
import numpy as np
import os



# ASL sign language 

# Goal: Have a functional ASL hand tracker that can detect the ASL signs

# - Download dataset to train the model

# - Mediapipe gives us the coordinates of each landmark

# - Train model for the ASL signs
#     - Inputs - the coordinates of the landmarks in the image
#     - Outputs - ASL signs 


def processframe(frame, hands):
    
    imagergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imagergb)
    mhl = results.multi_hand_landmarks

    pos = []

    if(mhl):
        for hand in mhl:
        
            for id, lm in enumerate(hand.landmark):
                pos.append([lm.x, lm.y, lm.z])
    
        pos = np.array(pos).flatten()
        return pos
    
cam = cv2.VideoCapture(1)


mphands = mp.solutions.hands

hands = mphands.Hands()

# while True:
#     success, frame = cam.read()
    
#     coords = processframe(frame, hands)
#     print(coords)

#     key = cv2.waitKey(1)
#     if(key == 27):
#         break


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


cwd = os.getcwd()

datapath = os.path.join(cwd,'data')

if not os.path.exists(datapath):  
    os.mkdir(datapath)
for i in alphabet: 
    path = os.path.join(datapath,i)
    if not os.path.exists(path): 
        os.mkdir(path)