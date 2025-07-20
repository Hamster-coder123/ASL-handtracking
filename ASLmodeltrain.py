import cv2
import mediapipe as mp
import time
import os
from fastai.tabular.all import *



mphands = mp.solutions.hands

hands = mphands.Hands()

currenttime = 0




from pathlib import Path   



def extractlandmarks(path):
    image = cv2.imread(path)
    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imagergb)
    if(results is not None):
        mhl = results.multi_hand_landmarks[0]
        landmarks = []
        for i in mhl:
            landmarks.append(i)
        return landmarks


path = Path("C:\Users\Bruger\Desktop\Vscode Python\ASL tracking\train")

def trainmodel():
    learn = tabular(dls,metrics = accuracy, layers = [200,100], embedding = 0.1)




# 
for i in path.iterdir():










pasttime = 0

d2flag = False
d3flag = False
d4flag = False



    
    

imagergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(imagergb)
mhl = results.multi_hand_landmarks
# print(mhl)


pos = []

if(mhl):
    for hand in mhl:
    
        for id, lm in enumerate(hand.landmark):
            

        

            if(id == 4 or id == 8 or id == 12 or id == 16 or id == 20):
                # print(id, X,Y)
                cv2.circle(frame, (X,Y), 5, (0,255,0), cv2.FILLED)
                pos.append((X, Y))



    p1 = pos[0] #index of the first pair (thumb)
    p2 = pos[1] #index of second (index)
    p3 = pos[2] #index of middle finger
    p4 = pos[3] #index of ring finger
    p5 = pos[4] #index of pinky




    




currenttime = time.time()
fps = int(1 / (currenttime - pasttime))
pasttime = currenttime

cv2.putText(frame, str(fps), (0,80), cv2.FONT_ITALIC, 3, (255,255,255), 3)

cv2.imshow("Camera",frame)

key = cv2.waitKey(1)
if(key == 27):
    break

    # print(success)
    
