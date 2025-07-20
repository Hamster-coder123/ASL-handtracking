import cv2
import mediapipe as mp
import time

'Track all the landmarks'
'make functions for different landmarks'
' track distance between the landmarks of fingers and thumb'
' call the functions for the corresponding finger'
''
''

cam = cv2.VideoCapture(0)

camW = 1280

camH = 720

cam.set(3,camW)
cam.set(4,camH)

mphands = mp.solutions.hands

hands = mphands.Hands()

currenttime = 0

pasttime = 0

d2flag = False
d3flag = False
d4flag = False



while True:
    success, frame = cam.read()
    
    

    imagergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imagergb)
    mhl = results.multi_hand_landmarks
    # print(mhl)


    pos = []

    if(mhl):
        for hand in mhl:
        
            for id, lm in enumerate(hand.landmark):
                X = int(lm.x * camW)
                Y = int(lm.y * camH)


         

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
    
