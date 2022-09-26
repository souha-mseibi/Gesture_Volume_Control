import mediapipe as mp
import cv2
import numpy as np
import uuid
import os 
import math 

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange() 

minVol = volRange[0]
maxVol = volRange [1]
vol=0
volBar=400




mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands 


cap = cv2.VideoCapture(0)
joint_list = [[8,7,6], [4,3,2]]

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2), )
                coords_0 = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)),
                [640,480]).astype(int))
                coords_1 = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y)),
                [640,480]).astype(int))
            
                cv2.circle(image , coords_0, radius =10, color =(0,255,0), thickness=-1 ) 
                cv2.circle(image , coords_1, radius =10, color =(0,255,0), thickness=-1 ) 
                cv2.line(image , coords_0 , coords_1 , color =(0,255,0) , thickness =3 )

                length = math.hypot (coords_1[0] - coords_0[0] , coords_1[1] - coords_0[1])
                
                #hand range = 50 - 300 
                #Volume range = -65 - 0 
                #change the range 
                vol=np.interp(length, [50 ,300] , [minVol , maxVol])
                volBar=np.interp(length, [50 ,300] , [400 , 150])
                print (vol)
                volume.SetMasterVolumeLevel(vol, None)
                cv2.putText(image, f'Volume :' , (20,120),cv2.FONT_HERSHEY_PLAIN , 2, (0,255,0), 3 )
                cv2.rectangle(image, (50,150),(85,400),(0,255,0),3)
                cv2.rectangle(image, (50,int(volBar)),(85,400),(121, 22, 200),cv2.FILLED)


# Save our image    
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

