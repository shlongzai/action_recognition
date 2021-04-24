import time
from DenseTrajectory.dense import DenseTrajectory
import cv2
import numpy as np
filepath=r'C:\Users\Administrator\Downloads\kick__Baddest_Fight_Scenes_EVER!_-_Kickboxer_-_Part_1_of_2_kick_f_cm_np1_ba_med_4.avi'
s=filepath.split('.')
file_change_path=s[0]+'change.'+s[1]
v=cv2.VideoCapture(filepath)
v2=cv2.VideoCapture(file_change_path)
width=int(v.get(4))
height=int(v.get(3))
m=np.zeros((width*int(v.get(7)),2*height,3),np.uint8)
i=0
print(v.get(7),v2.get(7))
while True:
    flag,frame=v.read()
    # print(type(m),i)
    flag2,frame2=v2.read()
    if flag2==False:
        break
    m[width*i:width*(i+1),:height]=frame
    m[width*i:width*(i+1),height:]=frame2
    i+=1
print(i)
cv2.imshow('imshow',m)
