import time
from DenseTrajectory.dense import DenseTrajectory
import cv2
import numpy as np
filepath=r'C:\Users\Administrator\Downloads\kick__Baddest_Fight_Scenes_EVER!_-_Kickboxer_-_Part_1_of_2_kick_f_cm_np1_ba_med_4.avi'
s=filepath.split('.')
file_change_path=s[0]+'change.'+s[1]
v=cv2.VideoCapture(filepath)
if v.isOpened():
  m=np.zeros((int(v.get(7)),int(v.get(4)),int(v.get(3)),3),np.uint8)
  i=0
  while True:
    flag,frame=v.read()
    # print(type(m),i)
    if flag==False:
      break
    m[i]=frame
    i+=1
print(i)
for i in range(m.shape[0]):
    cv2.imshow('imshow'+str(i),m[i])
