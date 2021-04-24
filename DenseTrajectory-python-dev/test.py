import time
##from DenseTrajectory.dense import DenseTrajectory
##from DenseTrajectory.dense_1 import DenseTrajectory as DenseTrajectory1
##from DenseTrajectory.dense_2 import  DenseTrajectory as DenseTrajectory2
##from DenseTrajectory.dense_3 import  DenseTrajectory as DenseTrajectory3
from DenseTrajectory.dense_4 import  DenseTrajectory as DenseTrajectory4
from DenseTrajectory.dense_6 import  DenseTrajectory as DenseTrajectory6
import cv2
filepath=r'C:\Users\Administrator\Downloads\kick__Baddest_Fight_Scenes_EVER!_-_Kickboxer_-_Part_1_of_2_kick_f_cm_np1_ba_med_4.avi'
s=filepath.split('.')
#file_change_path=s[0]+'change.'+s[1]
##extractor = DenseTrajectory()
##xx=extractor.compute(filepath)
##time2=time.time()
##print(time2-time1)
##
##extractor1 = DenseTrajectory1()
##xx1=extractor1.compute(filepath)
##time3=time.time()
##print(time3-time2)

##extractor2 = DenseTrajectory2()
##xx2=extractor2.compute(filepath)
##time4=time.time()
##print(time4-time3)
##
##extractor3= DenseTrajectory3()
##xx3=extractor3.compute(filepath)
time5=time.time()
##print(time5-time4)

extractor4= DenseTrajectory4()
xx4=extractor4.compute(filepath)
time6=time.time()
print(time6-time5)
print(xx4[0].shape)

extractor6= DenseTrajectory6()
xx6=extractor6.compute(filepath)
time7=time.time()
print(time7-time6)
print(xx6[0].shape)
