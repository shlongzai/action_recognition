import os,time,numpy
from DenseTrajectory.dense_4 import  DenseTrajectory as DenseTrajectory4

path1=r'E:\action_recognition\test'

extractor4=DenseTrajectory4()
time_list=[time.time()]
for i in os.listdir(path1):
    path2=os.path.join(path1,i)
    path3=path2.replace('.avi','_change.avi')
    if '.avi' in i and 'change' not in i and not os.path.exists(path3):
        npyfile=path2.replace('.avi','.npy')
        xx=extractor4.compute(path2,path3)
        numpy.save(npyfile,numpy.concatenate(xx,axis=1).astype(numpy.float16))
        time_list.append(time.time())
        print(time_list[1]-time_list[0])
        del time_list[0]
                   
