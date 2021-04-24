import os,time,numpy,multiprocessing
from DenseTrajectory.dense_4 import  DenseTrajectory as DenseTrajectory4

path1=r'E:\action_recognition\hmdb51'

extractor4=DenseTrajectory4()
time_list=[time.time()]
for i in os.listdir(path1):
    path2=os.path.join(path1,i)
    for j in os.listdir(path2):
        if '.avi' in j:
            path3=os.path.join(path2,j)
            npyfile=path3.replace('.avi','.npy')
            if not os.path.exists(npyfile):
                xx=extractor4.compute(path3)
                numpy.save(npyfile,numpy.concatenate(xx,axis=1).astype(numpy.float16))
                time_list.append(time.time())
                print(time_list[1]-time_list[0],xx[0].shape)
                del time_list[0]
