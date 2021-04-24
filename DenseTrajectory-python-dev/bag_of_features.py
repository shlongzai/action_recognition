import random,os,time,multiprocessing,json
#from numba import jit
#from sklearn.cluster import KMeans
import numpy as np
def svm():
  pass
def kmeans(num2):
  matrix_save_path=r'E:\action_recognition\hmdb51\kmeans_train.npy'
  X=np.load(matrix_save_path)
  central_save_path=r'E:\action_recognition\hmdb51\central_train.npy'
  #shape0,shape1=x.shape
  #central=np.zeros((num2,shape1),dtype=np.float16)
  kmeans = KMeans(n_clusters=4000, random_state=0)
  kmeans.fit(X)
##  central1=x[random.sample(range(shape0),num2)].astype(np.float64)
##  central0=np.zeros((1,num2,shape1),dtype=np.float64)
##  iteration_count=40
##  iteation_shape0=(shape0-1)//iteration_count+1
##  time0=time.time()
##  while np.sum(np.square(central0-central1[:,:shape1]))>0.001:
##    central0[:,:]=central1[:,:shape1]
##    central1=np.zeros((num2,shape1+1),dtype=np.float64)
##    loss=0
##    for i in range(iteration_count):
##        distance=np.sum(np.square(x[i*iteration_shape0:(i+1)*iteration_shape0]-central0),axis=2)
##        arg=np.argmin(distance,axis=1)
##        for j in range(iteration_shape0):
##            central1[arg[j],:-1]+=x[i]
##            central1[arg,-1]+=1
##            loss+=distance[arg]
##    central1[:,:-1]=central1[:,:-1]/central1[:,-1:]
##    print(loss)
##  time1=time.time()
##  print(time1-time0)
  np.save(central_save_path,kmeans.cluster_centers_)

def kmeans(num2):
  matrix_save_path=r'E:\action_recognition\hmdb51\kmeans_train.npy'
  X=np.load(matrix_save_path)
  central_save_path=r'E:\action_recognition\hmdb51\central_train.npy'
  #shape0,shape1=x.shape
  #central=np.zeros((num2,shape1),dtype=np.float16)
  kmeans = KMeans(n_clusters=4000, random_state=0)
  kmeans.fit(X)
##  central1=x[random.sample(range(shape0),num2)].astype(np.float64)
##  central0=np.zeros((1,num2,shape1),dtype=np.float64)
##  iteration_count=40
##  iteation_shape0=(shape0-1)//iteration_count+1
##  time0=time.time()
##  while np.sum(np.square(central0-central1[:,:shape1]))>0.001:
##    central0[:,:]=central1[:,:shape1]
##    central1=np.zeros((num2,shape1+1),dtype=np.float64)
##    loss=0
##    for i in range(iteration_count):
##        distance=np.sum(np.square(x[i*iteration_shape0:(i+1)*iteration_shape0]-central0),axis=2)
##        arg=np.argmin(distance,axis=1)
##        for j in range(iteration_shape0):
##            central1[arg[j],:-1]+=x[i]
##            central1[arg,-1]+=1
##            loss+=distance[arg]
##    central1[:,:-1]=central1[:,:-1]/central1[:,-1:]
##    print(loss)
##  time1=time.time()
##  print(time1-time0)
  np.save(central_save_path,kmeans.cluster_centers_)

def get_cluster_matrix():
  num1=10**5
  num2=4000
  path2=r"E:\action_recognition\hmdb51\numpyfile"
  file_distribute=[[None,-1]]
  time0=time.time()
  for i in os.listdir(path2):
    print(i)
    path3=os.path.join(path2,i)
    for j in os.listdir(path3):
      path4=os.path.join(path3,j)
      file_distribute.append([path4,file_distribute[-1][1]+np.load(path4).shape[0]])
  time1=time.time()
  print(time1-time0)
  shape1=30+96+300
  matrix_save_path=r'E:\action_recognition\hmdb51\kmeans_train.npy'
  l=file_distribute[-1][1]+1
  a=random.sample(range(l),num1%l)
  a.sort()
  i=0
  left=1
  matrix=np.zeros((num1,shape1),dtype=np.float16)
  while i<num1:
    while file_distribute[left][1]<a[i]:
      left+=1
    array=np.load(file_distribute[left][0])
    while i<num1 and a[i]<=file_distribute[left][1]:
      matrix[i]=array[a[i]-file_distribute[left-1][1]-1]
      i+=1
  time2=time.time()
  print(time2-time1)
  np.save(matrix_save_path,matrix)
#get_cluster_matrix()

#@jit(nopython=True)
def cal(array,index,iteration_num,max_num,class_num,central,shape0):
  feature=np.zeros((4000+class_num),dtype=np.int64)
  for i in range(iteration_num):
      arg=np.argmin(np.sum(np.square(array[i*max_num:min((i+1)*max_num,shape0)]-central),axis=2), axis=1)
      feature+=np.bincount(arg,minlength=4000+class_num)
  feature[4000+index]=1
  return feature
  

def bag_of_feature(process_name,feature_list,filename_list,not_dealed_file,
                   share_lock,central,index,filename_json,feature_json,max_num,class_num):
    #print(process_name)
    #print(len(not_dealed_file))
    while True:
        share_lock.acquire()
        if len(not_dealed_file)>0:
            file=not_dealed_file[0]
            del not_dealed_file[0]
            share_lock.release()
            time0=time.time()
            strtime=time.localtime(time0)
            #print('at %i:%i %s began dealing' %(strtime.tm_hour,strtime.tm_min,process_name))
            array=np.expand_dims(np.load(file).astype(np.float32),1)
            shape0=array.shape[0]
            #shape1=array.shape[2]
            iteration_num=(shape0-1)//max_num+1

            
            #feature=cal(array,index,iteration_num,max_num,class_num,central,shape0)

            
            feature=np.zeros((4000+class_num),dtype=np.int32)
            #array_broadcast=np.zeros((max_num,4000,shape1),np.float32)
            #print('array_broadcast shape:',array_broadcast.shape)            
            for i in range(iteration_num):
                #min_num=min((i+1)*max_num,shape0)
                #sub=min_num-i*max_num
                #array_broadcast[:sub]=np.tile(array[i*max_num:min_num],(1,4000,1))
                #arg=np.argmin(np.sum(np.square(array_broadcast[:sub]-central[:sub]),axis=2), axis=1)
                arg=np.argmin(np.sum(np.square(array[i*max_num:min((i+1)*max_num,shape0)]-central),axis=2),axis=1)
                feature+=np.bincount(arg,minlength=4000+class_num)
            feature[4000+index]=1
##

            
            time1=time.time()
            strtime=time.localtime(time1)
            #print('at %i:%i %s finish dealing,used time %i,shape:%i' %(strtime.tm_hour,strtime.tm_min,process_name,time1-time0,shape0))
            share_lock.acquire()
            filename_list.append(file)
            feature_list.append(feature.tolist())
            with open(filename_json,'w') as f:
                json.dump(list(filename_list),f)
            with open(feature_json,'w') as f:
                json.dump(list(feature_list),f)
            share_lock.release()
        else:
            share_lock.release()
            break
    #print('process %s exit' % process_name)
            
def main_process(num,max_num):
    # 列表声明方式
    # 声明一个进程级共享锁
    # 不要给多进程传threading.Lock()或者queue.Queue()等使用线程锁的变量，得用其进程级相对应的类
    # 不然会报如“TypeError: can't pickle _thread.lock objects”之类的报错
    share_lock = multiprocessing.Manager().Lock()
    share_flag = multiprocessing.Manager().Value('i',1)

    abspath=os.path.abspath('')
    
    dir_json=os.path.join(abspath,'dir.json')
    with open(dir_json,'r') as f:
        dir_list=json.load(f)
    class_num=len(dir_list)
    numpy_path=os.path.join(abspath,'numpyfile')
##    if len(dir_list)==0:
##        dir_list=os.listdir(numpy_path)
##        with open(dir_json,'w') as f:
##            json.dump(dir_list,f)
    central_file=os.path.join(abspath,'central_train.npy')
    #central=np.tile(np.expand_dims(np.load(central_file).astype(np.float32),0),(max_num,1,1))
    central=np.expand_dims(np.load(central_file).astype(np.float32),0)
    #print('central shape',central.shape)    
      
    not_dealed_file=multiprocessing.Manager().list()
    while True:
      flag=True
      process_path_list=os.listdir(numpy_path)
      for i in process_path_list:
          filename_json=os.path.join(abspath,'jsonfile1','filename_%s.json' %i)
          if os.path.exists(filename_json):
            with open(filename_json,'r') as f:
                filename_list=multiprocessing.Manager().list(json.load(f))
          else:
            filename_list=multiprocessing.Manager().list()
          path2=os.path.join(numpy_path,i)
          basename_list=[os.path.basename(file) for file in filename_list]
          for j in os.listdir(path2):
              path4=os.path.join(path2,j)
              if os.path.basename(path4) not in basename_list:
                  not_dealed_file.append(path4)
          print(i,len(not_dealed_file))
          if len(not_dealed_file)>1:
              flag=False
          else:
            continue
            
          feature_json=os.path.join(abspath,'jsonfile1','feature_%s.json' %i)
          if os.path.exists(feature_json):
            with open(feature_json,'r') as f:
                feature_list=multiprocessing.Manager().list(json.load(f))
          else:
            feature_list=multiprocessing.Manager().list()
          
          index=dir_list.index(i)
          process_list = []
          
          for i in range(num):
              process_name='process'+str(i)
              tmp_process = multiprocessing.Process(target=bag_of_feature,args=(process_name,feature_list,filename_list,not_dealed_file,
                                                          share_lock,central,index,filename_json,feature_json,max_num,class_num))
              process_list.append(tmp_process)
              tmp_process.start()
              #print('b')
          for process in process_list:
              process.join()
      if flag:
        break
if __name__ == "__main__":
    main_process(8,128)
