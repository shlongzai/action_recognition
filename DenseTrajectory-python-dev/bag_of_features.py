import random,os,time,multiprocessing,json
from sklearn.cluster import KMeans
import numpy as np


def kmeans(num2):
  matrix_save_path=r'E:\action_recognition\hmdb51\kmeans_train.npy'
  X=np.load(matrix_save_path)
  central_save_path=r'E:\action_recognition\hmdb51\central_train.npy'
  #shape0,shape1=x.shape
  #central=np.zeros((num2,shape1),dtype=np.float16)
  kmeans = KMeans(n_clusters=4000, random_state=0)
  kmeans.fit(X)
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

def bag_of_feature(process_name,feature_list,filename_list,not_dealed_file,
                   share_lock,central,index,filename_json,feature_json,max_num,class_num):
    while True:
        share_lock.acquire()
        if len(not_dealed_file)>0:
            file=not_dealed_file[0]
            del not_dealed_file[0]
            share_lock.release()
            time0=time.time()
            strtime=time.localtime(time0)
            array=np.expand_dims(np.load(file).astype(np.float32),1)
            shape0=array.shape[0]
            iteration_num=(shape0-1)//max_num+1
            
            feature=np.zeros((4000+class_num),dtype=np.int32)
            for i in range(iteration_num):
                arg=np.argmin(np.sum(np.square(array[i*max_num:min((i+1)*max_num,shape0)]-central),axis=2),axis=1)
                feature+=np.bincount(arg,minlength=4000+class_num)
            feature[4000+index]=1
            time1=time.time()
            strtime=time.localtime(time1)
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
            
def main_process(num,max_num):
    share_lock = multiprocessing.Manager().Lock()
    share_flag = multiprocessing.Manager().Value('i',1)
    abspath=os.path.abspath('')
    dir_json=os.path.join(abspath,'dir.json')
    with open(dir_json,'r') as f:
        dir_list=json.load(f)
    class_num=len(dir_list)
    numpy_path=os.path.join(abspath,'numpyfile')
    central_file=os.path.join(abspath,'central_train.npy')
    central=np.expand_dims(np.load(central_file).astype(np.float32),0)
      
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
