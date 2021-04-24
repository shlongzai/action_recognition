import os,time,numpy,multiprocessing
from DenseTrajectory.dense_4 import  DenseTrajectory as DenseTrajectory4

#path1=r'E:\action_recognition\hmdb51'

##extractor4=DenseTrajectory4()
##time_list=[time.time()]
##for i in os.listdir(path1):
##    path2=os.path.join(path1,i)
##    for j in os.listdir(path2):
##        if '.avi' in j:
##            path3=os.path.join(path2,j)
##            npyfile=path3.replace('.avi','.npy')
##            if not os.path.exists(npyfile):
##                xx=extractor4.compute(path3)
##                numpy.save(npyfile,numpy.concatenate(xx,axis=1).astype(numpy.float16))
##                time_list.append(time.time())
##                print(time_list[1]-time_list[0],xx[0].shape)
##                del time_list[0]
def get_filelist(path1,path2,share_list,share_lock,share_flag):
    share_lock.acquire()
    count=0
    flag=True
    for i in os.listdir(path1):
        path1_1=os.path.join(path1,i)
        for j in os.listdir(path1_1):
            path1_2=os.path.join(path1_1,j)
            npyfile=os.path.join(path2,i,j.replace('.avi','.npy'))
            if not os.path.exists(npyfile):
                share_list.append([path1_2,npyfile])
                if flag and len(share_list)>=20:
                    share_lock.acquire()
    print('get_function_finish1')
    share_flag=0
    print('get_list_length:%i' % len(share_list))
    print(count)
    print(len(share_list))
    print(sum([len(os.listdir(os.path.join(path1,i))) for i in os.listdir(path1)]))
    print(sum([len(os.listdir(os.path.join(path2,i))) for i in os.listdir(path2)]))
    share_lock.release()
        
def deal(process_name,share_list,share_lock,share_flag):
    extractor4=DenseTrajectory4()
    time0=time.time()
    sleeping_count=0
    while True:
        share_lock.acquire()
        if len(share_list)>0:
            sleeping_count=0
            path3,npyfile=share_list[0][0],share_list[0][1]
            del share_list[0]
            time0=time.time()
            strtime=time.localtime(time0)
            #npyfile=path3.replace('.avi','.npy')
            share_lock.release()
            print('at %i:%i %s began dealing' %(strtime.tm_hour,strtime.tm_min,process_name))
            xx=extractor4.compute(path3)
            try:
                numpy.save(npyfile,numpy.concatenate(xx,axis=1).astype(numpy.float16))
            except Exception as ee:
                print(ee)
            time1=time.time()
            strtime=time.localtime(time1)
            print('at %i:%i %s finish dealing,used time %i,shape:%i' %(strtime.tm_hour,strtime.tm_min,process_name,time1-time0,xx[0].shape[0]))
        else:
            share_lock.release()
            if share_flag==0:
                break
            time.sleep(10)
            sleeping_count+=1
            print('deal sleeping count',sleeping_count)
    print('process %s exit' % process_name)
def main_process(num):
    # 列表声明方式
    share_list = multiprocessing.Manager().list()
    # 声明一个进程级共享锁
    # 不要给多进程传threading.Lock()或者queue.Queue()等使用线程锁的变量，得用其进程级相对应的类
    # 不然会报如“TypeError: can't pickle _thread.lock objects”之类的报错
    share_lock = multiprocessing.Manager().Lock()
    share_flag = multiprocessing.Manager().Value('i',1)
    process_list = []
    path1=r'D:\action_recognition\hmdb51\videofile'
    path2=r'D:\action_recognition\hmdb51\numpyfile'
    #zippath=r'E:\action_recognition\hmdb51\zipfile'
    process_get=multiprocessing.Process(target=get_filelist,args=(path1,path2,share_list,share_lock,share_flag))
    process_get.start()
    print('a')
    process_get.join()
    print(len(share_list))
    for i in range(num):
        process_name='process'+str(i)
        tmp_process = multiprocessing.Process(target=deal,args=(process_name,share_list,share_lock,share_flag))
        process_list.append(tmp_process)
        tmp_process.start()
        print('b')
    print('get_function_finish2')
    time.sleep(20)
    for process in process_list:
        process.join()
    time.sleep(20)
if __name__ == "__main__":
    main_process(4)
