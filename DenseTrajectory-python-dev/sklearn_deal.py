import json,os,random
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

abspath=r'D:\action_recognition\hmdb51'
jsonfile=os.path.join(abspath,'jsonfile1')
with open(os.path.join(abspath,'dir.json'),'r') as f:
    dir_list=json.load(f)
    
    
random.seed(10)
test_rate=0.2
X_train=[]
y_train=[]
X_test=[]
y_test=[]
for i in dir_list:
    with open(os.path.join(jsonfile,'feature_%s.json'%i),'r') as f:
        data=json.load(f)
    for j in range(len(data)):
        #s=sum(data[j][:4000])
        s=1
        data[j]=[k/s for k in data[j][:4000]]
    length=len(data)
    test_length=int(length*test_rate)
    test_sample=random.sample(range(length),test_length)
    test_sample.sort(reverse=True)
    for j in test_sample:
        X_test.append(data.pop(j))
    X_train.extend(data)
    index=dir_list.index(i)
    y_test.extend([index]*test_length)
    y_train.extend([index]*(length-test_length))
    
    
    
def randomsearchbestmodel(clf,params,name,n_jobs,n_iter):
  clf_grid = RandomizedSearchCV(clf(), params, cv=5, scoring="accuracy",n_jobs=n_jobs,return_train_score=True,verbose=3,n_iter=n_iter)
  clf_grid.fit(np.array(X_train), y_train)
  print(clf_grid.best_params_)
  print(clf_grid.best_score_)
  print(clf_grid.scoring)
  print(clf_grid.cv_results_)
  best_clf=clf(**clf_grid.best_params_)
  best_clf.fit(np.array(X_train),y_train)
  best_clf_predict_train_y=best_clf.predict(np.array(X_train))
  best_clf_predict_test_y=best_clf.predict(np.array(X_test))
  print(sum([1 if i==j else 0 for i,j in zip(best_clf_predict_test_y,y_test)]))
  print(sum([1 if i==j else 0 for i,j in zip(best_clf_predict_train_y,y_train)]))

  d=dict([[i,j] if isinstance(j,list) else [i,j.tolist()] for i,j in clf_grid.cv_results_.items()])
  with open('%s_results.json' % name,'w') as f:
      json.dump(d,f)
      
  
  
from xgboost import XGBClassifier
  
xgboost_params={'n_estimators': [20, 25, 30,40, 50, 60,80,100,120,150,180,200,240,280,320,360],'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6],
                'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3],'subsample': [0.6, 0.7, 0.8, 0.9], 
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
  
randomsearchbestmodel(XGBClassifier,xgboost_params,'xgboost',8,50)



from sklearn.ensemble import RandomForestClassifier

randomforest_params={'n_estimators':[3,5,10,20,40,60,80,100,120,150,180,200,240,270,300,350,400],'max_depth':[6,8,12,15,18,20,24,28,30,None]
                     ,'max_features':[10,12,15,20,25,30,35,40,45,50,55,60,65,70,75,80],'min_samples_split':[2,5,10,15,20,30],'min_samples_leaf':[1,5,10,15,20]}


randomsearchbestmodel(RandomForestClassifier,randomforest_params,'randomforest',8,300)


