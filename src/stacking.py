# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:40:41 2016

@author: Murali and Nishant
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn import pipeline,metrics, grid_search 
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.ensemble import GradientBoostingClassifier as gbm
#import sklearn.svm.SVC

def read_from_txt(location,labels):
    ls=[]
    if labels==False:
       with open(location,"r") as f:
            for i in f.readlines():
               ls.append(float(i[:-2]))
       return ls
    elif labels==True:
        with open(location,"r") as f:
            for i in f.readlines():
               ls.append(float(i[:-1]))
        return ls
        


if __name__=="__main__":
   tf_idf_train_preds=read_from_txt(r"tfidf_train_prob_preds.txt",False)  
   tf_idf_test_preds=read_from_txt(r"tfidf_test_prob_preds.txt",False)
   S_train_preds=read_from_txt(r"sentiment_train_prob_preds.txt",False)
   S_test_preds=read_from_txt(r"sentiment_test_prob_preds.txt",False)
   pos_train_preds=read_from_txt(r"pos_train_prob_preds.txt",False)
   pos_test_preds=read_from_txt(r"pos_test_prob_preds.txt",False)
   topic_modelling_train_preds=read_from_txt(r"topic_train_prob_preds.txt",False)
   topic_modelling_test_preds=read_from_txt(r"topic_test_prob_preds.txt",False)
   #rf_tfidf_train_preds=read_from_txt(r"useful_rf_tfidf_train_prob_preds.txt",False)
   #rf_tfidf_test_preds=read_from_txt(r"useful_rf_tfidf_test_prob_preds.txt",False)
   rf_POS_train_preds=read_from_txt(r"rf_pos_train_prob_preds.txt",False)
   rf_POS_test_preds=read_from_txt(r"rf_pos_test_prob_preds.txt",False)
   rf_topics_train_preds=read_from_txt(r"rf_topics_train_prob_preds.txt",True)
   rf_topics_test_preds=read_from_txt(r"rf_topics_test_prob_preds.txt",True)
   train_labels=read_from_txt(r"train_labels.txt",True)
   test_labels=read_from_txt(r"test_labels.txt",True)
   train_preds=pd.concat([pd.Series(tf_idf_train_preds),pd.Series(S_train_preds),pd.Series(pos_train_preds),pd.Series(topic_modelling_train_preds),pd.Series(rf_topics_train_preds),pd.Series(rf_POS_train_preds)],axis=1)
   test_preds=pd.concat([pd.Series(tf_idf_test_preds),pd.Series(S_test_preds),pd.Series(pos_test_preds),pd.Series(topic_modelling_test_preds),pd.Series(rf_topics_test_preds),pd.Series(rf_POS_test_preds)],axis=1) 
   #train_preds=pd.concat([pd.Series(tf_idf_train_preds),pd.Series(rf_train_preds)],axis=1)

   #scl = StandardScaler()
   scl = StandardScaler()
   
   train=train_preds
   test=test_preds
   #train=scl.fit_transform(train)
   #test=scl.transform(test)
  
   y_train=pd.Series(train_labels)
   y_test=pd.Series(test_labels)
   """
   lr_model = lm.LogisticRegression(class_weight="balanced") 
   #lr_model=gbm()
   #adab= AdaBoostClassifier()
   #lr_model=adab
   """
   #lr_model = lm.LogisticRegression(class_weight="balanced") 
   lr_model=GradientBoostingClassifier()
   clf = pipeline.Pipeline([('scl',scl),
                    	     ('lr', lr_model)])
  
   param_grid = {'lr__loss':['deviance','exponential'],
                 'lr__learning_rate':[0.25,0.5,0.75,1,2,3,0.15,0.1],
                 'lr__n_estimators':[10,20,50,5],
                 'lr__max_depth':[2,3,4],
                 'lr__min_samples_split':[2,3,4],
                 'lr__min_samples_leaf':[1,2,3],
                 'lr__subsample':[0.25,0.4,0.5,0.75]
                }                        
                          
   """
   param_grid = {'lr__tol':[0.001,0.01,0.0001,0.00001,0.000001],
                 'lr__C': [0.08,0.1,0.09,0.01,0.125,0.5,0.75,1,0.001,0.15,2,3],
                  'lr__penalty':["l1","l2"]}
   """
   """
   param_grid = {'lr__n_estimators':[20,50,100,200,300],
                 'lr__learning_rate': [1.0,0.75,0.5,0.25,0.1]}                
   """              
    #f  Scorer 
   
   f_scorer = metrics.make_scorer(recall_score, greater_is_better = True)
    
    # Initialize Grid Search Model
   model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=f_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)
                                     
    # Fit Grid Search Model
   model.fit(train, y_train)
   print("Best score: %0.3f" % model.best_score_)
   print("Best parameters set:")
   best_parameters = model.best_estimator_.get_params()
   for param_name in sorted(param_grid.keys()):
   	  print("\t%s: %r" % (param_name, best_parameters[param_name]))
   best_model = model.best_estimator_
   best_model.fit(train_preds,y_train)
   preds = best_model.predict(test_preds)
   preds=list(preds)
   target_labels=list(y_test)
   """
   adab= AdaBoostClassifier(n_estimators=200)
   lr_model=adab
   lr_model.fit(train,y_train)
   preds=lr_model.predict(test)
   target_labels=list(y_test)
   """
   
   print f1_score(target_labels,preds,average="binary")
   print precision_score(target_labels,preds,average="binary")
   print recall_score(target_labels,preds,average="binary")
      