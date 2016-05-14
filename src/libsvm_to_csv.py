# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 00:09:26 2016

@author: Murali and Nishant
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm
from sklearn.metrics import f1_score,precision_score
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline,metrics, grid_search 
from numpy import genfromtxt
"""        
def reading_data_libsvm(data_location):
    ls=[]
    with open(data_location,"r") as f:
        ls.append(f.readlines())
    ls=ls[0]
    for i in range(len(ls)):
        ls[i]=ls[i].split()
        #ls[i][0]=int(ls[i][0])
        for j in range(0,len(ls[i])):
            ls[i][j]=float(ls[i][j][ls[i][j].find(":")+1:])
    return ls,np.array(ls)        
"""
def reading_data_libsvm(data_location):
    ls=[]
    storage_ls=[]
    max_feature=-1
    with open(data_location,"r") as f:
        ls.append(f.readlines())
    ls=ls[0]
    for i in range(len(ls)):
        temp_dct={}
        ls[i]=ls[i].split()
        #ls[i][0]=int(ls[i][0])
        for j in range(len(ls[i])):
            current_sub_feature=ls[i][j].split(":")
            temp_dct[int(current_sub_feature[0])]=1    
            if int(current_sub_feature[0])>max_feature:
                max_feature=int(current_sub_feature[0])
        storage_ls.append(temp_dct)        
    #print max_feature    
    new_ls=[]
    for i in range(len(ls)):
        new_sub_ls=[0]*(max_feature+1)
        cur_feature_dict=storage_ls[i]
        for j in cur_feature_dict.keys():
            #print j
            new_sub_ls[j]=1
        new_ls.append(new_sub_ls)
    res_df=pd.DataFrame(new_ls)
    res_df.to_csv("E:/Sarcasm detection/pos_tags_features_bigrams.csv",index=None)    
    return True
    
if __name__=="__main__":
   #clean_or_not=str(raw_input("Want to load clean tweets or the raw ones say clean for cleaned ones anything else for raw ones:"))
   full_pos_data=pd.read_csv(r"E:/Sarcasm detection/pos_tags_features_bigrams.csv",delimiter=",",header=None)
   full_data_temp=pd.read_csv(r"E:/Sarcasm detection/sarcasm-data-3000.tsv",delimiter="\t",header=None)
   #s=full_pos_data.shape
   #print s
   #print s[1]
   #full_data_temp.columns=[0,s[1],2]
   full_data=full_data_temp
   our_labels=full_data_temp[1]
   #aft_concat=pd.concat([full_pos_data,our_labels],axis=1)
   
   #full_data=np.delete(full_data,(0),axis=0)
   #full_data=aft_concat
   #tweets=read_tweets(clean_or_not,full_data)
   
   #print full_data.shape
   #full_data[2]=tweets
   targets=list(full_data[1])
   mod_targets=[1 if i=="SARCASM" else 0 for i in targets]
   full_data[1]=mod_targets
   #features_1_5=pd.read_csv("E:/Sarcasm detection/features1_5.csv",delimiter=",",header=None)
   #features_1_5=pd.read_csv("E:/Sarcasm detection/features1_5.csv",delimiter=",",header=None)
   #features_1_5=features_1_5.drop(features_1_5.index[[0]])
   #features_1_5=np.delete(features_1_5,(0),axis=0)
   """
   full_text=[]
   with open("D:/IE Project/new_cleaned_tweets.txt","r") as f:
        for i in f.readlines():
            full_text.append(i[:-2])
   """        
   #full_text=full_text[0]
   #full_data[2]=full_text    
   #full_text=tweets
   del full_data[0]
   #train_cols=range(1,s[1])
   x_train,x_test,y_train,y_test=train_test_split(full_pos_data,full_data[1],test_size=0.4,random_state=2,stratify=full_data[1])
   len_train=x_train.shape[0]
   len_test=x_test.shape[0] 
   #full_train=[pd.DataFrame(x_train),pd.DataFrame(x_test)]
   #tfidf_vec=TfidfVectorizer(analyzer="word",max_features=None,token_pattern=r'\w{1,}',strip_accents='unicode',lowercase=True,ngram_range=(1,1),min_df=3,use_idf=True,smooth_idf=True,norm="l2",sublinear_tf=True)
   #train_tfidf_matrix=tfidf_vec.fit_transform(x_train)
   #test_tf_idf_matrix=tfidf_vec.transform(x_test)
   #train=pd.DataFrame(train_tfidf_matrix.toarray())
   #test=pd.DataFrame(test_tf_idf_matrix.toarray())
   #train_features= features_1_5[:len_train]
   #test_features=features_1_5[len_train:]
   #print train_features.shape
   #print test_features.shape
   #train=np.append(train,features_1_5[:len_train],axis=1)
   #test=np.append(test,features_1_5[len_train:],axis=1)
   #print "After concat"
   train=x_train
   test=x_test
   print train.shape
   print test.shape
   print y_train.shape
   print y_test.shape
   #train=x_train
   """
   svd = TruncatedSVD(n_components=18,n_iter=9,algorithm='randomized', random_state=None, tol=0.0)
   svd.fit(train)
   train=svd.transform(train)
   test=svd.transform(test)
   scl = StandardScaler()
   train=scl.fit_transform(train)
   test=scl.transform(test)
   
   lr_model = lm.LogisticRegression(penalty="l2",C=15,class_weight="balanced",tol=0.0001) 
   
   """
   svd = TruncatedSVD(algorithm='randomized',random_state=5)
   scl = StandardScaler()
   lr_model = RandomForestClassifier(random_state=5) 
   
   clf = pipeline.Pipeline([('svd', svd),
    						 ('scl', scl),
   
                    	     ('lr', lr_model)])
   """                       
   param_grid = {'svd__n_components' : [5,15,18,25,32],
                 'svd__tol':[0.01,0.001,0.0001],
                 'lr__C': [2,3,5,6,7,8,9,10,11],
                  'lr__tol':[0.01,0.001,0.0001],
                  'lr__penalty':["l1","l2"]}
 
   """
   param_grid = {'svd__n_components' : [5,15,18],'svd__tol':[0.001,0.0001],
                'lr__n_estimators':[100,200],
                 'lr__max_features':["auto",None],
                 'lr__max_depth':[5,4,3],
                 'lr__min_samples_split':[1,2],
                 'lr__oob_score':[True,False],
                 'lr__class_weight':["balanced","balanced_subsample"]
           }   
   
   """
   clf = pipeline.Pipeline([
    						 ('scl', scl),
                    	     ('lr', lr_model)])
   param_grid = {
                 'lr__C': [5,6,7,8,9,10,11,12,13,14,15],
                'lr__penalty':["l1","l2"]}
   """
             
   #f  Scorer 
   f_scorer = metrics.make_scorer(f1_score, greater_is_better = True)
    
    # Initialize Grid Search Model
   model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=f_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=4)
                                     
    # Fit Grid Search Model
   model.fit(train, y_train)
   print("Best score: %0.3f" % model.best_score_)
   print("Best parameters set:")
   best_parameters = model.best_estimator_.get_params()
   for param_name in sorted(param_grid.keys()):
   	  print("\t%s: %r" % (param_name, best_parameters[param_name]))
   best_model = model.best_estimator_
   best_model.fit(train,y_train)
   preds = best_model.predict(test)
   preds=list(preds)
   target_labels=list(y_test)
   print f1_score(target_labels,preds,average="weighted")   
   print precision_score(target_labels,preds,average="weighted")
   
   """
   lr_model.fit(train,y_train)
   preds=lr_model.predict(test)
   preds=list(preds)
   target_labels=list(y_test)
   print f1_score(target_labels,preds,average="weighted")        
   
   """
   """
   trigrams ?
   Best score: 0.336
   Best parameters set:
	lr__C: 14
	lr__penalty: 'l1'
	svd__n_components: 100
	svd__n_iter: 7
   0.320302648172
   
   Bigrams 

Best score: 0.351
Best parameters set:
	lr__C: 13
	lr__penalty: 'l1'
	svd__n_components: 15
	svd__n_iter: 7
0.327272727273

Best score: 0.353
Best parameters set:
	lr__C: 14
	lr__penalty: 'l2'
	svd__n_components: 15
	svd__n_iter: 6
0.320175438596

Best score: 0.353
Best parameters set:
	lr__C: 5
	lr__penalty: 'l1'
	svd__n_components: 18
	svd__n_iter: 9
0.324675324675




unigrams with svd
Best score: 0.346
Best parameters set:
	lr__C: 15
	lr__penalty: 'l1'
	svd__n_components: 15
	svd__n_iter: 11
0.308740068104
   """