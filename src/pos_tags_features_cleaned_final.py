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

    
if __name__=="__main__":
   clean_or_not=str(raw_input("Want to load pos tags including stop words or not say no for no stopwords and st"))
   if clean_or_not=="no":
       with open(r"tags_twitter_model.txt","r") as f:
           full_pos_data=[]
           for i in f.readlines():
               full_pos_data.append(i[:-1])
   elif clean_or_not=="st": 
        with open(r"tags_twitter_model_with_stop.txt","r") as f:
           full_pos_data=[]
           for i in f.readlines():
               full_pos_data.append(i[:-1])           
   #full_pos_data=pd.read_csv(r"E:/Sarcasm detection/pos_tags_features_bigrams.csv",delimiter=",",header=None)
   full_data=pd.read_csv(r"sarcasm-data-3000.tsv",delimiter="\t",header=None)
   #full_data=full_data_temp
   #our_labels=list(full_data[1])
   #labels=[1 if i=="SARCASM" else 0 for i in our_labels] 
   targets=list(full_data[1])
   mod_targets=[1 if i=="SARCASM" else 0 for i in targets]
   #full_data[1]=mod_targets
   tfidf_vec=TfidfVectorizer(analyzer="word",max_features=None,strip_accents='unicode',token_pattern=r'\w{1,}',lowercase=True,ngram_range=(1,3),min_df=2,use_idf=True,smooth_idf=True,norm="l2",sublinear_tf=True)
   #full_tf_idf_matrix=tfidf_vec.fit_transform(full_pos_data)
   #full_array=pd.DataFrame(full_tfidf_matrix.toarray())
   #full_array=pd.DataFrame(full_tf_idf_matrix.toarray())
   x_train,x_test,y_train,y_test=train_test_split(full_pos_data,mod_targets,test_size=0.4,random_state=2,stratify=mod_targets)
   train=x_train
   test=x_test
   #print train.shape
   #print test.shape
   x_train=tfidf_vec.fit_transform(x_train)
   x_test=tfidf_vec.transform(x_test)
   x_train=x_train.toarray()
   x_test=x_test.toarray()
   train=x_train
   test=x_test
   #print y_train.shape
   #print y_test.shape
   #train=x_train
   """
   svd = TruncatedSVD(n_components=18,n_iter=9,algorithm='randomized', random_state=None, tol=0.0)
   svd.fit(train)
   train=svd.transform(train)
   test=svd.transform(test)
   scl = StandardScaler()
   train=scl.fit_transform(train)
   test=scl.transform(test)
   """
   
   lr_model=lm.LogisticRegression(C=0.35,penalty="l1",tol=0.01,class_weight="balanced") 
   lr_model.fit(train,y_train)
   preds=lr_model.predict(test)
   preds=list(preds)
   target_labels=list(y_test)
   print f1_score(y_train,lr_model.predict(train),average="weighted")   
   print f1_score(target_labels,preds,average="weighted")        

   
"""
tbr
5 fold cv

Best parameters set:
	lr__C: 0.35
	lr__penalty: 'l1'
	lr__tol: 0.001
 
st (1,3)

train f 0.424140821459
train p 0.323943661972
train r 0.614077669903

test f 0.396946564885
test p 0.305283757339
test r 0.567272727273

tbr rf 
#Best score: 0.342
Best parameters set:
	lr__class_weight: 'balanced_subsample'
	lr__max_depth: 5
	lr__max_features: 'auto'
	lr__min_samples_split: 1
	lr__n_estimators: 100
	lr__oob_score: True
0.377952755906
0.333333333333

train f,p,r
0.608515057113
0.531760435572
0.711165048544

test f,p,r
0.377952755906
0.333333333333
0.436363636364
"""