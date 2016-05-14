# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:23:06 2016

@author: Murali and Nishant
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
import nltk
import gensim as gs
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline,metrics, grid_search 
from numpy import genfromtxt
from sklearn.svm import SVC 
#import feature_extraction as fe
#stop=stopwords.words("english")

def multiple_hashtag_deletion(sentence,hashtag):
    sentence=sentence+" "
    lower_sent=sentence.lower()
    c=lower_sent.count(hashtag)
    ls_of_hashtags=[]
    while c>0:
        ind=lower_sent.find(hashtag)
        sp=lower_sent.find(" ",ind+1)
        ls_of_hashtags.append(sentence[ind:sp])
        sentence=sentence[:ind]+sentence[sp:]
        lower_sent=lower_sent[:ind]+lower_sent[sp:]
        c=c-1
    return sentence,ls_of_hashtags    
    
def read_tweets(cleannot,dataset):
    if cleannot=="clean":
       full_text=[]
       #with open("E:/Sarcasm detection/tweets_hashtag_expanded.txt","r") as f:
       with open("E:/Sarcasm detection/ultimate_combined_tweets.txt","r") as f:
           for i in f.readlines():
              full_text.append(i[:-1])
       print len(full_text)       
       return full_text 
    else:
        ls=list(dataset[2])
        new_ls=[]
        for i in ls:
            new_ls.append(multiple_hashtag_deletion(i.lower(),"#sarcasm")[0])
        return new_ls    
        
        
        
if __name__=="__main__":
   clean_or_not=str(raw_input("Want to load clean tweets or the raw ones say clean for cleaned ones anything else for raw ones:"))
   full_data=pd.read_csv(r"sarcasm-data-3000.tsv",delimiter="\t",header=None)
   #full_data=np.delete(full_data,(0),axis=0)
   tweets=read_tweets(clean_or_not,full_data)
   """
   phrases_inp=[i.split() for i in tweets]
   phrase_model=gs.models.Phrases(phrases_inp,min_count=2,threshold=5)
   phrases_op=list(phrase_model[phrases_inp])
   phrases_tweets=[" ".join(i) for i in phrases_op]
   print phrases_tweets[0]
   print phrases_tweets[1]
   print full_data.shape
   """
   full_data[2]=tweets
   targets=list(full_data[1])
   mod_targets=[1 if i=="SARCASM" else 0 for i in targets]
   full_data[1]=mod_targets
   features_1_5=pd.read_csv("features1_5.csv",delimiter=",",header=None)
   #features_1_5=pd.read_csv("E:/Sarcasm detection/features1_5.csv",delimiter=",",header=None)
   features_1_5=features_1_5.drop(features_1_5.index[[0]])
   #features_1_5=np.delete(features_1_5,(0),axis=0)
   """
   full_text=[]
   with open("D:/IE Project/new_cleaned_tweets.txt","r") as f:
        for i in f.readlines():
            full_text.append(i[:-2])
   """        
   #full_text=full_text[0]
   #full_data[2]=full_text    
   full_text=tweets
   del full_data[0]
   #x_train,x_test,y_train,y_test=train_test_split(full_data[2],full_data[1],test_size=0.4,random_state=2,stratify=full_data[1])
   #len_train=x_train.shape[0]
   #len_test=x_test.shape[0] 
   #full_train=[pd.DataFrame(x_train),pd.DataFrame(x_test)]
   #token_pattern=r'\w{1,}'
   tfidf_vec=TfidfVectorizer(analyzer="word",max_features=None,strip_accents='unicode',token_pattern=r'\w{1,}',lowercase=True,ngram_range=(1,2),min_df=2,use_idf=True,smooth_idf=True,norm="l2",sublinear_tf=True)
   #tfidf_vec=TfidfVectorizer(analyzer="word",max_features=None,lowercase=True,ngram_range=(1,2),min_df=2,use_idf=True,smooth_idf=True,norm="l2",sublinear_tf=True,stop_words="english")
   
   full_tfidf_matrix=tfidf_vec.fit_transform(full_data[2])
   full_array=pd.DataFrame(full_tfidf_matrix.toarray())
   
   #test_tf_idf_matrix=tfidf_vec.transform(x_test)
   #train=pd.DataFrame(train_tfidf_matrix.toarray())
   #test=pd.DataFrame(test_tf_idf_matrix.toarray())
   #train_features= features_1_5[:len_train]
   #test_features=features_1_5[len_train:]
   #print train_features.shape
   #print test_features.shape
   print full_array.shape
   print features_1_5.shape
   full_append=np.append(full_array,features_1_5,axis=1)
   #full_append=full_tfidf_matrix
   #print full_append.shape
   #print full_data[1].shape
   #full_append=np.append(full_append,np.array(full_data[1]).reshape(2980,1),axis=1)
   #np.random.shuffle(full_append)
   x_train,x_test,y_train,y_test=train_test_split(full_append,full_data[1],test_size=0.4,random_state=2,stratify=full_data[1])
   train=x_train
   test=x_test
   #train=np.append(train,features_1_5[:len_train],axis=1)
   #test=np.append(test,features_1_5[len_train:],axis=1)
   print "After concat"
   print train.shape
   print test.shape
   print y_train.shape
   print y_test.shape
   
   #85 best for clean

   svd = TruncatedSVD(n_components=190,algorithm='arpack', random_state=5, tol=0.01)
   svd.fit(train)
   train=svd.transform(train)
   test=svd.transform(test)
   scl = StandardScaler()
   train=scl.fit_transform(train)
   test=scl.transform(test)
   
   lr_model = lm.LogisticRegression(penalty="l1",C=1,class_weight="balanced",tol=0.001) 
   

   lr_model.fit(train,y_train)
   preds = lr_model.predict(test)
   preds=list(preds)
   target_labels=list(y_test)
   print f1_score(target_labels,preds,average="weighted")
   """
   preds=best_model.predict(test)
   preds=list(preds)
   target_labels=list(y_test)
   print f1_score(target_labels,preds,average="weighted")

with open("pos_train_bin_preds.txt","w") as f:
      for i in train_preds:
          print>>f,i
 
with open("pos_test_bin_preds.txt","w") as f:
      for i in preds:
          print>>f,i
   
THE BEST
consistent non random svd
80
0.0001 /0.001
14
l1
0.01

Best score: 0.606
Best parameters set:
	lr__C: 15
	lr__penalty: 'l1'
	lr__tol: 0.01
	svd__n_components: 100
	svd__n_iter: 2
	svd__tol: 0.001
0.610315186246

Best score: 0.591
Best parameters set:
	lr__C: 14
	lr__penalty: 'l1'
	lr__tol: 0.01
	svd__n_components: 90
	svd__tol: 0.0001
0.595375722543

Not cleaned  unigrams
Best score: 0.554
Best parameters set:
	lr__C: 11
	lr__penalty: 'l2'
	svd__n_components: 200
	svd__n_iter: 4
0.575221238938

cleaned unigrams
Best score: 0.521
Best parameters set:
	lr__C: 13
	lr__penalty: 'l2'
	svd__n_components: 250
	svd__n_iter: 4
0.512301013025

not cleaned unigrams and bigrams

Best score: 0.583
Best parameters set:
	lr__C: 17
	lr__penalty: 'l2'
	svd__n_components: 200
	svd__n_iter: 3
0.576368876081

cleaned unigrams and bigrams

Best score: 0.496
Best parameters set:
	lr__C: 12
	lr__penalty: 'l1'
	svd__n_components: 200
	svd__n_iter: 4
0.477011494253

"""   
"""
useful  cleaned 3

Best score: 0.532
Best parameters set:
	lr__C: 13
	lr__penalty: 'l1'
	lr__tol: 0.01
	svd__n_components: 110
	svd__tol: 0.001

0.523449319213

useful no 3
Best score: 0.523
Best parameters set:
	lr__C: 10
	lr__penalty: 'l1'
	lr__tol: 0.001
	svd__n_components: 200
	svd__tol: 0.01
0.528593508501

useful no 2
Best score: 0.524
Best parameters set:
	lr__C: 8
	lr__penalty: 'l1'
	lr__tol: 0.01
	svd__n_components: 90
	svd__tol: 0.01
0.531914893617

useful no 2
Best score: 0.524
Best parameters set:
	lr__C: 0.5
	lr__penalty: 'l1'
	lr__tol: 0.01
	svd__n_components: 200
	svd__tol: 0.01
0.533123028391
"""

"""
lty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8,tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=100, score=0.584615 -   7.4s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.01, lr__C=8, svd__n_components=110 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.01, lr__C=8, svd__n_components=110, score=0.597701 -   8.2s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=110 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=110, score=0.597701 -   8.2s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=120 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=120, score=0.575875 -   8.9s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.01, lr__C=8, svd__n_components=150 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.01, lr__C=8, svd__n_components=150, score=0.556863 -  11.5s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=150 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=150, score=0.556863 -  11.6s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=170 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=170, score=0.584000 -  13.4s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.01, lr__C=8, svd__n_components=200 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.01, lr__C=8, svd__n_components=200, score=0.561265 -  16.1s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=200 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=200, score=0.561265 -  15.7s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=90 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=90, score=0.504202 -   6.4s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=100 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=100, score=0.500000 -   7.3s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=100 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=100, score=0.500000 -   7.4s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=110 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=110, score=0.523077 -   8.1s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=120 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=120, score=0.517928 -   8.9s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=120 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=120, score=0.517928 -   9.1s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=150 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=150, score=0.540984 -  11.5s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=170 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=170, score=0.495868 -  13.6s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=170 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=170, score=0.495868 -  13.5s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=200 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=200, score=0.486957 -  16.3s
[CV] lr__penalty=l1, lr__tol=0.01, svd__tol=0.01, lr__C=10, svd__n_components=90 
[CV]  lr__penalty=l1, lr__tol=0.01, svd__tol=0.01, lr__C=10, svd__n_components=90, score=0.505837 -   6.6s
[CV] lr__penalty=l1, lr__tol=0.01, svd__tol=0.001, lr__C=10, svd__n_components=90 
[CV]  lr__penalty=l1, lr__tol=0.01, svd__tol=0.001, lr__C=10, svd__n_components=90, score=0.505837 -   6.6s
[CV] lr__penalty=l1, lr__tol=0.01, svd__tol=0.01, lr__C=10, svd__n_components=100 
[CV]  lr__penalty=l1, lr__tol=0.01, svd__tol=0.01, lr__C=101, svd__tol=0.01, lr__C=8, svd__n_components=100, score=0.513834 -   7.4s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=100 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=100, score=0.513834 -   7.4s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=110 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=110, score=0.523077 -   8.3s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.01, lr__C=8, svd__n_components=120 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.01, lr__C=8, svd__n_components=120, score=0.517928 -   9.2s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=120 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=120, score=0.517928 -   9.2s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=150 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=150, score=0.540984 -  11.8s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.01, lr__C=8, svd__n_components=170 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.01, lr__C=8, svd__n_components=170, score=0.495868 -  13.6s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=170 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.0001, lr__C=8, svd__n_components=170, score=0.495868 -  13.7s
[CV] lr__penalty=l2, lr__tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=200 
[CV]  lr__penalty=l2, lr__tol=0.001, svd__tol=0.001, lr__C=8, svd__n_components=200, score=0.486957 -  16.9s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=90 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=90, score=0.507813 -   6.6s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=90 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=90, score=0.500000 -   6.6s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=100 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=100, score=0.492308 -   7.5s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=110 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=110, score=0.476562 -   8.4s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=110 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=110, score=0.476562 -   8.4s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=120 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=120, score=0.513619 -   9.2s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=150 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=150, score=0.488189 -  11.9s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=150 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=150, score=0.488189 -  11.8s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=170 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.001, lr__C=8, svd__n_components=170, score=0.497992 -  13.7s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=200 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.01, lr__C=8, svd__n_components=200, score=0.561265 -  16.6s
[CV] lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=200 
[CV]  lr__penalty=l2, lr__tol=0.0001, svd__tol=0.0001, lr__C=8, svd__n_components=200, score=0.486957 -  16.5s
[CV] lr__penalty=l1, lr__tol=0.01, svd__tol=0.0001, lr__C=10, svd__n_components=90 
[CV]  lr__penalty=l1, lr__tol=0.01, svd__tol=0.0001, lr__C=10, svd__n_components=90, score=0.505837 -   8.0s
"""

"""
CV] lr__penalty=l1, lr__tol=0.0001, svd__tol=0.001, lr__C=11, svd__n_components=200 
[CV]  lr__penalty=l1, lr__tol=0.0001, svd__tol=0.001, lr__C=11, svd__n_components=200, score=0.555556 -  16.2s
[CV] lr__penalty=l2, lr__tol=0.01, svd__tol=0.01, lr__C=11, svd__n_components=90 
[CV]  lr__penalty=l2, lr__tol=0.01, svd__tol=0.01, lr__C=11, svd__n_components=90, score=0.590909 -   6.5s
[CV] lr__penalty=l2, lr__tol=0.01, svd__tol=0.001, lr__C=11, svd__n_components=90 
[CV]  lr__penalty=l2, lr__tol=0.01, svd__tol=0.001, lr__C=11, svd__n_components=90, score=0.590909 -   6.4s
[CV] lr__penalty=l2, lr__tol=0.01, svd__tol=0.01, lr__C=11, svd__n_components=100 
[CV]  lr__penalty=l2, lr__tol=0.01, svd__tol=0.01, lr__C=11, svd__n_components=100, score=0.584615 -   7.3s
[CV] lr__penalty=l2, lr__tol=0.01, svd__tol=0.0001, lr__C=11, svd__n_components=100 
[CV]  lr__penalty=l2, lr__tol=0.01, svd__tol=0.0001, lr__C=11, svd__n_components=100, score=0.584615 -   7.4s
[CV] lr__penalty=l2, lr__tol=0.01, svd__tol=0.001, lr__C=11, svd__n_components=110 
[CV]  lr__penalty=l2, lr__tol=0.01, svd__tol=0.001, lr__C=11, svd__n_components=110, score=0.597701 -   8.1s
[CV] lr__penalty=l2, lr__tol=0.01, svd__tol=0.01, lr__C=11, svd__n_components=120 
[CV]  lr__penalty=l2, lr__tol=0.01, svd__tol=0.01, lr__C=11, svd__n_components=120, score=0.573643 -   8.9s
[CV] lr__penalty=l2, lr__tol=0.01, svd__tol=0.0001, lr__C=11, svd__n_components=120 
[CV]  lr__penalty=l2, lr__tol=0.01, svd__tol=0.0001, lr__C=11, svd__n_components=120, score=0.573643 -   8.8s
[CV] lr__penalty=
"""

"""
CV]  lr__penalty=l1, lr__tol=0.01, svd__tol=0.01, lr__C=1, svd__n_components=120, score=0.593750 -  15.1s
[CV] lr__penalty=l1, lr__tol=0.01, svd__tol=0.0001, lr__C=1, svd__n_components=120 
[CV]  lr__penalty=l1, lr__tol=0.01, svd__tol=0.0001, lr__C=1, svd__n_components=120, score=0.593750 -  14.9s
[CV] lr__penalty=l1, lr__tol=0.01, svd__tol=0.001, lr__C=1, svd__n_components=150 
[CV]  lr__penalty=l1, lr__tol=0.01, svd__tol=0.001, lr__C=1, svd__n_components=150, score=0.603175 -  18.8s
"""
"""
lr__penalty=l2, lr__tol=0.01, svd__tol=0.001, lr__C=0.5, svd__n_components=150, score=0.600000 -  19.7s
 """