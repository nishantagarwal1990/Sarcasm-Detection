# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:19:09 2016

@author: Murali and Nishant
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm
from sklearn.metrics import f1_score,precision_score,recall_score
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline,metrics, grid_search 
from sklearn.ensemble import GradientBoostingClassifier as gbm
from numpy import genfromtxt
#import Ubaseline as bsl
import re
from gensim import corpora, models, similarities
import nltk

#full_data=pd.read_csv("only_hashtags_conv.txt")

def ls_to_txt(file_name,ls):
    with open(file_name,"w") as f:
        for i in ls:
            print>>f,i

def topics_creation(text_data_loc,num_of_topics):
    rexp=re.compile('[^a-zA-Z]')
    english_stemmer = nltk.stem.SnowballStemmer('english')
    stp=nltk.corpus.stopwords.words("english")
    stp=stp+["i","get","i'm","go","it","the","u"]
    uncleaned_data_ls=[]
    with open(text_data_loc,"r") as f:
        for i in f.readlines():
            uncleaned_data_ls.append(i)
    full_data=uncleaned_data_ls
    no_hashtags_text=[bsl.multiple_hashtag_deletion(i,"#")[0] for i in full_data]
    final_text=[[j.lower() for j in i.split() if j.lower() not in stp and rexp.sub("",j.lower())!=''] for i in no_hashtags_text]  
    final_text = [map(english_stemmer.stem,t) for t in final_text]
    dictionary = corpora.Dictionary(final_text)
    #print dictionary
    dictionary.save('tweets_lda.dict') # store the dictionary, for future reference
    corpus = [dictionary.doc2bow(text) for text in final_text]
    #print corpus[0]
    corpora.MmCorpus.serialize('tweets_lda_corpus.mm', corpus)
    model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=num_of_topics,alpha="auto",eval_every=1,iterations=100000)
    topics=[model[c] for c in corpus]
    print model.print_topic(0,topn=20)
    print model.print_topic(2,topn=20)
    #print topics[0]
    #print topics[2]
    return topics_to_df(topics,num_of_topics)
        
def topics_to_df(topics_from_lda,num):
    num_topics=num
    print num_topics
    ls=[]
    for i in topics_from_lda:
        temp_ls=[0.0]*num_topics
        for j in i:
            temp_ls[j[0]]=j[1]
        ls.append(temp_ls)
    topics_df=pd.DataFrame(ls)
    cols=[i for i in range(num_topics)]
    topics_df.columns=cols 
    topics_df.to_csv("topics.csv",index=None)
    return topics_df

if __name__=="__main__":
   #clean_or_not=str(raw_input("Want to load clean tweets or the raw ones say clean for cleaned ones anything else for raw ones:"))
   data_for_labels=pd.read_csv(r"D:/IE Project/sarcasm-data-3000.tsv",delimiter="\t",header=None)
   #topics_data=topics_creation(r"D:\sarcasmtemp\ultimate_combined_tweets.txt",400)
   topics_data=pd.read_csv(r"D:/sarcasmtemp/topics.csv",delimiter=",")
   
   sh=topics_data.shape
   targets=list(data_for_labels[1])
   #full_data=np.delete(full_data,(0),axis=0)
   #tweets=read_tweets(clean_or_not,full_data)
   
   #print full_data.shape
   #full_data[2]=tweets
   #targets=list(full_data[1])
   mod_targets=[1 if i=="SARCASM" else 0 for i in targets]
   #topics_data[sh[1]]=mod_targets
   data_for_labels[1]=mod_targets
   topics_data=topics_data.fillna(0.0)
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
   #del full_data[0]
   #train_cols=range(sh[1])
   
   x_train,x_test,y_train,y_test=train_test_split(topics_data,data_for_labels[1],test_size=0.4,random_state=2,stratify=data_for_labels[1])
   ti=list(y_test.index)
   with open("test_indices.txt","w") as f:
       for i in ti:
           print>>f,i
   
   """
   [ 353,  969,  576,  367,  803,  150, 2694,   65, 2015,  121,
            ...
              67, 2083, 1118, 1561, 2556, 1747,  838, 1706,  848, 2133],
           dtype='int64', length=1192)
   """
   
   #len_train=x_train.shape[0]
   #len_test=x_test.shape[0] 
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
   #print train.shape
   #print test.shape
   #print y_train.shape
   #print y_test.shape
   
   train=x_train
   test=x_test
   print train.shape
   print test.shape
   print y_train.shape
   print y_test.shape
   

   """
   svd = TruncatedSVD(n_components=50,tol=0.001,algorithm='arpack', random_state=5)
   train=svd.fit_transform(train)
   test=svd.transform(test)
   scl = StandardScaler()
   train=scl.fit_transform(train)
   test=scl.transform(test)
   lr_model = lm.LogisticRegression(class_weight="balanced",tol=0.01,C=2,penalty="l2") 
   #train=scl.fit_transform(train)
   #test=scl.transform(test)
   """
   
   lr_model= RandomForestClassifier(random_state=5)
   clf = pipeline.Pipeline([#('svd',svd),('scl', scl),
                            ('lr', lr_model)])
   """                         
   param_grid = {'svd__n_components':[5,10,20,30,40,50],'svd__tol':[0.001,0.01],
                 'lr__tol':[0.001,0.0001,0.01],
                 'lr__C': [1,2,3,4,5,10,2.5],
                  'lr__penalty':["l1","l2"]}
   
   """
   param_grid = {
                'lr__n_estimators':[50,70,75,80,100],
                 'lr__max_features':["auto",None],
                 'lr__max_depth':[5,4,3,2],
                 'lr__min_samples_split':[1,2],
                 'lr__oob_score':[True,False],
                 'lr__class_weight':["balanced","balanced_subsample"]
           }  
   
   """
   param_grid = {'lr__n_estimators':[100,200,300],
                 'lr__learning_rate': [1.0,0.75,0.5,0.25,0.1],
                  'lr__max_depth':[5,4,3,6,7,2],
                  'lr__subsample':[1.0,0.9,0.8],
                  'lr__max_features':["auto","sqrt","log2"],
                  'lr__warm_start':[True,False]  }                  
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
   train_prob=lr_model.predict_proba(train)[:,1]
   test_prob=lr_model.predict_proba(test)[:,1]
   train_preds=lr_model.predict(train)
   target_labels=list(y_test)  
   print f1_score(y_train,train_preds,average="weighted")
   print precision_score(y_train,train_preds,average="weighted")
   print recall_score(y_train,train_preds,average="weighted")
   print f1_score(target_labels,preds,average="weighted")
   print precision_score(target_labels,preds,average="weighted")
   print recall_score(target_labels,preds,average="weighted")

   
   

400 topics optimized for precision score
for the data in best topic models
Best score p: 0.363
f 0.43537414966
r 0.543689320388
Best parameters set:
	lr__C: 2
	lr__penalty: 'l2'
	lr__tol: 0.01
	svd__n_components: 50
	svd__tol: 0.001
 
0.416260162602
0.376470588235
0.465454545455

0.43537414966
0.363047001621
0.543689320388
0.416260162602
0.376470588235
0.465454545455

rf opt prec

#Best score: 0.466
Best parameters set:
	lr__class_weight: 'balanced'
	lr__max_depth: 2
	lr__max_features: None
	lr__min_samples_split: 1
	lr__n_estimators: 75
	lr__oob_score: True

train f,p,r
0.352941176471
0.573770491803
0.254854368932

test f,p,r
0.252747252747
0.516853932584
0.167272727273

"""
#topics_to_df(topics)    