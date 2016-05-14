# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 02:07:01 2016

@author: Murali and Nishant
"""

import pandas as pd
import gensim as gs
from textblob import TextBlob, Word
import numpy as np
import scipy.spatial.distance as ds
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline, grid_search
from afinn import Afinn
import re


eng_stopwords=stopwords.words("english")
#eng_stopwords=stopwords.words("english")
#domain_spec_stopwords=["press","foundations","trends","vol","editor","workshop","international","journal","research","paper","proceedings","conference","wokshop","acm","icml","sigkdd","ieee","pages","springer"]
#eng_stopwords=eng_stopwords+domain_spec_stopwords
    #normal_stopwords=[a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your]
with open(r"full_stopwords.txt","r") as f:
     comp_st=[]
     for i in f.readlines():
         comp_st.append(i[:-1])
compt_st=[i for i in comp_st if i!='']
eng_stopwords=eng_stopwords+comp_st  
afinn_model=Afinn()

with open(r"most_recent_ultimate_tweets.txt","r") as f:
    tweets_wo_hashtag=[]
    for i in f.readlines():
        tweets_wo_hashtag.append(i[:-1])
        
with open(r"best_hashtags_expanded.txt","r") as f:
     hashtags_alone=[]
     for i in f.readlines():
         hashtags_alone.append(i[:-1])

with open(r"AFINN-emoticon-8.txt","r") as f:
    ls_of_emoticons=[]
    for i in f.readlines():
        ls_of_emoticons.append(i.split()[0])
        
with open(r"optimal_splitted_tweets.txt","r") as f:        
    ls_of_sentences=[]
    for  i in f.readlines():
        if "splitter" in i.split():
            ls_of_sentences.append(i[:-1].split("splitter"))
        else:
            ls_of_sentences.append(i[:-1])

with open(r"additional_sarcasm_data_for_w2v.txt","r") as f:
    add_1=[]
    for i in f.readlines():
        add_1.append(i[:-1])
        
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

add_1_cleaned=[multiple_hashtag_deletion(i,"#")[0].lower() for i in add_1]        
        
        
#import math 
"""
def splitting_on_verb_position(ls_of_some_splitted_sentences):
    cur_ls=ls_of_some_splitted_sentences
    res_ls=[]
    c=0
    for i in cur_ls:
        if len(i)>1:
            res_ls.append(i)
        else:
            cur_tags=TextBlob(i[0]).tags
           # print cur_tags
            verb_tags=[j[0] for j in cur_tags if j[1]=="VB"]
            #print verb_tags
            if verb_tags==[]:
                res_ls.append(i)
            else:    
                c=c+1
                print c
                cur_sen_split=" ".join([k[0] for k in cur_tags])
                #print cur_sen_split
                cur_vb_indices=[cur_sen_split.index(j) for j in verb_tags]
                min_diff=len(cur_sen_split)
            #print cur_vb_indices
                for j in cur_vb_indices:
                    #print abs(float(len(cur_sen_split))/2-j)
                    if abs(float(len(cur_sen_split))/2-j)<=min_diff:
                       min_diff=abs(float(len(cur_sen_split))/2-j)
                       opt=j
                #print opt       
                res_ls.append(i[0].split(cur_sen_split[opt]))
    print c            
    return res_ls       

"""
def two_part_splitting(ls_of_some_splitted_sentences):
    cur_ls=ls_of_some_splitted_sentences
    res_ls=[]
    #c=0
    for i in cur_ls:
        #print i
        if len(i)>1 and isinstance(i,list):
            res_ls.append(i)
            
        elif isinstance(i,str) and i.split()!=[]:
            #print i[0].split()
            cur_sen_split=i.split()
            half_index=int(len(cur_sen_split)/2)
            word_to_be_splitted=cur_sen_split[half_index]
            now_ls=i.split(word_to_be_splitted)
            res_ls.append([now_ls[0]]+[word_to_be_splitted+now_ls[1]])
        elif isinstance(i,str) and i.split()==[]:
            #ind=cur_ls.index(i)
            #print i
            #print "here"
            #print i[0].split()
            #print full_raw_data[2][ind]
            res_ls.append(["empty"])
    return res_ls

 
full_raw_data=pd.read_csv(r"sarcasm-data-3000.tsv",delimiter="\t",header=None)

labels=list(full_raw_data[1])
labels=[1 if i=="SARCASM" else 0 for i in labels]

sarcasm_indices=[i for i in range(len(labels)) if labels[i]==1]

x_train,x_test,y_train,y_test=train_test_split(full_raw_data[2],labels,test_size=0.4,random_state=2,stratify=labels)  
     
final_splitted=two_part_splitting(ls_of_sentences)
train_indices=list(x_train.index)
test_indices=list(x_test.index)

not_sarcasm_train=[final_splitted[i] for i in train_indices if i not in sarcasm_indices]
sarcasm_only_train=[final_splitted[i] for i in sarcasm_indices if i in train_indices]
sarcasm_train_str=[" ".join(i) for i in sarcasm_only_train ]
sarcasm_train_ls_of_ls=[re.sub('[^a-zA-Z0-9]'," ",i).split() for i in sarcasm_train_str]

not_sarcasm_train_str=[" ".join(i) for i in not_sarcasm_train]
not_sarcasm_train_ls_of_ls=[re.sub('[^a-zA-Z0-9]'," ",  i).split() for i in not_sarcasm_train_str]

#w2v_sarcasm_model=gs.models.Word2Vec(sarcasm_train_ls_of_ls,size=100,window=5,min_count=1,iter=500,sg=1)
#print "sarcasm model loaded"
#w2v_not_sarcasm_model=gs.models.Word2Vec(not_sarcasm_train_ls_of_ls,size=100,window=5,min_count=1,iter=500,sg=1)
add_1_cleaned=[i.split() for i in add_1_cleaned]
print "after creating add1cleaned"
#print "non sarcasm model loaded"
sarcasm_inc_add=add_1_cleaned+sarcasm_train_ls_of_ls
phrase_model=gs.models.Phrases(sarcasm_inc_add,min_count=3,threshold=0.7)
print "phrase model defined"
sarcasm_inc_add_phrases=phrase_model[sarcasm_inc_add]
print "phrase model loaded"
w2v_sarcasm_inc_additional_model=gs.models.Word2Vec(sarcasm_inc_add_phrases,size=100,min_count=1,iter=50,sg=1)
print "additional sarcasm model loaded"
model=w2v_sarcasm_inc_additional_model
word_vectors = model.syn0
num_clusters = 700
kmeans_clustering = KMeans(n_clusters=num_clusters,n_init=5,max_iter=500,init="k-means++")
idx = kmeans_clustering.fit_predict( word_vectors )
word_centroid_map = dict(zip( model.index2word, idx ))
final_clusters=[]
print "In clusters list creation"
for cluster in xrange(0,num_clusters):
    #print "\nCluster %d" % cluster
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if word_centroid_map.values()[i] == cluster:
            words.append(word_centroid_map.keys()[i])
    final_clusters.append(words)

        
        #print words
#cluster_representation=[]          

"""
sample_sarcasm=w2v_sarcasm_model.most_similar("love",topn=200)        
final_sample_sarcasm=[i for i in sample_sarcasm if i[0] not in eng_stopwords]

sample_not_sarcasm=w2v_not_sarcasm_model.most_similar("love",topn=200)        
final_sample_not_sarcasm=[i for i in sample_not_sarcasm if i[0] not in eng_stopwords]
"""

#full_raw_data=pd.read_csv(r"E:/Sarcasm detection/sarcasm-data-3000.tsv",delimiter="\t",header=None)
emoticon_extractor_func=lambda dataset: [[j for j in ls_of_emoticons if i.find(j)!=-1] for i in dataset ]     

question_words=["who","when","how","why","when","whom","whose","what"]


extracted_emoticons=emoticon_extractor_func(list(full_raw_data[2]))

for i in range(len(extracted_emoticons)):
    if full_raw_data[2][i].find("http://")!=-1:
       #print full_raw_data[2][i]
       extracted_emoticons[i].remove("://")
       extracted_emoticons[i].remove(":/")
           
for i in range(len(tweets_wo_hashtag)):
    tweets_wo_hashtag[i]=tweets_wo_hashtag[i]+" "+" ".join(extracted_emoticons[i])
    #print extracted_emoticons[i]
    #print final_splitted[i]
    final_splitted[i]=final_splitted[i]+extracted_emoticons[i]
           
tb_pol=lambda x: TextBlob(x).polarity
tb_subj=lambda x: TextBlob(x).subjectivity
afinn_Sentiment=lambda x: afinn_model.score(x)
afinn_scores_tw=[afinn_Sentiment(i) for i in tweets_wo_hashtag]          

def sentiment_shift(tweet_splitted_into_sentences,method):
    if method=="afinn":
        diff_score=0
        for i in range(1,len(tweet_splitted_into_sentences)):
                diff_score+=abs(afinn_Sentiment(tweet_splitted_into_sentences[i])-afinn_Sentiment(tweet_splitted_into_sentences[i-1]))
        return diff_score
    elif method=="tb":
       diff_score=0
       for i in range(1,len(tweet_splitted_into_sentences)):
                diff_score+=abs(tb_pol(tweet_splitted_into_sentences[i])-tb_pol(tweet_splitted_into_sentences[i-1]))
       return diff_score

def tb_sub_shift(tweet_splitted_into_sentences):
    diff_score=0
    for i in range(1,len(tweet_splitted_into_sentences)):
        diff_score+=abs(tb_subj(tweet_splitted_into_sentences[i])-tb_subj(tweet_splitted_into_sentences[i-1]))
    return diff_score     
    

def sentiment_contrast_tweet_hashtag(tweet_splitted_into_sentences,corresponding_hashtag,method):
    max_contrast=0
    if corresponding_hashtag=="empty_hashtag":
        return 0.0
    else:
        if method=="afinn":
           for i in tweet_splitted_into_sentences:
               cur_diff=abs(afinn_Sentiment(i)-afinn_Sentiment(corresponding_hashtag))
               if cur_diff>max_contrast:
                   max_contrast=cur_diff
           return max_contrast
        elif method=="tb":
            for i in tweet_splitted_into_sentences:
                cur_diff=abs(tb_pol(i)-tb_pol(corresponding_hashtag))
                if cur_diff>max_contrast:
                    max_contrast=cur_diff
            return max_contrast

vb_tags=["VB","VBZ","VBD","VBP","VBN","VBG"]
adj_tags=["JJ","JJR","JJS"]
nn_tags=["NN","NNS","NNP","NNPS"]
adverb_tags=["RB","RBR","RBS"]

def extract_most_interesting_words(tweet_splitted_into_sentences):
    if len(tweet_splitted_into_sentences)==1:
        return ["empty"]
    else:    
        final_ls=[]
        #cur_vb=[]
        #cur_adj=[]
        #cur_nn=[]
        #print tweet_splitted_into_sentences
        #print " ".join(tweet_splitted_into_sentences)
        cur_tags=TextBlob(" ".join(tweet_splitted_into_sentences)).tags
        
        req_tags=[]
        for j in range(len(cur_tags)):
               if cur_tags[j][1] in vb_tags:
                   if j-1>=0:
                      if cur_tags[j-1][1] in adverb_tags:
                          if cur_tags[j-1][0] not in eng_stopwords:
                             req_tags.append(cur_tags[j-1][0])
                          if cur_tags[j][0] not in eng_stopwords:   
                             req_tags.append(cur_tags[j][0])
                      else:
                          if cur_tags[j][0] not in eng_stopwords:
                             req_tags.append(cur_tags[j][0])
               elif cur_tags[j][1] in adj_tags:
                   if cur_tags[j][0] not in eng_stopwords:
                      req_tags.append(cur_tags[j][0])           
        
        for i in tweet_splitted_into_sentences:
            temp_ls=[]
            #cur_tags=TextBlob(i).tags
            for j in i.split():
                for k in req_tags:
                    if k==j:
                        temp_ls.append(k)
                        req_tags.remove(k)
            final_ls.append(temp_ls)            
            #for j in cur_tags:
            #    if j[1] in vb_tags or j[1] in adj_tags or j[1] in nn_tags:
            #        if j[0] not in stopwords:
            #           temp_ls.append(j[0])
            #final_ls.append(temp_ls) 
    #print final_ls        
    return final_ls

#extracted_interesting_words=[extract_most_interesting_words(i) for i in sarcasm_only_train+not_sarcasm_train]        
                
def presence_of_adverb_verb(tweet_as_a_sentence):
   pos_tags=TextBlob(tweet_as_a_sentence).tags                
   for i in range(len(pos_tags)):
       if pos_tags[i][1] in vb_tags and pos_tags[i][0] not in eng_stopwords:
          if i-1>0:
             if pos_tags[i-1][1] in adverb_tags and pos_tags[i-1][0] not in eng_stopwords:
                 return 1
             else:
                 return 0

import cPickle
with open("D:\sarcasmtemp\most_int_words_train_split","r") as f:
      train_int_words=cPickle.load(f)
 
with open("D:\sarcasmtemp\most_int_words_test_split","r") as f:
      test_int_words=cPickle.load(f)


 
import itertools
from scipy.spatial.distance import cosine

train_int_words_str=[" ".join([k for j in i for k in j]) for i in train_int_words]
test_int_words_str=[" ".join([k for j in i for k in j]) for i in test_int_words]
train_int_words_str=[[k for j in i for k in j] for i in train_int_words]
test_int_words_str=[[k for j in i for k in j] for i in test_int_words]

#individual_cosine_comp= lambda v1,v2: cosine(v1,v2)


def word2vec_sim_calc(splitted_sentence,sarcasm_model,non_sarcasm_model,default_answer):
    ls_s=[]
    ls_ns=[]
    for i in splitted_sentence:
        inis=[0]*sarcasm_model.syn0.shape[1]
        inins=[0]*sarcasm_model.syn0.shape[1]
        for j in i.split():
            #print j
            if j[-1]=="!":
                j=j[:-1]
            if j in sarcasm_model.vocab.keys() and j in non_sarcasm_model.vocab.keys():
                if j not in eng_stopwords:
                   inis=inis+sarcasm_model[j]
                   inins=inins+non_sarcasm_model[j]          
        if sum(inis)==0.0 and sum(inins)==0.0:   
           print "empty"
           print i
           for j in i.split():
              if j in sarcasm_model.vocab.keys() and j in non_sarcasm_model.vocab.keys():
                   inis=inis+sarcasm_model[j]
                   inins=inins+non_sarcasm_model[j] 
        ls_s.append(inis)
        ls_ns.append(inins)
    #print len(ls_s)
    #print len(ls_ns)
    if len(ls_s)<=1 or len(ls_ns)<=1:
       return default_answer
    else:
        ls_s_comb=list(itertools.combinations(ls_s,2))
        ls_ns_comb=list(itertools.combinations(ls_ns,2))
        s_cosine=sum([cosine(i[0],i[1]) for i in ls_s_comb])
        ns_cosine=sum([cosine(i[0],i[1]) for i in ls_ns_comb])
        s_cosine=s_cosine/len(ls_s_comb)
        ns_cosine=ns_cosine/len(ls_ns_comb)
        if s_cosine>=ns_cosine:
            return 1
        else:   
            return 0

         
            
#if __name__=="__main__":
    
       
         