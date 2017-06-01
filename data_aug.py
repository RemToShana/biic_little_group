# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:37:39 2017

@author: Donkey
"""


from __future__ import print_function

#import tensorflow as tf
#from tensorflow.contrib import rnn
#from tensorflow.python.ops import rnn_cell
import numpy as np
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#imdb=input_data.read_data_sets('imdb_word_emb.npz',one_hot=True)
#==============================================================================
# 
# imdb = np.load('imdb.npz')
# X_train_s = imdb['x_train']
#==============================================================================


#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
################################################################################
import re
import json
import numpy as np
import pandas as pd
import gensim
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report

from IPython.display import display
#import joblib
import jieba
import os

folder_path = '.\\data\\'
#dataset = load_files(container_path=folder_path, shuffle=False)
#
#texts=dataset.data
#answer=dataset.target

alldata=[]
tanswer=[]
readpath='.\\data\\ASD32\\'
for root, subdir, files in os.walk(readpath, topdown=False):
    files.sort()
    for w in files:
        print(root+w)
        path=root+w
        try:
            one_file = open(path,'r',encoding='utf8')
            alldata.append(one_file.read())
            tanswer.append(1)
        except:
            one_file = open(path,'r')
            alldata.append(one_file.read())
            tanswer.append(1)
readpath='.\\data\\TD36\\'
for root, subdir, files in os.walk(readpath, topdown=False):
    files.sort()
    for w in files:
        print(root+w)
        path=root+w
        try:
            one_file = open(path,'r',encoding='utf8')
            alldata.append(one_file.read())
            tanswer.append(0)
        except:
            one_file = open(path,'r')
            alldata.append(one_file.read())
            tanswer.append(0)
def jiebacut(alltext):
    jieout=[]
    print('cutting')
    for i in range(0,len(alltext)):    
        print(i)
        tmp=[]
        this=alltext[i]
#        this= alltext[i].replace('\n','')
#        this= alltext[i].replace('　','')
#        this= alltext[i].replace(' ','')
    
        this = jieba.cut(this, cut_all=False)
        for word in this:
            tmp.append(word)
        jieout.append(tmp)
    return jieout

#jieout=[]
#for i in range(0,len(alltext)):    
#    tmp=[]
#    this= alltext[i].replace('\n','')
#    this= alltext[i].replace('　','')
#    this= alltext[i].replace(' ','')
#
#    this = jieba.cut(this, cut_all=False)
#    for word in this:
#        tmp.append(word)
#    jieout.append(tmp)
    
def eli(jieout):    
    punc='，。！？：／」「()（）％、１２３４５６７８９０0123456789'+' '
#    punc = punc.decode("utf-8")
    stop_word=set(['，','。','！','？','：','／','〔','．','；', '．','〕'])
    stop_words=set(punc)|stop_word
    final=[]
    print('clean')
    for j in range(0,len(jieout)):
        print('clean',j)
        filtered_sentence=[]
        for ind,w in enumerate(jieout[j]):
            wrong=0
            for in_w in w:
                if in_w in  stop_words:
#                    print(in_w)
                    wrong=1
                    break
            if wrong==0:    
                filtered_sentence.append(w)
        final.append(filtered_sentence)
    return final
####################################################### data augmentation (allaug) ##########################################################################
num_sen=5 #句子數 2 3 4 5
allanswer=[]
allaug=[]
for art in range(0,len(alldata)):
    example=alldata[art]
    
    example=re.sub('A','', example)
    example=re.sub('B','', example)
    exam_out=re.split('\W+', example) #吃掉所有非中英文
    
    aug=[]
    for i in range(len(exam_out)-(num_sen-1)):
#        aug.append(exam_out[i]+exam_out[i+1])
        toappend=''
        for j in range(0,num_sen):
            toappend=toappend+exam_out[i+j]
        if len(toappend)<2:
            toappend=''
            try:
                for j in range(0,num_sen+1):
                    toappend=toappend+exam_out[i+j]
            except:
                for j in range(-1,num_sen):
                    toappend=toappend+exam_out[i+j]
        
        allaug.append(toappend)
        allanswer.append(tanswer[art])

joblib.dump(allanswer,'./answer.pkl')     # TO INPUT
####################################################################################################################################
#allanswer=[]
#allaug=[]
#for art in range(0,len(alldata)):
#    example=alldata[art]
#    example=re.sub('A','', example)
#    example=re.sub('B','', example)
#    exam_out=re.split('\W+', example)
#    aug=[]
#    for i in range(len(exam_out)-1):
##        aug.append(exam_out[i]+exam_out[i+1])            
#        if len(exam_out[i]+exam_out[i+1])>1:
#            allaug.append(exam_out[i]+exam_out[i+1])
#        else:
#            allaug.append(exam_out[i-1]+exam_out[i]+exam_out[i+1])
#        allanswer.append(tanswer[art])
#
#joblib.dump(allanswer,'./answer.pkl')     # TO INPUT
############################################  make jieout article form ######################################################################
jieout=jiebacut(allaug)

jieout_art=[]
print('make seperate words into article with space')
for i in range(0,len(jieout)):
    jieout_art.append(" ".join(jieout[i]))
joblib.dump(jieout_art,'./jieart.pkl')
#jieout=eli(jieout)

print('Training word2vec ...\n')
size_w2=32 #word to vec size

###################################  train w2v #########################################################
model_w2v = gensim.models.Word2Vec(jieout_art, sg=1, size=size_w2, window=5, min_count=1, negative=5)
##
yes2000=[]
for art in range(len(jieout_art)):
    this_score=np.zeros(shape=(size_w2,1)).flatten()

    A=[]
    for words in jieout_art[art]:

        A.append(model_w2v[words])
        
    yes2000.append(A)
##
np.savez( "./w2v.npz", yes2000=yes2000 )
##
print("save npz")
#============================ get seqence length ==================================================
seqlen=[]
for a in range(0,len(yes2000)):
#    seqlen.append((len(yyy[a])-winlen+1)*winlen)
    seqlen.append(len(yes2000[a]))
joblib.dump(seqlen,'./seqlen.pkl')              # TO INPUT
#=========================  Same length w2v =====================================================
w2v=np.load('w2v.npz')
xxx=w2v['yes2000']

print("To same len")
sizew=len(xxx[0][0])
bla=np.zeros(shape=(1,size_w2)).flatten()
max_x=len(max(xxx,key=len))
for j in range(0,len(xxx)):
    while len(xxx[j])<max_x:
        xxx[j].append(bla)
                    ###########      to numpy to save ###############################
xx=list(xxx)
xxxnp=np.array(xx,dtype='f')
joblib.dump(xxxnp,'./xxxnp.pkl')               # TO INPUT






