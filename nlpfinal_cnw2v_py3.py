# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:49:43 2017

@author: Donkey
"""

from __future__ import print_function
import scipy
from sklearn.datasets import load_files
import jieba
import gensim
from gensim import corpora, models
import joblib
import numpy as np
import time
import os.path
from pathlib import Path
#==============================================================================
# This is a python 2 code 
#==============================================================================
# alltext 的形式是 一個2維的list
# [[原始文章1],[原始文章2],...,[原始文章N]]
#################################################################################
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

def eli(jieout):    
    punc='，。！？：／」「()（）％、１２３４５６７８９０0123456789'+' '
#    punc = punc.decode("utf-8")
#    stop_word=set(['，','。','！','？','：','／','〔','．','；', '．','〕','-','﹝','﹞'])
    stop_word='，。！？：／〔．； ．〕-﹝﹞'
#    stop_word=stop_word.decode('utf-8')
    stop_words=set(punc)|set(stop_word)
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

def getw2v(model_w2v,jieout):
    cn_w2v=[]
    for art in range(len(jieout)):
        this_score=np.zeros(shape=(size_w2,1)).flatten()
        print(art)
    
        A=[]
        for words in jieout[art]:
    
            A.append(model_w2v[words])
            
        cn_w2v.append(A)
    return cn_w2v



#==============================================================================
# alltext 的形式是 一個2維的list
# [[原始文章1],[原始文章2],...,[原始文章N]]
#==============================================================================

############讀故事進LIST###############
print("Make alltext")
alltext=[]
count = 0
read_place = "E:\\BIIC\\jdth\\"
while 1:
    count = count + 1;
    print(count)
    read_filename = read_place + str(count) + ".txt"
    if(os.path.isfile(read_filename)):
        alltext.append(open(read_filename,'r', encoding = 'utf-8').read())        
    else:
        print("List Done!")
        break
######################################

jieout=jiebacut(alltext)
jieout=eli(jieout)

print('Training word2vec ...\n')
size_w2=10
model_w2v = gensim.models.Word2Vec(jieout, sg=1, size=size_w2, window=5, min_count=1, negative=5) #only know the input is jieout , and size means the dimension of feature is okay
joblib.dump(model_w2v,'./model_w2v.pkl')

print('getting word embedding feature data')
w2vfeature=getw2v(model_w2v,jieout)
#############################################################
np.savez( "./cnw2v.npz", cnw2v=w2vfeature )
print("save npz")
a = np.load("./cnw2v.npz")
