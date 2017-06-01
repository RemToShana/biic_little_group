
from __future__ import print_function
import scipy
from sklearn.datasets import load_files
import jieba
import gensim
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer


movie_reviews_data_folder = '.\\traindata\\'
dataset = load_files(movie_reviews_data_folder, shuffle=False)
train_X=dataset.data
train_Y=dataset.target

movie_reviews_data_folder_test = '.\\testdata\\'
testdata = load_files(movie_reviews_data_folder_test, shuffle=False)
test_X=testdata.data
test_Y=testdata.target
import numpy as np
nptr=np.array(train_X)
npte=np.array(test_X)
s=np.concatenate((nptr,npte),axis=0)
alltext=list(s)


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
#
jieout=jiebacut(alltext)
jieout=eli(jieout)

jieout_art=[]
print('make seperate words into article with space')
for i in range(0,len(jieout)):
    jieout_art.append(" ".join(jieout[i]))

print('making TFIDF')
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(jieout_art)
words = vectorizer.get_feature_names()



    
trainX=vectorizer.transform(jieout_art[0:15000])
testX=vectorizer.transform(jieout_art[15000:16500])    


#tri=vectorizer.transform(jieout[0])
#print tfidf.shape
''' 暫時別用
print('train')
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
text_clf = Pipeline([('vect', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])
text_clf.fit(train_X, train_Y)
y_predicted = text_clf.predict(test_X)
import numpy as np
ACU=np.mean(y_predicted == test_Y) 
print ('acu:',ACU)
'''
#################################################################################
from sklearn import feature_selection
best_C=0.5 # 10 5 1 0.5 0.1 0.01
best_percent=0.7 #feature留幾趴
machine=LinearSVC(C=best_C, penalty="l1", dual=False,class_weight = 'balanced');
fs=feature_selection.SelectPercentile(feature_selection.f_classif, percentile = best_percent).fit(trainX, train_Y)            
trainX=fs.transform(trainX)

machine.fit(trainX, train_Y)

testX =fs.transform(testX) 
y_predicted = machine.predict(testX)
ACU=np.mean(y_predicted == test_Y) 
print ('acu:',ACU)

'''
raw_input("Press Enter to see grid search, it took very long time...")
print ("grid...")
#%%
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'vect__use_idf': (True,False),
              'clf__C': (1.0, 0.1),
}
grid_search = GridSearchCV(text_clf, parameters, n_jobs=-1)
grid_search = grid_search.fit(train_X, train_Y)

n_candidates = len(grid_search.cv_results_['params'])
for i in range(n_candidates):
    print(i, 'params - %s; mean - %0.2f; std - %0.2f'
             % (grid_search.cv_results_['params'][i],
                grid_search.cv_results_['mean_test_score'][i],
                grid_search.cv_results_['std_test_score'][i]))

y_predicted = grid_search.predict(test_X)

newACU=np.mean(y_predicted == test_Y)     
print (newACU)
'''

