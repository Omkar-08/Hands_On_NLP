# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

reviews = load_files('txt_sentoken')
X,y = reviews.data,reviews.target


with open('X.pickle','wb') as f:
    pickle.dump(X,f)

with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
corpus = [] 
for i in range(0,len(X)):
    reviews = re.sub(r'\W',' ',str(X[i]))
    reviews = reviews.lower()
    reviews = re.sub(r'\s+[a-z]\s+',' ',reviews)
    reviews = re.sub(r'^[a-z]\s+',' ',reviews)
    reviews = re.sub(r'\s+',' ',reviews)
    corpus.append(reviews)
    
 #BOW   
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

#Tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, sent_train, sent_test = train_test_split( X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,sent_train)

sent_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)

with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
    
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)


with open('classifier.pickle','rb') as f:
    clf = pickle.load(f)

with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
    
sample = ["The food is nice today."]
sample = tfidf.transform(sample).toarray()

print(clf.predict(sample))