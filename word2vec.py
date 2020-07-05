# -*- coding: utf-8 -*-
"""
Created on Sun May 24 01:18:44 2020

@author: Omkar
"""

import nltk
import bs4 as bs
import re
from nltk.corpus import stopwords
import urllib
from gensim.models import Word2Vec


source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming').read()

soup = bs.BeautifulSoup(source,'lxml')

text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text 
    
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'[#@%\$\&\(\)\<\>?\'\":;\[\]-]',' ',text)
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

sent = nltk.sent_tokenize(text)

sent = [nltk.word_tokenize(sent) for sent in sent]

for i in range(len(sent)):
    sent[i] = [word for word in sent[i] if word not in stopwords.words('english')]
    
model = Word2Vec(sent,min_count = 1)

words = model.wv.vocab

vector = model.wv.most_similar('global')