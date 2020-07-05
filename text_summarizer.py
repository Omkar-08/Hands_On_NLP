# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:33:03 2020

@author: Omkar
"""

import bs4 as bs
import urllib.request
import re
import nltk
nltk.download('stopwords')

src = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_Warming').read()

soup = bs.BeautifulSoup(src,'lxml')

text = ""

for para in soup.find_all('p'):
    text += para.text
    
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)

clean_text = text.lower()
clean_text = re.sub(r'\W',' ',clean_text)
clean_text = re.sub(r'\d',' ',clean_text)
clean_text = re.sub(r'\s+',' ',clean_text)

sentences = nltk.sent_tokenize(text)

stop_words = nltk.corpus.stopwords.words('english')

word2count = {}

for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for key in word2count.keys():
    word2count[key] = word2count[key]/max(word2count.values())
    
sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in  word2count.keys():
            if len(sentence.split(' ')) < 30:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]
                    
import heapq
best_sent = heapq.nlargest(5,sent2score,sent2score.get)

print('--------------------------')

for sent in best_sent:
    print(sent)