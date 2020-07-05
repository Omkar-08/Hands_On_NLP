# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:38:47 2020

@author: Omkar
"""

import nltk

sentence = "I am not going to eat that food"

words = nltk.word_tokenize(sentence)
newwords = []
temp_word = ""
for word in  words:
    
    if word == "not":
        temp_word = "not_"
    elif temp_word == "not_":
        word = temp_word + word
        temp_word = ""
    if word != "not":
        newwords.append(word)
        
sentence = " ".join(newwords)
            
    
