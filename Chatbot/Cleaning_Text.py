# -*- coding: utf-8 -*-

import pandas as pd
import nltk
import autocorrect
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from autocorrect import spell 

df = pd.read_csv('dataset.csv')
df

words_not_to_remove = ["not", "where", "why", "how", "what", "who", "which", "when", "whom"]
stopwords_list = [entries for entries in stopwords.words('english') if entries not in words_not_to_remove] 
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = " ".join([word for word in word_tokenize(text) if word not in stopwords_list])
    text = " ".join([w for w in word_tokenize(text) if w.isalnum()])
    text = " ".join([spell(w) for w in word_tokenize(text)])
    text = " ".join([ps.stem(w) for w in word_tokenize(text)])
    
    return text

df["processed_text"] = df["Expression"].apply(clean_text)

df