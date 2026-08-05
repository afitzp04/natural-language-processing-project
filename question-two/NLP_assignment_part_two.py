#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random

import nltk
import spacy
from pathlib import Path

import math
import re
from collections import Counter
import pandas as pd
from spacy.tokens import Doc

#detect environment and set project path automatically
try:
    #Colab - code developed by student here
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT = '/content/drive/MyDrive/cw-pack-2026'

except ImportError:
    #running locally - set to unzip file location
    PROJECT = str(Path(__file__).parent) if '__file__' in dir() else str(Path.cwd())

os.chdir(PROJECT)
print("Working in:", PROJECT)


# In[ ]:


#Part 2

#a) read and clean dataset
df = pd.read_csv(os.path.join(PROJECT, "texts", "hansard10000.csv"))

#a)i) rename Labour (Co-op) -> Labour
df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')

#a)ii) keep four most common parties and remove 'Speaker'
df = df[df['party'] != 'Speaker']
top_four = df['party'].value_counts().head(4).index
df = df[df['party'].isin(top_four)]

#a)iii) keep rows where speech_class is 'Speech'
df = df[df['speech_class'] == 'Speech']

#a)iv) keep rows where speech is at least 1000 characters
df = df[df['speech'].str.len() >= 1000]

print(f"Dataset shape after cleaning: {df.shape}")


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "part two reading and cleaning dataset"')


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report

#b) vectorise speeches
vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000)
X = vectorizer.fit_transform(df['speech'])
y = df['party']

#make the train/test split with stratified sampling where random seed is 26
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 26)

#random forest
rf = RandomForestClassifier(n_estimators = 300, random_state = 26)
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
print("Random Forest")
print("Macro F1:", f1_score(Y_test, rf_pred, average = 'macro'))
print(classification_report(Y_test, rf_pred, zero_division = 0))

#SVM with linear kernel
svm = LinearSVC(random_state = 26)
svm.fit(X_train, Y_train)
svm_pred = svm.predict(X_test)
print("SVM")
print("Macro F1:", f1_score(Y_test, svm_pred, average = 'macro'))
print(classification_report(Y_test, svm_pred, zero_division = 0))


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "vectorising speeches - baseline report"')


# In[ ]:


#c)
#adjust parameters to include unigrams, bigrams, trigrams
#range changes to 1,3 from part b, so all are considered
vectorizer_ngram = TfidfVectorizer(stop_words = 'english', max_features = 3000, ngram_range=(1,3))
X_ngram = vectorizer_ngram.fit_transform(df['speech'])

X_train_ng, X_test_ng, y_train_ng, y_test_ng = train_test_split(
    X_ngram, y, test_size = 0.2, stratify = y, random_state = 26
)

#Random Forest
rf_ng = RandomForestClassifier(n_estimators = 300, random_state = 26)
rf_ng.fit(X_train_ng, y_train_ng)
rf_ng_pred = rf_ng.predict(X_test_ng)
print("Random Forest with n-grams")
print("Macro F1:", f1_score(y_test_ng, rf_ng_pred, average = 'macro'))
print(classification_report(y_test_ng, rf_ng_pred, zero_division = 0))

#SVM
svm_ng = LinearSVC(random_state = 26)
svm_ng.fit(X_train_ng, y_train_ng)
svm_ng_pred = svm_ng.predict(X_test_ng)

#reports
print("SVM with n-grams")
print("Macro F1:", f1_score(y_test_ng, svm_ng_pred, average = 'macro'))
print(classification_report(y_test_ng, svm_ng_pred, zero_division = 0))


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "adjusted parameters for n-grams and reports"')


# In[ ]:


#"You can use this function in any way you like to try to achieve the best classification performance while keeping the number of features to no more than 3000, and using the same three classifiers as above"
#The question referencees three classifiers but only two (RF and SVM) are defined above in part b, so these two will be used below
#d)
import spacy
nlp = spacy.load("en_core_web_sm", disable = ["parser", "ner"])

#new tokeniser
def custom_tokenizer(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower() for token in doc
        #keep meaningful words
        if token.pos_ in ('NOUN', 'VERB', 'ADJ')
        #remove stopwords
        and not token.is_stop
        #remove punctuation
        and not token.is_punct
        #remove short tokens
        and len(token.lemma_) > 2
    ]
    return tokens

#custom tokeniser vectoriser
vectorizer_custom = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=3000)
X_custom = vectorizer_custom.fit_transform(df['speech'])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_custom, y, test_size = 0.2, stratify = y, random_state = 26
)

#run all classifiers and pick best
rf_c = RandomForestClassifier(n_estimators = 300, random_state = 26)
rf_c.fit(X_train_c, y_train_c)
rf_c_pred = rf_c.predict(X_test_c)
rf_c_f1 = f1_score(y_test_c, rf_c_pred, average = 'macro')

svm_c = LinearSVC(random_state = 26)
svm_c.fit(X_train_c, y_train_c)
svm_c_pred = svm_c.predict(X_test_c)
svm_c_f1 = f1_score(y_test_c, svm_c_pred, average = 'macro')

print(f"RF Macro F1: {rf_c_f1:.3f}")
print(f"SVM Macro F1: {svm_c_f1:.3f}")

#print reports
if svm_c_f1 >= rf_c_f1:
    print("\nBest classifier: SVM with custom tokenizer")
    print(classification_report(y_test_c, svm_c_pred))
else:
    print("\nBest classifier: Random Forest with custom tokenizer")
    print(classification_report(y_test_c, rf_c_pred, zero_division = 0))


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "custom tokeniser and report"')


# In[ ]:


#e) Function and performance explanation

#The custom tokeniser uses spaCy’s ability to lemmatise tokens, which means it retains only nouns, verbs, and adjectives. It removes stopwords, punctuation, and tokens shorter than three characters. This is to reduce meaningless vocabulary and reduce different forms into a lemma (eg, “running” or “ran” will become “run”). It focuses on meaningful content words that are more likely to carry political meaning and indicate the party label. This was motivated by  the need to reduce noise and improve efficiency because of the 3000-feature limit.

#Performance was compared across three strategies above. The baseline uni-gram TF-IDF model achieved macro F1 scores of 0.41 with Random Forest and 0.52 with SVM. This made the baseline SVM be the strongest overall classifier. Adding bi-grams and tri-grams slightly improved Random Forest to 0.44, which suggests that phrase-level features can capture some useful patterns. But, SVM performance dropped slightly to 0.50. This indicates that larger n-gram spaces may introduce sparsity as well as less useful features.

#The custom tokeniser achieved a macro F1 of 0.47 with SVM and 0.35 with Random Forest. This was weaker than the baseline SVM, which suggests that removing stop words and function words may discard useful stylistic signals that can help to distinguish parties. But it still performed well while using a linguistically-motivated feature selection process. This suggests that by restricting features to nouns, verbs, and adjectives, we may improve interpretability and reduce noise. However it may also have removed some discriminative signals like named entities and function words.

#SVM outperformed Random Forest across all experiments in this set up. This can be expected in sparser, high-dimensional text classification tasks because linear decision boundaries can often generalise better than ensemble tree methods.

#The dataset was also imbalanced. Conservative speeches dominated the test set with 250 samples, but Liberal Democrats had only 15. This affected minority-class performance: Liberal Democrats sometimes had zero recall, which shows the models struggled to learn enough distinguishing features for the smaller classes. For this reason, macro F1 is a better evaluation metric than accuracy, as it provides equal weight to all classes.

#The macro F1 scores (~0.47 - 0.52 for SVM) are reasonably acceptable for a multi-class, imbalanced speech classification task using TF-IDF features. Performance here has been limited by class imbalance and the sparse “bag of words” representation, especially with minority classes (eg, Liberal Democrats) particularly difficult to predict.

#Overall, the baseline SVM with uni-gram TF-IDF provided the best balance between efficiency and performance. The custom tokeniser was linguistically interesting and more selective, but it did not outperform the simpler baseline approach seen in part b.


# In[ ]:


get_ipython().system('git add .')
get_ipython().system('git commit -m "explanation for part e"')

