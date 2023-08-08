#!/usr/bin/env python
# coding: utf-8

# ### Data Scraping

# In[1]:


import pandas as pd
from bs4 import BeautifulSoup
import requests
from tqdm.notebook import tqdm, trange
import os
import numpy as np


# In[2]:


input_df = pd.read_excel("C:/Users/himan/Coding/Python/NLP/blackcoffer - assignment/20211030 Test Assignment/Input.xlsx")


# In[3]:


input_df


# In[4]:


file = "C:/Users/himan/Coding/Python/NLP/blackcoffer - assignment/20211030 Test Assignment/My Solution/scraped-text/"


# In[5]:


def text_extract_new_layout():
    input_df = pd.read_excel(r"C:\Users\himan\Coding\Python\NLP\blackcoffer - assignment\20211030 Test Assignment\Input.xlsx")
    for fn in trange(len(input_df)):
        url1 = input_df['URL'][fn]
        r = requests.get(url1, headers={"User-Agent": "ABC"})
        soup = BeautifulSoup(r.content, "lxml")
        results = soup.find(class_ = 'td-post-content tagdiv-type')
        if results is not None:
            results = results.find_all('p')
            title = soup.find_all("h1", class_ = 'entry-title')
            title = title[0].text
            content = ''
            for i in trange(len(results)):
                content+=results[i].text + ' '
            article = title + ' ' + content

            with open(file+str(input_df['URL_ID'][fn])+'.txt', 'w', encoding='utf-8') as f:
                f.write(article)
            f.close()


# In[6]:


#text_extract_new_layout()


# In[7]:


def text_extract_from_old_layout():
    input_df = pd.read_excel(r"C:\Users\himan\Coding\Python\NLP\blackcoffer - assignment\20211030 Test Assignment\Input.xlsx")
    for fn in trange(len(input_df)):
        url1 = input_df['URL'][fn]
        r = requests.get(url1, headers={"User-Agent": "ABC"})
        soup = BeautifulSoup(r.content, "lxml")
        results = soup.find_all(class_ = 'tdb-block-inner td-fix-index')
        if results is not None:
            title = soup.find_all("h1", class_='tdb-title-text')
            if len(title)<=0:
                continue
            title = title[0].text
            content = ''
            #print(results)
            results = results[14]
            results = results.find_all('p')
            for i in trange(len(results)):
                content+=results[i].text + ' '
            article = title + ' ' + content

            with open(str(input_df['URL_ID'][fn]) + '.txt', 'w', encoding='utf-8') as f:
                f.write(article)
            f.close()


# In[8]:


#text_extract_from_old_layout()


# In[9]:


url1 = 'https://insights.blackcoffer.com/rise-of-e-health-and-its-imapct-on-humans-by-the-year-2030-2/'
r = requests.get(url1, headers={"User-Agent": "ABC"})
soup = BeautifulSoup(r.content, "lxml")
results = soup.find_all(class_ = 'tdb-block-inner td-fix-index')


# In[10]:


results[14]


# In[11]:


import os

def get_file_names(folder_path):
    file_names = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            file_names.append(filename)
    return file_names

folder_path = 'C:/Users/himan/Coding/Python/NLP/blackcoffer - assignment/20211030 Test Assignment/My Solution/scraped-text/'
file_names = get_file_names(folder_path)

all_files = []

for filename in file_names:
    all_files.append(float(filename[:-4]))
    #print(filename[:-4])
    
len(input_df) - len(all_files)


# In[12]:


type(input_df['URL_ID'].tolist()[0])
type(all_files[0])


# In[13]:


for el in (input_df['URL_ID'].tolist()):
    if el not in all_files:
        print(el)


# ### Text Analysis

# In[14]:


import nltk
from nltk import word_tokenize, sent_tokenize
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import re


# #### Extracting Stop Words, Positive and Negative Words

# In[15]:


stop_words = []
for lst in tqdm(os.listdir('C:/Users/himan/Coding/Python/NLP/blackcoffer - assignment/20211030 Test Assignment/StopWords')):
    with open(f'C:/Users/himan/Coding/Python/NLP/blackcoffer - assignment/20211030 Test Assignment/StopWords/{lst}', 'r', encoding = 'latin-1') as f:
        lines = f.readlines()
    for line in lines:
        line = ''.join(line[:-1].split(' '))
        words = line.split('|')
        stop_words += words


# In[16]:


positive_words = []
with open('C:/Users/himan/Coding/Python/NLP/blackcoffer - assignment/20211030 Test Assignment/MasterDictionary/positive-words.txt', 'r') as f:
    words = f.readlines()
f.close()
for word in words:
    word = word[:-1]
    positive_words += [word]


# In[17]:


negative_words = []
with open('C:/Users/himan/Coding/Python/NLP/blackcoffer - assignment/20211030 Test Assignment/MasterDictionary/negative-words.txt', 'r') as f:
    words = f.readlines()
f.close()
for word in words:
    word = word[:-1]
    negative_words += [word]


# #### Removing duplicate words

# In[18]:


stop_words = list(set(stop_words))


# In[19]:


nltk_stopwords = stopwords.words('english')


# #### Text Preprocessing

# In[36]:


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
def preprocessing(article):
    lem_words = []
    words = []
    word_tokens = word_tokenize(article)
    for word in word_tokens:
        word = word.rstrip()
        word = word.lower()
        word = ''.join(re.findall('[a-zA-Z0-9]', word))
        tag = pos_tagger(nltk.pos_tag([word])[0][1])
        lem = WordNetLemmatizer()
        if word not in stop_words:
            if tag is None:
                lem_words.append(word) 
            else:
                lem_words.append(lem.lemmatize(word, tag)) 
        if word not in nltk_stopwords:
            if tag is None:
                words.append(word) 
            else:
                words.append(lem.lemmatize(word, tag)) 

    return lem_words, words


# #### Text Analysis

# In[37]:


def sentiment_scores(words): # Returns positive score, negative score, polarity score, subjectivity score
    pos_score = 0
    neg_score = 0
    for word in words:
        if word in positive_words:
            pos_score += 1
        elif word in negative_words:
            neg_score += -1

    neg_score *= -1
    
    
    return pos_score, neg_score, (pos_score - neg_score) / (pos_score + neg_score) + 0.000001, (pos_score + neg_score) / len(words) + 0.000001


# In[38]:


def syllables(words):
    return [word.count('a') + word.count('e') + word.count('i') + word.count('o') + word.count('u') if word[:-2] not in ['es', 'ed']  else word.count('a') + word.count('i') + word.count('o') + word.count('u') for word in words]


# In[39]:


def avg_word_len(words):
    lens = np.array([len(word) for word in words])
    return np.sum(lens) / len(lens)


# In[40]:


def num_personal_pronouns(words):
    sent = ' '.join(words)
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
    pronouns = pronounRegex.findall(sent)
    return len(pronouns)


# #### Running the modules

# In[41]:


output_df = pd.read_excel('C:/Users/himan/Coding/Python/NLP/blackcoffer - assignment/20211030 Test Assignment/Output Data Structure.xlsx')


# In[42]:


output_df = output_df.set_index('URL_ID')


# In[43]:


for article_id in tqdm(os.listdir('C:/Users/himan/Coding/Python/NLP/blackcoffer - assignment/20211030 Test Assignment/My Solution/scraped-text/')):
    with open(f'C:/Users/himan/Coding/Python/NLP/blackcoffer - assignment/20211030 Test Assignment/My Solution/scraped-text/{article_id}', 'r', encoding='utf-8') as f:
        article = f.readlines()
    f.close()
    id = float(article_id[:-4])
    article = ' '.join(article)
    sent_tokens = sent_tokenize(article)
    word_tokens = word_tokenize(article)
    output_df.at[id, 'AVG NUMBER OF WORDS PER SENTENCE'] = output_df.at[id, 'AVG SENTENCE LENGTH'] = len(word_tokens) / len(sent_tokens)
    clean_lemwords, clean_words = preprocessing(article)
    output_df.at[id, 'POSITIVE SCORE'], output_df.at[id, 'NEGATIVE SCORE'], output_df.at[id, 'POLARITY SCORE'], output_df.at[id, 'SUBJECTIVITY SCORE'] = sentiment_scores(clean_lemwords)
    output_df.at[id, 'WORD COUNT'] = len(clean_words)
    syllable_lst = np.array(syllables(clean_words))
    output_df.at[id, 'SYLLABLE PER WORD'] = np.sum(syllable_lst)/len(syllable_lst)
    output_df.at[id, 'COMPLEX WORD COUNT'] = len(np.where(syllable_lst>2)[0])
    output_df.at[id, 'PERCENTAGE OF COMPLEX WORDS'] = output_df.at[id, 'COMPLEX WORD COUNT']/output_df.at[id, 'WORD COUNT']
    output_df.at[id, 'FOG INDEX'] = 0.4*(output_df.at[id, 'AVG SENTENCE LENGTH'] + output_df.at[id, 'PERCENTAGE OF COMPLEX WORDS'])
    output_df.at[id, 'AVG WORD LENGTH'] = avg_word_len(clean_words)
    output_df.at[id, 'PERSONAL PRONOUNS'] = num_personal_pronouns(clean_words)
    


# In[44]:


output_df.to_excel('Output.xlsx')

