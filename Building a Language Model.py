#!/usr/bin/env python
# coding: utf-8

# # Building a Language Model

# In[83]:


import mpmath as mp
import os
import io
import sys
import psutil
import nltk 

from nltk.tokenize import word_tokenize
from nltk.collocations import *
from datetime import datetime
from collections import Counter


# In[70]:


from sklearn.model_selection import train_test_split
import numpy as np


# Creating a variable which stores the corpus. This makes the code relatively modular since only the value of the variable has to be changed in order to test with different corpora.
# 
# The corpus used for testing is the academic1 corpus in the Maltese set.

# In[71]:


path = 'Corpus/academic1.txt'


# The below function upon called returns the memory currently being used by python.exe in GBs

# In[72]:


def RAMusage():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    print('Memory Use: ', memoryUse, 'GB')


# Removing Symbols from the Corpus in order for the Language Models, Perplexities and Sentence Generators to focus solely on only words

# In[73]:


def RemoveSymbols(corpus):
    arr = []
    symbols = "“”‘’!\"#$€%&()*'+-,./:;<=>?@[\]^_`{|}~\n"
    
    for i in corpus:
        if i not in symbols:
            arr.append(i)
            
    return arr


# ## Other attempts to read corpus in a more efficient manner:
# 
# I tried to load the all of the corpora into one csv by loading the seperate corpora from my external hard drive. But unfortunately the below code despite reading the files from the path was not able to parse them. If the below worked, the testing would have been on the whole corpus which would have provided better results.
# 
# rawcorpus_dir = 'E:\Corpus\\'
# 
# output_dir = 'Corpus\KorpusMalti.csv'
# 
# csvout = pd.DataFrame()
# 
# for filename in os.listdir('Corpus'):
# 
#     data = pd.read_csv(filename, sep = ':', index_col = 0, header = None)
#     csvout.csvout.append(data)
#         
# csvout.to_csv(output_dir)
# 
# -------------------------------------------------------------------------------------------------------------------------------

# After the path is read, the file is stored into a variable 'corpus'. The variable is then preprocessed. This being that the corpus is tokenized and has the symbols removed. The time for the corpus to be loaded and preprocessed and the RAM Usage are monitored and printed. Considering that the corpus being tested in this case is rather small, the corresponing values for time and usage are slow. But the larger the corpus, the longer and less efficient this process becomes.

# In[74]:


extraction_start = datetime.now()

file = open(path)
corpus = file.read()

tokenize = word_tokenize(corpus)
tokens = RemoveSymbols(tokenize)

extraction_end = datetime.now()

extraction_time = dict()
extraction_time['extraction_time'] = extraction_end - extraction_start
RAMusage()
print('Extraction Time(HH::MM:SS:ms) - {}\n\n'.format(extraction_time['extraction_time']))


# The frequency distribution is calculated by the NLTK FreqDist function. But the argument passed is a bigram. The bigram is calculated by appending the iterator with the n word count for the length of the tokens in the corpus. Therefore, say NGrams(tokens, 5) would result in a 5Gram and NGrams(tokens, 3) would result in a Trigram, and so on...

# In[85]:


def NGrams(words, n):
    ngrams = []
    
    for i in range(0, len(words)):
        ngram = ' '.join(words[i:i + n])
        ngrams.append(ngram)
        
    return ngrams

bigram = NGrams(tokens, 2)
freqdist = nltk.FreqDist(bigram)

#The Below is commented in order to not attach a notebook with 30000 lines as a deliverable
#for i,j in freqdist.items():
    #print(i,j)


# The function 'Split' given a path as an argument returns the test and training sets obtained from the file. This is achieved by first storing the corpus into a list in order to be acceptable by the 'train_test_split' function, which as the name entails, splits the list into training and test sets of size 34% and 66% respectively.

# In[76]:


def Split(corpus):
    
    file = open(path)
    corpus = file.read()
    words = []
    
    for line in corpus:
        
        words.append(line)
        
    train, test = train_test_split(words, test_size = 0.66, train_size = 0.34, shuffle = False)
    
    return train, test

x, y = Split(path)


# Calculates the Perplexity for the NGram and model. Perplexity is defined as the nth root of 1/ the number of words in the dataset. After total perplexity, this value is raised to 1/N to define the final perplexity.

# In[86]:


def Perplexity(test, model):
    
    perp = mp.mpf(1)
    
    N = mp.mpf(0)
    
    for line in test:
        N += len(line)
        line = ' '.join(line)
        
        if model[line] > 0:
            perp = perp * (1/model[line])
        else:
            perp = perp * sys.maxsize
            
    perp = pow(perp, 1/float(N))
    return perp


# # Part 2 - Building a Language Model

# ## Vanilla

# In[78]:


def VanillaUnigram(train):
    
    model = Counter(train)
    
    for word in model:
        model[word] = model[word]/len(train)
        
    return model

def VanillaBigram(train):
    
    model = Counter([(word, train[i + 1]) for i, word in enumerate(train[:-1])])
    counter = Counter(train)
    
    for word in model:
        model[word] = model[word]/counter[word[0]]
        
    return model

def VanillaTrigram(train):
    
    bigram = Counter([(word, train[i + 1]) for i, word in enumerate(train[:-1])])
    trigram = Counter([(word, train[i + 1], train[i + 2]) for i, word in enumerate(train[:-2])])
    
    for word in trigram:
        trigram[word] = trigram[word]/bigram[(word[0], word[1])]
        
    return trigram


# In[79]:


a = VanillaUnigram(x)
p1 = (Perplexity(y, a))

b = VanillaBigram(x)
p2 = (Perplexity(y, b))

c = VanillaTrigram(x)
p3 = (Perplexity(y, c))

i1 = (Interpolation(a, b, c, ["<s>", "</s>"], "il-"))


# ## Laplace 
# 
# In essence the laplacian model is an enhanced version of the vanilla model. The enhancement comes from the addition of Laplacian smoothing. This is achieved by adding a + 1 after finding the size of the word, and hence reducing the dividend to a be smaller given a ratio from numerator to denominator. The Laplacian smoothing is applied to all three ngrams being the unigram, bigram and trigram.

# In[80]:


def LaplaceUnigram(train):
    
    model = Counter(train)
    
    for word in model:
        model[word] = (model[word]+1)/len(train)
        
    return model

def LaplaceBigram(train):
    
    model = Counter([(word, train[i + 1]) for i, word in enumerate(train[:-1])])
    counter = Counter(train)
    
    for word in model:
        model[word] = model[word] + 1/counter[word[0], word[1]]
        
    return model

def LaplaceTrigram(train):
    
    bigram = Counter([(word, train[i + 1]) for i, word in enumerate(train[:-1])])
    trigram = Counter([(word, train[i + 1], train[i + 2]) for i, word in enumerate(train[:-2])])
    
    for word in trigram:
        trigram[word] = trigram[word] + 1 /bigram[(word[0], word[1], word[2])]
        
    return trigram


# In[81]:


a2 = LaplaceUnigram(x)
p4 = (Perplexity(y, a2))

#b2 = LaplaceBigram(x)
#p5 = (Perplexity(y, b2))

#c2 = LaplaceTrigram(x)
#p6 = (Perplexity(y, c2))

#i2 = (Interpolation(a2, b2, c2, ["<s>", "</s>"], "il"))


# ## UNK
# 
# Although based on the Vanilla model (in fact both the bigram and the trigram return a Vanilla bigram and trigram respectively), the UNK Model processes some work before applying the Vanilla Model NGram on to the training set. This language model when presented with a word which is out of the corpus' vocabulary, the model swaps the word to be "<UNK>" which stands for an unknown variable.

# In[29]:


def UNKUnigram(train):
    
    counter = Counter(train)
    model = {}
    model["<UNK>"] = 0
    
    for word in counter:
        if counter[word] == 1:
            model["<UNK>"] += 1
            
        else:
            model[word] = counter[word]
            
    for word in model:
        model[word] = model[word]/len(train)
        
    return model

def UNKBigram(train):
    
    unigram = UNKUnigram(train)
    
    for i, word in enumerate(train):
        if not (word in unigram):
            train[i] = "<UNK>"
            
    return VanillaBigram(train)

def UNKTrigram(train):
    
    unigram = UNKUnigram(train)
    
    for i, word in enumerate(train):
        if not (word in unigram):
            train[i] = "<UNK>"
            
    return VanillaTrigram(train)


# In[50]:


a3 = UNKUnigram(x)
#p7 = (Perplexity(y, a3))
#The above commented line returns an understood error when running.

b3 = UNKBigram(x)
p8 = Perplexity(y, b3)

c3 = UNKTrigram(x)
p9 = (Perplexity(y, c3))

#i3 = (Interpolation(a3, b3, c3, ["<s>", "</s>"], "il"))


# The similarity of results shown in all perplexities of the models, raises some concern. A larger perplexity for the Unigram can be understood especially since there is very small accomodation, and this can be applied for the smaller perplexity for the Bigrams and Trigrams both. But the, similarity across all models note that there is something wrong with the processing of the corpus at hand.

# In[67]:


print("Vanilla Model: ", "Unigram: ", p1, "Bigram: ", p2, "Trigram: ", p3, "Interpolation: ", i1)
print("\n")
print("Laplace Model: ", "Unigram: ", p4, "Bigram: ", "Trigram: ", "Interpolation: ")
print("\n")
print("UNK Model: ", "Unigram: ", "Bigram: ", p8, "Trigram: ", p9, "Interpolation: ")


# ## Probability

# In[15]:


def UnigramProbability(unigram, sentence, word):
    return unigram[word]

def BigramProbability(bigram, sentence, word):
    if (sentence[-1], word) in bigram:
        return bigram[sentence[-1],word]
    
    else:
        return 0
    
def TrigramProbability(trigram, sentence, word): 
    if (sentence[-2],sentence[-1], word) in trigram:
        return trigram[sentence[-2],sentence[-1],word]
    
    else:
        return 0
    
def Interpolation(unigram, bigram, trigram, sentence, word):
    Unigram = 0.1*(unigram[word])
    Bigram = 0.3*(bigram[sentence[-1], word])
    Trigram = 0.6*(trigram[sentence[-2], sentence[-1], word])
    
    return Unigram+Bigram+Trigram


# ## Generate
# 
# When running the either Generate function on the notebook, the kernel crashes, the age of my laptop does not help either. But when running the .py file, the generate works perfectly. The function calls were commented to avoid any crashes when running.

# In[87]:


def UnigramGenerate(unigram, sentence, last = "", count = None):
    
    if(count != 0 and sentence[-1] != last):
        
        weights = np.array(list(unigram.values()))
        norm = weights/np.sum(weights)
        
        resample = np.random.multinomial(1, norm)
        key = list(resample).index(1)
        value = list(unigram.keys())[key]
        
        sentence.append(value)
        if count != None:
            UnigramGenerate(unigram, sentence, last, count-1)
        else:
            UnigramGenerate(unigram, sentence, last)
            
    return sentence

#print(UnigramGenerate(a, ["<s>"], "</s>"))


# In[89]:


def BigramGenerate(bigram, sentence, last, count = None):
    
    if(count != 0 and sentence != last):
        bigrams = []
        
    for word in bigram:
        if word[0] == sentence[-1]:
            bigrams[word] = bigram[word]
            
    if(bigrams == []):
        return sentence 
    
    weights = np.array(list(bigrams.values()))
    norm = weights / np.sum(weights)
    resample = np.random.multinomial(1, norm)
    key = list(resample).index(1)
    value = list(bigrams.keys())[key]
    
    sentence.append(value)
    
    if count != None:
        BigramGenerate(bigram, sentence, last, count-1)
    else:
        BigramGenerate(bigram, sentence, last)
        
    return sentence

#print(BigramGenerate(a, ["<s>"], "</s>"))


# In[88]:


def TrigramGenerate(bigram, trigram, sentence, last = "", count = None):
    if(len(sentence) == 1):
        sentence = BigramGenerate(bigram, sentence, last, count=1)
        
    if(count != 0 and sentence[-1] != last):
        trigrams = []
        
        for word in trigram:
            
            if(word[0] == sentence[-2] and word[1] == sentence[-1]):
                trigrams[word] = trigram[word]
                
        if(trigrams == []):
            return sentence
        
        weights = np.array(list(bigrams.values()))
        norm = weights / np.sum(weights)
        resample = np.random.multinomial(1, norm)
        key = list(resample).index(1)
        value = list(bigrams.keys())[key] 
        
        sentence.append(value[2])
        if count != None:
            TrigramGenerate(bigram, trigram, sentence, last, count-1)
            
        else:
            TrigramGenerate(bigram, trigram, sentence, last)
            
            
    return sentence

#print(TrigramGenerate(b, c, ["<s>"], "</s>"))


# # References 
# 
# https://web.stanford.edu/~jurafsky/slp3/3.pdf

# In[ ]:




