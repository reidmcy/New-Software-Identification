#!/usr/local/bin/python3

import metaknowledge as mk
import numpy as np
import pandas
import gensim
import nltk #For POS tagging
import sklearn #For generating some matrices
import pandas #For DataFrames
import numpy as np #For arrays
import matplotlib.pyplot as plt #For plotting
import seaborn #Makes the plots look nice
import IPython.display #For displaying images

import os #For looking through files
import os.path #For managing file paths
import re

mk.VERBOSE_MODE = False

#w2v = gensim.models.word2vec.Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary = True)

dataDir = 'data'
outputDir = 'outputs'

outputCSV = 'entries.csv'

targetTags = ['title', 'journal', 'keywords', 'abstract', 'id', 'year']

loadData = True

stop_words_nltk = nltk.corpus.stopwords.words('english')
snowball = nltk.stem.snowball.SnowballStemmer('english')

def normalizeTokens(tokenLst, stopwordLst = None, stemmer = None, lemmer = None, vocab = None):
    #We can use a generator here as we just need to iterate over it

    #Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    #Now we can use the semmer, if provided
    if stemmer is not None:
        workingIter = (stemmer.stem(w) for w in workingIter)

    #And the lemmer
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)

    #And remove the stopwords
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)

    #We will return a list with the stopwords removed
    if vocab is not None:
        vocab_str = '|'.join(vocab)
        workingIter = (w for w in workingIter if re.match(vocab_str, w))

    return list(workingIter)

def trainTestSplit(df, holdBackFraction = .2):
    df = df.reindex(np.random.permutation(df.index))
    holdBackIndex = int(holdBackFraction * len(df))
    train_data = df[holdBackIndex:].copy()
    test_data = df[:holdBackIndex].copy()

    return train_data, test_data

def generateVecs(df, sents = False):
    df['tokenized_text'] = df['text'].apply(lambda x: nltk.word_tokenize(x))
    df['normalized_text'] = df['tokenized_text'].apply(lambda x: normalizeTokens(x))

    if sents:
        df['tokenized_sents'] = df['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
        df['normalized_sents'] = df['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])

    ngCountVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, min_df=3, stop_words='english', norm='l2')
    newsgroupsVects = ngCountVectorizer.fit_transform([' '.join(l) for l in df['normalized_text']])
    df['vect'] = [np.array(v).flatten() for v in newsgroupsVects.todense()]

    return df

def main():
    if loadData:
        RC = mk.RecordCollection(dataDir)
        dfDict = {t : [] for t in targetTags}
        for R in RC:
            for t in targetTags:
                dfDict[t].append(R.get(t, None))
        df = pandas.DataFrame(dfDict)
        df.to_csv('{}/{}'.format(outputDir, outputCSV))
    else:
        df = pandas.read_csv('{}/{}'.format(outputDir, outputCSV))
    df['text'] = df['abstract']
    df = generateVecs(df.dropna().copy())

if __name__ == '__main__':
    main()
