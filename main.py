import gensim
import pandas
import nltk
import numpy as np

import neuralnet

import torch
import torch.nn as nn
from torch.autograd import Variable

import os

dataDir = 'data'
modelsDir = 'models'

littleFile = 'little.tsv'
someFile = 'some.tsv'
mostlyFile = 'mostly.tsv'

w2vFname = 'word2vec.bin'

regenW2V = False

eta = 0.001

def tokenizer(target):
    return [t for t in nltk.word_tokenize(target.lower()) if t != '.']

def sentinizer(sent):
    return [tokenizer(s) for s in nltk.sent_tokenize(sent)]

def genVecSeq(target, model):
    tokens = tokenizer(target)
    vecs = []
    for t in tokens:
        try:
            vecs.append(model.wv[t])
        except KeyError:
            #print(t)
            pass
    return vecs

def genWord2Vec(*dfs):
    vocab = []
    for df in dfs:
        vocab += list(df['title'].apply(lambda x: x.lower().split()))
        vocab += df['abstract'].apply(sentinizer).sum()

    model = gensim.models.Word2Vec(vocab,
        hs = 1, #Hierarchical softmax is better for infrequent words
        size = 200, #Dim
        window = 5, #Might want to increase this
        min_count = 0,
        max_vocab_size = None,
        workers = 8, #My machine has 8 hyperthreads
        )
    return model

def trainModel(dfPostive, dfNegative):
    dfPostive['vals'] = [np.array([1]) for i in range(len(dfPostive))]
    dfNegative['vals'] = [np.array([0]) for i in range(len(dfNegative))]

    df = dfPostive.append(dfNegative, ignore_index=True)

    from sklearn.utils import shuffle
    df = shuffle(df)


    N = neuralnet.BiRNN(200, 128, 2)
    N.cuda()

    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(N.parameters(), lr=eta)


    for i in range(100):
        losses = []
        for j in range(100):
            row = np.random.np.random.randint(0, len(df))

            xVec = Variable(torch.from_numpy(np.stack(df['abstract_tokenize'][row])).unsqueeze(0)).cuda()

            yVec = Variable(torch.from_numpy(df['vals'][row])).cuda()
            #print(yVec.data)

            optimizer.zero_grad()
            outputs = N(xVec)
            #print(outputs)
            loss = torch.nn.functional.cross_entropy(outputs, yVec)
            loss.backward()
            optimizer.step()
            losses.append(loss.data[0])
        print(i, end = ' : ')
        print(np.mean(losses))
    return N


def main():
    os.makedirs(dataDir, exist_ok = True)
    os.makedirs(modelsDir, exist_ok = True)

    dfs = {
        'little' : pandas.read_csv('data/little.tsv', sep='\t'),
        #'some' : pandas.read_csv('data/some.tsv', sep='\t'),
        'most' : pandas.read_csv('data/mostly.tsv', sep='\t'),
    }

    if regenW2V:
        print("Generating Word2Vec")
        w2v = genWord2Vec(*dfs.values())
        w2v.save('{}/{}'.format(modelsDir, w2vFname))
    else:
        w2v = gensim.models.Word2Vec.load('{}/{}'.format(modelsDir, w2vFname))
    print(w2v)
    for name, df in dfs.items():
        print("Generating vecs for: {}".format(name))
        df['title_tokenize'] = df['title'].apply(lambda x : genVecSeq(x, w2v))
        df['abstract_tokenize'] = df['abstract'].apply(lambda x : genVecSeq(x, w2v))

    trainModel(dfs['most'], dfs['little'])

if __name__ == '__main__':
    main()
