import gensim
import pandas
import nltk
import numpy as np

import neuralnet
import sklearn.utils
import torch
import torch.nn as nn
from torch.autograd import Variable

import utilities
import training

import os
import pickle

dataDir = 'data'
modelsDir = 'models'

rawFname = 'Stats-journs.tsv'
manualFname = 'classified.yaml'

w2vFname = 'word2vec.bin'
pickleFname = 'dfPickles.p'
regenW2V = False

eta = 0.001
numEpochs = 500
epochSize = 500

def main():
    os.makedirs(dataDir, exist_ok = True)
    os.makedirs(modelsDir, exist_ok = True)

    df, w2v = utilities.preprocesing(dataDir, rawFname, modelsDir, w2vFname, pickleFname, regen = regenW2V)

    dfTrain, dfTest = utilities.getTrainTest(df, dataDir, manualFname)

    print(len([c for c in dfTrain['class'] if c == 1]))
    print(len([c for c in dfTrain['class'] if c == 0]))
    print(len([c for c in dfTest['class'] if c == 1]))
    print(len([c for c in dfTest['class'] if c == 0]))
    #print(dfTrain)
    #print(dfTest)

    #Net = training.trainModel(df, w2v, eta, epochSize, numEpochs)

    #Net.save(modelsDir)

if __name__ == '__main__':
    main()
