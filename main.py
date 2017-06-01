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
numEpochs = 50
epochSize = 500

def main():
    os.makedirs(dataDir, exist_ok = True)
    os.makedirs(modelsDir, exist_ok = True)

    df, w2v = utilities.preprocesing(dataDir, rawFname, modelsDir, w2vFname, pickleFname, regen = regenW2V)


    dfTrain, dfTest = utilities.getTrainTest(df, dataDir, manualFname, w2v)

    Net = neuralnet.BiRNN(200, 256, 2, eta, modelsDir)

    e = training.trainModel(Net, dfTest, dfTrain, epochSize, numEpochs)

    print("Saving")
    Net.save()

    if e is not None:
        print("Error")
        raise e
    else:
        print("Done")

if __name__ == '__main__':
    main()
