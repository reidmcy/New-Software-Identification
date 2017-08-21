import os
from .utilities import preprocesing, getTrainTest
from .neuralnet import BiRNN
from .training import trainModel

def createClassifier(df, outputsDir = 'models', w2vFname = 'word2vec.bin', pickleFname = 'dfPickles.p', trainFname = 'training.csv', testFname = 'testing.csv', stepSize = .001, epochSize = 500, numEpochs = 50):
    #TODO: Explain defaults

    os.makedirs(outputsDir, exist_ok = True)

    #TODO: check df has right column names
    #Currently using: 'title', 'abstract' and 'class'

    df, w2v = preprocesing(df, outputsDir, w2vFname, pickleFname)

    #Currently assuming postives are rarer than negatives
    dfTrain, dfTest = getTrainTest(df, w2v, splitRatio = .1)

    print("Saving test-train data")
    dfTest[['authors', 'title', 'class']].to_csv("{}/{}".format(outputsDir, testFname))
    dfTrain[['authors', 'title', 'class']].to_csv("{}/{}".format(outputsDir, trainFname))

    #TODO: Test other NN sizes
    Net = BiRNN(200, #W2V size
                256, #Width
                2, #Height
                stepSize, #eta
                outputsDir #Autosave location
                )

    e = trainModel(Net, dfTest, dfTrain, epochSize, numEpochs)

    print("Done {} Epochs saving final model".format(numEpochs))
    Net.save()

def loadModel():
    with open(modelPath, 'rb') as f:
        N = torch.load(f)
    if torch.cuda.is_available():
        N.cuda()
