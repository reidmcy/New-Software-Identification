import torch
import torch.optim
import torch.nn
import torch.autograd
import torch.nn.functional
import numpy as np

import utilities

def trainModel(N, dfTest, dfTrain, epochSize, numEpochs):

    trainNp = len([c for c in dfTrain['class'] if c == 1])
    trainNn = len([c for c in dfTrain['class'] if c == 0])
    testNp = len([c for c in dfTest['class'] if c == 1])
    testNn = len([c for c in dfTest['class'] if c == 0])
    print("Training: {} postive, {} negative, {:.3f} percent".format(trainNp, trainNn, trainNp / (trainNn + trainNp)))
    print("Testing: {} postive, {} negative, {:.3f} percent".format(testNp, testNn, testNp / (testNn + testNp)))

    N.cuda()

    optimizer = torch.optim.Adam(N.parameters(), lr=N.eta)

    nTest = len(dfTest)

    try:
        for i in range(numEpochs):
            for j in range(epochSize):
                #import pdb; pdb.set_trace()
                if j % (epochSize // 20) == 0:
                    print("Epoch {}, Training: {:.0f}%".format(i, (j / epochSize) * 100), end = '\r')
                row = dfTrain.sample(n = 1).iloc[0]
                try:
                    abVec, tiVec, yVec = utilities.varsFromRow(row)
                except:
                    import pdb; pdb.set_trace()
                optimizer.zero_grad()
                outputs = N(abVec, tiVec)
                loss = torch.nn.functional.cross_entropy(outputs, yVec)
                loss.backward()
                optimizer.step()

            losses = []
            errs = []
            detectionRate = []
            falsePositiveRate = []

            for j in range(nTest):
                if j % (nTest // 20) == 0:
                    print("Epoch {}, Testing: {:.0f}%   ".format(i, (j / nTest) * 100), end = '\r')

                row = dfTest.iloc[j]

                abVec, tiVec, yVec = utilities.varsFromRow(row)

                outputs = N(abVec, tiVec)

                loss = torch.nn.functional.cross_entropy(outputs, yVec)
                losses.append(loss.data[0])
                pred = outputs.data.max(1)[1]

                errs.append(1 - pred.eq(yVec.data)[0][0])

                if dfTest['class'][j] == 1:
                    detectionRate.append(pred.eq(yVec.data)[0][0])
                else:
                    falsePositiveRate.append(1 - pred.eq(yVec.data)[0][0])

            print("Epoch {}, loss {:.3f}, error {:.3f}, detectionRate {:.3f}, falseP {:.3f}".format(i, np.mean(losses), np.mean(errs), np.mean(detectionRate),  np.mean(falsePositiveRate)))

            N.epoch += 1
            N.save()

    except KeyboardInterrupt as e:
        print("Exiting")
        return e
    N.cpu()
