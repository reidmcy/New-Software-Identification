
def varsFromRow(row):
    abVec = Variable(torch.from_numpy(np.stack(row['abstract_tokenize'])).unsqueeze(0)).cuda()

    tiVec = Variable(torch.from_numpy(np.stack(row['title_tokenize'])).unsqueeze(0)).cuda()

    yVec = Variable(torch.from_numpy(row['vals'])).cuda()

    return abVec, tiVec, yVec

def trainModel(dfPostive, dfNegative, numEpoch, numBatch):
    dfPostive['vals'] = [np.array([1]) for i in range(len(dfPostive))]
    dfNegative['vals'] = [np.array([0]) for i in range(len(dfNegative))]

    print("{} postive".format(len(dfPostive)))
    print("{} negative".format(len(dfNegative)))

    df = dfPostive.append(dfNegative, ignore_index=True).append(dfPostive, ignore_index=True)
    df = sklearn.utils.shuffle(df)
    df.index = range(len(df))
    splitIndex = len(df) // 10

    dfTrain = df[:-splitIndex]
    dfTest = df[-splitIndex:]

    dfTest.index = range(len(dfTest))

    N = neuralnet.BiRNN(200, 256, 2)
    N.cuda()

    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(N.parameters(), lr=eta)

    try:
        for i in range(numEpoch):
            for j in range(numBatch):

                row = dfTrain.sample(n = 1).iloc[0]

                abVec, tiVec, yVec = varsFromRow(row)


                optimizer.zero_grad()
                outputs = N(abVec, tiVec)
                loss = torch.nn.functional.cross_entropy(outputs, yVec)
                loss.backward()
                optimizer.step()

            losses = []
            errs = []
            detectionRate = []
            falsePositiveRate = []
            for j in range(len(dfTest)):
                row = dfTest.iloc[j]

                abVec, tiVec, yVec = varsFromRow(row)

                outputs = N(abVec, tiVec)

                loss = torch.nn.functional.cross_entropy(outputs, yVec)
                losses.append(loss.data[0])
                pred = outputs.data.max(1)[1]

                errs.append(1 - pred.eq(yVec.data)[0][0])

                if dfTest['vals'][j] == 1:
                    detectionRate.append(pred.eq(yVec.data)[0][0])
                else:
                    falsePositiveRate.append(1 - pred.eq(yVec.data)[0][0])

            print("Epoch {}, loss {:.3f}, error {:.3f}, detectionRate {:.3f}, falseP {:.3f}".format(i, np.mean(losses), np.mean(errs), np.mean(detectionRate),  np.mean(falsePositiveRate)))
    except KeyboardInterrupt:
        print("Exiting and saving")
    N.cpu()
    return N
