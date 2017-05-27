import gensim
import pandas
import nltk
import matplotlib.pyplot as plt
import seaborn

import pickle
import os.path


max_bytes = 2**31 - 1

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
        hs = 1, #Hierarchical softmax is slower, but better for infrequent words
        size = 200, #Dim
        window = 5, #Might want to increase this
        min_count = 0,
        max_vocab_size = None,
        workers = 8, #My machine has 8 hyperthreads
        )
    return model

def preprocesing(dataDir, dataDict, modelsDir, w2vFname, pickleFname, regen = False):
    if regen:
        print("Loading Data")
        dfs = {k : pandas.read_csv('{}/{}'.format(dataDir, v), sep='\t') for k, v in dataDict.items()}

        print("Generating Word2Vec")
        w2v = genWord2Vec(*dfs.values())
        w2v.save('{}/{}'.format(modelsDir, w2vFname))

        for name, df in dfs.items():
            print("Generating vecs for: {}".format(name))
            df['title_tokenize'] = df['title'].apply(lambda x : genVecSeq(x, w2v))
            df['abstract_tokenize'] = df['abstract'].apply(lambda x : genVecSeq(x, w2v))

        print("Saving pickle")
        bytes_out = pickle.dumps(dfs)
        n_bytes = len(bytes_out)
        with open('{}/{}'.format(modelsDir, pickleFname), 'wb') as f:
            for idx in range(0, n_bytes, max_bytes):
                f.write(bytes_out[idx:idx+max_bytes])

    else:
        w2v = gensim.models.Word2Vec.load('{}/{}'.format(modelsDir, w2vFname))

        print("Loading DFs")
        bytes_in = bytearray(0)
        input_size = os.path.getsize('{}/{}'.format(modelsDir, pickleFname))
        with open('{}/{}'.format(modelsDir, pickleFname), 'rb') as f:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f.read(max_bytes)
        dfs = pickle.loads(bytes_in)

    return dfs, w2v

def compareRows(rows, N, useTitle = True):
    fig, axes = plt.subplots(figsize = (20,15),
                             nrows = len(rows) + 1,
                             gridspec_kw = {'height_ratios': [5] * len(rows) + [1]})
    aLst = []
    for i, row in enumerate(rows):
        abVec, tiVec, yVec = main.varsFromRow(row)
        if useTitle:
            outLSTM, (h_n, c_n) = N.lstmTi(tiVec)
            s = row['title']
        else:
            outLSTM, (h_n, c_n) = N.lstmAb(abVec)
            s = row['abstract']
        out = N(abVec, tiVec)
        probNeg = np.exp(out.data[0][0])
        probPos = np.exp(out.data[0][1])
        probNeg = probNeg / (probNeg + probPos)
        probPos = probPos / (probNeg + probPos)

        a = np.array(outLSTM.data.tolist())
        aLst.append(a[0, -1:, :])
        a = a[:,:30,:]
        df = pandas.DataFrame(a[0, :, :])
        df.index = nltk.word_tokenize(s)[:a.shape[1]]
        seaborn.heatmap(df, ax = axes[i])
        axes[i].set_title("Article Title: '{}'\n$P_{{negative}} = {:.4f}, P_{{positive}} = {:.4f}$".format(row['title'], probNeg, probPos))
        axes[i].set_xticklabels([])


    dfDiff = pandas.DataFrame(aLst[0] - aLst[1])
    seaborn.heatmap(dfDiff, ax = axes[-1], xticklabels = [i if i in np.linspace(0, aLst[0].shape[1] - 1, num = 10, dtype='int') else '' for i in range(aLst[0].shape[1])])
    axes[-1].set_title('Difference in Final Output Vectors')

    return fig, axes
