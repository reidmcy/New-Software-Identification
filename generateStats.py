import metaknowledge as mk
import networkx as nx
import pandas
mk.VERBOSE_MODE = False

outputDir = 'nets'

def genNetStats(RC, name, saveToDisk = False):
    order = ('nodes', 'edges', 'isolates', 'loops', 'density', 'transitivity')
    results = {n : [] for n in order}
    names = []
    nets = {
        "{}-Citation".format(name) : RC.networkCitation(),
        "{}-Cocitation".format(name) : RC.networkCoCitation(),
        "{}-Coauthorship".format(name) : RC.networkCoAuthor(),
        "{}-keywords".format(name) : RC.networkOneMode('keywords'),
    }
    for n, G in nets.items():
        retTuple = mk.graphStats(G, makeString = False)
        names.append(n)
        for i, metric in enumerate(order):
            results[metric].append(retTuple[i])
        if saveToDisk:
            nx.write_graphml(G, "{}/{}-Citation".format(outputDir, name))
    return pandas.DataFrame(results, index = names)

def main():
    fullRC = mk.RecordCollection('data')
    journals = set([R.get('journal') for R in fullRC])
    jRCs = []

    for journal in journals:
        jRCs.append(mk.RecordCollection([R for R in fullRC if R.get('journal') == journal]))
        print(genNetStats(jRCs[-1], journal))
    print(genNetStats(fullRC, 'full'))
if __name__ == '__main__':
    main()
