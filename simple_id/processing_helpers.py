import pandas

def getSubClass(s):
    if len(s.split('.')) > 1:
        return int(s.split('.')[1][:2])
    else:
        return 0

def read_WOS_CLasses(path):
    df = pandas.read_csv(path)
    df['main_class'] = df['Description'].apply(lambda x: int(x[0]))
    df['sub_class'] = df['Description'].apply(getSubClass)
    df['Description'] = df['Description'].apply(lambda x : ' '.join(x.split(' ')[1:]))
    codeToName = {r['main_class'] if r['sub_class'] < 1 else -1 : r['Description'] for i, r in df.iterrows()}
    df['main_class'] = df['main_class'].apply(lambda x : codeToName[x])
    return df
