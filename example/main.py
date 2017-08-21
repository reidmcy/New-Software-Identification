import simple_id
import pandas

target = 'sample.csv'

def main():
    #Load your csv, with 'title', 'abstract' and 'class' columns
    df = pandas.read_csv(target)

    #Generates the W2V embedding and does all the other prepwork
    #saves a pickle of df for speeding up loading
    #Then starts training the model
    N = simple_id.createClassifier(df, numEpochs = 10)

    #Once trained the model can be loaded easily
    #N = simple_id.loadModel('models/BiRNN-2-128-10.pt')

    #This will iterate over the dataframe and give the models weights
    #for each row as a new dataframe with the same index
    dfRets = simple_id.analyseDF(df, N)

    #Now we can combine and save the results
    dfRets.join(df).to_csv('results.csv')

if __name__ == '__main__':
    main()
