import os
import pandas as pd

#note that this assumes that this script is run from the base Spam_Ham_ML directory and that the filestructure has not been modified
os.getcwd()

for dataset in ['dataset 1','dataset 2','dataset 3']:
    os.chdir(dataset)
    for folder in ['train','test']:
        os.chdir(folder)
        emails = []
        for classification in ['spam','ham']:
            os.chdir(classification)
            for file in os.listdir():
                filename = os.fsdecode(file)
                if filename.endswith(".txt"):
                    with open(filename,encoding='utf8',errors='ignore') as f:
                        emails.append({'text':f.read(),'class':classification})
            os.chdir('..')
        os.chdir('..')
        df = pd.DataFrame(emails)
        df.to_pickle(folder + ".pkl")
    os.chdir('..')