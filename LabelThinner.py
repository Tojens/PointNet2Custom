import glob
import pandas as pd
import numpy as np
i = 0


while i < 1200:
    files = glob.glob(r"/mnt/edisk/backup/dataset/semantic_raw/*.labels")
    #print(len(files[i]))
    #print(files[i][77:99])
    df = pd.read_csv(files[i], sep=",")
    df.columns = ['Classification']
    df['Classification'] = df.query('Classification != 4')
    #df['Classification'] = df['Classification' != 4]
    
    print(files[i], df['Classification'].unique())
    df['Classification'] = df['Classification'].replace([1,2,3,5,6,7,8],[1,2,3,4,5,6,5])
    df = df.dropna()
    df = df.astype('int8')
    print(files[i], df.dtypes, df['Classification'].dtypes, df['Classification'].unique())


    
    i += 1