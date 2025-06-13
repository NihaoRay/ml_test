import pandas as pd
import numpy as np


url = "Personal_table.csv"

df = pd.read_csv(url, error_bad_lines=False)
df.reset_index(inplace=True, drop=True)

cols_to_drop = ["Person ID", "Name"]

## Final dataset to work with
finaldf = df.drop(cols_to_drop, axis=1)
finaldf.to_csv('datasets/db_ms_fimu.csv',index=False)

col = ['age','workclass','fnlwgt','education','education-num',
               'marital-status','occupation','relationship','race','sex',
              'capital-gain','capital-loss','hours-per-week','native-country','salary']
df1 = pd.read_csv('datasets/adult.data' ,header=None)
df2 = pd.read_csv('datasets/adult.test', header=None)
df1.columns = col
df2.columns = col
df = pd.concat([df1,df2])


df.isin([' ?']).sum(axis=0)

set(df['salary'])

df['native-country'] = df['native-country'].replace(' ?',np.nan)
df['workclass'] = df['workclass'].replace(' ?',np.nan)
df['occupation'] = df['occupation'].replace(' ?',np.nan)
df['salary'].replace(' <=50K','<=50K',inplace=True)
df['salary'].replace(' >50K','>50K',inplace=True)
df['salary'].replace(' <=50K.','<=50K',inplace=True)
df['salary'].replace(' >50K.','>50K',inplace=True)

df.dropna(how='any',inplace=True)


for col in df.columns:
    val_att = set(df[col])
    if len(val_att) < 1000:
        print(col, set(df[col]), len(set(df[col])))
    else:
        print(col, len(set(df[col])))
    print()

# education-num is equal to education
# fnlwgt has too many values
cols_to_drop = ['fnlwgt','education-num']
finaldf = df.drop(cols_to_drop, axis=1)
finaldf.to_csv('datasets/db_adults.csv',index=False)



