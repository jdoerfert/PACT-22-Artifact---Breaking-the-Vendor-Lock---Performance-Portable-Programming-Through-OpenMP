import sys
import json
import os
import subprocess
import pandas as pd


with open(f'{sys.argv[1]}','r') as fd:
  status = json.load(fd)

df = pd.DataFrame.from_dict(status, orient='index')
di = {"Success": 1, "Fail" : 0 }
for c in df.columns:
  df[c] = df[c].map(di)

#print(df[(df['cudaomp'] == 0) & (df['cuda'] == 1)])

print(df.sum())
print(len(df))
