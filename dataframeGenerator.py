import pandas as pd
import numpy as np
import random

N=1000

d = {'C1': pd.Series([random.randint(1,10000) for n in range(N)], index=[n for n in range(N)]),
    'C2': pd.Series([np.random.uniform(0, 0.1) * 10000 for n in range(N)], index=[n for n in range(N)]),
    'C3': pd.Series([np.random.normal(0,0.1)*10000 for n in range(N)], index=[n for n in range(N)]),
    'C4': pd.Series([random.randint(1,10000) for n in range(N)], index=[n for n in range(N)]),
    'C5': pd.Series([random.randint(1,10000) for n in range(N)], index=[n for n in range(N)]),
    'C6': pd.Series([random.randint(1,10000) for n in range(N)], index=[n for n in range(N)]),
    'C7': pd.Series([random.randint(1,10000) for n in range(N)], index=[n for n in range(N)]),
    'C8': pd.Series([random.randint(1,10000) for n in range(N)], index=[n for n in range(N)]),


     }

# creates Dataframe.
dframe = pd.DataFrame(d)

# print the data.
print(dframe)

dframe.to_csv('losowe_liczby.csv', index=False)