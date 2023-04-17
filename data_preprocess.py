import numpy as np
import pandas as pd



data = pd.read_csv('data/wtbdata_245days.csv')

## handle outliers
def f(x):
    if x[12] < 0:
        return [x[0],x[1],x[2], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    elif x[3] == 0 and x[4] == 0 and x[5] == 0:
        return [x[0], x[1], x[2], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    elif x[4] > 180 or x[4] < -180:
        return [x[0], x[1],x[2], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    elif x[7] > 720 or x[7] < -720:
        return [x[0], x[1],x[2], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    else:
        return x
data = data.apply(f,axis=1,result_type='expand')

# Assume each turbine is independent to each other, so I fill nan value and normalize within turbine group.
df = data.drop(columns=['Tmstamp','Day']).groupby('TurbID').transform(lambda x: x.fillna(x.mean()))
df['TurbID'] = data['TurbID']

groups = df.groupby(['TurbID'])
mean, std = groups.transform("mean"), groups.transform("std")
normalized_df = (df[mean.columns] - mean) / std

normalized_df['TurbID'] = data['TurbID']
normalized_df['Day'] = data['Day']
normalized_df['Tmstamp'] = data['Tmstamp']

normalized_df.to_csv('data/clean_fill_data.csv',index=False)

