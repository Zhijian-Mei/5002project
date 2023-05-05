import numpy as np
import pandas as pd

data = pd.read_csv('data/wtbdata_245days.csv')


## handle outliers
def f(x):
    if x[12] < 0:
        return [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], 0.0]
    elif x[12] <= 0 and x[3] > 2.5:
        return [x[0], x[1], x[2], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    elif x[8] > 89 or x[9] > 89 or x[10] > 89:
        return [x[0], x[1], x[2], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    elif x[7] > 720 or x[7] < -720:
        return [x[0], x[1], x[2], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    elif x[4] > 180 or x[4] < -180:
        return [x[0], x[1], x[2], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    else:
        return x


data = data.apply(f, axis=1, result_type='expand')

# Assume each turbine is independent to each other, so I fill nan value and normalize within turbine group.
df = data.drop(columns=['Tmstamp', 'Day']).groupby('TurbID').transform(lambda x: x.fillna(x.mean()))
target = df['Patv']
df = df.drop(columns=['Patv'])
df['TurbID'] = data['TurbID']

groups = df.groupby(['TurbID'])
mean, std = groups.transform("mean"), groups.transform("std")

### for store mean and std for prediction
column_names = list(mean.columns)
column_names.append('TurbID')

mean_record = pd.DataFrame(columns=column_names)
std_record = pd.DataFrame(columns=column_names)

mean['TurbID'] = data['TurbID']
std['TurbID'] = data['TurbID']

means = list(mean.groupby('TurbID'))
stds = list(std.groupby('TurbID'))
for idx,item in means:
    mean_record = pd.concat([mean_record,item.iloc[[0]]])
mean_record = mean_record.reset_index(drop=True)
mean_record.to_csv('data/mean_record.csv',index=False)

for idx,item in stds:
    std_record = pd.concat([std_record,item.iloc[[0]]])
std_record = std_record.reset_index(drop=True)
std_record.to_csv('data/std_record.csv',index=False)

print(mean_record)
print(std_record)

quit()
###

normalized_df = (df[mean.columns] - mean) / std

normalized_df['Patv'] = target
normalized_df['TurbID'] = data['TurbID']
normalized_df['Day'] = data['Day']
normalized_df['Tmstamp'] = data['Tmstamp']

normalized_df.to_csv('data/clean_fill_data.csv', index=False)
