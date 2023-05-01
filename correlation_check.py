import pandas as pd

df = pd.read_csv('data/clean_fill_data.csv')

groups = df.groupby(['TurbID'])

for group in groups:
    id = group[0]
    data = group[1]
    print(f'Turb # {id}')
    print(data.corr(method='pearson').loc['Patv'] )
    print('------------------------------------------------')