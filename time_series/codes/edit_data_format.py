import datetime
import pandas as pd


DATA_PATH = '../data/'

print('Loading Data ...')
se_origin = pd.read_csv(DATA_PATH+'se.csv')
ethereum_origin = pd.read_csv(DATA_PATH+'ethereum.csv')

se = se_origin.copy()
ethereum = ethereum_origin.copy()

ethereum.columns = ['Date','Open','High','Low','Close','Volume','Market Cap']

cols = ['Date','Open','High','Low','Close','Volume']

se = se[cols]
ethereum = ethereum[cols]

ethereum['Date'] = \
    ethereum['Date'].apply(lambda x: datetime.datetime.strptime(x, '%b %d, %Y'))
ethereum[cols[1:]] = ethereum[cols[1:]].applymap(lambda x: float(x.replace(',', '')[1:]))

print('Saving Data ...')
se.to_csv(DATA_PATH+'se_mod.csv', index=False)
ethereum.to_csv(DATA_PATH+'ethereum_mod.csv', index=False)

print('Done!')
