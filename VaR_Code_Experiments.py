#%%Import basic stuff
import run_model as rm
import RiskTools as rt
import numpy as np
from numpy.linalg import eig
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

#Dates on which you want to start and end your data set
datastartdate = '20180101'
dataenddate = '20220930'

#cutoff date for the out of sample period
oosdate = '20220731'

#output folder for results.
datafldr = r'M:\Risk Management\Projects\2022\Public Funds VaR Model'
outfldr = r'M:\Risk Management\Projects\2022\Public Funds VaR Model\output'

datafldr = rt.scrubfldrname(datafldr)
outfldr = rt.scrubfldrname(outfldr)

mktDataFile = "MarketData.csv"
posDataFile = "position_data.pqt"

spread = 'z_spread'

#%%Create lists of portfolio IDs and print a list of available asset classes
(fund, public, publicIDs, publicIDstr, private, privateIDs, privateIDstr) = rt.importFunds(exclude=['AOU'])
#read in portfolio information
qry = f"""select distinct b.class from dbo.position a left join dbo.instrument b on a.cusip = b.cusip
where a.date between '{datastartdate}' and '{dataenddate}' and portfolioid in ({publicIDstr})"""
print(qry)

data = rt.read_data('AOCA', qry)
list(data['class'])

#%%Read in position data
posdata_raw = pd.read_parquet(datafldr + posDataFile)

#%%Scrub the raw position data
posdata_raw.columns = posdata_raw.columns.str.lower()
posdata_raw.sort_values("date", ascending=True, inplace=True)
print(posdata_raw.head())

#%%scrub position data
ClassFilter = ""
if ClassFilter !="":
    posdata = posdata_raw.loc[posdata_raw['class']==ClassFilter]
else:
    posdata = posdata_raw.copy()

posdata = posdata.loc[(posdata['date']>=datetime.strptime(datastartdate,'%Y%m%d')) & (posdata['date']<=datetime.strptime(dataenddate,'%Y%m%d'))].copy()
dates = posdata['date'].unique()

pos = pd.DataFrame()
data = posdata.loc[~posdata['marketvalue'].isna()]

for i in range(1, len(dates)):
    print(dates[i])
    prevdata = data.loc[data['date']==dates[i-1],['date','cusip','portfolioid','price', spread]]
    temp = data.loc[data['date']==dates[i]]
    temp = temp.merge(prevdata, how = 'inner', on = ['cusip','portfolioid'],suffixes=("","_prev"))
    temp['return']= temp['price']-temp['price_prev']
    temp['spread_chg'] = temp[spread]-temp[f'{spread}_prev']
    pos = pd.concat([pos,temp])

positions = pos.copy()
#positions = positions.merge(mktdata, on = 'date', how = "left")
positions.head(5)

#%%Read in market data
chunksize = 10 ** 6
mktdata = pd.DataFrame()
with pd.read_csv(datafldr + mktDataFile, chunksize=chunksize) as reader:
    for chunk in reader:
        mktdata = pd.concat([mktdata, chunk])
mktdata.columns = mktdata.columns.str.lower()
mktdata.sort_values("date",ascending=False, inplace=True)

print(mktdata.head(2))

#%%  Filter market data to days where we have position data

#filter the market data and calculate differences
mktdata = mktdata.loc[mktdata['date'].isin(dates)]

#Calculate the portfolio value for each day. The example portfolio used is 190132
#The portolio return is calculated as a relative change
sample_port = positions[positions['portfolioid'] == 190132].groupby(['date'])['marketvalue'].sum()
sample_port.sort_index(ascending=False, inplace=True)
sample_port_chg = sample_port.pct_change(-1)
sample_port_chg.dropna(inplace=True)

#Split the market data into two groups, one for which change is calculated as difference
#and the other for which the change is calculated as percentage change
mktdata_diff_col = ['date', '3m cmt', '1y cmt', '2y cmt', '3y cmt', '5y cmt', '7y cmt',\
                    '10y cmt', '20y cmt', '30y cmt', '1y swap', '2y swap', '3y swap',\
                    '5y swap', '7y swap', '10y swap', '20y swap', '30y swap',\
                    'cdx hy cdsi gen 5y sprd corp', 'cdx ig cdsi gen 5y sprd corp',\
                    'mba 30y fixed', 'mba 15y fixed', 'mb30a130 index', 'bbg agg oas_tsy',\
                    'bbg ig corp oas_tsy', 'bbg hy corp oas_tsy', 'bbg agg oas_swap',\
                    'bbg ig corp oas_swap', 'bbg hy corp oas_swap',' bloomberg u.s. mbs oas_tsy ',\
                    ' bloomberg u.s. mbs oas_swap ', 'bloomberg u.s. mbs z_spread', 'bloomberg u.s. mbs bid_spread']

mktdata_pct_col = ['date', 'markit liq lev loan', 'bbg agg_level', 'bbg hy corp_level', \
                   ' bloomberg u.s. mbs index_level ']

#Calculate the market data returns
mktdata_diff = mktdata[mktdata_diff_col]
mktdata_diff.set_index('date', inplace=True)
mktdata_diff = mktdata_diff.diff(-1)
mktdata_diff.drop(mktdata_diff.index[-1], axis=0, inplace=True)

mktdata_pct = mktdata[mktdata_pct_col]
mktdata_pct.set_index('date', inplace=True)
mktdata_pct = mktdata_pct.pct_change(-1)
mktdata_pct.drop(mktdata_pct.index[-1], axis=0, inplace=True)

mktdata_chg = mktdata_diff.join(mktdata_pct)
mktdata_chg.replace([np.inf, -np.inf], np.nan, inplace=True)
mktdata_chg.dropna(inplace=True)

#Making sure the dates match for both the portfolio data and market data
sample_port_chg = sample_port_chg[sample_port_chg.index.isin(mktdata_chg.index)]
mktdata_chg = mktdata_chg[mktdata_chg.index.isin(sample_port_chg.index)]

#Performing the PCA on the market data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sc = StandardScaler()
pca = PCA()

mktdata_chg_of = pca.fit_transform(sc.fit_transform(mktdata_chg))

expl_var = pca.explained_variance_ratio_
expl_var_cum = np.cumsum(expl_var)
print(expl_var)
print('\n')
print(expl_var_cum)

plt.bar(range(0, len(expl_var)), expl_var, label='Variance Explained by each PC')
plt.step(range(0, len(expl_var_cum)), expl_var_cum, label='Total Variance Explained')
plt.legend(loc='best')

#Regressing the portfolio returns on the first 6 PCs
#(First 6 PCs explain more that 90% of the variance)
X = train_data[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']]
y = train_data[['marketvalue']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print_model = model.summary()
print(print_model)
