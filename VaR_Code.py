#%%
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

spread = 'Z_Spread'

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
posdata_raw.sort_values("Date", ascending=False, inplace=True)


ClassFilter = ""
if ClassFilter !="":
    posdata_raw = posdata_raw.loc[posdata_raw['Class']==ClassFilter]

print(posdata_raw.head())

#%%scrub position data
posdata = posdata_raw.loc[(posdata_raw['Date']>=datetime.strptime(datastartdate,'%Y%m%d')) & (posdata_raw['Date']<=datetime.strptime(dataenddate,'%Y%m%d'))].copy()
dates = posdata['Date'].unique()

pos = pd.DataFrame()
data = posdata.loc[~posdata['marketvalue'].isna()]

for i in range(1, len(dates)):
    print(dates[i])
    prevdata = data.loc[data['Date']==dates[i-1],['Date','cusip','price', spread]]
    temp = data.loc[data['Date']==dates[i]]
    temp = temp.merge(prevdata, how = 'inner', on = 'cusip',suffixes=("","_prev"))
    temp['return']= temp['price']-temp['price_prev']
    temp['spread_chg'] = temp[spread]-temp[f'{spread}_prev']
    pos = pd.concat([pos,temp])

positions = pos.copy()
#positions = positions.merge(mktdata, on = 'Date', how = "left")
positions.head(5)


#%%Read in market data
chunksize = 10 ** 6
mktdata = pd.DataFrame()
with pd.read_csv(datafldr + mktDataFile, chunksize=chunksize) as reader:
    for chunk in reader:
        mktdata = pd.concat([mktdata, chunk])
mktdata.sort_values("Date",ascending=False, inplace=True)

print(mktdata.head(2))

#%%  Filter market data to days where we have position data

#filter the market data and calculate differences
mktdata = mktdata.loc[mktdata['Date'].isin(dates)]





