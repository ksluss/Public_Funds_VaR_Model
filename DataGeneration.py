#%%Import basic stuff
import run_model as rm
from RiskTools import useful_functions as rt
from RiskTools import GenerateMetrics as gm
import numpy as np
from numpy.linalg import eig
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import Market_Data as mkt

#Dates on which you want to start and end your data set
datastartdate = '20180101'
dataenddate = '20220930'

startdate = datetime.strptime(datastartdate,'%Y%m%d')
enddate = datetime.strptime(dataenddate,'%Y%m%d')

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


#%%Ratings
qry = f"""select distinct a.date, a.cusip, b.AOCA_Rating from dbo.position a 
    OUTER APPLY dbo.getInstrumentRating(a.Cusip, a.Date) b where a.date between '{datastartdate}' and '{dataenddate}' and portfolioid in ({publicIDstr})"""
print(qry)
ratings = rt.read_data('AOCA',qry)
ratings.columns = ratings.columns.str.lower()

#%%Read in position data
posdata_raw = pd.read_parquet(datafldr + posDataFile)
posdata_raw = posdata_raw.merge(ratings, on=['cusip','date'], how = 'left')
igornot = rt.getRatingsMap()
posdata_raw['is_ig'] = posdata_raw['aoca_rating'].map(igornot)


#%%Scrub the raw position data
posdata_raw.columns = posdata_raw.columns.str.lower()
posdata_raw.sort_values("date", ascending=True, inplace=True)
print(posdata_raw.head())
posdata_raw.to_parquet(datafldr + "position_data.pqt")