import pandas as pd
import statsmodels.api as sm

#Function for executing the statsmodels.api linregress function
def linregress(positions, y_vars, X_vars, model_start_date, oosdate):
    #split data into insample and out of sample data sets
    oos_data = positions.loc[(positions['Date']<=oosdate) & (positions['Date'] >= model_start_date)]
    ins_data = positions.loc[positions['Date']>oosdate]

    y = oos_data[y_vars]
    X = oos_data[X_vars]

    y_ins = ins_data[y_vars]
    X_ins = ins_data[X_vars]

    y= y_ins
    X= X_ins
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print_model = model.summary()
    print(print_model)
