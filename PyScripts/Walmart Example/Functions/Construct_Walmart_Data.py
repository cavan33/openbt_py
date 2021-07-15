"""
Get the Walmart data from CSV to numpy arrays and/or pandas dataframes
"""
def get_walmart_data():
     import pandas as pd
     import numpy as np
     # Example - the CO2 Plume data from Assignment 3
     # Fit the model
     walmart_pd = pd.read_csv('PyScripts/Walmart Example/Walmart_Store_sales.csv') # Might vary for your filesystem
     x_pd = walmart_pd.drop(labels = "Weekly_Sales", axis = 1) # All columns except sales:
     # Store, Date, Holiday Flag, Temperature, Fuel Price, CPI, Unemployment
     y_pd = walmart_pd["Weekly_Sales"] # Weekly sales column only
     
     # Convert the date row into an integer:
     # We'll use 1-1-2010 as day 1, since the data has only 2010 to 2012 data:
     from datetime import datetime
     # Also, add a "month" attribute:
     month = np.array([], dtype=int)
     for i in range(len(x_pd["Date"])):
          month = np.append(month, int(pd.to_datetime(x_pd["Date"][i], dayfirst = True).month))
          x_pd["Date"][i] = (pd.to_datetime(x_pd["Date"][i], dayfirst = True) - datetime(2009, 12, 31)).days    
     x_pd["Month"] = month
     # Rearrange columns:
     x_pd = x_pd[["Store", "Month", "Date", "Holiday_Flag", "Temperature",
                  "Fuel_Price", "CPI", "Unemployment"]]
     # Make these into numpy arrays:
     x = x_pd.to_numpy()
     y = y_pd.to_numpy()
     # x columns: Store, Month, Date, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment
     # y columns: Weekly_Sales
     return (x, y, x_pd, y_pd)
