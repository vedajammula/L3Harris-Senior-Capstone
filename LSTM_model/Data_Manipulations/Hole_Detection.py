import random
import numpy as np
import csv
import pandas as pd
from datetime import datetime
from datetime import date
from datetime import timedelta

class Hole_Detection:

    def detect_hole(df):
        df["Date"] = pd.to_datetime(df["Date"])
        dataf = pd.DataFrame({'Date':df.Date,'diff':df.Date-df.Date.shift(1)})
        dataf = dataf.drop(0)
        dataf["Is Weekday"] = dataf['Date'].dt.dayofweek <= 4
        temp = pd.DataFrame()
        temp['Date'] = dataf.loc[(dataf['Is Weekday'] == False) | (((dataf['diff'] >= timedelta(days=2)) & (dataf['Date'].dt.dayofweek != 0))), 'Date']
        return temp
    
