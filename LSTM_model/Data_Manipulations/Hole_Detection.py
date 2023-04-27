import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
from datetime import timedelta


def detect_hole(df):
    df['Dates'] = pd.to_datetime(df.index)
    dataf = pd.DataFrame({'Dates':df.Dates,'diff':df.Dates-df.Dates.shift(1)})
    dataf["Is Weekday"] = dataf['Dates'].dt.dayofweek <= 4
    temp = pd.DataFrame()
    temp['Dates'] = dataf.loc[(dataf['Is Weekday'] == False) | (((dataf['diff'] >= timedelta(days=2)) & (dataf['Dates'].dt.dayofweek != 0))), 'Dates']
    return temp