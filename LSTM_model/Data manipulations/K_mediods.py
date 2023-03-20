import random
import csv
from datetime import datetime
from datetime import date
from datetime import timedelta
import numpy as np
import random
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn_extra.cluster import KMedoids


class K_mediods:

    def compute_mediods(csv_name):
        df = pd.DataFrame(index= pd.date_range('2010-01-04','2017-01-03',freq='B'))
        df_temp = pd.read_csv('../../stock_data/nasdaq_all.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        df = df.join(df_temp)

        data = df.to_numpy()
        print(data)


km = K_mediods

km.compute_mediods('poop')




