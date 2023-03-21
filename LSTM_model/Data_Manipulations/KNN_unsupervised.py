import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
#source -> https://www.kaggle.com/code/kimchanyoung/simple-anomaly-detection-using-unsupervised-knn 

class KNN_unsupervised:

    def __init__(self, df):
        self.df = df


    def process_data(self):
        date_indexed = []
        #self.df['Date'] = pd.to_datetime(self.df['Date'])
        for i in range(len(self.df)):
            date_indexed.append(i)
        new_df = pd.DataFrame(data={'Date': date_indexed, 'Close': self.df['Close']})
        self.df = new_df
    
    def run_KNN(self):
        self.process_data()
        #create model
        neighbors = NearestNeighbors(n_neighbors = math.ceil(math.sqrt(len(self.df))))
        #fit model
        neighbors.fit(self.df)
        #get distances and indices of neighbors
        dists, indices = neighbors.kneighbors(self.df)
        #plot 
        plt.figure(figsize=(15, 7))
        plt.plot(dists.mean(axis =1))
        #get distance mean
        distances = pd.DataFrame(dists)
        distances_mean = distances.mean(axis =1)
        #get abnormal points
        quantile_75 = distances_mean.quantile(.75)
        threshold = math.ceil(quantile_75)
        outlier_index = np.where(distances_mean > threshold)
        outlier_vals = self.df.iloc[outlier_index]

        #plot outliers
        plt.figure(figsize=(20, 7))
        plt.plot(self.df["Date"], self.df["Close"], color = "b")
        # plot outlier values
        plt.scatter(outlier_vals["Date"], outlier_vals["Close"], color = "r")
        plt.show()

        return outlier_vals


df1 = pd.DataFrame(index= pd.date_range('2010-01-04','2017-01-03',freq='B'))
df_temp = pd.read_csv('../../stock_data/nasdaq_all.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
df1 = df1.join(df_temp)

# df1['Date'] = pd.to_datetime(df1['Date'])
df = df1.dropna()
#df.rename(['Date', 'Close'], axis='columns')

#print(df.head())

temp = KNN_unsupervised(df)
temp.run_KNN()