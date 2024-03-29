import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import streamlit as st
#source -> https://www.kaggle.com/code/morecoding/anomalydetection



class KNN_unsupervised():

    def __init__(self, filename, start_date, end_date):
        self.filename = filename
        self.start_date = start_date
        self.end_date = end_date
    
    def process_data(self):
        #read csv file and set the dates to what we want 
        df_temp = pd.read_csv('../stock_data/'+self.filename, usecols=['Date', 'Close'], na_values=['nan'])
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])

        df = df_temp[(df_temp['Date'] >= self.start_date) & (df_temp['Date'] <= self.end_date)]
        df.set_index('Date', inplace=True)

        return df


    def run_KNN(self):
        df = self.process_data()
        #compute percent change in stock of close price 
        #delta t = 100 * ((x_t - x_(t-1))/ x_(t-1))
        N,d = df.shape
        delta = pd.DataFrame(100*np.divide(df.iloc[1:,:].values-df.iloc[:N-1,:].values, df.iloc[:N-1,:].values), columns=df.columns, index=df.iloc[1:].index)

        #KNN algo with nearest neighbors
        knn = 4
        nbrs = NearestNeighbors(n_neighbors=knn, metric=distance.euclidean).fit(delta.values)
        distances, indices = nbrs.kneighbors(delta.values)
        graphing_dates = list(df.index)
        graphing_dates.pop(0)
        #create anomaly scores from distances 
        anomaly_score = distances[:,knn-1]

        st.text('Running KNN on ' + self.filename + ' with k neighbors=' + str(knn) )
        st.text('Colormap of outliers in this dataset')

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.set_title('Colormap of outliers')
        ax.set_xlabel('Time')
        ax.set_ylabel('Anomaly Score of Close Values')
        p = ax.scatter(graphing_dates ,list(delta.Close),c=anomaly_score,cmap='jet')

        fig.colorbar(p)
        st.pyplot(fig)


        #calculate and append anomaly scores 
        anom = pd.DataFrame(anomaly_score, index=delta.index, columns=['Anomaly_Score'])
        result = pd.concat((delta,anom), axis=1)

        #display largest anomaly scores 
        largest_anomalies = result.nlargest(5,'Anomaly_Score')

        st.text('Top 5 Largest Anomaly Scores of Data Entries')
        st.write(largest_anomalies)
        
        #the threshold will be the lowest anomaly score values
        threshold = largest_anomalies['Anomaly_Score'].min()
        indices = (result.index[result['Anomaly_Score'] >= threshold]).tolist()
        indices_closevals = [df.loc[ind, 'Close'] for ind in indices]
        #drop rows of the lowest anomaly score values
        for i in range(len(indices)):
            df = df.drop(indices[i])
        
        figure,axes = plt.subplots(figsize=(20, 15))
        axes.plot(df.index, df['Close'])
        axes.scatter(indices, indices_closevals, color = "r")
        plt.title(self.filename + ' data visualization ' + 'outliers')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        st.pyplot(figure)

        return df



########### To run and test
# filename = 'nasdaq_all.csv'
# start_date = '2010-01-04'
# end_date = '2017-01-03'

# k = KNN_unsupervised(filename, start_date, end_date)
# manipulated_data = k.run_KNN()
# print(manipulated_data)
