from LSTM_sim import LSTM_sim
from Data_Manipulations.Window import Window
import pandas as pd
import numpy as np
import streamlit as st
from Data_Manipulations.LOF import get_LOF
from Data_Manipulations.Hurst import get_hurst_diff
from Data_Manipulations import KNN_unsupervised
from Data_Manipulations.Hole_Detection import detect_hole
import matplotlib.pyplot as plt
import math



class Pipeline():

    def run_pipeline(self):
        filename = 'new_djdatafinal.csv'
        start_date = '2010-01-04'
        end_date = '2017-01-03'
        df_temp = pd.read_csv('../stock_data/'+filename, usecols=['Date', 'Close'], na_values=['nan'])
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        df = df_temp[(df_temp['Date'] >= start_date) & (df_temp['Date'] <= end_date)]
        df.set_index('Date', inplace=True)

        #preprocessing steps

        ###TEMP DATA INITIALIZIATION FOR MODEL
        """ dates = pd.date_range('2010-01-04','2017-01-03',freq='B')
        indices = ['djia_2012', 'nasdaq_all']
        df = pd.DataFrame(index=dates)
        df_temp = pd.read_csv('../stock_data/nasdaq_all.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        df = df.join(df_temp)
        #df_nas.head()
        df.fillna(method='pad') """
    
        st.set_page_config(page_title='L3Harris Senior Capstone', page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

        st.sidebar.title("Data Manipulations")

        st.sidebar.header("K-Nearest Neighbors")
        

        k = KNN_unsupervised.KNN_unsupervised(filename, start_date, end_date)
        manipulated_data = k.run_KNN()
        print(manipulated_data)

        win = Window(manipulated_data, 45, 10)
        outliers = pd.DataFrame()
        holes = pd.DataFrame()
        Hursts = []
        for i in range(win.numberOfWindows()):
            temp = pd.DataFrame()
            window, judge = win.nextWindow()
            #run detection on judge with window as training data window
            hole = detect_hole(judge)
            holes = pd.concat([holes,hole])
            temp = pd.concat([temp,hole])

            LOF = get_LOF(45, window, judge)  #LOF is now numpy list of numerical indices of outliers
            LOF = judge.iloc[LOF]
            outliers = pd.concat([outliers,LOF])
            temp = pd.concat([temp,LOF])
            

            color = get_hurst_diff(window, judge)
            Hursts.append(color)

            #get temp indices and change df to nan
            #use interpolate
            indices = temp.index.tolist()
            judge.loc[indices, 'Close'] = np.nan
            del temp

            #use df interpolate
            judge = pd.concat([window,judge]).interpolate(method='linear', order=2).tail(len(df. index))

            #returns modified dataframe cleaned to be resubmitted to the window
            #run fill missing value on cleaned df variable
            win.accepted(judge)

        cleaned = win.cleaned()


        st.sidebar.header("Hurst Exponent")
        fig = plt.figure(figsize=(50,15))
        ax = fig.add_subplot(111)
        ax.set_title('Hurst-based Trend Analysis')
        ax.set_xlabel('Time')
        ax.set_ylabel('Prices')
        graphing_dates = list(df.index)
        print(graphing_dates[0])
        ax.scatter(df.index,df["Close"], zorder=4)
        ax.axvspan(graphing_dates[0], graphing_dates[44], alpha=0.3,color='green',zorder=3)
        date_sliced = graphing_dates[44:]
        for i in range(len(date_sliced)):
            endDate = i+10 if i+10<len(date_sliced) else len(date_sliced)-1
            ax.axvspan(date_sliced[i], date_sliced[endDate], alpha=0.2,color=Hursts[math.floor(i/10)],zorder=3)
            i = endDate
        st.sidebar.pyplot(fig)

        print(Hursts)


        st.sidebar.header("Local Outlier Factor")
        st.sidebar
        st.sidebar.write(outliers)
        print(outliers)

        st.sidebar.header("Hole Detection")
        st.sidebar.write(holes)
        print(holes)
    

        
        


        #data manipulation steps

        #LSTM Model
        model_sim = LSTM_sim(cleaned)
        model_sim.simulation()

        #Model validity checks??

        #Testing Framework


###### RUN THE PIPELINE

pipeline = Pipeline()
pipeline.run_pipeline()
