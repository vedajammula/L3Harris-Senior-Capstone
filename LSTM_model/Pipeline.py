from LSTM_sim import LSTM_sim
from Window import Window
import pandas as pd
import streamlit as st


class Pipeline():

    def run_pipeline(self):
        #preprocessing steps

        ###TEMP DATA INITIALIZIATION FOR MODEL
        dates = pd.date_range('2010-01-04','2017-01-03',freq='B')
        indices = ['djia_2012', 'nasdaq_all']
        df = pd.DataFrame(index=dates)
        df_temp = pd.read_csv('../stock_data/nasdaq_all.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        df = df.join(df_temp)
        #df_nas.head()
        df.fillna(method='pad')

        win = Window(df, 20, 1)
        for i in range(win.numberOfWindows()):
            window, judge = win.Next()
            holes = detect_hole(judge)
            #run detection on judge with window as training data window
            #use df interpolate
            #returns modified dataframe cleaned to be resubmitted to the window
            #run fill missing value on cleaned df variable
            win.accepted(cleaned)
    
        st.set_page_config(page_title='L3Harris Senior Capstone', page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

        st.sidebar.title("Data Manipulations")

        st.sidebar.header("K-Nearest Neighbors")
        filename = 'nasdaq_all.csv'
        start_date = '2010-01-04'
        end_date = '2017-01-03'

        k = KNN_unsupervised(filename, start_date, end_date)
        manipulated_data = k.run_KNN()
        print(manipulated_data)


        st.sidebar.header("Hurst Exponent")

        st.sidebar.header("Local Outlier Factor")

        st.sidebar.header("Hole Detection")


        #data manipulation steps

        #LSTM Model
        model_sim = LSTM_sim(df)
        model_sim.simulation()

        #Model validity checks??

        #Testing Framework


###### RUN THE PIPELINE

pipeline = Pipeline()
pipeline.run_pipeline()
