from LSTM_sim import LSTM_sim
import pandas as pd
import streamlit as st
from Data_Manipulations.KNN_unsupervised import KNN_unsupervised 

class Pipeline():

    def run_pipeline(self):
        #preprocessing steps
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


        ###TEMP DATA INITIALIZIATION FOR MODEL
        # dates = pd.date_range('2010-01-04','2017-01-03',freq='B')
        # indices = ['djia_2012', 'nasdaq_all']
        # df = pd.DataFrame(index=dates)
        # df_temp = pd.read_csv('../stock_data/nasdaq_all.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        # df = df.join(df_temp)
        # #df_nas.head()
        # df.fillna(method='pad')

        #data manipulation steps

        #LSTM Model
        model_sim = LSTM_sim(manipulated_data)
        model_sim.simulation()

        #Model validity checks??

        #Testing Framework


###### RUN THE PIPELINE

pipeline = Pipeline()
pipeline.run_pipeline()
