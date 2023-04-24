from LSTM_sim import LSTM_sim
from Window import Window
import pandas as pd
import streamlit as st
from LOF import get_LOF
from Hurst import get_hurst_diff
from Hole_Detection import detect_hole


class Pipeline():

    def run_pipeline(self):
        #preprocessing steps
        win = Window(df, 45, 10)
        outliers = pd.DataFrame()
        hole = pd.DataFrame()
        Hursts = []
        for i in range(win.numberOfWindows()):
            temp = pd.DataFrame()
            window, judge = win.Next()
            #run detection on judge with window as training data window
            hole = detect_hole(judge)
            holes = holes.append(hole)
            temp = temp.append(hole)

            LOF = get_LOF(45, window, judge)
            LOF = np.add(LOF, judge.first_valid_index) #LOF is now numpy list of indices of outliers
            outliers = outliers.append(LOF)
            temp = temp.append(judgement.iloc[LOF])

            color = get_hurst_diff(window, judgement)
            Hursts.append(color)

            #get temp indices and change df to nan
            #use interpolate
            indices = temp.index.tolist()
            judgement.loc[indices, 'Close'] = np.nan
            del temp

            #use df interpolate
            judgement = judgement.interpolate(method='linear')

            #returns modified dataframe cleaned to be resubmitted to the window
            #run fill missing value on cleaned df variable
            win.accepted(judgement)
    
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
        print(Hursts)


        st.sidebar.header("Local Outlier Factor")
        print(outliers)

        st.sidebar.header("Hole Detection")
        print(Hursts)


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
