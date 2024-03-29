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
    def __init__(self, filename, start_date, end_date, data_flag):
        self.filename = filename
        self.start_date = start_date
        self.end_date = end_date
        self.data_flag = data_flag
        # if data_flag = 0, then data manipulations visualizations shown but the cleaned data is not passed thru the lstm model
        # if data_flag = 1, then data manipulations are shown and the cleaned data is passed thru the lstm model
        # if data_flag = 2, then only lstm model is run and no manipulations are run or visualized
    def run_pipeline(self):
        df_temp = pd.read_csv('../stock_data/'+self.filename, usecols=['Date', 'Close'], na_values=['nan'])
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        df = df_temp[(df_temp['Date'] >= self.start_date) & (df_temp['Date'] <= self.end_date)]
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
    

        st.title("Data Manipulations")

        if(self.data_flag == 0):
            st.write('The following data manipulations are for visualization purposes and are not passed into the LSTM model because this is real, original stock data that we should not have to manipulate. However, we can gain insights from these visualizations such as certain historical events which resulted in an anamalous close price for a given stock index.')
        if(self.data_flag == 1):
            st.write('The follow data manipulations will identify vulnerabilities in the simulated attack stock index datasets. Once vulnerabilities are identified, they will be resolved and the new manipulated dataset will be run through the LSTM close price prediction model. It is essential to compare the results of data manipulation and LSTM prediction of this tainted dataset to the tainted dataset without cleaning manipulations which can be seen on other tab of this page.')
        if(self.data_flag == 2):
            st.write('This page is to see just the LSTM model predictions on the tainted, simulated attack datasets. We use this page to compare prediction results with the following page where the cleaning manipulations are applied. ')
        
        if(self.data_flag != 2):
            st.header("K-Nearest Neighbors")
            k = KNN_unsupervised.KNN_unsupervised(self.filename, self.start_date, self.end_date)
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

            manipulated_data = win.cleaned()

            st.header("Hurst Exponent")
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
            st.pyplot(fig)

            print(Hursts)

            left_col, right_col = st.columns(2)
            
            with left_col:
                st.header("Local Outlier Factor")
                st.write("Outliers detected using LOF: ")
                st.write(outliers)
                st.write("Total number of outliers detected = ", len(outliers))
                print(outliers)

            with right_col:
                st.header("Hole Detection")
                st.write("Holes detected: ")
                st.write(holes)
                st.write("Total number of holes detected = ", len(holes))
                print(holes)
        
        
            if(self.data_flag == 1):
                st.header("Anomalies Caught")
                if self.filename == 'new_nasdaq.csv':
                    df_temp = pd.read_csv('../stock_data/nasdaq_all.csv', usecols=['Date', 'Close'], na_values=['nan'])
                elif self.filename == 'new_djdatafinal.csv':
                    df_temp = pd.read_csv('../stock_data/djia_2012.csv', usecols=['Date', 'Close'], na_values=['nan'])
                elif self.filename == 'new_russel.csv':
                    df_temp = pd.read_csv('../stock_data/russel2000_all.csv', usecols=['Date', 'Close'], na_values=['nan'])
                df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                original = df_temp[(df_temp['Date'] >= self.start_date) & (df_temp['Date'] <= self.end_date)]
                original.set_index('Date', inplace=True)
                false_value_indices = original.compare(df).index.values.tolist()
                found_value_indices = df.index.intersection(manipulated_data.index).values.tolist()
                false_set = set(false_value_indices)
                found_set = set(found_value_indices)
                true_positive = len(false_set & found_set)
                false_positive = len(found_set) - true_positive
                false_negative = len(false_set) - true_positive
                true_negative = len(graphing_dates) - false_negative - false_positive - true_positive
                
                st.subheader("True Positives: "+str(true_positive))
                st.subheader("False Negatives: "+str(false_negative))
                st.subheader("False Positives: "+str(false_positive))
                st.subheader("True Negatives: "+str(true_negative))

        
        


        #data manipulation steps

        #LSTM Model
        model_input_data = df
        if(self.data_flag == 1):
            model_input_data = manipulated_data

        model_sim = LSTM_sim(model_input_data, self.filename, self.start_date, self.end_date)
        model_sim.simulation()

        #Model validity checks??

        #Testing Framework


###### RUN THE PIPELINE
# filename = 'nasdaq_all.csv'
# start_date = '2010-01-04'
# end_date = '2017-01-03'

# pipeline = Pipeline(filename, start_date, end_date)
# pipeline.run_pipeline()
