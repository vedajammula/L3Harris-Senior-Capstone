#imports
import streamlit as st
import numpy as np
import random
import pandas as pd 
import matplotlib.pyplot as plt

from pandas import datetime
import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out



class LSTM_sim():

    def __init__(self, df):
        self.df = df

    def simulation(self):
        st.title("L3 Harris Senior Capstone - Data Input Manipulation of LSTM Model")

        self.get_data_timeseries()
        scaler = self.fill_missing_vals()
        left_col, right_col = st.columns(2)
        # x_train, y_train, x_test, y_test = self.create_train_test_sets()
        self.model_run(scaler)

    def get_data_timeseries(self):
        # dates = pd.date_range('2010-01-04','2017-01-03',freq='B')
        # df_nas = self.get_data(dates)
        # #df_nas.head()
        # df_nas.fillna(method='pad')

        st.subheader("NasDaq Data from 2010-2017, visualizing the close prices")
        left_col, right_col = st.columns(2)
        with left_col:
            st.write(self.df)
        #df_nas.plot(figsize=(10, 6), subplots=True)
        with right_col:
            print('cols', self.df.columns)
            st.line_chart(self.df)
    
    # def get_data(self, dates):
    #     indices = ['djia_2012', 'nasdaq_all']
    #     df = pd.DataFrame(index=dates)
    #     df_temp = pd.read_csv('../stock_data/nasdaq_all.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
    #     df = df.join(df_temp)
    #     return df
    
    def fill_missing_vals(self):
        self.df = self.df.fillna(method='ffill')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.df['Close'] = scaler.fit_transform(self.df['Close'].values.reshape(-1,1))
        return scaler
    
    # function to create train, test data given stock data and sequence length
    def load_data(self, look_back):
        data_raw = self.df.values # convert to numpy array
        data = []
        
        # create all possible sequences of length look_back
        for index in range(len(data_raw) - look_back): 
            data.append(data_raw[index: index + look_back])
        
        data = np.array(data);
        test_set_size = int(np.round(0.2*data.shape[0]));
        train_set_size = data.shape[0] - (test_set_size);
        
        x_train = data[:train_set_size,:-1,:]
        y_train = data[:train_set_size,-1,:]
        
        x_test = data[train_set_size:,:-1]
        y_test = data[train_set_size:,-1,:]
        
        return [x_train, y_train, x_test, y_test]
    
    
    def create_train_test_sets(self):
        look_back = 40 # choose sequence length
        x_train, y_train, x_test, y_test = self.load_data(look_back)
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        return [x_train, y_train, x_test, y_test]
    
    def model_run(self, scaler):
        input_dim = 1
        hidden_dim = 32
        num_layers = 2 
        output_dim = 1
        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        left_col, right_col = st.columns(2)

        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        with left_col:
            st.subheader("LSTM Model + Params:")
            st.write(model)
            st.write(len(list(model.parameters())))
        
            for i in range(len(list(model.parameters()))):
                st.write(list(model.parameters())[i].size())
        
        ##train model
        vals = self.create_train_test_sets()
        look_back = 40
        x_train = vals[0]
        y_train =vals[1]
        x_test = vals[2]
        y_test = vals[3]
        num_epochs = 100
        hist = np.zeros(num_epochs)

        # Number of steps to unroll
        seq_dim =look_back-1  
        with right_col:
            st.subheader("Training Model: ")
            for t in range(num_epochs):
                # Initialise hidden state
                # Don't do this if you want your LSTM to be stateful
                #model.hidden = model.init_hidden()
                
                # Forward pass
                y_train_pred = model(x_train)

                loss = loss_fn(y_train_pred, y_train)

                if t % 10 == 0 and t !=0:
                    st.write("Epoch ", t, "MSE: ", loss.item())
                hist[t] = loss.item()

                # Zero out gradient, else they will accumulate between epochs
                optimiser.zero_grad()

                # Backward pass
                loss.backward()

                # Update parameters
                optimiser.step()

        st.subheader('Looking at the shape of the train and test sets')
        st.write('x_train.shape = ',x_train.shape)
        st.write('y_train.shape = ',y_train.shape)
        st.write('x_test.shape = ',x_test.shape)
        st.write('y_test.shape = ',y_test.shape)
       
        st.subheader("Analyze Training Loss")
        st.line_chart(hist)

        

        ##make predictions
        y_test_pred = model(x_test)

        # invert predictions
        y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
        y_train = scaler.inverse_transform(y_train.detach().numpy())
        y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
        y_test = scaler.inverse_transform(y_test.detach().numpy())

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
        st.subheader('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
        st.subheader('Test Score: %.2f RMSE' % (testScore))

        # visualize results
        st.subheader("Final Results: Visualize our Predicted Stock Close Price v.s. Real Stock Close Price")
        figure, axes = plt.subplots(figsize=(20, 15))
        axes.xaxis_date()

        axes.plot(self.df[len(self.df)-len(y_test):].index, y_test, color = 'red', label = 'Real NasDaq Stock Price')
        axes.plot(self.df[len(self.df)-len(y_test):].index, y_test_pred, color = 'blue', label = 'Predicted NasDaq Stock Price')
        #axes.xticks(np.arange(0,394,50))
        plt.title('NasDaq Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('NasDaq Stock Price')
        plt.legend()
        plt.savefig('NasDaq_pred.png')
        st.pyplot(figure)
