import streamlit as st
import numpy as np
from pandas import datetime
import math
from operator import itemgetter
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.autograd import Variable

#questions:
# What is num epochs
# WHat is look back and Seq_dim

def train_and_test(x_train, y_train, x_test, y_test, model, scaler):
    ##train model
    look_back = 40
    num_epochs = 100
    #hist = np.zeros(num_epochs)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    # Number of steps to unroll
    seq_dim =look_back-1  
    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        #model.hidden = model.init_hidden()
                
        # Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred, y_train)

        #if t % 10 == 0 and t !=0:
        #    st.write("Epoch ", t, "MSE: ", loss.item())
        #hist[t] = loss.item()
        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

    #st.subheader('Looking at the shape of the train and test sets baseline')
    #st.write('x_train.shape = ',x_train.shape)
    #st.write('y_train.shape = ',y_train.shape)
    #st.write('x_test.shape = ',x_test.shape)
    #st.write('y_test.shape = ',y_test.shape)
   
    #st.subheader("Analyze Training Loss")
    #st.line_chart(hist)

        

    ##make predictions
    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    #st.subheader('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    #st.subheader('Test Score: %.2f RMSE' % (testScore))
    return(loss.item(), testScore, y_test_pred,y_test)
    

def forward_chaining_CV(X_train, y_train, folds, algorithm,scaler):
    """
    Given X_train and y_train (the test set is excluded from the Cross Validation),
    number of folds, the ML algorithm to implement and the parameters to test,
    the function acts based on the following logic: it splits X_train and y_train in a
    number of folds equal to number_folds. Then train on one fold and tests accuracy
    on the consecutive as follows:
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    Returns mean of test accuracies.
    """
    #print( 'Size train set: ', X_train.shape)
    
    # k is the size of each fold. It is computed dividing the number of 
    # rows in X_train by number_folds. This number is floored and coerced to int
    k = int(np.floor(float(X_train.shape[0]) / folds))
    st.write( 'Size of each fold: ', k)
    
    # initialize to zero the accuracies array. It is important to stress that
    # in the CV of Time Series if I have n folds I test n-1 folds as the first
    # one is always needed to train
    accuracies = np.zeros(folds-1)

    # loop from the first 2 folds to the total number of folds    
    for i in range(2, folds + 1):
        st.write( '')
        
        # the split is the percentage at which to split the folds into train
        # and test. For example when i = 2 we are taking the first 2 folds out 
        # of the total available. In this specific case we have to split the
        # two of them in half (train on the first, test on the second), 
        # so split = 1/2 = 0.5 = 50%. When i = 3 we are taking the first 3 folds 
        # out of the total available, meaning that we have to split the three of them
        # in two at split = 2/3 = 0.66 = 66% (train on the first 2 and test on the
        # following)
        split = float(i-1)/i
        
        # example with i = 4 (first 4 folds):
        #      Splitting the first       4        chunks at          3      /        4
        st.write( 'Splitting the first ' + str(i) + ' chunks at ' + str(i-1) + '/' + str(i) )
        
        # as we loop over the folds X and y are updated and increase in size.
        # This is the data that is going to be split and it increases in size 
        # in the loop as we account for more folds. If k = 300, with i starting from 2
        # the result is the following in the loop
        # i = 2
        # X = X_train[:(600)]
        # y = y_train[:(600)]
        #
        # i = 3
        # X = X_train[:(900)]
        # y = y_train[:(900)]
        # .... 
        X = X_train[:(k*i)]
        y = y_train[:(k*i)]
        st.write( 'Size of train + test: ', X.shape) # the size of the dataframe is going to be k*i

        # X and y contain both the folds to train and the fold to test.
        # index is the integer telling us where to split, according to the
        # split percentage we have set above
        index = int(np.floor(X.shape[0] * split))
        
        # folds used to train the model        
        X_trainFolds = X[:index]        
        y_trainFolds = y[:index]
        
        # fold used to test the model
        X_testFolds = X[(index + 1):]
        y_testFolds = y[(index + 1):]
        
        # i starts from 2 so the zeroth element in accuracies array is i-2. performClassification() is a function which takes care of a classification problem. This is only an example and you can replace this function with whatever ML approach you need.
        accuracies[i-2], testscore,y_test_pred, y_test = train_and_test(X_trainFolds, y_trainFolds, X_testFolds, y_testFolds, algorithm,scaler)
        
        # example with i = 4:
        #      Accuracy on fold         4     :    0.85423
        st.write( 'RMSE on fold ' + str(i) + ': ', accuracies[i-2])
    # the function returns the mean of the accuracy on the n-1 folds    
    return accuracies.mean()


"""
def k_fold(df, model):
    kfold = KFold(10,False,1)
    df = self.get_data_timeseries()
    #use k-fold CV to evaluate model
    #scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error',
                     #cv=cv, n_jobs=-1)
    #view mean absolute error
    np.mean(np.absolute(scores))
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
    """