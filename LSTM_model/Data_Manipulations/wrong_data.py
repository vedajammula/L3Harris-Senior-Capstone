import random
import numpy as np
import csv
import pandas as pd
import copy

#import os 
# dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)
# cwd = os.getcwd()
# print(cwd)

dj_df = pd.read_csv("stock_data/djia_2012.csv") 
# nas_df = pd.read_csv("stock_data/nasdaq_all.csv") 
# rus_df = pd.read_csv("stock_data/russel2000_all.csv") 

# Grab up to 10 random intervals of (randomly generated) n lines. Using Gaussian noise to make data in intervals progressively get worse. 
def gen_wrong(sample):
    df = pd.read_csv(sample) #get data col - can use iloc[:,0] to get Date col if needed
    df_copy = df.copy() #make copy of dataframe to apply change
    avg_length_crash = 342 #average length of stock market crash
    n = random.randint(5, 10) #num intervals
    noise_levels = [10, 20, 30, 40, 50] #for noise
    for i in range(n):
        num_rows = random.randint(1, avg_length_crash) #generate random number of rows for interval
        start_idx = random.randint(0, len(df) - num_rows) #start idx for intervall
        end_idx = start_idx + num_rows #last idx for interval
        interval = df_copy.iloc[start_idx:end_idx] #create interval
        for j in range(len(noise_levels)): #apply noise to copy of df
            noise = np.random.normal(0, noise_levels[j], num_rows) #generate random amt of noise
            interval.iloc[:, 1] += noise #add noise
            df_copy.iloc[start_idx:end_idx] = interval #add interval to df copy
    return df_copy

#testing
djdf_copy = gen_wrong("stock_data/djia_2012.csv")
dj_df.compare(djdf_copy)
# print(df.columns)
# print(df_copy.columns)