#!/usr/bin/env python
# coding: utf-8

# In[27]:


import random
import numpy as np
import csv
import pandas as pd

dj_df = pd.read_csv("stock_data/djia_2012.csv") 
nas_df = pd.read_csv("stock_data/nasdaq_all.csv", usecols=['Date', 'Adj Close']) 
rus_df = pd.read_csv("stock_data/russel2000_all.csv", usecols=['Date', 'Adj Close']) 

# dj_df["Date"] = pd.to_datetime(dj_df["Date"])
# nas_df["Date"] = pd.to_datetime(nas_df["Date"])
# nas_df["Date"] = pd.to_datetime(nas_df["Date"])

print(dj_df.head())
print(nas_df.head())
print(rus_df.head())

print(len(dj_df.index))
print(len(nas_df.index))
print(len(rus_df.index))

print(dj_df.columns)
print(nas_df.columns)
print(rus_df.columns)


# Grab up to 10 random intervals of (randomly generated) n lines. Using Gaussian noise to make data in intervals progressively get worse. 

# In[28]:


# len(dj_df.index)
# len(nas_df.index)
# len(rus_df.index)

def gen_wrong(sample):
    df = pd.read_csv(sample, parse_dates=['Date'])
    df_copy = df.copy()
    avg_length_crash = 342 #average length of stock market crash
    n = random.randint(5, 10) #num intervals
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5] #for noise
    for i in range(n):
        num_rows = random.randint(1, avg_length_crash) #generate random number of rows for wrong data gen
        start_idx = random.randint(0, len(df) - num_rows)
        end_idx = start_idx + num_rows
        interval = df_copy.iloc[start_idx:end_idx]
        for j in range(len(noise_levels)):
            noise = np.random.normal(0, noise_levels[j], num_rows)
            interval.iloc[:, 1] += noise
            df_copy.iloc[start_idx:end_idx] = interval
    return df_copy
        
# df = gen_wrong("stock_data/djia_2012.csv")
# print(df)


# In[30]:


import matplotlib.pyplot as plt

df = pd.read_csv("stock_data/djia_2012.csv", parse_dates=['Date'])
df_copy = gen_wrong("stock_data/djia_2012.csv")
print(df.columns)
print(df_copy.columns)

# Plot the original data and the copy - just had ai do this lol
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Dow Jones Industrial Average'], label='Original Data')
plt.plot(df_copy['Date'], df_copy['Dow Jones Industrial Average'], label='Data with Added Noise')
plt.legend()
plt.title('Comparison of Original Data and Data with Added Noise')
plt.xlabel('Date')
plt.ylabel('Stock Market Average')
plt.show()


# In[ ]:




