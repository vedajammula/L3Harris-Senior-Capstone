import random
import numpy as np
import csv
import pandas as pd

class Random_Data():

    def generate_random_data(data, fill, window, randomness, column_name):
        """

        Function to generate random data 

        Parameters:
        ----------
        data: pandas df
            stock data, untouched and to be altered
        fill: list
            list of row indicies that need to be changed
        window: int
            determines how far we look back -> range of random data generated
        randomness: int
            determines how random the data is, the smaller the number the more similar each generated data point will be to its previous
        column_name: string
            string for the column name in df, e.g. "Dow Jones Industrial Average"
       
            
        Returns: 
        -------
        pandas data frame
            This data frame is the result of applying random data generation to the orginal dataframe passed in
        """

        range = data.iloc[fill[0]-window:fill[0]+1]
        max = range[column_name].max()
        min = range[column_name].min()

        for i in fill:
            recent_max = data.at[i-1, column_name] + randomness
            recent_min = data.at[i-1, column_name] - randomness
            new_max = max if max <= recent_max else recent_max
            new_min = min if min >= recent_min else recent_min
            data.at[i, column_name] = np.random.uniform(new_min,new_max)

        return data


