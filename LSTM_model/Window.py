#imports
import pandas
import math

class Window():
    def __init__(self, dataframe, wSize = 20, stepSize = 1):
        self.df = dataframe
        self.wSize = wSize
        self.stepSize = stepSize
        self.index = 0
        self.size = len(self.df.index)
        #database, window size, and step size and index
    
    def numberOfWindows(self):
        return math.ceil((self.size-wSize) / self.stepSize)

    def nextWindow(self):
        limit = self.index + self.wSize + self.stepSize if self.index + self.wSize + self.stepSize <= self.size else self.size
        win = self.df.iloc[self.index : self.index + wSize]
        judge = self.df.iloc[self.index + wSize : limit]
        self.index = self.index + self.stepSize
        return (win, judge)

    def accepted(self, dataframe):
        cols = list(self.df.columns) 
        self.df.loc[self.df.date.isin(dataframe.date), cols] = dataframe[cols].values

    def cleaned(self):
        return self.df