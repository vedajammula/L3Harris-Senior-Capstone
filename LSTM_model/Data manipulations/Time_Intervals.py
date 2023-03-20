import random
import numpy as np
import csv
import pandas as pd

class Time_Intervals():

    def __init__(self, sample, n, k, d, size_max, size_min):
        """
            Initalize generating time intervals with necessary parameter values. 

            Parameters:
            ----------
            sample: String
                csv file of data
            n: int
                range
            k: int
                num of elements
            d: int (1)
                minimum distance
            size_max: int
                max size of dropped data entries
            size_min: int
                min size of dropped data entries
        """

        self.sample = sample
        self.n = n
        self.k = k
        self.d = d
        self.size_max = size_max
        self.size_min = size_min


    def generate_time_intervals(self):
        dropped_rows = self.rows_to_drop()
        self.sample.drop(dropped_rows)
        return self.sample

    def ranks(self):
        self.sample["Date"] = pd.to_datetime(self.sample["Date"])
        indices = sorted(range(len(self.sample)), key=lambda i: self.sample[i])
        return sorted(indices, key=lambda i: indices[i])

    def sample_with_minimum_distance(self):
        #k elements, range(n), minimum distance(d)
        sample = random.sample(range(self.n-(self.k-1)*(self.d-1)), self.k)
        return [s + (d-1)*r for s, r in zip(sample, self.ranks(sample))]

    def rows_to_drop(self):
        samples = self.sample_with_minimum_distance(self.n, self.k, self.d)
        temp = []
        for index, element in enumerate(samples):
            size = random.randint(self.size_min, self.size_max)
            temp.extend(list(range(element+1, element+size)))
        samples.extend(temp)
        return samples