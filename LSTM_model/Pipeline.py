from LSTM_sim import LSTM_sim
from Window import Window

class Pipeline():

    def run_pipeline(self):
        #preprocessing steps
        win = Window(dataframe, 20, 1)
        for i in range(win.numberOfWindows()):
            window, judge = win.Next()
            #run detection on judge with window as training data window
    
        #data manipulation steps

        #LSTM Model
        model_sim = LSTM_sim()
        model_sim.simulation()

        #Model validity checks??

        #Testing Framework


###### RUN THE PIPELINE

pipeline = Pipeline()
pipeline.run_pipeline()
