from LSTM_sim import LSTM_sim
from Window import Window

class Pipeline():

    def run_pipeline(self):
        #preprocessing steps
        win = Window(dataframe, 20, 1)
        for i in range(win.numberOfWindows()):
            window, judge = win.Next()
            holes = detect_hole(judge)
            #run detection on judge with window as training data window
            #use df interpolate
            #returns modified dataframe cleaned to be resubmitted to the window
            #run fill missing value on cleaned df variable
            win.accepted(cleaned)
    
        #data manipulation steps

        #LSTM Model
        model_sim = LSTM_sim()
        model_sim.simulation()

        #Model validity checks??

        #Testing Framework


###### RUN THE PIPELINE

pipeline = Pipeline()
pipeline.run_pipeline()
