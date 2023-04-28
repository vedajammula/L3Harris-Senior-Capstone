import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

def get_LOF(n, window, nextWindow):
    window['Dates'] = pd.to_datetime(window.index)
    window["Dates"] = window["Dates"].apply(lambda x: x.toordinal())    
    nextWindow['Dates'] = pd.to_datetime(nextWindow.index)
    nextWindow['Dates'] = nextWindow['Dates'].apply(lambda x: x.toordinal())

    win = window.loc[:,['Dates', 'Close']].to_numpy()
    next = nextWindow.loc[:,['Dates', 'Close']].to_numpy()

    if int(win.size/2) < n:
        n = int(win.size/2)
    clf = LocalOutlierFactor(n_neighbors=n, novelty=True)
    clf.fit(win)
    pred = clf.predict(next)
    outliers = np.nonzero(pred == -1)
    return outliers