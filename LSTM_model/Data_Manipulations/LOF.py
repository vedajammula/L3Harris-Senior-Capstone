import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

def get_LOF(n, window, nextWindow):
    if int(window.size/2) < n:
        n = int(window.size/2)
    clf = LocalOutlierFactor(n_neighbors=n, novelty=True)
    clf.fit(window)
    pred = clf.predict(nextWindow)
    outliers = np.nonzero(pred == -1)
    return outliers