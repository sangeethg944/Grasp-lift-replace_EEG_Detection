from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def AddNoActivity(data):
    no_activity = []
    i = 0
    while i < len(data):
        if sum(data.iloc[i][34:]) == 0:
            no_activity.append(1)
            i = i + 1
        else:
            no_activity.append(0)
            i = i + 1
    data['NoActivity'] = no_activity

    return data


def scaling(data):
    cols = data.columns
    scaler = StandardScaler()
    data = np.array(data.astype(float))
    data1 = scaler.fit_transform(data)
    data = pd.DataFrame(data1, columns = cols)
    return data

