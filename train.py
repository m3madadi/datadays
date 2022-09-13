import os
import pandas as pd
import numpy as np
# from pyod.models.xgbod import XGBOD
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, recall_score, precision_score
import joblib


input_path = '/home/madadi/Project/datadays/data/'


def plot_outlier(predicted_df):
    fig, ax = plt.subplots(figsize=(12,4))
    outlier_points = predicted_df.loc[predicted_df['prediction'] == 1] #anomaly
    ax.plot(predicted_df.index, predicted_df['value'], color='blue', label = 'Normal')
    ax.scatter(outlier_points.index, outlier_points['value'], color='red', label = 'Anomaly')
    plt.legend()
    plt.show()

def print_metrics(original_df, predicted_df):
    print("F1 Score:", f1_score(original_df['label'], predicted_df['prediction']))
    print("Recall:", recall_score(original_df['label'], predicted_df['prediction']))
    print("Precision:", precision_score(original_df['label'], predicted_df['prediction']))


def process_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp')
    # df = df.asfreq(freq='T')
    # df = df.interpolate()
    df['label'] = df.label.astype(int)

    df['lag_1'] = df['value'].shift(1)
    df['lag_2'] = df['value'].shift(2)
    df['lag_3'] = df['value'].shift(3)
    df['lag_4'] = df['value'].shift(4)
    df.fillna(0, inplace=True)

    decompose = STL(df['value'], period=2).fit()
    df['resid'] = decompose.resid
    df['trend'] = decompose.trend
    df['seasonal'] = decompose.seasonal

    return df


# xgb_clf = XGBOD(n_jobs=8, silent=False)
if_clf = IsolationForest(max_samples=0.9, warm_start=True)
for filename in os.listdir(input_path):
    input_df = pd.read_csv(os.path.join(input_path, filename))
    df = process_data(input_df)
    data = df[['value', 'lag_1', 'lag_2', 'resid']]
    if_clf.fit(data)
    if_clf.n_estimators += 10
joblib.dump(if_clf, 'if_detector.sav')

loaded_clf = joblib.load('if_detector.sav')
input_df = pd.read_csv('data/0.csv')

df = process_data(input_df)
data = df[['value', 'lag_1', 'lag_2', 'resid']]

prediction_df = data.copy()
# prediction_df['score'] = xgb_clf.decision_scores_ # outlier score
prediction_df['anomaly'] = loaded_clf.predict(data)
prediction_df['prediction'] = np.where(prediction_df['anomaly']==-1, 1, 0)
# prediction_df['prediction'].value_counts()
# df['label'].value_counts()

# plot_outlier(prediction_df)

print_metrics(df, prediction_df)