import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score
# from pyod.models.auto_encoder import AutoEncoder


# input_path = os.environ.get('DATA_104_PATH') + '/test_data'
output_path = 'output'

input_path = '/home/madadi/Project/datadays/data/'


# PrerProcess Data
def preprocess(df, drop_label):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp')
    # df = df.asfreq(freq='T')
    # df = df.interpolate()
    if drop_label:
        df = df.drop(['label'], axis=1)
    else:
        df['label'] = df.label.astype(int)


    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df['value'].values.reshape(-1, 1))
    df['scaled'] = np_scaled
    df['lag_1'] = df['value'].shift(1)
    df['lag_2'] = df['value'].shift(2)
    df.fillna(0, inplace=True)
    return df

# Each method should apply in a function

# Isolation Forest Method
def isolation_forest(df):
    outliers_fraction = float(.01)
    data = df[['value']]

    model =  IsolationForest(contamination=outliers_fraction)
    df['anomaly'] = model.fit_predict(data)
    df['label'] = np.where(df['anomaly']==-1, 1, 0)

    return df

# DBSCAN Method
def auto_encoder(df):
    # data = df['scaled'].to_frame()

    # Load Trained AutoEncoder Model


    # ae_clf = AutoEncoder(hidden_neurons =[1, 10, 10, 1], epochs=2)
    # ae_clf.fit(data)
    # data['score'] = ae_clf.decision_scores_ # outlier score
    # df['label'] = np.where(data['score'] < 1.54, 0, 1)

    # df['label'] = np.where(df['anomaly']==-1, 1, 0)

    return df



class TimeSeriesAnomalyDetector:
    def __call__(self, df):
        processed_df = preprocess(df, True)

        final_df = isolation_forest(processed_df)
        # final_df = auto_encoder(processed_df)

        return final_df
    
anomaly_detector = TimeSeriesAnomalyDetector()


if __name__ == '__main__':
    f1_list = []
    for filename in os.listdir(input_path):
        input_df = pd.read_csv(os.path.join(input_path, filename))
        print(filename, len(input_df))
        result = anomaly_detector(input_df)
        result.to_csv(os.path.join(output_path, filename))
        print(f'item {filename} processed.')

        f1_list.append(f1_score(input_df['label'], result['label']))
    print(np.array(f1_list).mean())