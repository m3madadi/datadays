import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from adtk.detector import PcaAD, PersistAD
from adtk.visualization import plot
from statsmodels.tsa.seasonal import STL

send_for_robo_epics = True


if send_for_robo_epics:
    input_path = os.environ.get('DATA_104_PATH') + '/test_data'
    output_path = '/output'
    drop_label = False
else:
    input_path = '/home/madadi/Project/datadays/data/'
    output_path = 'output'
    drop_label = True


# PrerProcess Data
def preprocess(df, drop_label):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp')
    # df = df.asfreq(freq='T')
    # df = df.interpolate()
    if drop_label:
        df = df.drop(['label'], axis=1)


    # scaler = StandardScaler()
    # np_scaled = scaler.fit_transform(df['value'].values.reshape(-1, 1))
    # df['scaled'] = np_scaled
    decompose = STL(df['value'], period=2).fit()
    df['resid'] = decompose.resid
    df['trend'] = decompose.trend
    df['seasonal'] = decompose.seasonal
    df['lag_1'] = df['value'].shift(1)
    df['lag_2'] = df['value'].shift(2)
    df['lag_3'] = df['value'].shift(3)
    df['lag_4'] = df['value'].shift(4)
    df.fillna(0, inplace=True)
    return df

# Each method should apply in a function

# Isolation Forest Method
def isolation_forest(df):
    outliers_fraction = float(.01)
    data = df[['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'trend', 'seasonal']]

    model =  IsolationForest(contamination=outliers_fraction)
    df['anomaly'] = model.fit_predict(data)
    df['label'] = np.where(df['anomaly']==-1, 1, 0)

    return df[['value', 'label']]


def pca(df):
    data = df[['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'trend', 'seasonal']]

    pca_ad = PcaAD(k=2)
    anomalies = pca_ad.fit_detect(data)
    df['label'] = np.where(anomalies==True, 1, 0)

    return df[['value', 'label']]


def persist(df):
    outliers_fraction = float(5.0)
    data = df[['value', 'lag_1', 'lag_2', 'resid']]

    persist_ad = PersistAD(c=outliers_fraction, side='both')
    persist_ad.window = 3
    anomalies = persist_ad.fit_detect(data)
    df['label'] = np.where(anomalies['resid']==True, 1, 0)

    return df[['value', 'label']]


class TimeSeriesAnomalyDetector:
    def __call__(self, df):
        processed_df = preprocess(df, drop_label)

        # final_df = isolation_forest(processed_df)
        # final_df = auto_encoder(processed_df)
        final_df = pca(processed_df)
        # final_df = persist(processed_df)

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

    #     f1_list.append(f1_score(input_df['label'], result['label']))
    # print(np.array(f1_list).mean())