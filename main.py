import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

# input_path = os.environ.get('DATA_104_PATH') + '/test_data'
output_path = 'output'

input_path = '/home/madadi/Project/datadays/data/'


# Each method should apply in a function
# Isolation Forest Method
def isolation_forest(df):
    scaler = StandardScaler()
    outliers_fraction = float(.01)
    np_scaled = scaler.fit_transform(df['value'].values.reshape(-1, 1))
    data = pd.DataFrame(np_scaled)
    model =  IsolationForest(contamination=outliers_fraction)
    df['label'] = model.fit_predict(data)
    df['label'].mask(df['label'] == 1, 0, inplace=True)
    df['label'].mask(df['label'] == -1, 1, inplace=True)
    return df



class TimeSeriesAnomalyDetector:
    def __call__(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        # df = df.set_index('timestamp')
        # df = df.asfreq(freq='T')
        df = df.drop(['label'], axis=1)
        # df = df.resample('H').mean().interpolate()
        # df = df.interpolate()

        final_df = isolation_forest(df)

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