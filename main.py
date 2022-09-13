import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from pyod.models.knn import KNN
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
# from adtk.detector import PcaAD, PersistAD
# from adtk.visualization import plot
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm

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
    decompose = STL(df['value'], period=4).fit()
    df['resid'] = decompose.resid
    df['trend'] = decompose.trend
    df['seasonal'] = decompose.seasonal
    df['lag_1'] = df['value'].shift(1)
    df['lag_2'] = df['value'].shift(2)
    df['lag_3'] = df['value'].shift(3)
    df['lag_4'] = df['value'].shift(4)
    df['lag+1'] = df['value'].shift(-1)
    df['lag+2'] = df['value'].shift(-2)
    df['lag+3'] = df['value'].shift(-3)
    df.fillna(0, inplace=True)
    return df

# Each method should apply in a function

# Isolation Forest Method
def isolation_forest(df):
    outliers_fraction = float(.06)
    data = df[['value', 'resid', 'lag_1', 'lag_2', 'lag_3','lag_4','seasonal']]

    # new_data = PCA().fit_transform(data)
    # arima = sm.tsa.arima.ARIMA(data['value'], order=(9, 1, 4)).fit()
    # data['arima_resid'] = arima.resid

    model =  IsolationForest(contamination=outliers_fraction, n_estimators=50)
    df['anomaly'] = model.fit_predict(new_data)
    df['label'] = np.where(df['anomaly']==-1, 1, 0)

    return df[['value', 'label']]

def ecod(df):
    outliers_fraction = float(.06)
    data = df[['value', 'resid', 'lag_1', 'lag_2', 'lag_3','lag_4','seasonal']]

    # new_data = PCA(n_components=3).fit_transform(data)
    # arima = sm.tsa.arima.ARIMA(data['value'], order=(9, 1, 4)).fit()
    # data['arima_resid'] = arima.resid

    model =  ECOD(contamination=outliers_fraction)
    df['anomaly'] = model.fit_predict(data)
    df['label'] = np.where(df['anomaly']==1, 1, 0)

    return df[['value', 'label']]


def knn(df):
    outliers_fraction = float(.06)
    data = df[['value', 'resid', 'lag_1', 'lag_2', 'lag_3','lag_4','seasonal']]

    model =  KNN(contamination=outliers_fraction)
    df['anomaly'] = model.fit_predict(data)
    df['label'] = np.where(df['anomaly']==1, 1, 0)

    return df[['value', 'label']]


def pca(df):
    data = df[['value', 'resid', 'lag_1', 'lag_2', 'lag_3','lag_4','seasonal']]

    pca_ad = PcaAD(k=2, c=2.5)
    anomalies = pca_ad.fit_detect(data)
    df['label'] = np.where(anomalies==True, 1, 0)

    return df[['value', 'label']]


def persist(df):
    outliers_fraction = float(3.0)
    data = df[['value', 'lag_1', 'lag_2', 'resid']]

    persist_ad = PersistAD(c=outliers_fraction, side='both', window=20)
    anomalies = persist_ad.fit_detect(data)
    df['label'] = np.where(anomalies['value']==True, 1, 0)

    return df[['value', 'label']]


class TimeSeriesAnomalyDetector:
    def __call__(self, df):
        processed_df = preprocess(df, drop_label)

        # final_df = isolation_forest(processed_df)
        final_df = ecod(processed_df)
        # final_df = copod(processed_df)
        # final_df = auto_encoder(processed_df)
        # final_df = pca(processed_df)
        # final_df = persist(processed_df)
        # final_df = knn(processed_df)

        return final_df
    
anomaly_detector = TimeSeriesAnomalyDetector()


if __name__ == '__main__':
    f1_list = []
    precision_list = []
    recall_list = []
    for filename in os.listdir(input_path):
        input_df = pd.read_csv(os.path.join(input_path, filename))
        print(filename, len(input_df))
        result = anomaly_detector(input_df)
        result.to_csv(os.path.join(output_path, filename))
        print(f'item {filename} processed.')

    #     f1_list.append(f1_score(input_df['label'], result['label']))
    #     precision_list.append(precision_score(input_df['label'], result['label']))
    #     recall_list.append(recall_score(input_df['label'], result['label']))
    # print("F1 Score:", np.array(f1_list).mean())
    # print("Precision Score:", np.array(precision_list).mean())
    # print("Recall Score:", np.array(recall_list).mean())




# Reports
# IsolationForest(contamination=0.05) -> Final Score: 0.42531 - Features: ['value', 'resid', 'lag_1', 'lag_2']
# IsolationForest(contamination=0.09) -> Final Score: 0.40677 - Features: ['value', 'resid', 'lag_1', 'lag_2']
# IsolationForest(contamination=0.07) -> Final Score: 0.39193 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'trend', 'seasonal']
# IsolationForest(contamination=0.07) -> Final Score: 0.43533 - Features: ['value', 'resid', 'lag_1', 'lag_2']
# IsolationForest(contamination=0.07) -> Final Score: 0.43346 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4']
# IsolationForest(contamination=0.07) -> Final Score: 0.41407 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'trend']
# IsolationForest(contamination=0.06) -> Final Score: 0.44024 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3']
# IsolationForest() -> Final Score: 0.29377 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3']
# IsolationForest(contamination=0.06, n_estimators=500) -> Final Score: 0.42656 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3']
# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.45001 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3']
# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.45423 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'seasonal']
# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.42298 - Features: ['value', 'lag_1', 'lag_2', 'lag_3', 'seasonal']
# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.44895 - Features: ['resid', 'lag_1', 'lag_2', 'lag_3', 'seasonal']
# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.45074 - Features: ['scaled', 'resid', 'lag_1', 'lag_2', 'lag_3', 'seasonal'] (All features scaled except resid & seasonal)
# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.43733 - Features: ['scaled', 'resid', 'lag_1', 'lag_2', 'lag_3', 'seasonal'] (All features scaled)
# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.47006 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'seasonal'] - STL period=4 ### BEST ###
# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.44298 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'seasonal'] - STL period=9
# IsolationForest(contamination=0.06, n_estimators=25) -> Final Score: 0.45929 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'seasonal'] - STL period=4
# IsolationForest(contamination=0.05, n_estimators=50) -> Final Score: 0.36865 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'seasonal'] - STL period=4
# IsolationForest(contamination=0.06, n_estimators=200, max_samples=1500) -> Final Score: 0.40370 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'seasonal'] - STL period=4
# IsolationForest(contamination=0.065, n_estimators=100, max_samples=50) -> Final Score: 0.42349 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'seasonal'] - STL period=4
# IsolationForest(contamination=0.06, n_estimators=50, max_samples=50) -> Final Score: 0.41291 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'seasonal'] - STL period=4
# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.45638 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3','lag_4','seasonal', 'lag+1', 'lag+2', 'lag+3'] - STL period=4

# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.41467 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3','lag_4', 'seasonal', 'pca'] - STL period=4
# IsolationForest(contamination=0.06, n_estimators=50) -> Final Score: 0.38620 - Features: ['value', 'pca' - 'value'] - STL period=4
# IsolationForest(contamination=0.05, n_estimators=50) -> Final Score: 0.33792 - Features: ['pca1', 'pca2', 'pca3'] - STL period=4


# PcaAd(k=2, c=3.0) -> Final Score: 0.20631 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3']

# PersistAD(c=3.0, side='both', window=20) -> Final Score: 0.22138 - Features: ['value']

# KNN(contamination=0.06) -> Final Score: 0.26736 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3']

# COPOD(contamination=0.06) -> Final Score: 0.26780 - Features: ['value', 'resid', 'lag_1', 'lag_2', 'lag_3', 'seasonal']