import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from darts.utils.statistics import extract_trend_and_seasonality, ModelMode, remove_from_series, check_seasonality, remove_trend
from darts import TimeSeries
from statsmodels.tsa.seasonal import STL, seasonal_decompose

# input_path = os.environ.get('DATA_104_PATH') + '/test_data'
input_path = '/Users/madadi/Projects/datadays/data'
output_path = '/output'


class TimeSeriesAnomalyDetector:
    def __call__(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp')
        df = df.asfreq(freq='60S')
        df = df.drop(['label'], axis=1)
        df = df.resample('H').mean().interpolate()

        # res = STL(series).fit()
        # res.plot()
        # plt.show()

        scaler = StandardScaler()
        series = TimeSeries.from_dataframe(df)
        is_sesonal, seasonality_period = check_seasonality(ts = series, m = None, max_lag = 100, alpha = 0.05)
        if is_sesonal:
            trend, seasonality = extract_trend_and_seasonality(ts = series, model = ModelMode.ADDITIVE, method = "STL", freq = seasonality_period)
            series = remove_from_series(ts = series, other = trend, model = ModelMode.ADDITIVE)
            series = remove_from_series(ts = series, other = seasonality, model = ModelMode.ADDITIVE)
        else:
            series = remove_trend(ts = series, model = ModelMode.ADDITIVE, method = "STL")
        

        outliers_fraction = float(.01)
        dataframe_style = series.pd_dataframe() 
        np_scaled = scaler.fit_transform(dataframe_style.value.to_numpy().reshape(-1, 1))
        data = pd.DataFrame(np_scaled)

        model =  IsolationForest(contamination=outliers_fraction)
        model.fit(data)
        dataframe_style['label'] = model.predict(data)


        return dataframe_style
    
anomaly_detector = TimeSeriesAnomalyDetector()


if __name__ == '__main__':
    for filename in os.listdir(input_path):
        input_df = pd.read_csv(os.path.join(input_path, filename))
        print(filename, len(input_df))
        result = anomaly_detector(input_df)
        result.to_csv(os.path.join(output_path, filename))
        print(f'item {filename} processed.')