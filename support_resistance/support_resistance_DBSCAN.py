import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_prep.data_hunter_futures import DataHunterFutures
from data_prep.data_hunter import DataHunter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = DataHunterFutures(symbol='ETHUSDT', start_date='1 Jan 2018', end_date='2024-04-08 00:00:00',
                       step='1h').prepare_data_online()
from_beginning = True
price_range_percent = 1
price_range_low = df.iloc[-1]['close'] - price_range_percent * df.iloc[-1]['close']
price_range_high = df.iloc[-1]['close'] + price_range_percent * df.iloc[-1]['close']
print(f'The support and resistance lines between {price_range_low} and {price_range_high}')
# if not from_beginning:
#     start_dp = df[df['timestamp'] == start_date].index[0]
#     df = df[start_dp:]
x_low_df = df[(df['low'] >= price_range_low - price_range_low * .001) & (
        df['low'] <= price_range_high + price_range_high * .001)][['timestamp', 'low']]
x_high_df = df[(df['high'] >= price_range_low - price_range_low * .001) & (
        df['high'] <= price_range_high + price_range_high * .001)][['timestamp', 'high']]
x_low = np.array(x_low_df['low']).reshape(-1, 1)
x_high = np.array(x_high_df['high']).reshape(-1, 1)
x_low_nor = StandardScaler().fit_transform(x_low)
x_high_nor = StandardScaler().fit_transform(x_high)

df_filter_nor = np.array(df[-1000:].filter(['open', 'high', 'low', 'close']))
# df_filter_nor = StandardScaler().fit_transform(df_filter)
db_low = DBSCAN(eps=0.005, min_samples=1)
db_low.fit(x_low_nor)
db_high = DBSCAN(eps=0.5, min_samples=1)
db_high.fit(x_high_nor)
db = DBSCAN(eps=0.5, min_samples=10)
db.fit(df_filter_nor)
x_low_df['labels'] = db_low.labels_
x_high_df['labels'] = db_high.labels_
# plt.scatter(df_filter_nor[:, 2], df_filter_nor[:, 1], c=db.labels_, s=40, cmap=plt.cm.Spectral)
# plt.show()
# plt.scatter(x_low, kmeans_low.labels_, c=kmeans_low.labels_, s=40, cmap=plt.cm.Spectral)
# plt.show()
