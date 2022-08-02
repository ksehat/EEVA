import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_prep.data_hunter_futures import DataHunterFutures
from data_prep.data_hunter import DataHunter
from sklearn.cluster import KMeans
from utils.utils import equal_date_pointer


def find_nearest_trend_line(df, df_highs_center_importance, df_lows_center_importance):
    highs_price_trend_diff_df = df.iloc[-1]['close'] - df_highs_center_importance['center']
    lower_high_idx = highs_price_trend_diff_df[highs_price_trend_diff_df >= 0].idxmin()
    higher_high_idx = highs_price_trend_diff_df[highs_price_trend_diff_df <= 0].idxmax()
    lows_price_trend_diff_df = df.iloc[-1]['close'] - df_lows_center_importance['center']
    lower_low_idx = lows_price_trend_diff_df[lows_price_trend_diff_df >= 0].idxmin()
    higher_low_idx = lows_price_trend_diff_df[lows_price_trend_diff_df <= 0].idxmax()
    lower_high = df_highs_center_importance.iloc[lower_high_idx]['center']
    higher_high = df_highs_center_importance.iloc[higher_high_idx]['center']
    lower_low = df_lows_center_importance.iloc[lower_low_idx]['center']
    higher_low = df_lows_center_importance.iloc[higher_low_idx]['center']
    return lower_high, higher_high, lower_low, higher_low


def opt_num_clusters(wcss, k_means_diff_percent):
    optimum_k = len(wcss) - 1
    for i in range(0, len(wcss) - 1):
        diff = abs(wcss[i + 1] - wcss[i]) / wcss[i]
        if diff < k_means_diff_percent:
            optimum_k = i
            break
    print("Optimum K is " + str(optimum_k + 1))
    optimum_clusters = optimum_k + 1
    return optimum_clusters


def equal_dp(df, ts, dp_old):
    try:
        dp = df[df['timestamp'] == ts].index.values[0]
    except:
        print('the time stamp cannot found.')
        dp = dp_old + 1
    dp_old = dp
    return dp, dp_old


def sup_res_finder(df, dp, from_beginning, start_date, price_range_percent,
                   k_means_diff_percent):
    df = df
    price_range_low = df.iloc[-1]['close'] - price_range_percent * df.iloc[-1]['close']
    price_range_high = df.iloc[-1]['close'] + price_range_percent * df.iloc[-1]['close']
    print(f'The support and resistance lines between {price_range_low} and {price_range_high}')
    if not from_beginning:
        start_dp = df[df['timestamp'] == start_date].index[0]
        df = df[start_dp:]
    x_low = df[(df['low'] >= price_range_low - price_range_low * .001) & (
            df['low'] <= price_range_high + price_range_high * .001)]['low']
    x_high = df[(df['high'] >= price_range_low - price_range_low * .001) & (
            df['high'] <= price_range_high + price_range_high * .001)]['high']
    x_low = np.array(x_low).reshape(-1, 1)
    x_high = np.array(x_high).reshape(-1, 1)
    x_all = np.concatenate((x_high, x_low))
    wcss_high = []
    wcss_low = []
    wcss_all = []
    k_models_high = []
    k_models_low = []
    k_models_all = []

    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(x_high)
        wcss_high.append(kmeans.inertia_)
        k_models_high.append(kmeans)
        kmeans = KMeans(n_clusters=i, random_state=0).fit(x_low)
        wcss_low.append(kmeans.inertia_)
        k_models_low.append(kmeans)
        kmeans = KMeans(n_clusters=i, random_state=0).fit(x_all)
        wcss_all.append(kmeans.inertia_)
        k_models_all.append(kmeans)

    k_opt_high = opt_num_clusters(wcss_high, k_means_diff_percent)
    k_opt_low = opt_num_clusters(wcss_low, k_means_diff_percent)
    k_opt_all = opt_num_clusters(wcss_all, k_means_diff_percent)

    # print(df[-win_width:-1])
    kmeans_high = KMeans(n_clusters=k_opt_high, random_state=0).fit(x_high)
    highs_list = kmeans_high.cluster_centers_
    highs_list = [x[0] for x in highs_list]
    df_x_high_labels = pd.DataFrame({
        'x_high': x_high.reshape(-1),
        'label': kmeans_high.labels_,
    })
    highs_min_list = []
    highs_max_list = []
    highs_center_list = []
    num_of_members_list = []
    for l in np.unique(kmeans_high.labels_):
        # highs_min_list.append(min(df_x_high_labels['x_high'][df_x_high_labels['label'] == l]))
        # highs_max_list.append(max(df_x_high_labels['x_high'][df_x_high_labels['label'] == l]))
        highs_center_list.append(kmeans_high.cluster_centers_[l][0])
        num_of_members_list.append(len(df_x_high_labels['x_high'][df_x_high_labels['label'] == l]))
    df_highs_center_importance = pd.DataFrame({
        'center': highs_center_list,
        'num of members': num_of_members_list
    })
    df_highs_center_importance.sort_values(by='center', ascending=True, inplace=True)
    print('====================')
    print(df_highs_center_importance)
    print('====================')
    kmeans_low = KMeans(n_clusters=k_opt_low, random_state=0).fit(x_low)
    lows_list = kmeans_low.cluster_centers_
    lows_list = [x[0] for x in lows_list]
    df_x_low_labels = pd.DataFrame({
        'x_low': x_low.reshape(-1),
        'label': kmeans_low.labels_,
    })
    lows_min_list = []
    lows_max_list = []
    lows_center_list = []
    num_of_members_list = []
    for l in np.unique(kmeans_low.labels_):
        # lows_min_list.append(min(df_x_low_labels['x_low'][df_x_low_labels['label'] == l]))
        # lows_max_list.append(max(df_x_low_labels['x_low'][df_x_low_labels['label'] == l]))
        lows_center_list.append(kmeans_low.cluster_centers_[l][0])
        num_of_members_list.append(len(df_x_low_labels['x_low'][df_x_low_labels['label'] == l]))
    df_lows_center_importance = pd.DataFrame({
        'center': lows_center_list,
        'num of members': num_of_members_list
    })
    df_lows_center_importance.sort_values(by='center', ascending=True, inplace=True)
    print(df_lows_center_importance)
    print('====================')

    kmeans_all = KMeans(n_clusters=k_opt_all, random_state=0).fit(x_all)
    all_list = kmeans_all.cluster_centers_
    all_list = [x[0] for x in all_list]
    df_x_all_labels = pd.DataFrame({
        'x_all': x_all.reshape(-1),
        'label': kmeans_all.labels_,
    })
    alls_center_list = []
    num_of_mem_list = []
    for l in np.unique(kmeans_all.labels_):
        alls_center_list.append(kmeans_all.cluster_centers_[l][0])
        num_of_mem_list.append(len(df_x_all_labels[df_x_all_labels['label'] == l]))
    df_alls_center_importance = pd.DataFrame({
        'center': alls_center_list,
        'num of members': num_of_mem_list
    })
    df_alls_center_importance.sort_values(by='center', ascending=True, inplace=True)
    print(df_alls_center_importance)
    print('====================')

    lower_high, higher_high, lower_low, higher_low = find_nearest_trend_line(df,
                                                                             df_highs_center_importance,
                                                                             df_lows_center_importance)
    print(f'lower_high = {lower_high}')
    print(f'higher_high = {higher_high}')
    print(f'lower_low = {lower_low}')
    print(f'higher_low = {higher_low}')

    # plt.scatter(x_high, kmeans_high.labels_, c=kmeans_high.labels_, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    # plt.scatter(x_low, kmeans_low.labels_, c=kmeans_low.labels_, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    return lower_high, higher_high, lower_low, higher_low


def main():
    dp_old = 0
    df = DataHunterFutures(symbol='ETHUSDT', start_date='01 Nov 2021',
                           end_date='2022-06-04 00:00:00', step='4h').prepare_data()
    # ts = '2022-05-07 04:00:00'
    # dp, dp_old = equal_dp(df, ts, dp_old)
    dp = -1
    lower_high, higher_high, lower_low, higher_low = sup_res_finder(df, dp, from_beginning=True,
                                                                    start_date='2021-12-01 00:00:00',
                                                                    price_range_percent=.2,
                                                                    k_means_diff_percent=0.3)
    below_trend = max(lower_high, lower_low)
    upper_trend = min(higher_high, higher_low)
    print('====================')
    print(f'below_trend = {below_trend}')
    print(f'upper_trend = {upper_trend}')


if __name__ == '__main__':
    main()
