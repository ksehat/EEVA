import os
from xgboost import XGBClassifier as xgb
import pandas as pd

files_name = [x for x in os.listdir() if 'Classified' in x]
for file_name in files_name:
    df1 = pd.read_csv(file_name, index_col=0)
    X = df1.iloc[100:, :-4]
    Y = df1.iloc[100:, -1]
    x_train = X[:-100]
    y_train = Y[:-100]
    x_test = X[-100:]
    y_test = Y[-100:]
    model = xgb(scale_pos_weight=.82)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    result_df = pd.DataFrame({
        'PL': df1.iloc[-100:, -2],
        'y_test': y_test,
        'pred': pred
    })
    result_df.to_csv('results_df.csv')
    print(pred)
