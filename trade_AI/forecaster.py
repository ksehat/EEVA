import pandas as pd
import math
import pickle
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('df_15m.csv').dropna()
train_end = 8640*2
train = df.iloc[:-train_end,:]
# val = df.iloc[-4000:-2000,:]
test = df.iloc[-train_end:,:]
x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
# x_val = val[:][:-1]
# y_val = val[:][-1]
x_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]


model = rfc(class_weight={0:0.001,1:100})
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print('===========')
result_cm = confusion_matrix(y_test, y_pred)
win_rate = result_cm[1][1]/(result_cm[0][1] + result_cm[1][1])
profit = math.pow((1+0.015-0.0012),result_cm[1][1])*math.pow((1-0.005-0.0012),result_cm[0][1])
print('win rate is:',win_rate)
print('profit is:',profit)

# save the model to disk
pickle.dump(model, open('model.sav', 'wb'))
# load the model from disk
# loaded_model = pickle.load(open('model.sav', 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)